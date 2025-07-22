"""
从 Swin 推理得到 token, 并将 token 转化为图像 patch 格式
Dense ViT features proposed in "Vision Transformers for Dense Prediction"
Modified from: https://github.com/intel-isl/DPT
"""
import torch
import torch.nn as nn
import types
import math
import torch.nn.functional as F


# vit 前向推理, 并将中间层结果转化为图像, 然后输出
def forward_swin(pretrained, x):
    """
    pretrained: including model & post-process
    x: input in image shape: b, 3, h, w
    """
    b, c, h, w = x.shape

    # ViT 前向推理, glob 为x对应的最终特征, (B,N,C)
    pretrained.model.eval()     # 冻结 dropout & drop path
    with torch.no_grad():
        final, intermediates = pretrained.model.forward_flex(x)

    # 将 token 转为图像, 得到的 layer 特征大小为 [1/8, 1/16, 1/32, 1/32], c=[128, 256, 512, 1024]
    unflatten1 = nn.Unflatten(2, torch.Size([
        h // pretrained.model.patch_size[1] // 2 ** 1,
        w // pretrained.model.patch_size[0] // 2 ** 1,
    ]))
    unflatten2 = nn.Unflatten(2, torch.Size([
        h // pretrained.model.patch_size[1] // 2 ** 2,
        w // pretrained.model.patch_size[0] // 2 ** 2,
    ]))
    unflatten3 = nn.Unflatten(2, torch.Size([
        h // pretrained.model.patch_size[1] // 2 ** 3,
        w // pretrained.model.patch_size[0] // 2 ** 3,
    ]))

    layer_1 = unflatten1(intermediates[0].transpose(1,2))
    layer_2 = unflatten2(intermediates[1].transpose(1,2))
    layer_3 = unflatten3(intermediates[2].transpose(1,2))
    layer_4 = unflatten3(intermediates[3].transpose(1,2))
    intermediates = [layer_1, layer_2, layer_3, layer_4]
    final = unflatten3(final.transpose(1,2))

    return final, intermediates


# position embedding, 根据输入进行 resize
# def _resize_pos_embed(self, posemb, gs_h, gs_w):
#     posemb_tok, posemb_grid = (
#         posemb[:, : self.start_index],
#         posemb[0, self.start_index :],
#     )
#
#     gs_old = int(math.sqrt(len(posemb_grid)))
#
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
#
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#
#     return posemb


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    # 使用 dinov2 的小改进
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))
    if gs_old**2 == gs_h*gs_w and gs_h == gs_w:     # 和训练情况一致, 无需插值
        return posemb

    h_scale = (gs_h + 0.1) / math.sqrt(len(posemb_grid))
    w_scale = (gs_w + 0.1) / math.sqrt(len(posemb_grid))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid,
        scale_factor=(h_scale, w_scale),
        mode="bicubic"
    )
    assert gs_h == posemb_grid.shape[-2] and gs_w == posemb_grid.shape[-1]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    patch_W, patch_H = self.patch_size
    assert x.shape[2] % patch_H == 0, f"Input image height {x.shape[2]} is not a multiple of patch height {patch_H}"
    assert x.shape[3] % patch_W == 0, f"Input image width {x.shape[3]} is not a multiple of patch width: {patch_W}"

    # normalize: ImageNet归一化参数与当前归一化结果一致, 无需转换

    # 参考 Swin 源代码, 首先对输入进行 patch embedding
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) # (B, Ph*Pw, C)
    if getattr(self.patch_embed, "norm", None) is not None:
        x = self.patch_embed.norm(x)

    # position embedding
    if self.ape:
        pos_embed = self._resize_pos_embed(
            self.absolute_pos_embed, h // self.patch_size[1], w // self.patch_size[0]
        )
        x = x + pos_embed
    x = self.pos_drop(x)

    # 将 [cls] token 加入到 token 序列中: Swin Transformer 不存在 [CLS] Token, 无此处理, 跳过
    # pass

    # 前馈
    intermediates = []
    for i, stage in enumerate(self.layers):
        x = stage(x, h//(patch_H*2**i), w//(patch_W*2**i))
        intermediates.append(x)     # B, L, C, c=embeded dim
    x = self.norm(x)                # B, L, C

    return x, intermediates


# 对预定义的网络增加中间输出节点, 并将 token unflatten 成为图像
def _make_swin_b16_backbone(
    model,
    psize=[16, 16],                     # patch size
    hooks=[2, 5, 8, 11],                # 在 hocks 的位置定义输出
    vit_features=768,                   # transformer blocks 的 dim
    use_readout="ignore",               # 如何处理 [cls] 与其他 token 的关系
    start_index=0,                      # [cls] 起始位置, swin 没有 [CLS] token
    enable_attention_hooks=False,
    aggregation="sparse",
):
    pretrained = nn.Module()

    # 将模型传给 pretrained 模型容器, 并插入输出层, 由于 swin 对每个尺度的层数做了封装, 因此 hooks=[1,1,1,1]
    pretrained.model = model
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = psize
    pretrained.model.hooks = hooks
    pretrained.model.aggregation = aggregation
    pretrained.patch_size = psize[0]
    pretrained.vit_features = vit_features

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_swinv1_base_384(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    from .swin_transformer import SwinTransformer
    from .config import get_config
    config = get_config("Models/UniFormer/extractor/Swinv1/configs/swin_base_patch4_window12_384_finetune.yaml")
    model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            num_classes=21841,  # 21841 for ImageNet-22k, 1000 for ImageNet-1k
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN.APE,
                            norm_layer=nn.LayerNorm,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=False,
                            fused_window_process=config.FUSED_WINDOW_PROCESS)

    # 加载权重
    path = "foundation_model_weights/swin_base_patch4_window12_384_22k.pth"
    states = torch.load(path, map_location="cpu")
    weights = states['model']
    model.load_state_dict(weights, strict=True)

    # 保证 hooks 的划分与原生的 stages 划分一致
    hooks = []
    for i in range(4):
        hooks.append(sum(config.MODEL.SWIN.DEPTHS[:i+1]) - 1)

    vit_features = [256, 512, 1024, 1024]

    return _make_swin_b16_backbone(
        model,
        psize=[config.MODEL.SWIN.PATCH_SIZE] * 2,
        hooks=hooks,
        vit_features=vit_features,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )

def _make_pretrained_swinv1_base_224(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    from .swin_transformer import SwinTransformer
    from .config import get_config
    config = get_config("Models/UniFormer/extractor/Swinv1/configs/swin_base_patch4_window7_224_22k.yaml")
    model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            num_classes=21841,  # 21841 for ImageNet-22k, 1000 for ImageNet-1k
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN.APE,
                            norm_layer=nn.LayerNorm,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=False,
                            fused_window_process=config.FUSED_WINDOW_PROCESS)

    # 加载权重
    path = "foundation_model_weights/swin_base_patch4_window7_224_22k.pth"
    states = torch.load(path, map_location="cpu")
    weights = states['model']
    model.load_state_dict(weights, strict=True)

    # 保证 hooks 的划分与原生的 stages 划分一致
    hooks = []
    for i in range(4):
        hooks.append(sum(config.MODEL.SWIN.DEPTHS[:i+1]) - 1)

    vit_features = [256, 512, 1024, 1024]

    return _make_swin_b16_backbone(
        model,
        psize=[config.MODEL.SWIN.PATCH_SIZE] * 2,
        hooks=hooks,
        vit_features=vit_features,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
