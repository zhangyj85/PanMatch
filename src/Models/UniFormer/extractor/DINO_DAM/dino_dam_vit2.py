"""
从 ViT 推理得到 token, 并将 token 转化为图像 patch 格式
Dense ViT features proposed in "Vision Transformers for Dense Prediction"
Modified from: https://github.com/intel-isl/DPT
"""
import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F


activations = {}    # 全局变量, 用来给 get_activation() 存放 hock 输出


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}      # 全局变量, 用来给 get_attention() 存放 hock 输出


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


# vit 前向推理, 并将中间层结果转化为图像, 然后输出
def forward_vit(pretrained, x):
    """
    pretrained: including model & post-process
    x: input in image shape: b, 3, h, w
    """
    b, c, h, w = x.shape

    # ViT 前向推理, glob 为x对应的最终特征, (B,N,C)
    # pretrained.model.eval()     # 冻结 dropout & drop path
    # with torch.no_grad():
    final, intermediates = pretrained.model.forward_flex(
        x,
        hooks=pretrained.hooks,
        aggregation=pretrained.aggregation
    )

    unflatten = nn.Unflatten(
        2,
        torch.Size(
            [
                h // pretrained.patch_size,
                w // pretrained.patch_size,
            ]
        ),
    )

    # 丢弃 [CLS] token, BNC -> BCN
    post_final = pretrained.act_postprocess(final)
    # 将 token 转为图像, 得到的 layer 特征统一为 1/16, c=Dim
    post_final = unflatten(post_final)
    post_intermediates = []
    for feat in intermediates:
        post_feat = pretrained.act_postprocess(feat)
        post_feat = unflatten(post_feat)
        post_intermediates.append(post_feat.contiguous())

    return post_final.contiguous(), post_intermediates


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


# def forward_flex(self, x, hooks=[1,3,5,7], aggregation="sparse"):
#     b, c, h, w = x.shape

#     # 对输入进行 position embedding (适应任意方形图像)
#     pos_embed = self._resize_pos_embed(
#         self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
#     )

#     B = x.shape[0]

#     # 部分网络对 patch embedding 采用了 resnet
#     if hasattr(self.patch_embed, "backbone"):
#         x = self.patch_embed.backbone(x)
#         if isinstance(x, (list, tuple)):
#             x = x[-1]  # last feature if backbone outputs list/tuple of features

#     patch_W, patch_H = self.patch_size
#     assert x.shape[2] % patch_H == 0, f"Input image height {x.shape[2]} is not a multiple of patch height {patch_H}"
#     assert x.shape[3] % patch_W == 0, f"Input image width {x.shape[3]} is not a multiple of patch width: {patch_W}"
#     x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

#     # 将 [cls] token 加入到 token 序列中
#     if getattr(self, "dist_token", None) is not None:
#         cls_tokens = self.cls_token.expand(
#             B, -1, -1
#         )  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#     else:
#         cls_tokens = self.cls_token.expand(
#             B, -1, -1
#         )  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)

#     x = x + pos_embed
#     if hasattr(self, "pos_drop"):   # dinov2 没有 pos_drop
#         x = self.pos_drop(x)

#     intermediates = []
#     accumulator = 0
#     num_accumulated = 0
#     for i, blk in enumerate(self.blocks):
#         x = blk(x)
#         if aggregation == "dense":
#             accumulator = accumulator + x
#             num_accumulated += 1
#         if i in hooks:
#             if aggregation == "dense":
#                 x_ = accumulator / num_accumulated
#                 num_accumulated = 0
#                 accumulator = 0
#             else:
#                 x_ = x
#             intermediates.append(x_)

#     x = self.norm(x)

#     return x, intermediates


@torch.no_grad()
def forward_flex(self, x, hooks=[1,3,5,7], aggregation="sparse"):
    b, c, h, w = x.shape

    # 对输入进行 position embedding (适应任意方形图像)
    x = self.prepare_tokens_with_masks(x, masks=None)

    intermediates = []
    accumulator = 0
    num_accumulated = 0
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if aggregation == "dense":
            accumulator = accumulator + x
            num_accumulated += 1
        if i in hooks:
            if aggregation == "dense":
                x_ = accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
            else:
                x_ = x
            intermediates.append(x_)

    x = self.norm(x)

    return x, intermediates


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


from foundation_model_repos.facebookresearch_dinov2_main.dinov2.layers.attention import Attention
class PytorchMemEffAttention(Attention):
    # 直接继承 dinov2 原本的 attn, 因此在属性上不存在差异, 参数可以正常加载, 仅影响前馈的具体计算过程
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p).transpose(1, 2)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def use_memeffattn_in_model(model):

    for i in range(len(model.blocks)):

        slow_attn = model.blocks[i].attn

        meff_attn = PytorchMemEffAttention(
            dim=slow_attn.qkv.in_features,
            num_heads=slow_attn.num_heads,
        )
        meff_attn.qkv = slow_attn.qkv
        meff_attn.attn_drop = slow_attn.attn_drop
        meff_attn.proj = slow_attn.proj
        meff_attn.proj_drop = slow_attn.proj_drop

        model.blocks[i].attn = meff_attn


# 对预定义的网络增加中间输出节点, 并将 token unflatten 成为图像
def _make_vit_b16_backbone(
    model,
    psize=[16, 16],                     # patch size
    hooks=[2, 5, 8, 11],                # 在 hocks 的位置定义输出
    vit_features=768,                   # transformer blocks 的 dim
    use_readout="ignore",               # 如何处理 [cls] 与其他 token 的关系
    aggregation="sparse",               # 如何处理额外的 ViT Tokens
    start_index=1,                      # [cls] 起始位置
    enable_attention_hooks=False,
    fast_attn=True
):
    pretrained = nn.Module()

    # 将模型传给 pretrained 模型容器, 并插入输出层
    pretrained.model = model

    # 将 model 中的 attention 更换为 flash attn
    if fast_attn: use_memeffattn_in_model(pretrained.model)

    # 更新后的 dinov2 引入 registers 来解决 features 中的高频噪声. 设计兼容性判断语句
    if getattr(model, "register_tokens", None) is not None:
        num_register_tokens = model.register_tokens.shape[1]
    else:
        num_register_tokens = 0

    # 将 token 转为 FPN 特征
    pretrained.act_postprocess = nn.Sequential(
        Slice(start_index+num_register_tokens),                                     # ignore [cls] token
        Transpose(1, 2),                                                            # BNC -> BCN
    )   # layer1, 1/4

    # 参数设置
    # pretrained.model.start_index = start_index
    # pretrained.model.patch_size = psize
    pretrained.vit_features = [vit_features]*4
    pretrained.hooks = hooks
    pretrained.patch_size = psize[0]
    pretrained.aggregation = aggregation

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


# 构造适配特定参数的网络结构
# def _make_pretrained_dinov2_vitg14_518(
#     use_readout="ignore",
#     hooks=None,
#     enable_attention_hooks=False
# ):
#     # DINOv2 = vit-large, 14*14, 518*518
#     repo_path = '/home/zhangyj85/.cache/torch/hub/facebookresearch_dinov2_main'
#     model = torch.hub.load(
#         repo_or_dir=repo_path,
#         model='dinov2_vitg14',
#         source='local',
#     )
#     # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
#
#     hooks = [5, 11, 17, 23] if hooks == None else hooks
#     return _make_vit_b16_backbone(
#         model,
#         psize=[model.patch_size]*2,
#         hooks=hooks,
#         vit_features=model.embed_dim,       # vit transformer 的通道数
#         use_readout=use_readout,
#         enable_attention_hooks=enable_attention_hooks,
#     )
#
#
# def _make_pretrained_dinov2_vitl14_518(
#     use_readout="ignore",
#     hooks=None,
#     enable_attention_hooks=False
# ):
#     # DINOv2 = vit-large, 14*14, 518*518, loader 方式1
#     # repo_path = '/home/zhangyj85/.cache/torch/hub/facebookresearch_dinov2_main'
#     # model = torch.hub.load(
#     #     repo_or_dir=repo_path,
#     #     model='dinov2_vitl14',
#     #     source='local',
#     # )
#     # loader 方式2
#     # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
#     # loader 方式3
#     repo_path = 'facebookresearch_dinov2_main'
#     model = torch.hub.load(
#         repo_or_dir=repo_path,
#         model='dinov2_vitl14',
#         pretrained=False,   # dinov2的接口
#         source='local',
#     )
#     weights = torch.load("foundation_model_weights/dinov2_vitl14_pretrain.pth", map_location="cpu")
#     model.load_state_dict(weights, strict=True)
#
#     hooks = [5, 11, 17, 23] if hooks == None else hooks
#     return _make_vit_b16_backbone(
#         model,
#         psize=[model.patch_size]*2,
#         hooks=hooks,
#         vit_features=model.embed_dim,       # vit transformer 的通道数
#         use_readout=use_readout,
#         enable_attention_hooks=enable_attention_hooks,
#     )
#
#
# def _make_pretrained_dinov2_vitb14_518(
#     use_readout="ignore",
#     hooks=None,
#     enable_attention_hooks=False
# ):
#     repo_path = '/home/zhangyj85/.cache/torch/hub/facebookresearch_dinov2_main'
#     model = torch.hub.load(
#         repo_or_dir=repo_path,
#         model='dinov2_vitb14',
#         source='local',
#     )
#     # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#
#     hooks = [2, 5, 8, 11] if hooks == None else hooks
#     return _make_vit_b16_backbone(
#         model,
#         psize=[model.patch_size]*2,
#         hooks=hooks,
#         vit_features=model.embed_dim,       # vit transformer 的通道数
#         use_readout=use_readout,
#         enable_attention_hooks=enable_attention_hooks,
#     )
#
#
# def _make_pretrained_dinov2_vits14_518(
#     use_readout="ignore",
#     hooks=None,
#     enable_attention_hooks=False
# ):
#     repo_path = '/home/zhangyj85/.cache/torch/hub/facebookresearch_dinov2_main'
#     model = torch.hub.load(
#         repo_or_dir=repo_path,
#         model='dinov2_vits14',
#         source='local',
#     )
#     # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#
#     hooks = [1, 3, 5, 7] if hooks == None else hooks
#     return _make_vit_b16_backbone(
#         model,
#         psize=[model.patch_size]*2,
#         hooks=hooks,
#         vit_features=model.embed_dim,       # vit transformer 的通道数
#         use_readout=use_readout,
#         enable_attention_hooks=enable_attention_hooks,
#     )


def _make_pretrained_depth_anything_vitl14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True,
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitl14',
        pretrained=False,   # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DAM_weights = torch.load("foundation_model_weights/depth_anything_vitl14.pth", map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in DAM_weights.items():
        if "depth_head." in k:
            # 排除无关权重
            continue
        prefix = "pretrained."
        name = k[len(prefix):] if prefix in k else k
        new_state_dict[name] = v

    # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
    model.load_state_dict(new_state_dict, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_depth_anything_vitb14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitb14',
        pretrained=False,   # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DAM_weights = torch.load("foundation_model_weights/depth_anything_vitb14.pth", map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in DAM_weights.items():
        if "depth_head." in k:
            # 排除无关权重
            continue
        prefix = "pretrained."
        name = k[len(prefix):] if prefix in k else k
        new_state_dict[name] = v

    # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
    model.load_state_dict(new_state_dict, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_depth_anything_vits14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vits14',
        pretrained=False,   # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DAM_weights = torch.load("foundation_model_weights/depth_anything_vits14.pth", map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in DAM_weights.items():
        if "depth_head." in k:
            # 排除无关权重
            continue
        prefix = "pretrained."
        name = k[len(prefix):] if prefix in k else k
        new_state_dict[name] = v

    # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
    model.load_state_dict(new_state_dict, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation="sparse",
        fast_attn=fast_attn
    )


def _make_pretrained_depth_anything_v2_vitl14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitl14',
        pretrained=False,   # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DAM_weights = torch.load("foundation_model_weights/depth_anything_v2_vitl.pth", map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in DAM_weights.items():
        if "depth_head." in k:
            # 排除无关权重
            continue
        prefix = "pretrained."
        name = k[len(prefix):] if prefix in k else k
        new_state_dict[name] = v

    # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
    model.load_state_dict(new_state_dict, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_depth_anything_v2_vitb14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitb14',
        pretrained=False,   # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DAM_weights = torch.load("foundation_model_weights/depth_anything_v2_vitb.pth", map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in DAM_weights.items():
        if "depth_head." in k:
            # 排除无关权重
            continue
        prefix = "pretrained."
        name = k[len(prefix):] if prefix in k else k
        new_state_dict[name] = v

    # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
    model.load_state_dict(new_state_dict, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_depth_anything_v2_vits14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/facebookresearch_dinov2_main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vits14',
        pretrained=False,   # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DAM_weights = torch.load("foundation_model_weights/depth_anything_v2_vits.pth", map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in DAM_weights.items():
        if "depth_head." in k:
            # 排除无关权重
            continue
        prefix = "pretrained."
        name = k[len(prefix):] if prefix in k else k
        new_state_dict[name] = v

    # 对比发现, depth anything 的权重和 DINOv2 很像, 误差接近零
    model.load_state_dict(new_state_dict, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )



def _make_pretrained_dinov2reg_vitg14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/dinov2-main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitg14_reg',      # 使用优化后的DINOv2, https://github.com/facebookresearch/dinov2
        pretrained=False,               # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DINOv2_weights = torch.load("foundation_model_weights/dinov2_vitg14_reg4_pretrain.pth", map_location="cpu")
    model.load_state_dict(DINOv2_weights, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_dinov2reg_vitg14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/dinov2-main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitg14_reg',      # 使用优化后的DINOv2, https://github.com/facebookresearch/dinov2
        pretrained=False,               # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DINOv2_weights = torch.load("foundation_model_weights/dinov2_vitg14_reg4_pretrain.pth", map_location="cpu")
    model.load_state_dict(DINOv2_weights, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_dinov2_vits14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/dinov2-main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vits14',          # DINOv2, https://github.com/facebookresearch/dinov2
        pretrained=False,               # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DINOv2_weights = torch.load("foundation_model_weights/dinov2_vits14_pretrain.pth", map_location="cpu")
    model.load_state_dict(DINOv2_weights, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_dinov2_vitb14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/dinov2-main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitb14',          # DINOv2, https://github.com/facebookresearch/dinov2
        pretrained=False,               # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DINOv2_weights = torch.load("foundation_model_weights/dinov2_vitb14_pretrain.pth", map_location="cpu")
    model.load_state_dict(DINOv2_weights, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )


def _make_pretrained_dinov2_vitl14_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True
):
    # 加载模型 backbone
    repo_path = 'foundation_model_repos/dinov2-main'
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='dinov2_vitl14',          # DINOv2, https://github.com/facebookresearch/dinov2
        pretrained=False,               # dinov2的接口
        source='local',
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    DINOv2_weights = torch.load("foundation_model_weights/dinov2_vitl14_pretrain.pth", map_location="cpu")
    model.load_state_dict(DINOv2_weights, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )
