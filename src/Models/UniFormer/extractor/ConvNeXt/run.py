"""
从 ConvNeXt 推理得到 features
Modified from: https://github.com/intel-isl/DPT
"""
import torch
import torch.nn as nn
import types
import math
import torch.nn.functional as F


# vit 前向推理, 并将中间层结果转化为图像, 然后输出
def forward_convnext(pretrained, x):
    """
    pretrained: including model & post-process
    x: input in image shape: b, 3, h, w
    """
    b, c, h, w = x.shape

    # ViT 前向推理, glob 为x对应的最终特征, (B,N,C)
    pretrained.model.eval()     # 冻结 dropout & drop path
    with torch.no_grad():
        final, intermediates = pretrained.model.forward_flex(x)

    return final, intermediates


def forward_flex(self, x):

    patch_W, patch_H = self.patch_size
    assert x.shape[2] % patch_H == 0, f"Input image height {x.shape[2]} is not a multiple of patch height {patch_H}"
    assert x.shape[3] % patch_W == 0, f"Input image width {x.shape[3]} is not a multiple of patch width: {patch_W}"

    # normalize: ImageNet归一化参数与当前归一化结果一致, 无需转换

    # ConvNeXt 无 patch embedding, 无 position embedding

    # 前馈
    intermediates = []
    for i in range(4):
        x = self.downsample_layers[i](x)
        x = self.stages[i](x)
        intermediates.append(x)     # B, C, H, W, c=embeded dim
    x = self.norm(x.mean([-2, -1])) # B, C
    N, L = x.shape
    x = x.reshape(N, L, 1, 1)       # B, C, 1, 1

    return x, intermediates


# 对预定义的网络增加中间输出节点, 并将 token unflatten 成为图像
def _make_convnext_b32_backbone(
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

    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    return pretrained


def _make_pretrained_convnext_base_224(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False
):
    from .convnext import ConvNeXt
    model = ConvNeXt(num_classes=21841,             # ImageNet 22k 类别数
                     depths=[3, 3, 27, 3],          # base 模型的参数
                     dims=[128, 256, 512, 1024], 
                     drop_path_rate=0.2,            # 官网使用的参数
                     layer_scale_init_value=1e-6,   # 默认参数
                     head_init_scale=1.,)

    # 加载权重
    path = "foundation_model_weights/convnext_base_22k_224.pth"
    states = torch.load(path, map_location="cpu")
    weights = states['model']
    model.load_state_dict(weights, strict=True)

    # 保证 hooks 的划分与原生的 stages 划分一致
    hooks = [2, 5, 32, 35]
    vit_features = [128, 256, 512, 1024]

    return _make_convnext_b32_backbone(
        model,
        psize=[4, 4],
        hooks=hooks,
        vit_features=vit_features,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
