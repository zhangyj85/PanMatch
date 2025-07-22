from __future__ import print_function

from modulefinder import Module

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

from .convnext import Block as ConvXBlock
from .convnext import LayerNorm, trunc_normal_


def LayerNorm2d(features):
    return nn.GroupNorm(num_groups=1, num_channels=features)


def LayerNorm3d(features):
    return nn.GroupNorm(num_groups=1, num_channels=features)


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_cosine(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = F.cosine_similarity(fea1, fea2, dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, norm=None):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    if norm == "cosine":
        fn = groupwise_correlation_cosine
    elif norm == "l2":
        fn = groupwise_correlation_norm
    elif norm == "none":
        fn = groupwise_correlation
    else:
        raise NotImplementedError

    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = fn(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = fn(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_cosine_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = -1 * refimg_fea.new_ones([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_cosine(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups=1)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_cosine(refimg_fea, targetimg_fea, num_groups=1)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class learnable_upsample(nn.Module):
    """Upsample features according to its features"""
    def __init__(self, in_chans, out_chans, rate=4):
        super().__init__()
        # map the input into the output channels
        self.mlp = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, stride=1, padding=0, bias=False),     # 通道压缩
            nn.InstanceNorm2d(out_chans), nn.GELU(),                                            # 增加非线性
            nn.Conv2d(out_chans, out_chans, 1, stride=1, padding=0),
        )
        # upsample flow
        self.mask = nn.Sequential(
            nn.Conv2d(in_chans, 256, 1, stride=1, padding=0, bias=False),  # 通道压缩
            nn.InstanceNorm2d(out_chans), nn.GELU(),
            nn.Conv2d(256, (rate ** 2) * 9, 3, stride=1, padding=1, bias=False),
        )
        self.rate = rate

    def forward(self, x):
        rate = self.rate
        flow = self.mlp(x)
        mask = self.mask(x)

        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)  # 注意, 由于不是光流/视差, 对上采样的结果不需要乘以倍率
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(N, C, rate * H, rate * W)
        return up_flow.contiguous()


class guide_upsample(nn.Module):
    """Using original features with attention mechanism to guide the low resolution features for upsampling"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge  = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, low_feats, guide_feats):
        # 首先将图像转为序列
        b, _, H, W = guide_feats.shape
        _, _, h, w = low_feats.shape
        guide_feats = guide_feats.permute(0, 2, 3, 1).reshape(b, H*W, -1)   # (B,C,H,W) -> (B,HW,C)
        low_feats = low_feats.permute(0, 2, 3, 1).reshape(b, h*w, -1)       # (B,C,h,w) -> (B,hw,C)

        # 对 q,k,v 分别进行线性映射
        query = self.q_proj(guide_feats).reshape(b, H*W, 1, self.nhead, self.dim)    # (B,HW,C) -> (B,HW,1,n,c)

        # key & value 进行邻域聚合与上采样
        key = self.k_proj(low_feats).reshape(b, h, w, -1).permute(0, 3, 1, 2)                               # 还原低分辨率图像, (B,C,h,w)
        key = F.pad(key.float(), pad=(1,1,1,1), mode="replicate")                                           # 转为 float32 后进行padding, 保持图像边缘的数据均值与整体相近, 进而避免上采样后的特征图周围有一圈数据的均值显著小于图像中心
        key = F.unfold(key, kernel_size=3, padding=0, stride=1, dilation=1).reshape(b, -1, h, w)     # 邻域聚合, (B,C*9,h,w)
        key = F.interpolate(key, size=(H, W), mode="nearest").reshape(b, -1, 9, H, W)                       # 最近邻插值, (B,C,9,H,W)
        key = key.permute(0, 3, 4, 2, 1).reshape(b, H*W, 9, self.nhead, self.dim)                           # (B,H*W,N,n,c)

        value = self.v_proj(low_feats).reshape(b, h, w, -1).permute(0, 3, 1, 2)                                 # 还原低分辨率图像
        value = F.pad(value.float(), pad=(1, 1, 1, 1), mode="replicate")
        value = F.unfold(value, kernel_size=3, padding=0, stride=1, dilation=1).reshape(b, -1, h, w)     # 邻域聚合
        value = F.interpolate(value, size=(H, W), mode="nearest").reshape(b, -1, 9, H, W)                       # 最近邻插值
        value = value.permute(0, 3, 4, 2, 1).reshape(b, H*W, 9, self.nhead, self.dim)                           # (B,H*W,N,n,c)

        # 计算注意力
        query = query.permute(0, 1, 3, 2, 4)        # (B,HW,n,1,c)
        key = key.permute(0, 1, 3, 4, 2)            # (B,HW,n,c,N)
        value = value.permute(0, 1, 3, 2, 4)        # (B,HW,n,N,c)
        # 为了避免半精度数值溢出, 改用数值友好的方式实现 attn
        delta = self.dim ** 0.5
        score = F.softmax(torch.matmul(query / delta, key / delta) * delta, dim=-1)     # (B,HW,n,1,N)
        message = torch.matmul(score, value).squeeze(-2)                            # (B,HW,n,c)
        message = self.merge(message.reshape(b, H*W, self.nhead * self.dim))

        # TODO: 补充归一化手段, 注意 InstanceNorm1d 输入/输出为 (N,C,L)
        message = self.norm1(message).permute(0, 2, 1)

        return message.reshape(b, -1, H, W).contiguous()


class GuideNet(nn.Module):
    def __init__(self, features, depths=[1,1,1,1], n_downsample=3):
        super().__init__()
        self.layer1 = self._make_layer(BasicBlock, inplanes=3 * 4**2, planes=features, blocks=depths[0], stride=1, pad=1, dilation=1)   # 1/4
        self.layer2 = self._make_layer(BasicBlock, inplanes=features, planes=features, blocks=depths[1], stride=1+(n_downsample>0), pad=1, dilation=1)   # 1/8
        self.layer3 = self._make_layer(BasicBlock, inplanes=features, planes=features, blocks=depths[2], stride=1+(n_downsample>1), pad=1, dilation=1)   # 1/16
        self.layer4 = self._make_layer(BasicBlock, inplanes=features, planes=features, blocks=depths[3], stride=1+(n_downsample>2), pad=1, dilation=1)   # 1/32

    def _make_layer(self, block, inplanes, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = convbn(inplanes, planes, kernel_size=3, stride=stride, pad=1, dilation=1)
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, pad, dilation))
        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, _, h, w = x.shape
        x4 = F.unfold(x, kernel_size=4, stride=4, padding=0).reshape(b, -1, h // 4, w // 4).contiguous()
        x4 = self.layer1(x4)
        x8 = self.layer2(x4)
        x16 = self.layer3(x8)
        x32 = self.layer4(x16)
        return x32, x16, x8, x4


class DefaultFlowGuideNet(nn.Module):
    # 根据 ConvNeXt 进行改写
    def __init__(self, features, depths=[2,2,2,2], n_downsample=3):
        super().__init__()
        # 光流任务通常在 1/8 尺度下进行, 因此基准特征分别率(layer1)设置为 1/8
        self.firstconv = nn.Sequential(
            nn.Conv2d(3*4**2, features, kernel_size=7, stride=1, padding=3, dilation=1),
            LayerNorm(features, eps=1e-6, data_format="channels_first")
        )
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = self._make_layer(ConvXBlock, inplanes=features, planes=features, blocks=depths[i], downsample=n_downsample>2+i)  # base = 1/8
            self.stages.append(stage)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, downsample=False):
        layers = []
        if downsample:
            downsample_layer = nn.Sequential(
                LayerNorm(inplanes, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
            )
            layers.append(downsample_layer)
        for i in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.unfold(x, kernel_size=4, stride=4, padding=0).reshape(b, -1, h // 4, w // 4).contiguous()
        x = self.firstconv(x)
        output = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            output.append(x)
        return output


class HighResGuideNet(nn.Module):
    # 根据 ConvNeXt 进行改写
    def __init__(self, features, depths=[2,2,2,2]):
        super().__init__()
        # 特征尺度为 [1/2, 1/4, 1/8, 1/16]
        self.firstconv = nn.Sequential(
            nn.Conv2d(3*2**2, features, kernel_size=7, stride=1, padding=3, dilation=1),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )
        self.stages = nn.ModuleList()
        downsample_list = [False, True, True, True]
        self.scales = [2, 4, 8, 16]
        for i in range(len(depths)):
            stage = self._make_layer(ConvXBlock, inplanes=features, planes=features, blocks=depths[i], downsample=downsample_list[i])  # base = 1/8
            self.stages.append(stage)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, downsample=False):
        layers = []
        if downsample:
            downsample_layer = nn.Sequential(
                LayerNorm(inplanes, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
            )
            layers.append(downsample_layer)
        for i in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.unfold(x, kernel_size=2, stride=2, padding=0).reshape(b, -1, h // 2, w // 2).contiguous()
        x = self.firstconv(x)
        output = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            output.append(x)
        return output


#
# class ChannelAttentionEnhancement(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttentionEnhancement, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                 nn.ReLU(),
#                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttentionExtractor(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttentionExtractor, self).__init__()
#
#         self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.samconv(x)
#         return self.sigmoid(x)
#
#
# class CSAM(nn.Module):
#     """add the lost high-frequency features into ViT all-purposed features"""
#     def __init__(self, features):
#         super().__init__()
#         """projection for scale consistency"""
#         # self.context_proj = nn.Sequential(
#         #     nn.Conv2d(features, features, kernel_size=1, padding=0, bias=False),
#         #     nn.InstanceNorm2d(features),
#         # )
#         self.align_proj = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(features)
#         )
#         # 使用两层 MLP 来判断两者的差异
#         self.difference = nn.Sequential(
#             nn.Conv2d(features*2, features, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
#         )
#         """contextual spatial attention module"""
#         self.CAE = ChannelAttentionEnhancement(in_planes=features)
#         self.SAE = SpatialAttentionExtractor()
#
#     def forward(self, context, details, eps=0.1):
#         diff = self.difference(torch.cat([context, details], dim=1))
#         attn = self.SAE(self.CAE(diff) * diff)
#         attn_scaled = (attn - 0.5) * (1 - eps) + 0.5
#         # attn 是一个稀疏向量, 仅在高频信息区域有较大响应
#         output = (1 - attn_scaled) * context + attn_scaled * self.align_proj(details)
#         return output, attn
#
#
# class ShortCutFusion(nn.Module):
#     """add the lost high-frequency features into ViT all-purposed features"""
#     def __init__(self, features):
#         super().__init__()
#         """projection for scale consistency"""
#         self.align_proj = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(features),    # 注意同时归一化
#         )
#         self.register_buffer(name="weight", tensor=torch.zeros(1))
#         self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.momentum = 0.999
#
#     def _update_params(self):
#         alpha = 0.5 * F.sigmoid(self.alpha)
#         self.weight = self.momentum * self.weight + (1 - self.momentum) * alpha.detach()
#
#     def forward(self, context, details):
#         alpha = 0.5 * F.sigmoid(self.alpha)     # alpha in (0, 0.5), 通过动量更新, self.weight 最终会收敛到 alpha 的稳定值, 简单证明一下即可: y(t) = A * y(t-1) + (1 - A) * B, y(t) 收敛于 B
#         if self.training:
#             weight = self.momentum * self.weight + (1 - self.momentum) * alpha
#             # self.weight = weight        # 更新权值, 不能在这里更新, 否则每次前馈都会导致 weight 发生改变
#         else:
#             weight = self.weight
#
#         output = (1 - weight) * context + weight * self.align_proj(details)
#         return output
#
#
# class RaftConvGRU(nn.Module):
#     def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
#         super(RaftConvGRU, self).__init__()
#         self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
#         self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
#         self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
#
#     def forward(self, h, x):
#         hx = torch.cat([h, x], dim=1)
#
#         z = torch.sigmoid(self.convz(hx))
#         r = torch.sigmoid(self.convr(hx))
#         q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
#
#         h = (1-z) * h + z * q
#         return h
#
#
# class FuseConvGRU(nn.Module):
#     def __init__(self, features):
#         super().__init__()
#         """projection for scale consistency"""
#         self.context_proj = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(features),
#             nn.Tanh(),
#         )
#         self.details_proj = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(features),
#             nn.ReLU(),
#         )
#         self.fuseBlock = RaftConvGRU(hidden_dim=features, input_dim=features, kernel_size=3)
#
#     def forward(self, context, details):
#         return self.fuseBlock(self.context_proj(context), self.details_proj(details))
#
#
# class FeatureCompletion(nn.Module):
#     def __init__(self, features):
#         super().__init__()
#         self.combine = nn.Sequential(
#             nn.Conv2d(features * 2, features, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.ReLU(inplace=True),
#         )
#         self.delta = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.Tanh()
#         )
#         self.weight = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, context, details):
#         x = self.combine(torch.cat([context, details], dim=1))
#         w = self.weight(x)
#         y = self.delta(x)
#         output = context + w * y
#         return output


@torch.no_grad()
def depth2coords(source_depth, source_intrinsics, source_pose, target_intrinsics, target_pose, clamp_min_depth=1e-3):
    """
    Input:
        source depth: (B,1,H,W)
        source intrinsics: (B,3,3), 相机内参
        source pose: (B,4,4), 世界坐标系到相机坐标系的变换矩阵 (相机位姿)
    Output:
        source correspondent coords: (B,2,H,W)
    """
    assert source_intrinsics.size(1) == source_intrinsics.size(2) == 3
    assert source_pose.size(1) == source_pose.size(2) == 4
    assert source_depth.dim() == 4

    b, _, h, w = source_depth.shape
    device = source_depth.device

    # create standard pixel grid coords
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))  # [H, W]
    z = torch.ones_like(x)
    grid = torch.stack([x, y, z], dim=0).float()
    grid = grid.view(1, 3, h, w).repeat(b, 1, 1, 1)     # [B,3,H,W]

    # back project to 3D world coords
    points = torch.inverse(source_intrinsics).bmm(grid.view(b, 3, -1)) * source_depth.view(b, 1, h * w)  # [B, 3, H*W]
    points = torch.inverse(source_pose[:, :3, :3]).bmm(points - source_pose[:, :3, -1:])

    # transform viewpoint
    points = torch.bmm(target_pose[:, :3, :3], points) + target_pose[:, :3, -1:]
    points = torch.bmm(target_intrinsics, points)   # [B,3,H*W]
    pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)
    pixel_coords = pixel_coords.reshape(b, 2, h, w).contiguous()

    # flow
    flow = pixel_coords - grid[:, :2]

    return pixel_coords, flow


def skew(x):
    """
    Input: x is a 3x1 vector in (N, 3,)
    Output: x_ is a 3x3 metric in (N, 3, 3)
    x = [a; b; c]
    x_ = [[0, -c,  b],
          [c,  0, -a],
          [-b, a,  0],]
    vec(0) = bmm(x_, x)
    """
    N, _ = x.shape
    x_ = x.new_zeros([N, 3, 3])
    x_[:, 0, 1] = -x[:, 2]
    x_[:, 0, 2] =  x[:, 1]
    x_[:, 1, 0] =  x[:, 2]
    x_[:, 1, 2] = -x[:, 0]
    x_[:, 2, 0] = -x[:, 1]
    x_[:, 2, 1] =  x[:, 0]
    return x_


@torch.no_grad()
def flow2disp(source_flow):
    return - source_flow[:, :1]


@torch.no_grad()
def flow2depth(source_flow, source_intrinsics, source_pose, target_intrinsics, target_pose, clamp_min_value=1e-6):
    assert source_intrinsics.size(1) == source_intrinsics.size(2) == 3
    assert source_pose.size(1) == source_pose.size(2) == 4
    assert source_flow.dim() == 4

    b, _, h, w = source_flow.shape
    device = source_flow.device

    # create standard pixel grid coords
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))  # [H, W]
    z = torch.ones_like(x)
    grid = torch.stack([x, y, z], dim=0).float()
    grid = grid.view(1, 3, h, w).repeat(b, 1, 1, 1)  # [B,3,H,W]

    pixel_coords = torch.zeros_like(grid)
    pixel_coords[:, :2] = source_flow + grid[:, :2]
    pixel_coords[:, 2:] = grid[:, 2:]               # [B,3,H,W]

    source_grid = grid.permute(0, 2, 3, 1).reshape(-1, 3, 1)                          # (bhw,3,1)
    target_skew_grid = skew(pixel_coords.permute(0, 2, 3, 1).reshape(-1, 3))        # (bhw,3,3)

    source_intrinsics = source_intrinsics.view(b, 1, 3, 3).repeat(1, h * w, 1, 1).reshape(-1, 3, 3)
    source_pose = source_pose.view(b, 1, 4, 4).repeat(1, h * w, 1, 1).reshape(-1, 4, 4)
    target_intrinsics = target_intrinsics.view(b, 1, 3, 3).repeat(1, h * w, 1, 1).reshape(-1, 3, 3)
    target_pose = target_pose.view(b, 1, 4, 4).repeat(1, h * w, 1, 1).reshape(-1, 4, 4)

    R = torch.bmm(target_pose[:, :3, :3], torch.inverse(source_pose[:, :3, :3]))    # (n,3,3)
    P = torch.bmm(target_skew_grid, target_intrinsics)  #(n,3,3)
    B = P.bmm(R).bmm(torch.inverse(source_intrinsics)).bmm(source_grid)
    A = P.bmm(R).bmm(source_pose[:, :3, -1:]) - P.bmm(target_pose[:, :3, -1:])
    depth_y = A[:, 0] / (B[:, 0] + clamp_min_value)
    depth_x = A[:, 1] / (B[:, 1] + clamp_min_value)

    # 为了保持数值稳定, 选择 flow 更大的方向来计算深度
    depth = torch.zeros_like(depth_x)
    source_flow_abs = source_flow.abs().permute(0, 2, 3, 1).reshape(-1, 2)
    x_fill = (source_flow_abs[:, 0] >= source_flow_abs[:, 1])
    y_fill = (source_flow_abs[:, 0] <  source_flow_abs[:, 1])
    depth[x_fill] = depth_x[x_fill]
    depth[y_fill] = depth_y[y_fill]
    depth = depth.reshape(b, 1, h, w).contiguous()

    return depth
