"""
为了规避两次会议中提到的 zero-shot claim 的问题, 默认的模型采用 RADIO (ViT-H/16), trained on Datacomp-1B (多模态数据集）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DINO_DAM.dino_dam_vit2 import (
    _make_pretrained_depth_anything_vitl14_518,
    _make_pretrained_depth_anything_v2_vitl14_518,
    _make_pretrained_depth_anything_v2_vitb14_518,
    _make_pretrained_depth_anything_v2_vits14_518,
    _make_pretrained_dinov2reg_vitg14_518,
    _make_pretrained_dinov2_vits14_518,
    _make_pretrained_dinov2_vitb14_518,
    forward_vit,
)

from .SAM.vit import (
    _make_pretrained_sam_vitb16_1024,
    _make_pretrained_sam_vitl16_1024,
    _make_pretrained_sam_vith16_1024,
    forward_sam
)

from .Swinv1.run import (
    _make_pretrained_swinv1_base_224,
    forward_swin,
)

from .ConvNeXt.run import (
    _make_pretrained_convnext_base_224,
    forward_convnext,
)

# from .RADIO import (
#     _mask_pretrained_radio_vith16_432,
#     forward_radio,
# )

from .convnext import Block as ConvXBlock
from .utils import build_gwc_volume

hooks = {
    "swinv1_base12_224": [1, 3, 21, 23],  # Pyramid ViT, no need for specification
    "dinov2_vitg14_518": [5, 11, 17, 23],
    "dinov2_vitl14_518": [5, 11, 17, 23],   # (24 // 4) * i - 1, i=1,2,3,4
    "dinov2_vitb14_518": [2, 5, 8, 11],     # (12 // 4) * i - 1
    "dinov2_vits14_518": [1, 3, 5, 7],      # ( 8 // 4) * i - 1
    "dam_vitl14_518":    [5, 11, 17, 23],
    "damv2_vitl14_518":  [5, 11, 17, 23],
    "damv2_vitb14_518":  [2, 5, 8, 11],
    "damv2_vits14_518":  [1, 3, 5, 7],
    "dinov2reg_vitg14_518": [9, 19, 29, 39],
    "sam_vitb16_1024":   [2, 5, 8, 11],
    "sam_vitl16_1024":   [5, 11, 17, 23],
    "sam_vith16_1024":   [7, 15, 23, 31],
    "radio_vith16_432":  [7, 15, 23, 31],
    "convnext_base32_224": [2, 5, 32, 35],

}

forward_fn = {
    "dam_vitl14_518": forward_vit,
    "damv2_vitl14_518": forward_vit,
    "damv2_vitb14_518": forward_vit,
    "damv2_vits14_518": forward_vit,
    "dinov2reg_vitg14_518": forward_vit,
    "dinov2_vits14_518": forward_vit,
    "dinov2_vitb14_518": forward_vit,
    # "radio_vith16_432": forward_radio,
    "sam_vitb16_1024": forward_sam,
    "sam_vitl16_1024": forward_sam,
    "sam_vith16_1024": forward_sam,
    "swinv1_base12_224": forward_swin,
    "convnext_base32_224": forward_convnext,
}


def _make_encoder(
    backbone,
    hooks=None,
    use_readout="ignore",
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=True,
):
    if backbone == "dam_vitl14_518":
        pretrained = _make_pretrained_depth_anything_vitl14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "damv2_vitl14_518":
        pretrained = _make_pretrained_depth_anything_v2_vitl14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "damv2_vitb14_518":
        pretrained = _make_pretrained_depth_anything_v2_vitb14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "damv2_vits14_518":
        pretrained = _make_pretrained_depth_anything_v2_vits14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "dinov2reg_vitg14_518":
        pretrained = _make_pretrained_dinov2reg_vitg14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "dinov2_vits14_518":
        pretrained = _make_pretrained_dinov2_vits14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "dinov2_vitb14_518":
        pretrained = _make_pretrained_dinov2_vitb14_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    elif backbone == "radio_vith16_432":
        pretrained = _mask_pretrained_radio_vith16_432(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
        )
    elif backbone == "sam_vitb16_1024":
        pretrained = _make_pretrained_sam_vitb16_1024(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
        )
    elif backbone == "sam_vitl16_1024":
        pretrained = _make_pretrained_sam_vitl16_1024(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
        )
    elif backbone == "sam_vith16_1024":
        pretrained = _make_pretrained_sam_vith16_1024(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
        )
    elif backbone == "swinv1_base12_224":
        pretrained = _make_pretrained_swinv1_base_224(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    elif backbone == "convnext_base32_224":
        pretrained = _make_pretrained_convnext_base_224(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
    else:
        # TODO: 增加 ViT-g, ViT-s
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained


class ViTEncoder(nn.Module):
    def __init__(
        self,
        backbone="dam_vitl14_518",
        features=128,
        output_corr_chans=[32,64,96,128],   # corr feature channels
        output_feat_chans=[32,64,96,128],   # ref feature channels
        n_downsample=2,                     # 最大下采样到 1 / 2**(2+n_downsample),
        aggregation="sparse"                # 特征聚合方式, (sparse, dense)
    ):
        super().__init__()

        # 实例化专家视觉大模型
        expert = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
        )
        for p in expert.model.parameters():
            p.requires_grad = False
        expert.model.eval()
        self.expert = [expert]                              # 永久隐藏参数
        self.expert_forward = forward_fn[backbone]          # 前馈 function
        vit_features = self.expert[0].vit_features          # 特征维度

        # 对 ViT 的特征进行压缩投影处理
        self.expert_proj = nn.Sequential(
            nn.Conv2d(vit_features, features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features, affine=False),    # 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        )

        self.learnable_encoder = False
        if self.learnable_encoder:
            # 可学习的 encoder
            self.encoder = _make_encoder(
                backbone="damv2_vits14_518",
                hooks=[7],
                aggregation=aggregation,
                use_readout="ignore",
                enable_attention_hooks=False,
            )
            del self.encoder.model.mask_token
            self.encoder_forward = forward_fn["damv2_vits14_518"]
            vit_features = self.encoder.vit_features
            self.encoder_proj = nn.Sequential(
                nn.Conv2d(vit_features, features, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=features, affine=False),    # 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
            )

            self.weight = nn.parameter.Parameter(torch.ones(1), requires_grad=True)
            self.bias = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)

        # shortcut branch
        self.shortcut = nn.Identity()
        self.deepconv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
        )
        self.concat = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        )

    def train(self, mode=True):
        super().train(mode)
        self.expert[0].eval()
        if self.learnable_encoder:
            self.expert_proj.eval()
            self.shortcut.eval()
            self.deepconv.eval()
            # self.concat.eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def forward(self, img):
        with torch.no_grad():
            self.expert[0] = self.expert[0].to(img.device).to(img.dtype)
            final, intermediate = self.expert_forward(self.expert[0], img)

        expert_features = self.expert_proj(final)

        x = self.shortcut(expert_features)
        s = self.deepconv(expert_features)

        loss = 0.
        if self.learnable_encoder:
            aux_final, aux_intermediate = self.encoder_forward(self.encoder, img)
            aux_features = self.encoder_proj(aux_final)
            x = self.weight * x + aux_features + self.bias
            # 希望仅使用达模型的情况下, 其特征表征与引入可学习的表征尽可能相似
            loss = ((x.detach() - expert_features) ** 2).mean()

        output = self.concat(torch.cat([x, s], dim=1))

        return {"feat": output, "loss": loss}


class Formerplusplus(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            output_corr_chans=[32,64,96,128],   # corr feature channels
            output_feat_chans=[32,64,96,128],   # ref feature channels
            n_downsample=3,                     # 最大下采样到 1 / 2**(n_downsample),
            aggregation="sparse",                # 特征聚合方式, (sparse, dense)
            aux_enable=False,                   # 是否采用辅助网络捕获额外的信息
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
        )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_fn[backbone]         # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features, features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features, affine=False),
        ) for _ in range(4)])

        # 构造 CNN 引导多尺度特征
        from .utils import DefaultFlowGuideNet as GuideNet
        self.cnn_encoder = GuideNet(features, depths=[2,2,2,2], n_downsample=n_downsample)     # 最大降采样到 1/16
        self.n_downsample = n_downsample

        # 构造引导上采样模块, 输出为 norm 后的结果
        from .utils import guide_upsample
        self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])

        # 多尺度融合
        from .fpn1 import fpn_decoder
        self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        # output correspondent projection
        self.output_corr_chans = output_corr_chans
        self.corr_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_corr_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_corr_chans))])

        # output features projection
        self.output_feat_chans = output_feat_chans
        self.feat_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_feat_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_feat_chans))])

        # single-view reconstruction
        self.recon_head = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, 3*8**2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=8),
        ) for _ in range(4)])

        # context consistency
        self.context_head = nn.Sequential(# base resolution is 1/8
            nn.Conv2d(features, features, kernel_size=2, stride=2, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(features, vit_features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=vit_features),
        )
        self.context_list = []

        # 使用额外的 CNN 网络获取 ViT 中不可获取的信息
        self.aux_enable = aux_enable
        if self.aux_enable:
            self.aux_encoder = GuideNet(features, depths=[2,2,2,2], n_downsample=n_downsample)
            self.weight = nn.Parameter(torch.zeros(4), requires_grad=True)
            # self.bias = nn.Parameter(torch.zeros(4), requires_grad=True)
            for p in self.linear.parameters():
                p.requires_grad = False
            self.linear.eval()
            for p in self.resize.parameters():
                p.requires_grad = False
            self.resize.eval()
            for p in self.cnn_encoder.parameters():
                p.requires_grad = False
            self.cnn_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()
        if self.aux_enable:
            for module in self.linear:
                module.eval()
            for module in self.resize:
                module.eval()
            self.cnn_encoder.eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            final, intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates
        self.context_list.append(final.detach())

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 引导特征
        x_cnn = x
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)

        # 构造特征金字塔: multi-layers -> multi-scale
        layer_1 = self.resize[0](layer_1, guide_1)  # 1/4
        layer_2 = self.resize[1](layer_2, guide_2)  # 1/8
        layer_3 = self.resize[2](layer_3, guide_3)  # 1/16
        layer_4 = self.resize[3](layer_4, guide_4)  # 1/32

        return layer_4, layer_3, layer_2, layer_1

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):

        input = layer_4

        if self.aux_enable:
            aux1, aux2, aux3, aux4 = self.aux_encoder(img)
            layer_1 = aux1 * self.weight[0] + layer_1 #+ self.bias[0]
            layer_2 = aux2 * self.weight[1] + layer_2 #+ self.bias[1]
            layer_3 = aux3 * self.weight[2] + layer_3 #+ self.bias[2]
            layer_4 = aux4 * self.weight[3] + layer_4 #+ self.bias[3]

        layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])

        return layer_4, layer_3, layer_2, layer_1

    def forward(self, x, y):
        # obtain multi-scale decoded features
        self.context_list.clear()  # 清空列表(包括内存)
        if self.aux_enable:
            with torch.no_grad():
                left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.encoder(x)
                right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.encoder(y)
        else:
            left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.encoder(x)
            right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.encoder(y)
        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.decoder(left_layer_4, left_layer_3, left_layer_2, left_layer_1, x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.decoder(right_layer_4, right_layer_3, right_layer_2, right_layer_1, y)

        # TODO: 将 context 和 corr 分开(类似 RAFT 的 share backbone), context 用于图像重建和语义理解, corr 用于跨视角匹配
        left_fpn, right_fpn = [], []
        for i in range(len(self.output_corr_chans)):
            left_fpn.append(self.corr_proj[i](eval('left_layer_' + str(i + 1))))
            right_fpn.append(self.corr_proj[i](eval('right_layer_' + str(i + 1))))

        context_fpn = []
        for i in range(len(self.output_feat_chans)):
            context_fpn.append(self.feat_proj[i](eval('left_layer_' + str(i + 1))))

        loss = 0.
        if self.training:

            # single-view reconstruction constrain
            left1 = F.interpolate(x, scale_factor=1/2**(min(self.n_downsample-3, 1)), mode="bilinear", align_corners=True)
            recon_left1 = self.recon_head[0](left_layer_1)
            left2 = F.interpolate(x, scale_factor=1/2**(min(self.n_downsample-3, 2)), mode="bilinear", align_corners=True)
            recon_left2 = self.recon_head[1](left_layer_2)
            left3 = F.interpolate(x, scale_factor=1/2**(min(self.n_downsample-3, 3)), mode="bilinear", align_corners=True)
            recon_left3 = self.recon_head[2](left_layer_3)
            left4 = F.interpolate(x, scale_factor=1/2**(min(self.n_downsample-3, 4)), mode="bilinear", align_corners=True)
            recon_left4 = self.recon_head[3](left_layer_4)
            alpha = 0.9
            recon_loss = alpha ** (min(self.n_downsample-2, 1)) * F.mse_loss(recon_left1, left1, size_average=True) + \
                         alpha ** (min(self.n_downsample-2, 2)) * F.mse_loss(recon_left2, left2, size_average=True) + \
                         alpha ** (min(self.n_downsample-2, 3)) * F.mse_loss(recon_left3, left3, size_average=True) + \
                         alpha ** (min(self.n_downsample-2, 4)) * F.mse_loss(recon_left4, left4, size_average=True)

            # context consistency
            left_context = self.context_head(left_layer_1)
            _, _, ref_h, ref_w = self.context_list[0].shape
            left_context = F.interpolate(left_context, size=(ref_h, ref_w), mode="bilinear", align_corners=True)
            context_loss = 0.9 * (1. - F.cosine_similarity(left_context, self.context_list[0], dim=1).mean()) + \
                           0.1 * F.l1_loss(left_context, self.context_list[0], size_average=True)

            loss += recon_loss + context_loss

        return {
            "left_fpn": left_fpn, "right_fpn": right_fpn, "context_fpn": context_fpn,
            "loss": loss,
        }


class HighResFormerplusplus(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求. 统一输出特征金字塔的分辨率为 [1/2, 1/4, 1/8, 1/16]
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            output_corr_chans=[32,64,96,128],   # corr feature channels
            output_feat_chans=[32,64,96,128],   # ref feature channels
            aggregation="sparse",               # 特征聚合方式, (sparse, dense)
            aux_enable=False,                   # 是否采用辅助网络捕获额外的信息
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
        )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_fn[backbone]         # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features, features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features),
        ) for _ in range(4)])

        # 构造 CNN 引导多尺度特征
        from .utils import HighResGuideNet as GuideNet
        self.cnn_encoder = GuideNet(features, depths=[2,2,2,2])     # 最大降采样到 1/16

        # 构造引导上采样模块, 输出为 norm 后的结果
        from .utils import guide_upsample
        self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])

        # 多尺度融合
        from .fpn1 import fpn_decoder
        self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        # output correspondent projection
        # output_corr_chans = output_corr_chans + [features] * (4 - len(output_corr_chans))   # 保证每个尺度都有关联损失参与监督
        self.output_corr_chans = output_corr_chans
        self.corr_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_corr_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_corr_chans))])

        # output features projection
        self.output_feat_chans = output_feat_chans
        self.feat_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_feat_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_feat_chans))])

        # single-view reconstruction
        self.recon_head = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, 3*4**(i+1), kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2**(i+1)),
        ) for i in range(4)])

        # context consistency
        self.context_head = nn.Sequential(# base resolution is 1/2
            nn.Conv2d(features, features, kernel_size=6, stride=2, padding=2, groups=8), nn.ReLU(),  # 1/4
            nn.Conv2d(features, features, kernel_size=6, stride=2, padding=2, groups=8), nn.ReLU(),  # 1/8
            nn.Conv2d(features, features, kernel_size=6, stride=2, padding=2, groups=8), nn.ReLU(),  # 1/16
            nn.Conv2d(features, vit_features, kernel_size=1)
        )
        self.context_list = []

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            final, intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates
        self.context_list.append(final.detach())

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 引导特征
        x_cnn = x
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)

        # 构造特征金字塔: multi-layers -> multi-scale
        layer_1 = self.resize[0](layer_1, guide_1)  # 1/2
        layer_2 = self.resize[1](layer_2, guide_2)  # 1/4
        layer_3 = self.resize[2](layer_3, guide_3)  # 1/8
        layer_4 = self.resize[3](layer_4, guide_4)  # 1/16

        return layer_4, layer_3, layer_2, layer_1

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):

        input = layer_4
        layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])

        return layer_4, layer_3, layer_2, layer_1

    def forward(self, x, y):
        # obtain multi-scale decoded features
        self.context_list.clear()  # 清空列表(包括内存)
        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.encoder(x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.encoder(y)

        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.decoder(left_layer_4, left_layer_3, left_layer_2, left_layer_1, x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.decoder(right_layer_4, right_layer_3, right_layer_2, right_layer_1, y)

        # 计算 corr features
        left_fpn, right_fpn = [], []
        for i in range(len(self.output_corr_chans)):
            left_fpn.append(self.corr_proj[i](eval('left_layer_' + str(i + 1))))
            right_fpn.append(self.corr_proj[i](eval('right_layer_' + str(i + 1))))

        # 计算 context features
        context_fpn = []
        for i in range(len(self.output_feat_chans)):
            context_fpn.append(self.feat_proj[i](eval('left_layer_' + str(i + 1))))

        loss = 0.
        if self.training:

            # single-view reconstruction constrain
            alpha = 0.9
            recon_loss = 0.
            for i in range(4):
                recon_x = self.recon_head[i](eval('left_layer_' + str(i + 1)))
                recon_loss += alpha**i * F.mse_loss(recon_x, x, reduction='mean')

            # context consistency
            left_context = self.context_head(left_layer_1)
            _, _, ref_h, ref_w = self.context_list[0].shape
            left_context = F.interpolate(left_context.float(), size=(ref_h, ref_w), mode="bilinear", align_corners=True)
            context_loss = 0.9 * (1. - F.cosine_similarity(left_context, self.context_list[0], dim=1).mean()) + \
                           0.1 * F.l1_loss(left_context, self.context_list[0], size_average=True)

            loss += recon_loss + context_loss

            # pre-count stereoInfo correlation
            # cross_constrain = []
            # for i in range(1, 4):
            #     b, _, h, w = left_fpn[i].shape
            #     cross_constrain.append(build_gwc_volume(left_fpn[i], right_fpn[i], w,  1, "cosine").permute(0, 1, 3, 4, 2).reshape(b, h*w, w))
            # cross_constrain_1 = build_gwc_volume(left_fpn[0], right_fpn[0], x.shape[-1] // 2,  1, "cosine")
            # cross_constrain_2 = build_gwc_volume(left_fpn[1], right_fpn[1], x.shape[-1] // 4,  1, "cosine")
            # cross_constrain_3 = build_gwc_volume(left_fpn[2], right_fpn[2], x.shape[-1] // 8,  1, "cosine")
            # cross_constrain_4 = build_gwc_volume(left_fpn[3], right_fpn[3], x.shape[-1] // 16, 1, "cosine")

        return {
            "left_fpn": left_fpn, "right_fpn": right_fpn, "context_fpn": context_fpn,
            "loss": loss, #"cross_constrain": cross_constrain
        }


class HighResFormerplusplus_v2(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求. 
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            GuideNet=None,                      # 引导网络, nn.Module
            output_corr_chans=[32,64,96,128],   # corr feature channels
            output_feat_chans=[32,64,96,128],   # ctex feature channels
            aggregation="sparse",               # 特征聚合方式, (sparse, dense)
            fast_attn=True,
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
            fast_attn=fast_attn,
        )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_fn[backbone]         # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features[i], features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features),
        ) for i in range(4)])

        # 构造 CNN 引导多尺度特征
        if GuideNet is None:
            from .utils import HighResGuideNet as DefaultGuideNet
            self.cnn_encoder = DefaultGuideNet(features, depths=[2,2,2,2])     # 最大降采样到 1/16
            self.scales = [2, 4, 8, 16]
        else:
            self.cnn_encoder = GuideNet                 # 已经实例化的网络
            self.scales = GuideNet.scales               # 指明尺度

        # 构造引导上采样模块, 输出为 norm 后的结果
        from .utils import guide_upsample
        self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])

        # 多尺度融合
        from .fpn1 import fpn_decoder
        self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        # output correspondent projection
        # output_corr_chans = output_corr_chans + [features] * (4 - len(output_corr_chans))   # 保证每个尺度都有关联损失参与监督
        self.output_corr_chans = output_corr_chans
        self.corr_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_corr_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_corr_chans))])

        # output features projection
        self.output_feat_chans = output_feat_chans
        self.feat_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_feat_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_feat_chans))])

        # single-view reconstruction
        self.recon_head = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, 3*self.scales[i]**2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=self.scales[i]),
        ) for i in range(4)])

        # context consistency
        summary_head = []
        summary_iter = self.scales[0]
        while summary_iter < 16:
            summary_head.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=6, stride=2, padding=2, groups=8), nn.ReLU(),
            ))
            summary_iter *= 2
        summary_head.append(nn.Conv2d(features, vit_features[0], kernel_size=1))   # for dino
        # summary_head.append(nn.Conv2d(features, 256, kernel_size=1))   # for sam
        # summary_head.append(nn.Conv2d(features, 1024, kernel_size=1))   # for swin, convnext
        self.context_head = nn.Sequential(*summary_head)
        self.context_list = []

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            final, intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates
        self.context_list.append(final.detach())

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 引导特征
        x_cnn = x
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)

        # 构造特征金字塔: multi-layers -> multi-scale
        layer_1 = self.resize[0](layer_1, guide_1)  # 1/2
        layer_2 = self.resize[1](layer_2, guide_2)  # 1/4
        layer_3 = self.resize[2](layer_3, guide_3)  # 1/8
        layer_4 = self.resize[3](layer_4, guide_4)  # 1/16

        return layer_4, layer_3, layer_2, layer_1

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):

        input = layer_4
        layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])

        return layer_4, layer_3, layer_2, layer_1

    def forward(self, x, y):
        # obtain multi-scale decoded features
        self.context_list.clear()  # 清空列表(包括内存)
        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.encoder(x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.encoder(y)

        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.decoder(left_layer_4, left_layer_3, left_layer_2, left_layer_1, x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.decoder(right_layer_4, right_layer_3, right_layer_2, right_layer_1, y)

        # 计算 corr features
        left_fpn, right_fpn = [], []
        for i in range(len(self.output_corr_chans)):
            left_fpn.append(self.corr_proj[i](eval('left_layer_' + str(i + 1))))
            right_fpn.append(self.corr_proj[i](eval('right_layer_' + str(i + 1))))

        # 计算 context features
        context_fpn = []
        for i in range(len(self.output_feat_chans)):
            context_fpn.append(self.feat_proj[i](eval('left_layer_' + str(i + 1))))

        loss = 0.
        if self.training:

            # single-view reconstruction constrain
            alpha = 0.9
            recon_loss = 0.
            for i in range(4):
                recon_x = self.recon_head[i](eval('left_layer_' + str(i + 1)))
                recon_w = alpha ** math.log2(self.scales[i] / self.scales[0])
                recon_loss += recon_w * F.mse_loss(recon_x, x, reduction='mean')

            # context consistency
            left_context = self.context_head(left_layer_1)
            _, _, ref_h, ref_w = self.context_list[0].shape
            left_context = F.interpolate(left_context.float(), size=(ref_h, ref_w), mode="bilinear", align_corners=True)
            context_loss = 0.9 * (1. - F.cosine_similarity(left_context, self.context_list[0], dim=1).mean()) + \
                           0.1 * F.l1_loss(left_context, self.context_list[0], size_average=True)

            loss += recon_loss + context_loss

            # pre-count stereoInfo correlation
            # cross_constrain = []
            # for i in range(1, 4):
            #     b, _, h, w = left_fpn[i].shape
            #     cross_constrain.append(build_gwc_volume(left_fpn[i], right_fpn[i], w,  1, "cosine").permute(0, 1, 3, 4, 2).reshape(b, h*w, w))
            # cross_constrain_1 = build_gwc_volume(left_fpn[0], right_fpn[0], x.shape[-1] // 2,  1, "cosine")
            # cross_constrain_2 = build_gwc_volume(left_fpn[1], right_fpn[1], x.shape[-1] // 4,  1, "cosine")
            # cross_constrain_3 = build_gwc_volume(left_fpn[2], right_fpn[2], x.shape[-1] // 8,  1, "cosine")
            # cross_constrain_4 = build_gwc_volume(left_fpn[3], right_fpn[3], x.shape[-1] // 16, 1, "cosine")

        return {
            "left_fpn": left_fpn, "right_fpn": right_fpn, "context_fpn": context_fpn,
            "loss": loss, #"cross_constrain": cross_constrain
        }
    

class HighResFormerplusplus_v2_remove_aux(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求. 
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            GuideNet=None,                      # 引导网络, nn.Module
            output_corr_chans=[32,64,96,128],   # corr feature channels
            output_feat_chans=[32,64,96,128],   # ctex feature channels
            aggregation="sparse",               # 特征聚合方式, (sparse, dense)
            fast_attn=True,
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
            fast_attn=fast_attn,
        )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_fn[backbone]         # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features[i], features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features),
        ) for i in range(4)])

        # 构造 CNN 引导多尺度特征
        if GuideNet is None:
            from .utils import HighResGuideNet as DefaultGuideNet
            self.cnn_encoder = DefaultGuideNet(features, depths=[2,2,2,2])     # 最大降采样到 1/16
            self.scales = [2, 4, 8, 16]
        else:
            self.cnn_encoder = GuideNet                 # 已经实例化的网络
            self.scales = GuideNet.scales               # 指明尺度

        # 构造引导上采样模块, 输出为 norm 后的结果
        from .utils import guide_upsample
        self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])

        # 多尺度融合
        from .fpn1 import fpn_decoder
        self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        # output correspondent projection
        # output_corr_chans = output_corr_chans + [features] * (4 - len(output_corr_chans))   # 保证每个尺度都有关联损失参与监督
        self.output_corr_chans = output_corr_chans
        self.corr_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_corr_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_corr_chans))])

        # output features projection
        self.output_feat_chans = output_feat_chans
        self.feat_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_feat_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_feat_chans))])

        # single-view reconstruction
        # self.recon_head = nn.ModuleList([nn.Sequential(
        #     ConvXBlock(features),
        #     nn.Conv2d(features, 3*self.scales[i]**2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PixelShuffle(upscale_factor=self.scales[i]),
        # ) for i in range(4)])

        # context consistency
        # summary_head = []
        # summary_iter = self.scales[0]
        # while summary_iter < 16:
        #     summary_head.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=6, stride=2, padding=2, groups=8), nn.ReLU(),
        #     ))
        #     summary_iter *= 2
        # summary_head.append(nn.Conv2d(features, vit_features, kernel_size=1))   # for dino
        # summary_head.append(nn.Conv2d(features, 256, kernel_size=1))   # for sam
        # summary_head.append(nn.Conv2d(features, 1024, kernel_size=1))   # for swin, convnext
        # self.context_head = nn.Sequential(*summary_head)
        # self.context_list = []

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            final, intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates
        # self.context_list.append(final.detach())

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 引导特征
        x_cnn = x
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)

        # 构造特征金字塔: multi-layers -> multi-scale
        layer_1 = self.resize[0](layer_1, guide_1)  # 1/2
        layer_2 = self.resize[1](layer_2, guide_2)  # 1/4
        layer_3 = self.resize[2](layer_3, guide_3)  # 1/8
        layer_4 = self.resize[3](layer_4, guide_4)  # 1/16

        return layer_4, layer_3, layer_2, layer_1

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):

        input = layer_4
        layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])

        return layer_4, layer_3, layer_2, layer_1

    def forward(self, x, y):
        # obtain multi-scale decoded features
        # self.context_list.clear()  # 清空列表(包括内存)
        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.encoder(x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.encoder(y)

        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.decoder(left_layer_4, left_layer_3, left_layer_2, left_layer_1, x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.decoder(right_layer_4, right_layer_3, right_layer_2, right_layer_1, y)

        # 计算 corr features
        left_fpn, right_fpn = [], []
        for i in range(len(self.output_corr_chans)):
            left_fpn.append(self.corr_proj[i](eval('left_layer_' + str(i + 1))))
            right_fpn.append(self.corr_proj[i](eval('right_layer_' + str(i + 1))))

        # 计算 context features
        context_fpn = []
        for i in range(len(self.output_feat_chans)):
            context_fpn.append(self.feat_proj[i](eval('left_layer_' + str(i + 1))))

        # loss = 0.
        # if self.training:

        #     # single-view reconstruction constrain
        #     alpha = 0.9
        #     recon_loss = 0.
        #     for i in range(4):
        #         recon_x = self.recon_head[i](eval('left_layer_' + str(i + 1)))
        #         recon_w = alpha ** math.log2(self.scales[i] / self.scales[0])
        #         recon_loss += recon_w * F.mse_loss(recon_x, x, reduction='mean')

        #     # context consistency
        #     left_context = self.context_head(left_layer_1)
        #     _, _, ref_h, ref_w = self.context_list[0].shape
        #     left_context = F.interpolate(left_context.float(), size=(ref_h, ref_w), mode="bilinear", align_corners=True)
        #     context_loss = 0.9 * (1. - F.cosine_similarity(left_context, self.context_list[0], dim=1).mean()) + \
        #                    0.1 * F.l1_loss(left_context, self.context_list[0], size_average=True)

        #     loss += recon_loss + context_loss

            # pre-count stereoInfo correlation
            # cross_constrain = []
            # for i in range(1, 4):
            #     b, _, h, w = left_fpn[i].shape
            #     cross_constrain.append(build_gwc_volume(left_fpn[i], right_fpn[i], w,  1, "cosine").permute(0, 1, 3, 4, 2).reshape(b, h*w, w))
            # cross_constrain_1 = build_gwc_volume(left_fpn[0], right_fpn[0], x.shape[-1] // 2,  1, "cosine")
            # cross_constrain_2 = build_gwc_volume(left_fpn[1], right_fpn[1], x.shape[-1] // 4,  1, "cosine")
            # cross_constrain_3 = build_gwc_volume(left_fpn[2], right_fpn[2], x.shape[-1] // 8,  1, "cosine")
            # cross_constrain_4 = build_gwc_volume(left_fpn[3], right_fpn[3], x.shape[-1] // 16, 1, "cosine")

        return {
            "left_fpn": left_fpn, "right_fpn": right_fpn, "context_fpn": context_fpn,
            "loss": 0., #"cross_constrain": cross_constrain
        }
    

class HighResFormerplusplus_v2_only_guided_upsample_block(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求. 
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            GuideNet=None,                      # 引导网络, nn.Module
            output_corr_chans=[32,64,96,128],   # corr feature channels
            output_feat_chans=[32,64,96,128],   # ctex feature channels
            aggregation="sparse",               # 特征聚合方式, (sparse, dense)
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
        )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_fn[backbone]         # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features[i], features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features),
        ) for i in range(4)])

        # 构造 CNN 引导多尺度特征
        if GuideNet is None:
            from .utils import HighResGuideNet as DefaultGuideNet
            self.cnn_encoder = DefaultGuideNet(features, depths=[2,2,2,2])     # 最大降采样到 1/16
            self.scales = [2, 4, 8, 16]
        else:
            self.cnn_encoder = GuideNet                 # 已经实例化的网络
            self.scales = GuideNet.scales               # 指明尺度

        # 构造引导上采样模块, 输出为 norm 后的结果
        from .utils import guide_upsample
        self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])

        # 多尺度融合
        # from .fpn1 import fpn_decoder
        # self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        # output correspondent projection
        # output_corr_chans = output_corr_chans + [features] * (4 - len(output_corr_chans))   # 保证每个尺度都有关联损失参与监督
        self.output_corr_chans = output_corr_chans
        self.corr_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_corr_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_corr_chans))])

        # output features projection
        self.output_feat_chans = output_feat_chans
        self.feat_proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_feat_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_feat_chans))])

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            final, intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates
        # self.context_list.append(final.detach())

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 引导特征
        x_cnn = x
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)

        # 构造特征金字塔: multi-layers -> multi-scale
        layer_1 = self.resize[0](layer_1, guide_1)  # 1/2
        layer_2 = self.resize[1](layer_2, guide_2)  # 1/4
        layer_3 = self.resize[2](layer_3, guide_3)  # 1/8
        layer_4 = self.resize[3](layer_4, guide_4)  # 1/16

        return layer_4, layer_3, layer_2, layer_1

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):

        # input = layer_4
        # layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])

        return layer_4, layer_3, layer_2, layer_1

    def forward(self, x, y):
        # obtain multi-scale decoded features
        # self.context_list.clear()  # 清空列表(包括内存)
        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.encoder(x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.encoder(y)

        left_layer_4, left_layer_3, left_layer_2, left_layer_1 = self.decoder(left_layer_4, left_layer_3, left_layer_2, left_layer_1, x)
        right_layer_4, right_layer_3, right_layer_2, right_layer_1 = self.decoder(right_layer_4, right_layer_3, right_layer_2, right_layer_1, y)

        # 计算 corr features
        left_fpn, right_fpn = [], []
        for i in range(len(self.output_corr_chans)):
            left_fpn.append(self.corr_proj[i](eval('left_layer_' + str(i + 1))))
            right_fpn.append(self.corr_proj[i](eval('right_layer_' + str(i + 1))))

        # 计算 context features
        context_fpn = []
        for i in range(len(self.output_feat_chans)):
            context_fpn.append(self.feat_proj[i](eval('left_layer_' + str(i + 1))))

        return {
            "left_fpn": left_fpn, "right_fpn": right_fpn, "context_fpn": context_fpn,
            "loss": 0., #"cross_constrain": cross_constrain
        }
    

class HighResFormerplusplus_v3_block(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求. 
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
       更新: 使用分离式的 encoder, 确保 corr 和 ctex 的信息不互相干扰. 同时能够有效利用不同模型的信息: DINOv2 对匹配的感知能力更强, DAM和SAM对context的感知能力更强
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            GuideNet=None,                      # 引导网络, nn.Module
            output_chans=[32,64,96,128],   # corr feature channels
            aggregation="sparse",               # 特征聚合方式, (sparse, dense)
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=hooks[backbone],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
        )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_fn[backbone]         # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features[i], features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features),
        ) for i in range(4)])

        # 构造 CNN 引导多尺度特征
        if GuideNet is None:
            from .utils import HighResGuideNet as DefaultGuideNet
            self.cnn_encoder = DefaultGuideNet(features, depths=[2,2,2,2])     # 最大降采样到 1/16
            self.scales = [2, 4, 8, 16]
        else:
            self.cnn_encoder = GuideNet                 # 已经实例化的网络
            self.scales = GuideNet.scales               # 指明尺度

        # 构造引导上采样模块, 输出为 norm 后的结果
        from .utils import guide_upsample
        self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])

        # 多尺度融合
        from .fpn1 import fpn_decoder
        self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        self.proj = nn.ModuleList([nn.Sequential(
            ConvXBlock(features),
            nn.Conv2d(features, output_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        ) for i in range(len(output_chans))])

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            final, intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 引导特征
        x_cnn = x
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)

        # 构造特征金字塔: multi-layers -> multi-scale
        layer_1 = self.resize[0](layer_1, guide_1)  # 1/2
        layer_2 = self.resize[1](layer_2, guide_2)  # 1/4
        layer_3 = self.resize[2](layer_3, guide_3)  # 1/8
        layer_4 = self.resize[3](layer_4, guide_4)  # 1/16

        return layer_4, layer_3, layer_2, layer_1

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):

        input = layer_4
        layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])

        return layer_4, layer_3, layer_2, layer_1

    def forward(self, x):
        # obtain multi-scale decoded features
        x_layer_4, x_layer_3, x_layer_2, x_layer_1 = self.encoder(x)
        y_layer_4, y_layer_3, y_layer_2, y_layer_1 = self.decoder(x_layer_4, x_layer_3, x_layer_2, x_layer_1, x)

        # 计算 corr features
        out_fpn = []
        for i in range(len(self.proj)):
            out_fpn.append(self.proj[i](eval('y_layer_' + str(i + 1))))

        return out_fpn
    

class HighResFormerplusplus_v3(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求. 
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
    """
    def __init__(
            self,
            backbone="damv2_vitl14_518",
            features=256,
            GuideNet=None,                      # 引导网络, nn.Module
            output_corr_chans=[32,64,96,128],   # corr feature channels
            output_feat_chans=[32,64,96,128],   # ctex feature channels
            aggregation="sparse",               # 特征聚合方式, (sparse, dense)
    ):
        super().__init__()
        if not isinstance(backbone, list):
            backbone = [backbone]
        
        self.feature = HighResFormerplusplus_v3_block(
            backbone=backbone[0],
            features=features,
            GuideNet=GuideNet,
            output_chans=output_corr_chans,
            aggregation=aggregation
        )
        if len(output_feat_chans) > 0:
            self.context = HighResFormerplusplus_v3_block(
                backbone=backbone[-1],
                features=features,
                GuideNet=GuideNet,
                output_chans=output_feat_chans,
                aggregation=aggregation
            )
        else:
            self.context = nn.Identity()

    def forward(self, im0, im1):
        feature0 = self.feature(im0)
        feature1 = self.feature(im1)
        context0 = self.context(im0)
        return {
            "left_fpn": feature0, "right_fpn": feature1, "context_fpn": context0, "loss": 0.
        }