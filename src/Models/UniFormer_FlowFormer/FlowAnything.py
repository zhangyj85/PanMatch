import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from Models.UniFormer.extractor.utils import HighResGuideNet
DAMv2_Small = {
    "backbone": "damv2_vits14_518", "features": 96,
    "GuideNet": HighResGuideNet(96),
    "output_corr_chans": [96] * 4,
    "output_feat_chans": [96] * 4,
}

DAMv2_Base = {
    "backbone": "damv2_vitb14_518", "features": 128,
    "GuideNet": HighResGuideNet(128),
    "output_corr_chans": [128] * 4,
    "output_feat_chans": [128] * 4,
}

DAMv2_Large = {
    "backbone": "damv2_vitl14_518", "features": 192,
    "GuideNet": HighResGuideNet(192),
    "output_corr_chans": [192] * 4,
    "output_feat_chans": [192] * 4,
}

SAM_Base = {
    "backbone": "sam_vitb16_1024", "features": 128,
    "GuideNet": HighResGuideNet(128),
    "output_corr_chans": [128] * 4,
    "output_feat_chans": [128] * 4,
}

SAM_Large = {
    "backbone": "sam_vitl16_1024", "features": 192,
    "GuideNet": HighResGuideNet(192),
    "output_corr_chans": [192] * 4,
    "output_feat_chans": [192] * 4,
}

SAM_Huge = {
    "backbone": "sam_vith16_1024", "features": 256,
    "GuideNet": HighResGuideNet(256),
    "output_corr_chans": [256] * 4,
    "output_feat_chans": [256] * 4,
}

Swin_Base = {
    "backbone": "swinv1_base12_224", "features": 128,
    "GuideNet": HighResGuideNet(128),
    "output_corr_chans": [128] * 4,
    "output_feat_chans": [128] * 4,
}

ConvNeXt_Base = {
    "backbone": "convnext_base32_224", "features": 128,
    "GuideNet": HighResGuideNet(128),
    "output_corr_chans": [128] * 4,
    "output_feat_chans": [128] * 4,
}

DINOv2_Reg_Giant = {
    "backbone": "dinov2reg_vitg14_518", "features": 256,
    "GuideNet": HighResGuideNet(256),
    "output_corr_chans": [256] * 4,
    "output_feat_chans": [256] * 4,
}

DINOv2_Base = {
    "backbone": "dinov2_vitb14_518", "features": 128,
    "GuideNet": HighResGuideNet(128),
    "output_corr_chans": [128] * 4,
    "output_feat_chans": [128] * 4,
}

seperate_base = {
    "backbone": ["dinov2_vitb14_518", "damv2_vitb14_518"], 
    "features": 256,
    "GuideNet": HighResGuideNet(256),
    "output_corr_chans": [256] * 4,
    "output_feat_chans": [256] * 4,
}


def SingleScale_Patchembedding(inplanes, outplanes):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=4, stride=4, padding=0, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class MultiScales_PatchEmbedding(nn.Module):
    def __init__(self, in_planes=[128]*4, out_planes=256):
        super(MultiScales_PatchEmbedding, self).__init__()
        self.proj1 = nn.Conv2d(in_planes[0], out_planes // 4, kernel_size=4, stride=4, padding=0)           # 1/2 -> 1/8
        self.proj2 = nn.Conv2d(in_planes[1], out_planes // 4, kernel_size=2, stride=2, padding=0)           # 1/4 -> 1/8
        self.proj3 = nn.Conv2d(in_planes[2], out_planes // 4, kernel_size=1, stride=1, padding=0)           # 1/8 -> 1/8
        self.proj4 = nn.ConvTranspose2d(in_planes[3], out_planes // 4, kernel_size=2, stride=2, padding=0)
        self.merge = nn.GroupNorm(num_groups=4, num_channels=out_planes)
    def forward(self, x_list):
        x1 = self.proj1(x_list[0])
        x2 = self.proj2(x_list[1])
        x3 = self.proj3(x_list[2])
        x4 = self.proj4(x_list[3])
        x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.merge(x_)
        return x_


class MultiScales_PatchEmbedding_v2(nn.Module):
    def __init__(self, in_planes=[128]*4, out_planes=256, merge=True):
        super(MultiScales_PatchEmbedding_v2, self).__init__()
        self.proj1 = nn.Conv2d(in_planes[0], out_planes // 4, kernel_size=4, stride=4, padding=0)           # 1/2 -> 1/8
        self.proj2 = nn.Conv2d(in_planes[1], out_planes // 4, kernel_size=2, stride=2, padding=0)           # 1/4 -> 1/8
        self.proj3 = nn.Conv2d(in_planes[2], out_planes // 4, kernel_size=1, stride=1, padding=0)           # 1/8 -> 1/8
        self.proj4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                     nn.Conv2d(in_planes[3], out_planes // 4, kernel_size=3, stride=1, padding=1))          # 1/16 -> 1/8
        self.norm_ = nn.GroupNorm(num_groups=4, num_channels=out_planes)
        if merge:
            self.merge = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.merge = nn.Identity()

    def forward(self, x_list):
        x1 = self.proj1(x_list[0].float())
        x2 = self.proj2(x_list[1].float())
        x3 = self.proj3(x_list[2].float())
        x4 = self.proj4(x_list[3].float())
        x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.merge(self.norm_(x_))
        return x_
    

class MultiScales_PatchEmbedding_v2_bias(nn.Module):
    def __init__(self, in_planes=[128]*4, out_planes=256, merge=True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_planes[0], out_planes // 8 * 1, kernel_size=4, stride=4, padding=0)           # 1/2 -> 1/8
        self.proj2 = nn.Conv2d(in_planes[1], out_planes // 8 * 2, kernel_size=2, stride=2, padding=0)           # 1/4 -> 1/8
        self.proj3 = nn.Conv2d(in_planes[2], out_planes // 8 * 4, kernel_size=1, stride=1, padding=0)           # 1/8 -> 1/8
        self.proj4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                     nn.Conv2d(in_planes[3], out_planes // 8 * 1, kernel_size=3, stride=1, padding=1))          # 1/16 -> 1/8
        self.norm_ = nn.GroupNorm(num_groups=4, num_channels=out_planes)                                        # 不使用 relu 等激活函数, 以免信息丢失
        if merge:
            self.merge = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.merge = nn.Identity()

    def forward(self, x_list):
        x1 = self.proj1(x_list[0].float())
        x2 = self.proj2(x_list[1].float())
        x3 = self.proj3(x_list[2].float())
        x4 = self.proj4(x_list[3].float())
        x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.merge(self.norm_(x_))
        return x_
    

class MultiScales_PatchEmbedding_v2_bias2(nn.Module):
    def __init__(self, in_planes=[128]*4, out_planes=256, merge=True):
        super().__init__()
        ratio = [1, 2, 4, 1]
        num_groups = sum(ratio)
        self.proj1 = nn.Conv2d(in_planes[0], out_planes // num_groups * ratio[0], kernel_size=4, stride=4, padding=0)           # 1/2 -> 1/8
        self.proj2 = nn.Conv2d(in_planes[1], out_planes // num_groups * ratio[1], kernel_size=2, stride=2, padding=0)           # 1/4 -> 1/8
        self.proj3 = nn.Conv2d(in_planes[2], out_planes // num_groups * ratio[2], kernel_size=1, stride=1, padding=0)           # 1/8 -> 1/8
        self.proj4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                     nn.Conv2d(in_planes[3], out_planes // num_groups * ratio[3], kernel_size=3, stride=1, padding=1))          # 1/16 -> 1/8
        self.norm_ = nn.GroupNorm(num_groups=num_groups, num_channels=out_planes)                                        # 不使用 relu 等激活函数, 以免信息丢失
        if merge:
            self.merge = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.merge = nn.Identity()

    def forward(self, x_list):
        x1 = self.proj1(x_list[0].float())
        x2 = self.proj2(x_list[1].float())
        x3 = self.proj3(x_list[2].float())
        x4 = self.proj4(x_list[3].float())
        x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.merge(self.norm_(x_))
        return x_
    

class MultiScales_PatchEmbedding_v2_ablation(nn.Module):
    def __init__(self, in_planes=[128]*4, out_planes=256, merge=True):
        super(MultiScales_PatchEmbedding_v2_ablation, self).__init__()
        # self.proj1 = nn.Conv2d(in_planes[0], out_planes // 4, kernel_size=4, stride=4, padding=0)           # 1/2 -> 1/8
        # self.proj2 = nn.Conv2d(in_planes[1], out_planes // 4, kernel_size=2, stride=2, padding=0)           # 1/4 -> 1/8
        # self.proj3 = nn.Conv2d(in_planes[2], out_planes // 4, kernel_size=1, stride=1, padding=0)           # 1/8 -> 1/8
        # self.proj4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #              nn.Conv2d(in_planes[3], out_planes // 4, kernel_size=3, stride=1, padding=1))          # 1/16 -> 1/8
        # self.norm_ = nn.GroupNorm(num_groups=4, num_channels=out_planes)
        # if merge:
        self.merge = nn.Conv2d(in_planes[0], out_planes, kernel_size=4, stride=4, padding=0)
        # else:
        #     self.merge = nn.Identity()

    def forward(self, x_list):
        # x1 = self.proj1(x_list[0].float())
        # x2 = self.proj2(x_list[1].float())
        # x3 = self.proj3(x_list[2].float())
        # x4 = self.proj4(x_list[3].float())
        # x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.merge(x_list[0])
        return x_


class FlowAnything(nn.Module):
    def __init__(self, args):
        super(FlowAnything, self).__init__()
        self.bn = not args['model']['freeze_bn']
        self.conf_forward = args['model']['conf_forward']

        from .CorrTransformer import CorrTransformer as CorrDecoder
        self.decoder = CorrDecoder(
            pretrained=False,
            conf_forward=self.conf_forward,
        )
        # best model 移除了 aux loss, 使用改进的 patch embedding, 使用 SAM_Huge 来保证大尺度图像的正常训练
        from Models.UniFormer.extractor.models import HighResFormerplusplus_v2_remove_aux as LVMextracotr
        selective_model = DINOv2_Reg_Giant
        self.encoder = LVMextracotr(**selective_model)
        self.feature_embedding = MultiScales_PatchEmbedding_v2_bias2(in_planes=selective_model['output_corr_chans'], out_planes=256)
        self.context_embedding = MultiScales_PatchEmbedding_v2_bias2(in_planes=selective_model['output_feat_chans'], out_planes=256)

    def freeze_bn(self):
        if not self.bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                    m.eval()

    def forward(self, image1, image2, flow_init=None):
        """ Estimate optical flow between pair of frames """
        _, _, h, w = image1.shape

        # 临时冻结BN来解决小batch size情况下的训练问题
        self.freeze_bn()

        sync_context = torch.no_grad if self.conf_forward else nullcontext
        with sync_context():
            feat_dict = self.encoder(image1, image2)
            fmap1 = self.feature_embedding(feat_dict['left_fpn'])
            fmap2 = self.feature_embedding(feat_dict['right_fpn'])
            context = self.context_embedding(feat_dict['context_fpn'])
        corr1_dict = self.decoder(fmap1, fmap2, context, flow_init)

        output = {}
        output['flow'] = corr1_dict['flow_predictions'][-1]
        output['corr'] = [fmap1, fmap2]     # 用于构造 cost volume 的特征
        if self.training:
            output['flow_predictions'] = corr1_dict['flow_predictions']
            output['info_predictions'] = corr1_dict['info_predictions']
            output['auxiliary'] = feat_dict['loss']

        if self.conf_forward:
            w1 = torch.exp(corr1_dict['info_predictions'][-1][:,0:1])
            w2 = torch.exp(corr1_dict['info_predictions'][-1][:,1:2])
            output['conf'] = w2 / (w1 + w2)

        return output
