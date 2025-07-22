import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import Block as ConvXBlock
from .convnext import LayerNorm, trunc_normal_


class FuseBlock(nn.Module):
    def __init__(self, features):
        super(FuseBlock, self).__init__()
        self.conv_ups = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, stride=1, padding=1),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )
        self.conv_out = ConvXBlock(features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, out, res):
        # 尺度对齐
        _, _, h, w = res.shape
        out = F.interpolate(out, size=(h, w), mode="nearest")
        out = self.conv_ups(out)

        output = self.conv_cat(torch.cat([res, out], dim=1))
        output = self.conv_out(output)
        return output


class fpn_decoder(nn.Module):
    def __init__(self, features, num_res_blocks=[1,1,1,1]):
        super().__init__()
        """fpn 的输入为 guided scaled features, 均经过 LayerNorm 处理, 因此使用 Norm"""
        self.features = features

        self.deepconv = ConvXBlock(features)
        self.fuseconv = nn.ModuleList()
        self.iresconv = nn.ModuleList()
        for i in range(len(num_res_blocks)):
            self.fuseconv.append(FuseBlock(features))
            self.iresconv.append(self._make_fuse_layers(num_res_blocks[i]))

    def _make_fuse_layers(self, num_res):
        layers = []
        for i in range(1, num_res):
            layers.append(ConvXBlock(self.features))
        return nn.Sequential(*layers)

    def forward(self, x, rem_list):

        # 注意, 所有输入均为 norm 后的结果, 此外, 该 decoder 保证所有输出都是 norm 后的结果
        x = self.deepconv(x)
        out = []
        for i in range(len(rem_list)):
            x = self.fuseconv[i](x, rem_list[i])
            x = self.iresconv[i](x)
            out.append(x)
        return out
