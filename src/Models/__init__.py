"""
Description:
(1) 构建算法总体框架，包括输入双目图像的预处理、选用匹配网络、视差结果后处理
(2) input: stereo images
    output: disparity map & confidence (options)
(3) This file including some utils funcs
"""
import importlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tile import *


def lcm(x, y):
    xy_gcd = math.gcd(x, y)
    xy_lcm = x * y // xy_gcd
    return xy_lcm


__Model_Dict__ = {
    "UniFormer_FlowFormer": {
        "dr": lcm(16, 14),              # 最大降采样比例
        "pad_method": "resize",         # padding 方法
        "unpad_method": "restore",      # unpadding 方法
    },
}


# 导入模型, 格式: 文件夹 + 模型 (两者同名)
def import_model(model_name: str):
    # load the module, will raise ImportError if module cannot be loaded
    if model_name not in __Model_Dict__.keys():
        raise ValueError(
            f"Model {model_name} not in MODELS list. Valid models are {__Model_Dict__.keys()}"
        )
    m = importlib.import_module("Models." + model_name)     # 得到Models文件夹下的module文件
    return getattr(m, model_name)                           # 得到class的定义


class IMG_Processer(object):
    def __init__(self, d_rate):
        # padding方式
        self.rate = d_rate          # 最大降采样倍率

    def padding(self, img):
        # 图像padding, 左边和上边做padding
        _, _, H, W = img.shape
        h_pad = ((H // self.rate + 1) * self.rate - H) % self.rate
        w_pad = ((W // self.rate + 1) * self.rate - W) % self.rate
        ### Improving: top reflect for context preserve, right zero pad for imambugurous matching
        # self.size = (0, w_pad, h_pad, 0)
        # img = F.pad(img, pad=(0, w_pad, 0, 0), mode="constant")   # top right pad, usually for disparity
        # img = F.pad(img, pad=(0, 0, h_pad, 0), mode="reflect")
        
        ### following flow padding mode, e.g., used in FlowFormer, unimatch
        self.size = (w_pad//2, w_pad-w_pad//2, 0, h_pad)    # for kitti
        # self.size = (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2)    # for sintel
        img = F.pad(img, pad=self.size, mode="replicate")
        return img

    def unpadding(self, img):
        # 去除padding区域
        h, w = img.shape[-2:]
        pad_wl, pad_wr, pad_ht, pad_hb = self.size
        img = img[..., pad_ht:h-pad_hb, pad_wl:w-pad_wr]
        return img

    def resize(self, img, update=True):
        # 将图像 resize 到合适的尺寸进行推理. 如果要对图像进行 resize, 需在 channel 前2维 padding 0
        _, _, H, W = img.shape
        if update:
            # 更新当前输入 batch 的原尺寸大小
            self.size = (H, W)
        resize_h = math.ceil(H / self.rate) * self.rate
        resize_w = math.ceil(W / self.rate) * self.rate
        img = F.interpolate(img, size=(resize_h, resize_w), mode="bilinear", align_corners=True)
        img[:, 0] *= resize_w / W 
        img[:, 1] *= resize_h / H
        return img

    def restore(self, flow):
        _, _, h, w = flow.shape
        flow = F.interpolate(flow, size=self.size, mode="bilinear", align_corners=True)
        (H, W) = self.size
        flow[:, 1] *= H / h
        flow[:, 0] *= W / w
        return flow


# 总模型定义
class MODEL(nn.Module):
    def __init__(self, config):
        super(MODEL, self).__init__()
        self.config = config
        model_name = config['model']['name']
        self.backbone = import_model(model_name)(config)
        self.model_name = model_name
        self.processer = IMG_Processer(d_rate=__Model_Dict__[model_name]["dr"])

    def forward(self, imgL, imgR, training_size=[384,384], flow_init=None):

        if self.training:
            output = self.backbone(imgL, imgR)

        else:
            pad_fn = getattr(self.processer, __Model_Dict__[self.model_name]["pad_method"])
            unpad_fn = getattr(self.processer, __Model_Dict__[self.model_name]["unpad_method"])
            if self.config['model']['tile_forward']:
                IMAGE_SIZE = imgL.shape[-2:]
                try:
                    TRAIN_SIZE = __Model_Dict__[self.model_name]["tile"]
                except:
                    TRAIN_SIZE = training_size
                # print("tile size is ", TRAIN_SIZE)
                TRAIN_SIZE[0] = min(TRAIN_SIZE[0], IMAGE_SIZE[0])  # 避免溢出
                TRAIN_SIZE[1] = min(TRAIN_SIZE[1], IMAGE_SIZE[1])

                min_overlap = (224,224)#(TRAIN_SIZE[0]-128, TRAIN_SIZE[1]-128)  # 重叠大小
                hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap=min_overlap)
                weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma=0.05)
                train_size = TRAIN_SIZE
                image_size = IMAGE_SIZE

                print(f"Using tiling strategy, tile size: {train_size}, overlap: {min_overlap}")

                # 基于全图(全局)预测先验, 再进行 tile 细化估计 (效果提升不明显, 先验作为初值并不能被网络很好地感知和利用起来)
                flows = 0
                flow_count = 0
                for idx, (h, w) in enumerate(hws):
                    image1_tile = imgL[:, :, h:h + train_size[0], w:w + train_size[1]]  # 截取图像
                    image2_tile = imgR[:, :, h:h + train_size[0], w:w + train_size[1]]

                    # 如果存在光流先验
                    if flow_init is not None:
                        flow_tile = flow_init[:, :, h:h + train_size[0], w:w + train_size[1]]
                        all_tile_pad = pad_fn(torch.cat([flow_tile, image1_tile, image2_tile], dim=1))
                        flow_tile_pad, image1_tile_pad, image2_tile_pad = all_tile_pad[:,0:2], all_tile_pad[:,2:5], all_tile_pad[:,5:8]
                    else:
                        flow_tile = torch.zeros_like(image1_tile)[:,:2]
                        all_tile_pad = pad_fn(torch.cat([flow_tile, image1_tile, image2_tile], dim=1))
                        flow_tile_pad, image1_tile_pad, image2_tile_pad = None, all_tile_pad[:,2:5], all_tile_pad[:,5:8]
                    
                    output = self.backbone(image1_tile_pad, image2_tile_pad, flow_tile_pad)            # 对 padding 后的截取图像进行推理
                    flow_pred_pad = output['flow']
                    if "conf" in output.keys():
                        flow_conf_pad = output['conf']
                        flow_comb_pad = torch.cat([flow_pred_pad, flow_conf_pad], dim=1)
                        flow_comb = unpad_fn(flow_comb_pad)                 # 去 padding
                        flow_pre  = flow_comb[:, :2]
                        flow_conf = flow_comb[:, 2:]
                        # 置信度设置最小阈值, 避免后续除零导致 NAN
                        flow_conf = flow_conf.clamp(min=1e-6)
                    else:
                        flow_pre = unpad_fn(flow_pred_pad)                  # 去 padding
                        flow_conf = weights[idx]

                    padding = (w, image_size[1] - w - train_size[1], h, image_size[0] - h - train_size[0], 0, 0)
                    flows += F.pad(flow_pre * flow_conf, padding)
                    flow_count += F.pad(flow_conf, padding)                          # 计算当前区域的光流和加权权重

                flow_pre = flows / flow_count

                # output = {"flow": flow_pre, "conf": flow_count}
                output = {"flow": flow_pre, "conf": torch.ones_like(flow_pre)[:,:1]}

            else:
                # 输入图像预处理, padding以适合网络
                imgL = pad_fn(imgL)
                imgR = pad_fn(imgR)
                output = self.backbone(imgL, imgR)

                # 对 padding 区域进行后处理
                # if self.config['task'] == "stereo":
                #     disp = output['disparity']
                #     zero = torch.zeros_like(disp)
                #     flow = torch.cat([disp, zero], dim=1)
                #     flow = self.processer.restore(flow)
                #     output['disparity'] = - flow[:, 0:1]
                #
                # elif self.config['task'] == "flow":
                output['flow'] = unpad_fn(output['flow'])
                # output['conf'] = F.interpolate(output['conf'], size=(output['flow'].shape[-2:]), mode="bilinear", align_corners=True) if 'conf' in output.keys() else torch.ones_like(output['flow'])[:,:1]
                output['conf'] = torch.ones_like(output['flow'])[:,:1]

            # else:
            #     raise NotImplementedError
            # TODO: 增加视差后处理模块(基于交叉熵的置信度估计滤波)
        return output


class LOSS(nn.Module):
    def __init__(self, config):
        super(LOSS, self).__init__()
        self.config = config
        model_name = config['model']['name']
        # 动态导入对应模型的损失函数模块
        m = importlib.import_module("Models." + model_name)
        loss_module = getattr(m, "loss_func")
        self.loss = loss_module(config)

    def forward(self, label_dict, training_output):
        return self.loss(label_dict, training_output)
