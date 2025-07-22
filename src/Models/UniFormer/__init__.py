import importlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


supported_baselines = [
    "PSMNet", "selective_IGEV",
    "SKFlow", "unimatch", "FlowFormer",
]


# 导入模型, 格式: 文件夹 + 模型 (两者同名)
def import_model(model_name: str):
    # load the module, will raise ImportError if module cannot be loaded
    if model_name not in supported_baselines:
        raise ValueError(
            f"Model {model_name} not in MODELS list. Valid models are {supported_baselines}"
        )
    m = importlib.import_module("." + model_name)           # 得到Models文件夹下的module文件
    return getattr(m, model_name)                           # 得到class的定义



# 总模型定义
class MODEL(nn.Module):
    def __init__(self, config):
        super(MODEL, self).__init__()
        self.config = config
        model_name = config['model']['name']
        self.backbone = import_model(model_name)(config)
        self.processer = IMG_Processer(d_rate=__DownSampleRate__[model_name])  # TODO: 将最大下采样与模型绑定

    def forward(self, imgL, imgR, training_size=[448,448]):

        if self.training:
            output = self.backbone(imgL, imgR)

        else:
            if self.config['model']['tile_forward']:
                IMAGE_SIZE = imgL.shape[-2:]
                TRAIN_SIZE = training_size
                TRAIN_SIZE[0] = min(TRAIN_SIZE[0], IMAGE_SIZE[0])  # 避免溢出
                TRAIN_SIZE[1] = min(TRAIN_SIZE[1], IMAGE_SIZE[1])

                hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap=224)
                weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma=0.05)
                train_size = TRAIN_SIZE
                image_size = IMAGE_SIZE

                flows = 0
                flow_count = 0
                for idx, (h, w) in enumerate(hws):
                    image1_tile = imgL[:, :, h:h + train_size[0], w:w + train_size[1]]  # 截取图像
                    image2_tile = imgR[:, :, h:h + train_size[0], w:w + train_size[1]]
                    image1_tile_pad = self.processer.padding(image1_tile)               # 截取的图像可能存在尺寸小于训练尺寸的问题, 因此需要进行 padding
                    image2_tile_pad = self.processer.padding(image2_tile)
                    output = self.backbone(image1_tile_pad, image2_tile_pad)            # 对 padding 后的截取图像进行推理
                    flow_pred_pad = output['flow']
                    flow_pre = self.processer.unpadding(flow_pred_pad)                  # 去 padding

                    padding = (w, image_size[1] - w - train_size[1], h, image_size[0] - h - train_size[0], 0, 0)
                    flows += F.pad(flow_pre * weights[idx], padding)
                    flow_count += F.pad(weights[idx], padding)                          # 计算当前区域的光流和加权权重

                flow_pre = flows / flow_count

                output = {"flow": flow_pre}

            else:
                # 输入图像预处理, padding以适合网络
                imgL = self.processer.padding(imgL)
                imgR = self.processer.padding(imgR)
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
                output['flow'] = self.processer.unpadding(output['flow'])

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
