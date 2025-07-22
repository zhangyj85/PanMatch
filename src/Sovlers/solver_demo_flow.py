"""
Description:
    Author: Yongjian Zhang
    E-mail: zhangyj85@mail2.sysu.edu.cn
"""
import torch
import shutil
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Models import MODEL
from utils.tools import *
from utils.logger import ColorLogger
from utils.matcher_utiles import dense_warp
from cross_task_eval.match.kde import kde

from PIL import Image
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def draw_star(image, center, size, color):
    """
    在图像上绘制一个实心且尖角朝上的标准五角星
    :param image: 要绘制星星的图像
    :param center: 星星的中心坐标 (x, y)
    :param size: 星星的大小（外圆半径）
    :param color: 星星的颜色 (B, G, R)
    :return: 绘制好星星的图像
    """
    # 计算内圆半径，标准五角星内圆半径与外圆半径有固定比例关系
    inner_radius = size * 0.382
    outer_radius = size
    # 起始角度，让星星尖角朝上
    start_angle = -np.pi / 2
    points = []
    for i in range(5):
        # 计算外顶点的坐标
        outer_angle = start_angle + i * 2 * np.pi / 5
        outer_x = int(center[0] + outer_radius * np.cos(outer_angle))
        outer_y = int(center[1] + outer_radius * np.sin(outer_angle))
        points.append((outer_x, outer_y))

        # 计算内顶点的坐标
        inner_angle = outer_angle + np.pi / 5
        inner_x = int(center[0] + inner_radius * np.cos(inner_angle))
        inner_y = int(center[1] + inner_radius * np.sin(inner_angle))
        points.append((inner_x, inner_y))

    # 将顶点转换为适合 OpenCV 的格式
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))

    # 绘制实心星星
    cv2.fillPoly(image, [points], color)
    return image


# 后处理：去除 plt 坐标轴和空白
def remove_axes_and_whitespace():
    # 获取当前坐标轴
    ax = plt.gca()
    
    # 移除坐标轴
    ax.axis('off')
    
    # 移除所有装饰
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 移除标题（如果需要）
    ax.set_title("")
    
    # 设置坐标轴位置为覆盖整个图形
    ax.set_position([0, 0, 1, 1])
    
    # 获取当前图形并调整
    fig = plt.gcf()
    fig.set_frameon(False)  # 移除图形边框
    
    # 调整边距
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


def scale_camera_intrinsics(K, s_x, s_y):
    """
    根据图像的缩放比例调整相机内参矩阵。

    参数:
    K (numpy.ndarray): 原始相机内参矩阵，形状为 (3, 3)。
    s_x (float): 图像在宽度方向上的缩放比例。
    s_y (float): 图像在高度方向上的缩放比例。

    返回:
    numpy.ndarray: 缩放后的相机内参矩阵，形状为 (3, 3)。
    """
    # 提取原始相机内参
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    # 计算缩放后的相机内参
    f_x_new = s_x * f_x
    f_y_new = s_y * f_y
    c_x_new = s_x * c_x
    c_y_new = s_y * c_y

    # 构建缩放后的相机内参矩阵
    K_new = np.array([[f_x_new, 0, c_x_new],
                      [0, f_y_new, c_y_new],
                      [0, 0, 1]])

    return K_new


class DemoSolver(object):
    def __init__(self, config):
        self.config = config
        log_path = os.path.join(self.config['record']['path'], self.config['model']['name'])
        self.logger = ColorLogger(log_path, 'logger.log')

        # 获取模型
        self.model = MODEL(self.config)

        # 创建工具包
        image_mean = self.config["data"]["mean"]                                # RGB图, 三通道的归一化数值
        image_std  = self.config["data"]["std"]
        if len(image_std) < 3:
            image_mean, image_std = image_mean * 3, image_std * 3               # 灰度图, 三通道的归一化数值相同
        self.Imagetool = TensorImageTool(mean=image_mean, std=image_std)

    def load_checkpoint(self):
        ckpt_full = self.config['train']['resume']
        states = torch.load(ckpt_full, map_location="cpu")
        self.model.load_state_dict(states['model_state'], strict=True)
        self.tile_size = states['training_config']['data']['crop_size']         # 读取模型训练期间采用的 crop size

    def sample(self, matches, certainty, num=10000):
        # ref: https://github.com/Parskatt/RoMa/blob/main/romatch/models/matcher.py
        self.sample_mode = "threshold_balanced"     # 当置信度大于最大阈值, 则视为完全可靠; balanced 用于根据密度进行采样, 如果密度很低时, 需要换成 "threshhold"
        self.sample_thresh = 0.999                  # 对齐 RoMa 的 setting
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)    # Kernel Density Estimation, 返回 density (N,) 表示匹配点与其他点的空间邻近性, 输出密度值以反映匹配的可靠性，常用于离群点检测或数据去噪
        p = 1 / (density+1)                     # density 的范围为 (0,N), 密度太大就表示冗余太多, 这些点就会被以更小的概率选中
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]
    
    def flow2match(self, flow, certainty):
        b, _, hs, ws = flow.shape
        device = flow.device

        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            ),
            indexing = 'ij'
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)

        # 计算坐标对应关系
        im_B_coords = 0.5 * (im_A_coords + 1) * torch.tensor([ws, hs], device=device).view(1,2,1,1) + flow      # 逆归一化后, 加上光流, 得到 im_A 匹配点在 im_B 图像中的具体坐标
        im_B_coords = 2.0 * (im_B_coords / torch.tensor([ws, hs], device=device).view(1,2,1,1)) - 1             # 坐标归一化到 (-1,1)
        if (im_B_coords.abs() > 1).any() and True:
            # 将超出图像显示范围的估计结果去除掉, 限制坐标极值为 (-1,1)
            wrong = (im_B_coords.abs() > 1).sum(dim=1) > 0
            certainty[wrong[:,None]] = 0
        im_B_coords = torch.clamp(im_B_coords, -1, 1)

        warp = torch.cat([im_A_coords, im_B_coords], dim=1)
        return warp.permute(0, 2, 3, 1), certainty

    def save_match_results(self, imA, imB, output, save_path):
        ### 前馈并筛选有效点
        output_AtoB = output
        output_BtoA = self.model(imB, imA, self.tile_size)
        flow_AtoB, conf_AtoB = output_AtoB['flow'], output_AtoB['conf']
        flow_BtoA, conf_BtoA = output_BtoA['flow'], output_BtoA['conf']

        # 得到 warp 图像
        imA = self.Imagetool.renormalize(imA, fprint=False).permute(0, 3, 1, 2)
        imB = self.Imagetool.renormalize(imB, fprint=False).permute(0, 3, 1, 2)
        # 将图像归一化到 (0,1); 由于使用了 tile 策略, 置信度可能超过 1, 将超过 1 的区域置为 1
        dense_imA, conf_imA, dense_imB, conf_imB = dense_warp(imA / 255., imB / 255., flow_AtoB, flow_BtoA, conf_AtoB.clamp(max=1), conf_BtoA.clamp(max=1), coff=0.5)

        """keypoint sparse matching"""
        ### flow(B,2,H,W) -> matches(B,H,W,4),  pixel range -> grid range(-1,1)
        dense_matches, dense_certainty = self.flow2match(flow_AtoB, conf_imA)
        try:
            sparse_matches, sparse_certainty = self.sample(
                dense_matches, dense_certainty, 500
            )
        except:
            # 置信度全部为零, 随机选点
            sparse_matches, sparse_certainty = self.sample(
                dense_matches, 0.01 * torch.ones_like(dense_certainty), 500
            )

        offset = 0.5
        _, _, h1, w1 = imA.shape
        _, _, h2, w2 = imB.shape
        kpts1 = sparse_matches[:, :2].data.cpu().numpy()   # (N,2), range (-1,1)
        kpts1 = (
            np.stack(
                (
                    w1 * (kpts1[:, 0] + 1) / 2 - offset,
                    h1 * (kpts1[:, 1] + 1) / 2 - offset,
                ),
                axis=-1,
            )
        )
        kpts2 = sparse_matches[:, 2:].data.cpu().numpy()
        kpts2 = (
            np.stack(
                (
                    w2 * (kpts2[:, 0] + 1) / 2 - offset,
                    h2 * (kpts2[:, 1] + 1) / 2 - offset,
                ),
                axis=-1,
            )
        )
        import kornia.feature as KF
        from kornia_moons.viz import draw_LAF_matches
        inner_thr = 0.1     # 置信度大于 0.1 的算做正确匹配
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(kpts1[None]).cpu()),
            KF.laf_from_center_scale_ori(torch.from_numpy(kpts2[None]).cpu()),
            np.concatenate([np.arange(len(kpts1)).reshape(-1,1), np.arange(len(kpts2)).reshape(-1,1) ], axis=1),
            imA[0].permute(1,2,0).contiguous().data.cpu().numpy().astype(np.uint8),
            imB[0].permute(1,2,0).contiguous().data.cpu().numpy().astype(np.uint8),
            (sparse_certainty > inner_thr).data.cpu().numpy().astype(bool),
            draw_dict={"inlier_color": (0.2, 1, 0.2, 0.2), "tentative_color": ( 1, 0.2, 0.3, 0.2), "feature_color": None, "vertical": False}    # vertical 控制两张图的排列方向
        )
        # 保存当前的 sparse matching 结果
        remove_axes_and_whitespace()
        plt.savefig(os.path.join(save_path, "sparse_matching.png"), 
            bbox_inches='tight',                # 紧密边界框
            pad_inches=-0.01,                   # -0.01 用来解决可能产生的 1 像素透明边界
            transparent=True,                   # 透明背景（可选）
        )
        plt.close()
        
        """dense matching"""
        dense_imA = (dense_imA * 255).type(torch.uint8)
        dense_imB = (dense_imB * 255).type(torch.uint8)
        # 组合图像
        im0 = torch.cat([imA, dense_imA], dim=2)
        self.Imagetool.ImageSave(im0[0].permute(1, 2, 0), save_path + '/match_im0.png')  
        im1 = torch.cat([imB, dense_imB], dim=2)
        self.Imagetool.ImageSave(im1[0].permute(1, 2, 0), save_path + '/match_im1.png')  
        # 输出不确定度
        conf_imA = conf_imA.mul(255).type(torch.uint8)
        self.Imagetool.ImageSave(conf_imA[0].permute(1, 2, 0), save_path + '/confidence_im0.png')  
        conf_imB = conf_imB.mul(255).type(torch.uint8)
        self.Imagetool.ImageSave(conf_imB[0].permute(1, 2, 0), save_path + '/confidence_im1.png')  

        return output_BtoA
        
    def save_flow_results(self, imgL, imgR, flow_pt, save_path):

        color_flow_mag = np.percentile((flow_pt**2).sqrt().view(-1).data.cpu().numpy(), 97)  # 设置可视化的显示范围

        # save color estimated disparity
        color_pt_flow = self.Imagetool.colorize_flow(flow_pt, color_flow_mag)
        color_pt_flow = color_pt_flow.type(torch.uint8)
        self.Imagetool.ImageSave(color_pt_flow[0], save_path + "/ref_color_flow_pt.png")
        return color_pt_flow

    def save_disp_results(self, imgL, imgR, flow_pt, save_path):

        disp_pt = -flow_pt[:, :1, ...]
        disp_pt = -disp_pt if disp_pt.max() <= 0 else disp_pt   # 当左右图反转时, 用绝对值表示视差
        max_color_disp = np.percentile(disp_pt.view(-1).data.cpu().numpy(), 97)
        min_color_disp = np.percentile(disp_pt.view(-1).data.cpu().numpy(), 3)

        # save color estimated disparity
        color_pt_disp = self.Imagetool.colorize_disp(disp_pt, max_color_disp, min_color_disp, fprint=False)
        color_pt_disp = color_pt_disp.type(torch.uint8)
        self.Imagetool.ImageSave(color_pt_disp[0], save_path + "/left_color_-xdisp_pt.png")
        return color_pt_disp
    
    def show_query_heatmap(self, img1, img2, feat1, feat2, flow, save_path, query=(224,224), t=0.07):
        # 计算全局响应
        B, c, h, w = feat1.shape
        feat1 = feat1.permute(0, 2, 3, 1).reshape(B, h*w, c)
        feat2 = feat2.reshape(B, c, h*w)
        affinity = torch.bmm(feat1, feat2)
        norm1 = torch.norm(feat1, dim=-1, keepdim=True) # (B, hw, 1)
        norm2 = torch.norm(feat2, dim=-2, keepdim=True) # (B, 1, hw)
        norm_affinity = affinity / (torch.bmm(norm1, norm2) + 1e-8) # (B, hw, hw), dim=-1 means the proposal
        score = norm_affinity
        score = score.reshape(B, h, w, h, w)

        # 还原图像到 cv2 格式
        img1 = self.Imagetool.renormalize(img1, fprint=False)[0].data.cpu().numpy().astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = self.Imagetool.renormalize(img2, fprint=False)[0].data.cpu().numpy().astype(np.uint8)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # 查找对应位置的响应
        img1_mark = draw_star(img1, center=query, size=10, color=(0,0,255))
        cv2.imwrite(os.path.join(save_path, "marked_reference.png"), img1_mark)
        
        H, W, _ = img1.shape
        drate = H // h
        query_h, query_w = query[1] // drate, query[0] // drate
        score_slice = score[0, query_h, query_w]
        score_slice = F.interpolate(score_slice[None, None], scale_factor=drate, mode="nearest")[0,0]
        norm_score = (score_slice - score_slice.min()) / (score_slice.max() - score_slice.min())
        heatmap = (norm_score.data.cpu().numpy() * 255).astype(np.uint8)
        heatmap = plt.get_cmap('jet')(heatmap)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        alpha = 0.4 # 透明度
        # heatmap 是直接利用特征的相似性得到的, 其尺寸经过 padding 处理. 因此要对 img 进行对应的 resize / padding
        temp_img2 = cv2.resize(img2.astype(np.float32), (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)
        img2_heat = cv2.addWeighted(temp_img2.astype(np.uint8), 1 - alpha, heatmap, alpha, 0)
        # 添加预测光流认为的匹配位置
        target_position = (query[0] + flow[0, 0, query[1], query[0]],   # query in (w,h) format
                           query[1] + flow[0, 1, query[1], query[0]], )
        img2_heat_mark = draw_star(img2_heat, center=target_position, size=10, color=(0,0,255))
        cv2.imwrite(os.path.join(save_path, "marked_target_heatmap.png"), img2_heat_mark)


    def run(self):
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.logger.info("{} Model Inference {}".format("*"*20, "*"*20))     # 输出表头

        if self.config["train"]["resume"] is not None:
            # 若提供了预训练模型, 则加载预训练权重
            self.load_checkpoint()
            self.logger.info('Model loaded: {}, checkpoint: {}.'.format(self.config["model"]["name"], self.config["train"]["resume"]))
        else:
            print("No Model loaded.")

        self.model.eval()
        with torch.no_grad():

            # 处理输入图像对
            path_im0 = self.config['data']['im0']
            path_im1 = self.config['data']['im1']
            im0 = TF.normalize(TF.to_tensor(Image.open(path_im0).convert('RGB')), self.config["data"]["mean"], self.config["data"]["std"], inplace=True)
            im1 = TF.normalize(TF.to_tensor(Image.open(path_im1).convert('RGB')), self.config["data"]["mean"], self.config["data"]["std"], inplace=True)

            im0 = im0[None].to('cuda', non_blocking=True)
            im1 = im1[None].to('cuda', non_blocking=True)
            
            K0 = np.array(self.config['data']['K0'], dtype=np.float32)
            K1 = np.array(self.config['data']['K1'], dtype=np.float32)
            
            # 若不使用 tile, 需要对图像进行缩放
            if not self.config['model']['tile_forward']:
                _, _, h0, w0 = im0.shape
                _, _, h1, w1 = im1.shape
                flag1 = (h0 < self.tile_size[0]) & (h1 < self.tile_size[0]) & (w0 < self.tile_size[1]) & (w1 < self.tile_size[1])
                flag2 = (h0 == h1) & (w0 == w1)
                if not (flag1 & flag2):
                    im0 = F.interpolate(im0, size=self.tile_size, mode="bilinear", align_corners=True)
                    im1 = F.interpolate(im1, size=self.tile_size, mode="bilinear", align_corners=True)
                    # 记得对相机内参进行缩放
                    K0 = scale_camera_intrinsics(K0, s_x=self.tile_size[1]/w0, s_y=self.tile_size[0]/h0)
                    K1 = scale_camera_intrinsics(K1, s_x=self.tile_size[1]/w1, s_y=self.tile_size[0]/h1)
            
            # 如果图像分辨率太大, 则适当进行图像降采样
            # scale_factor = 1/4
            # im0 = F.interpolate(im0, scale_factor=scale_factor, mode="bilinear", align_corners=True)
            # im1 = F.interpolate(im1, scale_factor=scale_factor, mode="bilinear", align_corners=True)
            # K0 = scale_camera_intrinsics(K0, s_x=scale_factor, s_y=scale_factor)
            # K1 = scale_camera_intrinsics(K1, s_x=scale_factor, s_y=scale_factor)
            output = self.model(im0, im1, self.tile_size)
                
            flow_pred = output["flow"]  # 输出结果限制在有效范围内

            subpath = self.config['record']['path']
            os.makedirs(subpath, exist_ok=True)
            shutil.copyfile(path_im0, os.path.join(subpath, "im0_"+os.path.basename(path_im0)))     # 避免重名导致覆盖问题
            shutil.copyfile(path_im1, os.path.join(subpath, "im1_"+os.path.basename(path_im1)))

            if not self.config['model']['tile_forward']:
                # 使用 tiling 策略不方便通过特征相关来构建热力图
                fea0, fea1 = output['corr']
                self.show_query_heatmap(im0, im1, fea0, fea1, flow_pred, subpath, query=self.config['data']['query'])      # query = (w,h)
            self.save_disp_results(im0, im1, flow_pred, subpath)
            self.save_flow_results(im0, im1, flow_pred, subpath)
            output_BtoA = self.save_match_results(im0, im1, output, subpath)

            """Depth Estimation"""
            # 计算相对位姿, 作为深度估计的输入
            K_A = K0
            K_B = K1
            # NOTE: 使用相对位姿来简化计算. 当绝对位姿已知时, 可以准确计算绝对深度 (注意, 当光流非常小的时候, 深度估计的结果不可靠)
            # NOTE: 如果只知道相对位姿, 那么只能得到相对深度. 需要已知单位长度对应的真实长度才能还原场景的真实深度. 例如在 stereo 中对深度乘以基线长度, 可以得到绝对深度
            R1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            t1 = np.reshape(np.array([0., 0., 0.,]), (3, 1))
            T_AtoB = flow2pose(flow_pred, output_BtoA['flow'], K1=K_A, K2=K_B, mask_1to2=None)
            print(f"relative pose:\n {T_AtoB}")
            R2 = T_AtoB[:3,:3] @ R1
            t2 = T_AtoB[:3,3:] + T_AtoB[:3,:3] @ t1
            # 根据相对位姿和估计的光流, 计算深度, 并保存
            depth  = flow2absdep(flow_pred, K_A, R1, t1, K_B, R2, t2, vis=True)
            try:
                dep_max = np.percentile(depth[depth>0].data.cpu().numpy(), 97) # 取 97% 的最大值, 注意深度可能是负数, 因为内参不完全正确
                dep_min = np.percentile(depth[depth>0].data.cpu().numpy(), 3)  # 取  3% 的最小值
                depth_color = self.Imagetool.colorize(depth, dep_max=dep_max, dep_min=dep_min, fprint=False, cmap='plasma_r')     # 近处为亮色, 远出为暗色
                depth_color = depth_color.float() * (depth>0).float().permute(0, 2, 3, 1)  # 将无效深度置为 0
                depth_color = depth_color.type(torch.uint8)
                self.Imagetool.ImageSave(depth_color[0], subpath + "/reference_color_depth.png")
            except Exception as e:
                print(e)
