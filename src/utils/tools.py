import torch
import cv2
import os
import numpy as np
from .visualization import disp_color_func
from .flow_viz import flow_to_image

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Dep2PcdTool(object):
    """docstring for Dep2PcdTool， 将深度图转为点云"""
    def __init__(self, valid_dep=1e-5,
                 mean = [0.485, 0.456, 0.406],
                 std  = [0.229, 0.224, 0.225]):
        super(Dep2PcdTool, self).__init__()
        self.valid_dep = valid_dep  # 最小有效深度的大小, 默认0.01mm
        self.mean = mean            # 3通道均值, 单通道图像需要重复输入
        self.std  = std             # 3通道标准差

    def renormalize(self, color, fprint=True):
        """
        input:  tensor, bchw, torch.normalize 之后的结果
        output: tensor, bhwc, (0,1)
        """
        mean = torch.tensor(self.mean)
        std  = torch.tensor(self.std)
        color = color * std[None, :, None, None].cuda() + mean[None, :, None, None].cuda()
        if fprint: print("Renormalize Image in shape [B, H, W, 3],", color.shape)
        return color.permute(0,2,3,1) * 255

    def colorize(self, dep, dep_max=None, dep_min=None, fprint=True):
        """
        input:  tensor, b1hw, (dmin, dmax)
        output: tensor, bhw3, (0-255)
        """
        mask = (dep > self.valid_dep).float()           # 深度小于 0.01mm, 认为是无效深度

        if dep_max == None: dep_max = dep.max()
        if dep_min == None: dep_min = dep[mask.bool()].min()

        # 有效深度内的归一化, 无效深度为0
        dep_norm = (dep - dep_min) / (dep_max - dep_min + 1e-8) * 255 * mask

        device = dep.device
        if device != 'cpu': dep_norm = dep_norm.to('cpu')

        dep_out = []
        for i in range(dep.shape[0]):
            dep_color = cv2.applyColorMap(cv2.convertScaleAbs(dep_norm[i].permute(1, 2, 0).data.numpy(), alpha=1.0), cv2.COLORMAP_JET)
            dep_color = dep_color * (mask[i].permute(1,2,0).data.cpu().numpy().astype(np.uint8))       # 无效深度的颜色是黑色
            dep_color = cv2.cvtColor(dep_color, cv2.COLOR_BGR2RGB)                  # opencv 默认格式为 BGR, 为了正常显示, 需要转换为 RGB
            dep_out.append(torch.tensor(dep_color))
        dep = torch.stack(dep_out)

        if fprint: print("Colorize Image in shape [B, H, W, 3],", dep.shape)
        return dep.to(device)

    def pcd2ply(self, rgb, dep, calib, ply_file):
        """
        dep: numpy array (H, W, 1), 0-100
        rgb: numpy array (H, W, 3), 0-255
        """
        rgb = np.array(rgb, dtype="float32")
        dep = np.array(dep, dtype="float32")
        pcd = self.rgbd2pcd(rgb, dep, calib)
        # f"{}" replace the contain with variable
        header = "ply\n" + \
                 "format ascii 1.0\n" + \
                 f"element vertex {pcd.shape[0]}\n" +\
                 "property float32 x\n" + \
                 "property float32 y\n" + \
                 "property float32 z\n" + \
                 "property uint8 red\n" + \
                 "property uint8 green\n" + \
                 "property uint8 blue\n" + \
                 "end_header\n"
        with open(ply_file, 'w+') as f:
            f.write(header)
            for i in range(pcd.shape[0]):
                x, y, z, r, g, b = pcd[i,:]
                line = '{:.5f} {:.5f} {:.5f} {:.0f} {:.0f} {:.0f}\n'.format(x,y,z,r,g,b)
                f.write(line)

    def rgbd2pcd(self, rgb, dep, calib):
        """
        rgb: numpy array (H, W, 3), (0,1)
        dep: numpy array (H, W, 1), (0,192)
        pcd: numpy array (N, 6)
        """
        xyz = self.dep2xyz(dep, calib)  # (N, 3), N=HW
        rgb = rgb.reshape(-1, 3)        # (N, 3)
        valid = (dep > self.valid_dep).reshape(-1, )
        pcd = np.concatenate([xyz, rgb], axis=1)
        pcd = pcd[valid, :]     # 仅保留有效深度的像素点
        return pcd                      # (N, 6)

    def dep2xyz(self, dep, calib):
        """
        dep: numpy.array (H, W, 1)
        cal: numpy.array (3, 3)
        xyz: numpy.array (N, 3)
        """
        # 参数输出以便确认
        print("Intrinsic:\n", calib)

        # 生成图像坐标
        u, v = np.meshgrid(np.arange(dep.shape[1]), np.arange(dep.shape[0]))    # (H, W, 2)
        u, v = u.reshape(-1), v.reshape(-1)                                     # (H*W,), (H*W,)

        # 构成所需的坐标矩阵
        img_coord = np.stack([u, v, np.ones_like(u)])   # (3, H*W), (u,v,1)
        cam_coord = np.linalg.inv(calib) @ img_coord    # (3,3)^(-1) * (3, HW)
        xyz_coord = cam_coord * dep[v, u, 0]            # (3, HW)
        return xyz_coord.T                              # (HW, 3)


class TensorImageTool(object):
    """ImageTool Function, 图像处理"""
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(TensorImageTool, self).__init__()
        self.mean = torch.tensor(mean)
        self.std  = torch.tensor(std)

    def renormalize(self, color, fprint=True):
        """
        input:  tensor, bchw, torch.normalize 之后的结果
        output: tensor, bhwc, (0,255.)
        """
        mean = self.mean
        std  = self.std
        color = color * std[None, :, None, None].to(color.device) + mean[None, :, None, None].to(color.device)
        color = color.permute(0,2,3,1) * 255
        if fprint:
            print("Renormalize Image in shape [B, H, W, C],", color.shape, "data type:", color.dtype)
        return color

    def colorize(self, dep, dep_max=None, dep_min=None, cmap="jet", fprint=True):
        """
        input:  tensor, b1hw, (dmin, dmax)
        output: tensor, bhw3, (0-255)
        """
        if dep_max == None:
            dep_max = np.percentile(dep[dep > 1e-3].data.cpu().numpy(), 97) # 取 97% 的最大值
        if dep_min == None:
            dep_min = np.percentile(dep[dep > 1e-3].data.cpu().numpy(), 3)  # 取  3% 的最小值

        dep_norm = (dep - dep_min) / (dep_max - dep_min + 1e-8) * 255
        dep_norm = dep_norm.clamp(0, 255)

        device = dep.device
        if device != 'cpu':
            dep_norm = dep_norm.to('cpu')

        dep_out = []
        for i in range(dep.shape[0]):
            # dmax * alpha + beta = 255
            # dmin * alpha + beta = 0
            # alpha = 255.0 / (dep_max - dep_min)
            # beta  = - dep_min * alpha
            # dep_color = cv2.applyColorMap(cv2.convertScaleAbs(dep_norm[i].permute(1,2,0).data.numpy(), alpha=alpha, beta=beta), cv2.COLORMAP_JET)
            dep_ = dep_norm[i, 0].data.cpu().numpy().astype(np.uint8)
            dep_ = plt.get_cmap(cmap)(dep_)[:, :, :3]
            dep_ = (dep_ * 255).astype(np.uint8)
            # dep_ = cv2.cvtColor(dep_, cv2.COLOR_RGB2BGR)
            dep_out.append(torch.tensor(dep_))
        dep = torch.stack(dep_out)

        if fprint:
            print("Colorize Image in shape [B, H, W, 3],", dep.shape)
        return dep.to(device)

    def colorize_disp(self, disp, disp_max=None, disp_min=None, fprint=True):
        """
        input:  tensor, b1hw, (dmin, dmax)
        output: tensor, bhw3, (0-255)
        """
        device = disp.device
        if device != 'cpu':
            disp = disp.to('cpu')

        color_disp = []
        for i in range(disp.shape[0]):
            temp_disp = disp[i]     # (1,H,W)
            # 对视差进行归一化
            if disp_max == None:
                disp_max = np.percentile(temp_disp[temp_disp > 1e-3].data.cpu().numpy(), 97) # 取 97% 的最大值
            if disp_min == None:
                disp_min = np.percentile(temp_disp[temp_disp > 1e-3].data.cpu().numpy(), 3)  # 取  3% 的最小值
            temp_disp = (temp_disp - disp_min) / (disp_max - disp_min)
            temp_disp = temp_disp.clamp(0, 1)

            # 视差图上色
            _, H, W = temp_disp.shape
            temp_disp = temp_disp.reshape(H*W, 1).data.numpy()
            color_temp_disp = disp_color_func(temp_disp)
            color_temp_disp = color_temp_disp.reshape(H, W, 3) * 255.
            color_disp.append(torch.from_numpy(color_temp_disp))
        color_disp = torch.stack(color_disp)

        if fprint:
            print("Colorize Image in shape [B, H, W, 3],", color_disp.shape)
        return color_disp.to(device)

    def colorize_flow(self, flow, rad_max=None):
        b = flow.shape[0]
        device = flow.device
        if device != 'cpu':
            flow = flow.to('cpu')

        stack = []
        for i in range(b):
            np_flow = flow[i].permute(1,2,0).data.numpy()
            np_img = flow_to_image(np_flow, rad_max=rad_max, convert_to_bgr=False)
            stack.append(torch.from_numpy(np_img))
        color_flow = torch.stack(stack, dim=0)
        return color_flow.to(device)

    def ImageSave(self, img, save_path=None):
        """
        input: tensor, chw, color save, uint8
        """
        if img.device != 'cpu':
            img = img.to('cpu')
        img = img.data.numpy()   # np.array, HWC
        if save_path == None:
            save_path = './TempImage.png'


        # 注意, 如果使用 Image 打开(RGB), 需要对通道进行转换(RGB -> BGR)
        cv2.imwrite(save_path, img[..., ::-1])

    def DepthSave(self, depth, scale=1, save_path=None):
        """
        input: tensor, hw1, depth save, 16bit png
        需要再检查一下, 存在 0.1mm 的误差
        """
        if depth.device != 'cpu':
            depth = depth.to('cpu')
        depth = depth.data.numpy() / scale      # np.array, HWC
        depth = depth.astype(np.uint16)
        if save_path == None:
            save_path = './Temp-16bit.png'
        cv2.imwrite(save_path, depth)


def Images2Video(ImgList, savepath, fps=10):
    """
    ImgList: [(H,W,C), ...]
    size: (W, H)
    """
    H, W, _ = ImgList[0].shape
    size = (W, H)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    os.makedirs(savepath, exist_ok=True)
    video = cv2.VideoWriter(os.path.join(savepath, "video.mp4"), fourcc, fps, size)
    for img in ImgList:
        video.write(img[..., ::-1])
    video.release()
    cv2.destroyAllWindows()
    print('Finish create video file.')


def flow2absdep(flow_AtoB, K_A, R_A, T_A, K_B, R_B, T_B, vis=False):

    # 构建 reference view 的坐标
    b, _, h, w = flow_AtoB.shape
    device = flow_AtoB.device
    # reference 坐标图
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    grid_x = grid_x.float().unsqueeze(0).expand(b, -1, -1)  # [B, H, W]
    grid_y = grid_y.float().unsqueeze(0).expand(b, -1, -1)  # [B, H, W]
    grid_A = torch.stack([grid_x, grid_y], dim=1)     # reference view standard grid, [B, 2, H, W]
    grid_B = grid_A + flow_AtoB                       # target view   

    # 计算闭式解的辅助参数
    H =  K_B @ R_B @ np.linalg.inv(R_A) @ np.linalg.inv(K_A)          # (3,3)
    B = -K_B @ R_B @ np.linalg.inv(R_A) @ T_A + K_B @ T_B        # (3,1)

    # 注意, 在针孔模型中, (u,v) 是按照 w,h 的顺序表示的
    u1, v1 = grid_A[:,:1], grid_A[:,1:]
    u2, v2 = grid_B[:,:1], grid_B[:,1:]
    eps = 1e-8
    B_u = (B[2,0] * u2 - B[0,0])
    H_u = ((H[0,0] * u1 + H[0,1] * v1 + H[0,2]) - (H[2,0] * u1 + H[2,1] * v1 + H[2,2]) * u2 + eps)
    # d_u = B_u / H_u

    B_v = (B[2,0] * v2 - B[1,0])
    H_v = ((H[1,0] * u1 + H[1,1] * v1 + H[1,2]) - (H[2,0] * u1 + H[2,1] * v1 + H[2,2]) * v2 + eps)
    # d_v = B_v / H_v

    # 设定超定线性方程, 计算最小二乘解
    matrix_A = torch.cat([H_u, H_v], dim=1).permute(0, 2, 3, 1).reshape(b*h*w, 2, 1)   # (bN x 2 x 1)
    matrix_b = torch.cat([B_u, B_v], dim=1).permute(0, 2, 3, 1).reshape(b*h*w, 2, 1)   # (bN x 2 x 1)
    d_lsm = torch.bmm(torch.bmm(torch.inverse(torch.bmm(matrix_A.transpose(1,2), matrix_A)), matrix_A.transpose(1,2)), matrix_b)
    d_lsm = d_lsm.reshape(b, 1, h, w)

    if not vis:
        # 针对容易出错的地方进行滤波处理
        denominator = H_u ** 2 + H_v ** 2
        thr = 0.4 * K_A[0,0]   # 根据场景自适应的域值
        valid = d_lsm.isfinite() & (denominator > thr)
        d_lsm[~valid] = 0.
    
    # # 当分母不横跨 0 时, 所得的结果相对可信
    # m_u = H_u
    # m_v = H_v

    # thr_diff = 0.3   # 0.3m 的最大深度差异误差
    # thr_data = 0.1
    # d_mask = ((d_u.abs() + d_v.abs()) / 2) * ((d_u - d_v).abs() < thr_diff).float() * (m_u.abs() > thr_data * m_u.abs().max()).float() * (m_v.abs() > thr_data * m_v.abs().max()).float()

    # if vis:
    #     # 选择一致性最好的结果作为可视化结果
    #     flag_u = m_u.max() * m_u.min() > 0
    #     flag_v = m_v.max() * m_v.min() > 0
    #     if flag_u and flag_v:
    #         # 同时满足, 则取最小值最大的
    #         if m_u.min() >= m_v.min():
    #             return d_u.abs() * (m_u.abs() > thr_data * m_u.abs().max()).float()
    #         else:
    #             return d_v.abs() * (m_v.abs() > thr_data * m_v.abs().max()).float()
    #     elif flag_u:
    #         return d_u.abs() * (m_u.abs() > thr_data * m_u.abs().max()).float()
    #     elif flag_v:
    #         return d_v.abs() * (m_v.abs() > thr_data * m_v.abs().max()).float()
    return d_lsm


def flow2absdepv2(flow_AtoB, K_A, R_A, t_A, K_B, R_B, t_B):

    # 构建 reference view 的坐标
    b, _, h, w = flow_AtoB.shape
    device = flow_AtoB.device
    # reference 坐标图
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    grid_x = grid_x.float().unsqueeze(0).expand(b, -1, -1)  # [B, H, W]
    grid_y = grid_y.float().unsqueeze(0).expand(b, -1, -1)  # [B, H, W]
    grid_A = torch.stack([grid_x, grid_y], dim=1)     # reference view standard grid, [B, 2, H, W]
    grid_B = grid_A + flow_AtoB                       # target view   

    # 计算相对位姿: 从 B 到 A
    R_B2A = R_A @ R_B.T
    t_B2A = t_A - R_B2A @ t_B

    # 计算闭式解的参数
    grid_A_homogeneous = torch.cat([grid_A, torch.ones(b,1,h,w, device=device)], dim=1).permute(1, 0, 2, 3).reshape(3, b*h*w)
    xyz_A = np.linalg.inv(K_A) @ grid_A_homogeneous.data.cpu().numpy()
    grid_B_homogeneous = torch.cat([grid_B, torch.ones(b,1,h,w, device=device)], dim=1).permute(1, 0, 2, 3).reshape(3, b*h*w)
    xyz_B = R_B2A @ np.linalg.inv(K_B) @ grid_B_homogeneous.data.cpu().numpy()

    d_u = (t_B2A[0] * xyz_B[2] - t_B2A[2] * xyz_A[0]) / (xyz_B[0] * xyz_B[2] - xyz_B[2] * xyz_A[0] + 1e-8)
    d_u = torch.tensor(d_u, device=device).reshape(b,1,h,w)
    d_v = (t_B2A[1] * xyz_B[2] - t_B2A[2] * xyz_A[1]) / (xyz_B[1] * xyz_B[2] - xyz_B[2] * xyz_A[1] + 1e-8)
    d_v = torch.tensor(d_v, device=device).reshape(b,1,h,w)
    
    # 选择光流更大的分量计算深度
    d_mask = flow_AtoB[:,0:1].abs() > flow_AtoB[:,1:2].abs()
    d_mask = (~d_mask).long()   # 这里的逻辑有点拗口, 注意下别搞错了. 目前的写法没错
    d = torch.cat([d_u, d_v], dim=1)
    d = torch.gather(d, dim=1, index=d_mask)

    return d


def flow2reldep(flow_AtoB, K_A, R_A, T_A, K_B, R_B, T_B):

    # 构建 reference view 的坐标
    b, _, h, w = flow_AtoB.shape
    device = flow_AtoB.device
    # reference 坐标图
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    grid_x = grid_x.float().unsqueeze(0).expand(b, -1, -1)  # [B, H, W]
    grid_y = grid_y.float().unsqueeze(0).expand(b, -1, -1)  # [B, H, W]
    grid_A = torch.stack([grid_x, grid_y], dim=1)     # reference view standard grid, [B, 2, H, W]
    grid_B = grid_A + flow_AtoB                       # target view   

    # 计算闭式解的辅助参数
    H =  K_B @ R_B @ np.linalg.inv(R_A) @ np.linalg.inv(K_A)          # (3,3)
    B = -K_B @ R_B @ np.linalg.inv(R_A) @ T_A + K_B @ T_B        # (3,1)

    # 注意, 在针孔模型中, (u,v) 是按照 w,h 的顺序表示的
    # 像素坐标可以得到绝对深度, 但是容易出现 NAN; 换成归一化坐标后能获得稳定的相对深度的计算结果
    u1, v1 = grid_A[:,:1]/w, grid_A[:,1:]/h
    u2, v2 = grid_B[:,:1]/w, grid_B[:,1:]/h
    eps = 0
    d_u = (B[2,0] * u2 - B[0,0]) / \
          ((H[0,0] * u1 + H[0,1] * v1 + H[0,2]) - (H[2,0] * u1 + H[2,1] * v1 + H[2,2]) * u2 + eps)
    d_v = (B[2,0] * v2 - B[1,0]) / \
          ((H[1,0] * u1 + H[1,1] * v1 + H[1,2]) - (H[2,0] * u1 + H[2,1] * v1 + H[2,2]) * v2 + eps)
    
    # d_u 和 d_v 都是相对深度, 哪个没有 NAN 输出哪个
    invalid_flag = torch.isnan(d_u).any() or torch.isinf(d_u).any()
    if invalid_flag:
        return d_v
    return d_u


from utils.matcher_utiles import forward_backward_consistency_check
from cross_task_eval.match.utils import estimate_pose
import torch.nn.functional as F
def flow2pose(flow_1to2, flow_2to1, K1, K2, mask_1to2=None):
    B, _, h1, w1 = flow_1to2.shape
    _, _, h2, w2 = flow_2to1.shape
    diff_Eu = forward_backward_consistency_check(flow_1to2, flow_2to1)
    conf_check = 1. - F.tanh(0.1 * diff_Eu)     # 0.1 是缩放系数, 减缓在 diff 很小时置信度快速衰减的速率
    final_certainty = mask_1to2 * conf_check if mask_1to2 is not None else conf_check

    ### flow(B,2,H,W) -> matches(B,H,W,4),  pixel range -> grid range(-1,1)
    from cross_task_eval.match.scannet_test1500_benchmark import ScanNetBenchmark
    benchmark = ScanNetBenchmark()
    dense_matches, dense_certainty = benchmark.flow2match(flow_1to2, final_certainty)
    try:
        sparse_matches, sparse_certainty = benchmark.sample(
            dense_matches, dense_certainty, 10000
        )
    except:
        sparse_matches = torch.empty(0, 4, device=dense_matches.device)  # 稀疏匹配结果为空, 表示匹配失败

    ### 将图像缩放到 (480,640), ref: RoMa, DKM, LoFTR, SuperGlue
    scale1 = 480 / min(w1, h1)
    scale2 = 480 / min(w2, h2)
    w1, h1 = scale1 * w1, scale1 * h1
    w2, h2 = scale2 * w2, scale2 * h2
    K1 = K1 * scale1    # NOTE: ROMA 中的这个写法是不对的, 但是因为后续过程不使用 K[2:, :], 所以不影响结果
    K2 = K2 * scale2

    offset = 0.5
    kpts1 = sparse_matches[:, :2].data.cpu().numpy()
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
    for _ in range(5):
        shuffling = np.random.permutation(np.arange(len(kpts1)))
        kpts1 = kpts1[shuffling]
        kpts2 = kpts2[shuffling]
        try:
            norm_threshold = 0.5 / (
            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
            R_est, t_est, mask = estimate_pose(
                kpts1,
                kpts2,
                K1,
                K2,
                norm_threshold,
                conf=0.99999,
            )
            T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
            break
        except:
            continue
    return T1_to_2_est


def main():
    from PIL import Image
    import torchvision.transforms.functional as TF
    disp, _ = pfm_imread("/media/zhangyj85/Dataset/Stereo Datasets/Middlebury/2021/data/artroom1/disp0.pfm")
    disp = np.ascontiguousarray(disp, dtype=np.float32)
    disp = Image.fromarray(disp.astype('float32'), mode='F')
    K1 = np.array([[1733.74, 0, 792.27],
          [0, 1733.74, 541.89],
          [0,0,1]])
    K2 = np.array([[1733.74, 0, 792.27],
          [0, 1733.74, 541.89],
          [0,0,1]])
    baseline = 536.62
    dep = (K1[0, 2] - K2[0, 2]) + K1[0, 0] * baseline / (np.array(disp) + 1e-8)
    dep = Image.fromarray(dep)
    dep = TF.to_tensor(np.array(dep))
    dep = dep.unsqueeze(0) # tensor, B1HW
    PCDTool = Dep2PcdTool()
    PCDTool.pcd2ply(torch.ones_like(dep).repeat(1, 3, 1, 1)[0].data.cpu().numpy(), dep[0].permute(1, 2, 0).data.cpu().numpy(), K1, '/home/zhangyj85/Desktop/demo.ply')



def pfm_imread(filename):
    import re
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def PCA(tensor, n_components=5, savepath=None, cmap='viridis'):
    from sklearn import decomposition
    assert len(tensor.shape) == 4, "Only support 4D tensor with the shape of [B,C,H,W] where B = 1"
    n, c, h, w = tensor.shape
    feat = tensor[0].squeeze().reshape(c, h*w).permute(1, 0)  # [hw,c]
    feat = feat.data.cpu().numpy()
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(feat)
    pca_features = pca.transform(feat).reshape(h, w, -1)
    norm_feat = np.sqrt(np.sum(pca_features ** 2, axis=2))
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        import matplotlib.pyplot as plt
        for i in range(n_components):
            plt.imsave(os.path.join(savepath, 'pca_fea_{}.svg'.format(i)), pca_features[:,:,i], cmap=cmap)
        plt.imsave(os.path.join(savepath, 'pca_fea_sqrt.svg'), norm_feat, cmap=cmap)
    return pca_features, norm_feat


if __name__ == "__main__":
    main()