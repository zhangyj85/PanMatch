import cv2
import numpy as np
import torch
import torch.nn.functional as F


def forward_backward_consistency_check(flow1, flow2, img1=None, img2=None):
    """
    前后一致性检查：计算光流的前后一致性误差。
    
    参数：
    - flow1 (Tensor): 从img1到img2的光流，形状为 [B, 2, H, W]
    - flow2 (Tensor): 从img2到img1的光流，形状为 [B, 2, H, W]
    - img1 (Tensor): 第一帧图像，形状为 [B, 3, H, W]
    - img2 (Tensor): 第二帧图像，形状为 [B, 3, H, W]
    
    返回：
    - consistency_error (Tensor): 光流前后一致性误差，形状为 [B, H, W]
    """
    
    # 获取批大小、图像高度和宽度
    B, _, H, W = flow1.shape
    device = flow1.device
    # assert B == 1, "Only support B = 1"

    # 坐标图
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    idea_img = torch.stack([grid_x, grid_y], dim=1)     # [B, 2, H, W]

    # 将flow2应用到img1上（即根据flow2从img2迁移到img1）
    grid1 = flow_to_grid(flow2)
    warp_img_1to2 = F.grid_sample(idea_img, grid1, align_corners=True)  # 默认使用 mode='bilinear', padding_mode='zeros'
    
    # 将flow1应用到img2上（即根据flow1从img1迁移到img2）
    grid2 = flow_to_grid(flow1)
    warp_img_1to2to1 = F.grid_sample(warp_img_1to2, grid2, align_corners=True)

    # 计算前后一致性误差：通过重采样图像后比较两张图像的差异
    forward_backward_error = torch.abs(idea_img - warp_img_1to2to1)  # flow1对img2的迁移误差
    
    return torch.sum(forward_backward_error ** 2, dim=1, keepdim=True).sqrt() 

def flow_to_grid(flow):
    """
    将光流转换为采样网格。
    
    参数：
    - flow (Tensor): 光流，形状为 [B, 2, H, W]
    
    返回：
    - grid (Tensor): 变换后的采样网格，形状为 [B, H, W, 2]
    """
    B, _, H, W = flow.shape
    device = flow.device
    # 创建网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # 将网格转换到[-1, 1]范围
    grid_x = (grid_x + flow[:, 0]) / W * 2 - 1  # flow[:, 0]是x方向的光流
    grid_y = (grid_y + flow[:, 1]) / H * 2 - 1  # flow[:, 1]是y方向的光流
    
    grid = torch.stack([grid_x, grid_y], dim=-1)  # 形状为 [B, H, W, 2], range(-1,1)
    
    return grid

def flow_for_warp(target, flow):
    """
    根据光流将 target 图像 warp 到 ref 平面
    """
    grid = flow_to_grid(flow)
    warp = F.grid_sample(target, grid, align_corners=True)
    return warp

def dense_warp(img1, img2, flow1, flow2, conf1, conf2, coff=1e-1):
    # 创建画布
    B, _, H, W = img1.shape
    white_im = torch.ones((B,1,H,W),device=img1.device)

    # 计算前后一致性
    diff_Eu = forward_backward_consistency_check(flow1, flow2, img1, img2)
    conf_check = 1. - F.tanh(coff * diff_Eu)     # coff=0.1 是缩放系数, 减缓在 diff 很小时置信度快速衰减的速率. coff 越小, 则前后一致性允许的误差越大
    
    # 从 img2 得到 warp img1, img2 in (0,1)
    warp_img1 = flow_for_warp(img2, flow1)
    confidence1 = conf1 * conf_check
    dense_im1 = confidence1 * warp_img1 + (1 - confidence1) * white_im

    # 计算前后一致性
    diff_Eu = forward_backward_consistency_check(flow2, flow1, img1, img2)
    conf_check = 1. - F.tanh(coff * diff_Eu)

    # 从 img1 得到 warp img2
    warp_img2 = flow_for_warp(img1, flow2)
    confidence2 = conf2 * conf_check
    dense_im2 = confidence2 * warp_img2 + (1 - confidence2) * white_im
    return dense_im1, confidence1, dense_im2, confidence2

def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

def sample(matches, certainty, max_num=10000, threshold=1, balanced=True):
    """
    max_num: 限制最大 key points 数量
    threshold: 将置信度大于 threshold 的置信度置为 1
    balanced: 在图像平面上均匀采样, 而不聚焦到某个集中的位置上
    """
    certainty = certainty.clone()
    certainty[certainty > threshold] = 1
    matches, certainty = (
        matches.reshape(-1, 4),     # [w1, h1, w2, h2]
        certainty.reshape(-1),
    )
    expansion_factor = 4 if balanced else 1
    good_samples = torch.multinomial(certainty, 
        num_samples = min(expansion_factor * max_num, len(certainty)),     # 保证最大采样点数量不超过 certainty 的数量
        replacement=False
    )
    good_matches, good_certainty = matches[good_samples], certainty[good_samples]
    if balanced:
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
            num_samples = min(max_num, len(good_certainty)), 
            replacement=False
        )
        good_matches, good_certainty = good_matches[balanced_samples], good_certainty[balanced_samples]
    return good_matches, good_certainty

# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    """
    K0, K1: [3, 3], Camera Intrinsics
    kpts0, kpts1: [-1, 2], wh order
    """
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0 = (K0inv @ (kpts0-K0[None,:2,2]).T).T     # 从图像坐标 (0,h) / (0,w) 转为归一化坐标 (-1, 1)
    kpts1 = (K1inv @ (kpts1-K1[None,:2,2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


if "__name__" == "__main__":
    # 示例用法：
    # 假设 flow1, flow2 是两个光流估计，img1 和 img2 是输入图像
    B, C, H, W = 1, 3, 256, 256  # 示例尺寸
    flow1 = torch.randn(B, 2, H, W)  # 从img1到img2的光流
    flow2 = torch.randn(B, 2, H, W)  # 从img2到img1的光流
    img1 = torch.randn(B, C, H, W)  # 第一帧图像
    img2 = torch.randn(B, C, H, W)  # 第二帧图像

    # 计算一致性误差
    consistency_error = forward_backward_consistency_check(flow1, flow2, img1, img2)
    print(consistency_error.shape)  # 输出一致性误差形状
