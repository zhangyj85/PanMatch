from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def disp2coord_dist(disp, mask=None, kernel=4, topk=4):
    # 从原图分辨率下的 disp map 中, 得到当前降采样分辨率下的视差频率统计, 近似为视差分布
    B, _, H, W = disp.shape
    assert (H % kernel == 0) and (W % kernel == 0), "Kernel {} cannot be divided by the shape of {}.".format((kernel, kernel), (H, W))
    h, w = H // kernel, W // kernel
    num_neighbors = kernel ** 2

    # 得到 matching coord 统计频数
    coords_left = torch.arange(0, W, device=disp.device, requires_grad=False).view(1, 1, 1, -1)     # (1,1,1,W)
    coords_right = coords_left - disp
    coords_right_unfold = F.unfold(coords_right, kernel_size=kernel, stride=kernel).view(B, num_neighbors, 1, h, w)
    coords_right_unfold /= kernel       # 坐标尺度缩放

    # mask 用于统计 kernel 内有效像素的个数, 将频数转为概率
    coords_valid = mask.float() if mask is not None else torch.ones_like(disp)
    coords_valid_unfold = F.unfold(coords_valid, kernel_size=kernel, stride=kernel).view(B, num_neighbors, 1, h, w)

    # 将频数映射到频率 volume (B,N,w,h,w)
    coords_proposals = torch.arange(0, W, step=kernel, device=disp.device, requires_grad=False)
    coords_proposals = (coords_proposals + 1/2 * (kernel - 1)) / kernel
    coords_proposals = coords_proposals.view(1, 1, -1, 1, 1).repeat(B, num_neighbors, 1, h, w)

    coords_volume = 1 - torch.abs(coords_right_unfold - coords_proposals)       # (B,N,w,h,w), 坐标到候选坐标的距离, 越小越靠近真值. 1-距离得到权重
    coords_volume[coords_volume < 0] = 0.                                       # 权重非负
    coords_volume *= coords_valid_unfold                                        # 去除 kernel 内真值无效的区域
    coords_volume = torch.sum(coords_volume, dim=1, keepdim=False)              # 沿着 num_neighbor 维度进行加权求和, 得到每个候选坐标维度(dim=2)的频数, (B,w,h,w)
    coords_volume /= (torch.sum(coords_valid_unfold, dim=1, keepdim=False) + 1e-8)      # 将频数转为概率, 此时沿着 dim=1 的 w 维度求和应为 1
    sorted_score_volume, ind = coords_volume.sort(dim=1, descending=True)       # 降序排列, 则权重靠前的为正样本

    # 得到正样本/负样本的 index
    positive_ind = ind[:, :2, :, :]                                             # (B,2,H,W), 仅考虑前两个作为正例, 因为第二个开始权重就已经锐减了
    negative_ind = ind[:, topk:, :, :]                                          # (B,W-topk,H,W), 允许 topk 之后的作为负样本
    valid = torch.sum(coords_valid_unfold, dim=1, keepdim=False) > 0            # 沿着 kernel 维度求和大于 0 的才存在正样本, 设置 0.99, 保证所在区域都能找得到 matching pixels
    return positive_ind, negative_ind, valid, sorted_score_volume, coords_volume


# @torch.no_grad()
# def flow2coord_dist(flow, kernel=4, topk=6):
#     B, _, H, W = flow.shape
#     assert (H % kernel == 0) and (W % kernel == 0), "Kernel {} cannot be divided by the shape of {}.".format((kernel, kernel), (H, W))
#     h, w = H // kernel, W // kernel
#     num_neighbors = kernel ** 2

#     # 得到统计频数
#     coord_H, coord_W = torch.meshgrid(
#         torch.arange(0, H, device=flow.device, requires_grad=False),
#         torch.arange(0, W, device=flow.device, requires_grad=False),
#         indexing='ij'
#     )
#     coord_reference = torch.stack([coord_W.unsqueeze(0), coord_H.unsqueeze(0)], dim=0).view(1, 2, H, W)
#     coord_match = coord_reference + flow     # (B,2,H,W)
#     coord_match_unfold = F.unfold(coord_match, kernel_size=kernel, stride=kernel).view(B, 2, num_neighbors, 1, h, w)
#     coord_match_unfold /= kernel            # 坐标轴缩放

#     # 将频数映射到频率 volume, 由于显存不够, 使用 for 循环, 时间换空间
#     num_splits = 64     # 1/8 尺度下, 拆分成 64 份. 实验发现, 就算把这个数值缩小到 4, 也不能显著提速, 因此还是以节省显存为主, 默认设置为 64
#     num_neighbors_split = num_neighbors // num_splits
#     coord_h, coord_w = torch.meshgrid(
#         torch.arange(0, H, step=kernel, device=flow.device, requires_grad=False),
#         torch.arange(0, W, step=kernel, device=flow.device, requires_grad=False),
#         indexing='ij'
#     )
#     # 构造低分辨率相对原图的中心位置坐标
#     coord_h = (coord_h + 1/2 * (kernel - 1)) / kernel
#     coord_w = (coord_w + 1/2 * (kernel - 1)) / kernel
#     # (2,h,w) -> (2,hw) -> (B,2,N,hw,h,w)
#     coord_proposals = torch.stack([coord_w, coord_h], dim=0).view(1, 2, 1, h*w, 1, 1).repeat(B, 1, num_neighbors_split, 1, h, w)
#     coord_volume_total = 0.
#     coord_match_unfold_splits = torch.split(coord_match_unfold, dim=2, split_size_or_sections=num_neighbors_split)
#     for n in range(num_splits):
#         coord_volume = 1 - torch.abs(coord_match_unfold_splits[n] - coord_proposals)    # (B,2,n,hw,h,w), 坐标到候选坐标的距离, 越小越靠近真值. 1-距离得到权重
#         invalid = (coord_volume[:,0] < 0).float() + (coord_volume[:,1] < 0).float()     # 权值非负
#         coord_volume_score = torch.sum(coord_volume, dim=1, keepdim=False)              # (B,n,hw,h,w)
#         coord_volume_score[invalid.bool()] = 0.                                         # 负权值置零, 仅保留有效的 weights
#         coord_volume_total += coord_volume_score                                        # (B,n,hw,h,w)
#     coord_volume_total = torch.sum(coord_volume_total, dim=1, keepdim=False)            # (B,hw,h,w)
#     coord_volume_total /= 4*num_neighbors
#     _, ind = coord_volume_total.sort(dim=1, descending=True)

#     # 输出正负样本对
#     positive_ind = ind[:, :2, :, :]     # (B,2,H,W), 仅考虑前两个作为正例, 因为第二个开始权重就已经锐减了
#     negative_ind = ind[:, topk:, :, :]  # (B,W-topk,H,W), 允许 topk 之后的作为负样本
#     valid = torch.sum(coord_volume_total, dim=1, keepdim=True) > 0.5  # 沿着 hw 维度求和大于 0 的才存在正样本, 设置 0.99, 保证所在区域都能找得到 matching pixels

#     """使用 stereo 可视化验证算法, 已验证"""
#     # temp_h = torch.arange(0, h, device=flow.device).view(1, 1, -1, 1).repeat(B, 1, 1, w)
#     # temp_w = torch.arange(0, w, device=flow.device).view(1, 1, 1, -1).repeat(B, 1, h, 1)
#     # plt.figure(), plt.imshow((((temp_h + temp_w - positive_ind) % w) * valid.float())[0,1].data.cpu()), plt.show()
#     return positive_ind, negative_ind, valid


@torch.no_grad()
def flow2coord_dist_fast(flow, mask=None, kernel=4, topk=6):
    """assuming flow w and flow h is independent, respectively."""
    B, _, H, W = flow.shape
    assert (H % kernel == 0) and (W % kernel == 0), (
        "Kernel {} cannot be divided by the shape of {}.".format((kernel, kernel), (H, W))
    )
    h, w = H // kernel, W // kernel
    num_neighbors = kernel ** 2

    # create ref coords
    coord_H, coord_W = torch.meshgrid(
        torch.arange(0, H, device=flow.device, requires_grad=False),
        torch.arange(0, W, device=flow.device, requires_grad=False),
        indexing='ij'
    )
    coord_reference = torch.stack([coord_W.unsqueeze(0), coord_H.unsqueeze(0)], dim=0).view(1, 2, H, W)

    # calculate the target coords, and then unfold it
    coord_match = coord_reference + flow  # (B,2,H,W)
    coord_match_unfold = F.unfold(coord_match, kernel_size=kernel, stride=kernel).view(B, 2, num_neighbors, 1, h, w)
    coord_match_unfold /= kernel  # 坐标轴缩放

    # mask 用于统计 kernel 内有效像素的个数, 将频数转为概率
    coord_valid = mask.float() if mask is not None else torch.ones_like(flow)[:,:1]
    coord_valid_unfold = F.unfold(coord_valid, kernel_size=kernel, stride=kernel).view(B, num_neighbors, 1, h, w)

    """首先计算 flow w 的 distribution"""
    coord_match_unfold_w = coord_match_unfold[:, 0]     # (B,N,1,h,w)

    # 将频数映射到频率 volume (B,N,w,h,w)
    coords_proposals_w = torch.arange(0, W, step=kernel, device=flow.device, requires_grad=False)
    coords_proposals_w = (coords_proposals_w + 1 / 2 * (kernel - 1)) / kernel                       # 坐标对齐到网格中心
    coords_proposals_w = coords_proposals_w.view(1, 1, -1, 1, 1).repeat(B, num_neighbors, 1, h, w)  # (B,N,w,h,w)

    coords_volume_w = 1 - torch.abs(coord_match_unfold_w - coords_proposals_w)  # (B,N,w,h,w), 坐标到候选坐标的距离, 越小越靠近真值. 1-距离得到权重
    coords_volume_w[coords_volume_w < 0] = 0.                                   # 权重非负
    coords_volume_w *= coord_valid_unfold                                       # 去除 kernel 内真值无效的区域
    coords_volume_w = torch.sum(coords_volume_w, dim=1, keepdim=False)          # 沿着 num_neighbor 维度进行加权求和, 得到每个候选坐标维度(dim=2)的频数, (B,w,h,w)
    coords_volume_w /= (torch.sum(coord_valid_unfold, dim=1, keepdim=False) + 1e-8)      # 将频数转为概率, 此时沿着 dim=1 的 w 维度求和应为 1
    del coords_proposals_w

    """重复计算 flow h 的 distribution"""
    coord_match_unfold_h = coord_match_unfold[:, 1]

    # 将频数映射到频率 volume (B,N,h,h,w)
    coords_proposals_h = torch.arange(0, H, step=kernel, device=flow.device, requires_grad=False)
    coords_proposals_h = (coords_proposals_h + 1 / 2 * (kernel - 1)) / kernel  # 坐标对齐到网格中心
    coords_proposals_h = coords_proposals_h.view(1, 1, -1, 1, 1).repeat(B, num_neighbors, 1, h, w)  # (B,N,h,h,w)

    coords_volume_h = 1 - torch.abs(coord_match_unfold_h - coords_proposals_h)  # (B,N,h,h,w), 坐标到候选坐标的距离, 越小越靠近真值. 1-距离得到权重
    coords_volume_h[coords_volume_h < 0] = 0.                                   # 权重非负
    coords_volume_h *= coord_valid_unfold                                       # 去除 kernel 内真值无效的区域
    coords_volume_h = torch.sum(coords_volume_h, dim=1, keepdim=False)          # 沿着 num_neighbor 维度进行加权求和, 得到每个候选坐标维度(dim=2)的频数, (B,w,h,w)
    coords_volume_h /= (torch.sum(coord_valid_unfold, dim=1, keepdim=False) + 1e-8)      # 将频数转为概率, 此时沿着 dim=1 的 w 维度求和应为 1
    del coords_proposals_h

    """将 flow w & flow h 组合"""
    coords_volume = torch.bmm(coords_volume_h.permute(0, 2, 3, 1).reshape(B*h*w, h, 1),
                              coords_volume_w.permute(0, 2, 3, 1).reshape(B*h*w, 1, w))
    # TODO: 通过可视化验证 coords_volume: 对于任意ref位置 (h0,w0), 其响应图为 hxw, 响应值仅出现在 target 坐标上
    # 调整 coords_volume
    coords_volume = coords_volume.reshape(B, h, w, h*w).permute(0, 3, 1, 2).contiguous()

    # 得到最终的正负样本集合
    sorted_score_volume, ind = coords_volume.sort(dim=1, descending=True)  # 降序排列, 则权重靠前的为正样本
    positive_ind = ind[:, :2, :, :]  # (B,2,H,W), 仅考虑前两个作为正例, 因为第二个开始权重就已经锐减了
    negative_ind = ind[:, topk:, :, :]  # (B,W-topk,H,W), 允许 topk 之后的作为负样本
    valid = torch.sum(coord_valid_unfold, dim=1, keepdim=False) > 0        # 只有 kernel 内存在有效的频数计算时, 才认为该区域的概率是有效的
    
    """TODO: check visualization, 已验证, 算法正确"""
    ref_ind = torch.arange(0, h*w, device=flow.device, requires_grad=False)
    del_ind = positive_ind - ref_ind.view(1, 1, h, w)           # 相对位置偏移, 即光流, 用一维坐标表示
    w_shift = torch.arange(0, w, device=flow.device).view(1, 1, 1, w).repeat(1, 1, h, 1)
    flow_ind_y = (w_shift + del_ind) // w                       # w_shift 用来保证取整运算能够根据当前 w 列的位置进行调整, 左加右减, 平移轴
    flow_ind_x = del_ind - flow_ind_y * w
    flow_ind = torch.stack([flow_ind_x, flow_ind_y], dim=2)     # (B,N,2,H,W), N 表示 posotive 个数
    return positive_ind, negative_ind, valid, sorted_score_volume, coords_volume


def infoNCE_lossfunc(feat1, feat2, gt, mask, format="stereo", t=0.07, topk=4):
    # feat1 is the reference view features, in shape of (B,c,h,w). feat2 is the target view features
    # gt is the reference view in shape of (B,2,H,W)
    # mask in shape of (B,1,H,W)
    B, _, H, W = gt.shape
    _, c, h, w = feat1.shape
    kernel = H // h
    assert kernel == W//w, "Downsample ratio should be the same at space dimension."

    if format == "stereo":
        # 只需要沿着 w 维度进行相似性计算
        feat1 = feat1.permute(0, 2, 3, 1).reshape(B*h, w, c)
        feat2 = feat2.permute(0, 2, 1, 3).reshape(B*h, c, w)
        affinity = torch.bmm(feat1, feat2)
        norm1 = torch.norm(feat1, dim=-1, keepdim=True) # (Bh, w, 1)
        norm2 = torch.norm(feat2, dim=-2, keepdim=True) # (Bh, 1, w)
        norm_affinity = affinity / (torch.bmm(norm1, norm2) + 1e-8) # (Bh, w, w)
        norm_affinity = norm_affinity.reshape(B, h, w, w).permute(0, 3, 1, 2).contiguous()
        positive_ind, negative_ind, valid, sorted_score_volume, score_volume = disp2coord_dist(gt, mask, kernel=kernel, topk=topk)

    elif format == "flow":
        # 只需要沿着 w 维度进行相似性计算
        feat1 = feat1.permute(0, 2, 3, 1).reshape(B, h*w, c)
        feat2 = feat2.reshape(B, c, h*w)
        affinity = torch.bmm(feat1, feat2)
        norm1 = torch.norm(feat1, dim=-1, keepdim=True) # (B, hw, 1)
        norm2 = torch.norm(feat2, dim=-2, keepdim=True) # (B, 1, hw)
        norm_affinity = affinity / (torch.bmm(norm1, norm2) + 1e-8) # (B, hw, hw)
        norm_affinity = norm_affinity.reshape(B, h, w, h*w).permute(0, 3, 1, 2).contiguous()
        positive_ind, negative_ind, valid, sorted_score_volume, score_volume = flow2coord_dist_fast(gt, mask, kernel=kernel, topk=topk)
    else:
        raise NotImplementedError

    # 构造正负样本对
    positive_samples = torch.gather(norm_affinity, dim=1, index=positive_ind)
    negative_samples = torch.gather(norm_affinity, dim=1, index=negative_ind)

    # mask 取最小区域, 保证不会引入噪声监督信号
    # TODO: 考虑遮挡: 去除了遮挡区域的误匹配; 不考虑遮挡: 让特征通过上下文来实现遮挡情况下的匹配
    mask_thr = (2/kernel)**2    # 为了保证有效性, kernel 内至少需要 4 个有效像素
    mask = F.avg_pool2d(mask.float(), kernel_size=kernel, stride=kernel)
    mask = (mask > mask_thr).float()
    mask = mask.reshape(B * h * w)
    if not mask.bool().any():
        return 0. * norm_affinity.mean()

    # info NCE loss
    lables = torch.zeros(B * h * w, dtype=torch.long, device=norm_affinity.device)              # (B*h*w,1), indicate the 0-th is gt
    p_score_list = torch.chunk(positive_samples, chunks=positive_samples.shape[1], dim=1)       # 切分成 top-k 个正例
    infoNCE_loss = 0
    for i, p_score_sample in enumerate(p_score_list):
        # TODO: 考虑将 n_score 进行 detach, 解除负样本的梯度
        logits = torch.cat([p_score_sample, negative_samples], dim=1)  # (B, D-topk+1, H, W)
        logits = logits.permute(0, 2, 3, 1).reshape(B * h * w, -1)
        loss = F.cross_entropy(logits.float() / t, lables, reduction="none")  # 注意转为 float32 避免半精度溢出, 可视化loss: loss.reshape(h,w)
        weight = sorted_score_volume[:,i].reshape(B * h * w)                  # weight 表示真实的概率分布
        loss_mean = (weight * loss * mask).sum() / mask.sum()                 # 有效区域内进行平均
        infoNCE_loss += loss_mean

    return infoNCE_loss


def infoNCE_lossfunc2(feat1, feat2, gt, mask, format="stereo", t=0.07, topk=4):
    # feat1 is the reference view features, in shape of (B,c,h,w). feat2 is the target view features
    # gt is the reference view in shape of (B,2,H,W)
    # mask in shape of (B,1,H,W)
    # topk 在这里无效
    B, _, H, W = gt.shape
    _, c, h, w = feat1.shape
    kernel = H // h
    assert kernel == W//w, "Downsample ratio should be the same at space dimension."

    if format == "stereo":
        # 只需要沿着 w 维度进行相似性计算
        feat1 = feat1.permute(0, 2, 3, 1).reshape(B*h, w, c)
        feat2 = feat2.permute(0, 2, 1, 3).reshape(B*h, c, w)
        affinity = torch.bmm(feat1, feat2)
        norm1 = torch.norm(feat1, dim=-1, keepdim=True) # (Bh, w, 1)
        norm2 = torch.norm(feat2, dim=-2, keepdim=True) # (Bh, 1, w)
        norm_affinity = affinity / (torch.bmm(norm1, norm2) + 1e-8) # (Bh, w, w)
        norm_affinity = norm_affinity.reshape(B, h, w, w).permute(0, 3, 1, 2).contiguous()
        positive_ind, negative_ind, valid, sorted_score_volume, score_volume = disp2coord_dist(gt, mask, kernel=kernel, topk=topk)

    elif format == "flow":
        # 只需要沿着 w 维度进行相似性计算
        feat1 = feat1.permute(0, 2, 3, 1).reshape(B, h*w, c)
        feat2 = feat2.reshape(B, c, h*w)
        affinity = torch.bmm(feat1, feat2)
        norm1 = torch.norm(feat1, dim=-1, keepdim=True) # (B, hw, 1)
        norm2 = torch.norm(feat2, dim=-2, keepdim=True) # (B, 1, hw)
        norm_affinity = affinity / (torch.bmm(norm1, norm2) + 1e-8) # (B, hw, hw)
        norm_affinity = norm_affinity.reshape(B, h, w, h*w).permute(0, 3, 1, 2).contiguous()
        positive_ind, negative_ind, valid, sorted_score_volume, score_volume = flow2coord_dist_fast(gt, mask, kernel=kernel, topk=topk)
    else:
        raise NotImplementedError

    # mask 取最小区域, 保证不会引入噪声监督信号
    # TODO: 考虑遮挡: 去除了遮挡区域的误匹配; 不考虑遮挡: 让特征通过上下文来实现遮挡情况下的匹配
    mask_thr = (2/kernel)**2    # 为了保证有效性, kernel 内至少需要 4 个有效像素
    mask = F.avg_pool2d(mask.float(), kernel_size=kernel, stride=kernel)
    mask = (mask > mask_thr).float()
    # mask *= valid.reshape(B, 1, h, w)   # 这一步是必须的, 尽管 score_volume 本身也考虑了匹配点超出图像的情况 (这种情况下的概率分布为全0), 但是 mask 与 valid 为交集而非真包含关系. 这一步主要是判断最终的 mask 是否全零, 避免 bool_mask 才发现全零导致 nan
    if not mask.bool().any():
        return 0. * norm_affinity.mean()

    labels = score_volume
    exp_scores = torch.exp(norm_affinity / t)
    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    bool_mask = (labels * mask) > 1e-2     # 当 top-k 较大时, 即采样了较多正样本时, ground-truth positive samples 中可能存在离散概率趋零的位置, 因此需要通过域值处理以避免数值不稳定
    inner_element = torch.log(torch.masked_select(exp_scores / denominator, bool_mask))
    # infoNCE 无法可视化, 因为 bool_mask 对每个 (u,v) 所选择的 k 个正样本的个数并不相同
    infoNCE_loss = -torch.sum(inner_element * torch.masked_select(labels, bool_mask)) / (torch.sum(torch.masked_select(labels, bool_mask)) + 1e-8)  # 避免由于 bool_mask 全零导致的 nan

    return infoNCE_loss