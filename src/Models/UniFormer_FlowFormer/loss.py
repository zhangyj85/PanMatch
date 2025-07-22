import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.UniFormer.loss3 import infoNCE_lossfunc2 as infoNCE_lossfunc


def sequence_loss(flow_preds, info_list, flow_gt, mask, gamma=0.9):
    if mask.float().mean() < 0.25:
        # 当前无效信息大于有效信息, 则直接返回 0. 保证 DDP 过程持续且有效
        loss = 0.
        for flow in flow_preds:
            loss += flow.mean()
        return 0. * loss

    mask = mask.repeat(1, 2, 1, 1).bool()
    assert mask.shape == flow_gt.shape, [mask.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[mask]).any()

    flow_loss = 0.0
    n_predictions = len(flow_preds)

    # raft loss
    weights = [gamma**(n_predictions - i - 1) for i in range(n_predictions)]
    all_losses = []
    for flow, weight in zip(flow_preds, weights):
        all_losses.append(weight * F.l1_loss(flow[mask], flow_gt[mask], size_average=True))
    flow_loss += sum(all_losses)
    return flow_loss


def sequence_laplace_loss(flow_preds, info_preds, flow_gt, mask, gamma=0.9):
    # 在训练早期使用这个loss, 会导致大多数区域被判定为困难区域, 导致模型的优化效果停滞
    if mask.float().mean() < 0.25:
        # 当前无效信息大于有效信息, 则直接返回 0. 保证 DDP 过程持续且有效
        loss = 0.
        for flow, info in zip(flow_preds, info_preds):
            loss += flow.mean() + info.mean()
        return 0. * loss

    mask = mask.repeat(1, 2, 1, 1).bool()
    assert mask.shape == flow_gt.shape, [mask.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[mask]).any()

    n_predictions = len(flow_preds)
    weights = [gamma ** (n_predictions - i - 1) for i in range(n_predictions)]
    all_losses = []

    var_max, var_min = 10, 0
    for flow, info, weight in zip(flow_preds, info_preds, weights):
        alpha = info[:, :2]
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        # Large b Component
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)      # 根据这里的定义, log_b1, alpha1 对应易预测区域, log_b0, alpha0 对应难预测区域
        # term2: [N, 2, m, H, W]
        term2 = ((flow_gt - flow).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        # term1: [N, m, H, W]
        term1 = alpha - math.log(2) - log_b
        nf_loss = torch.logsumexp(alpha, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        final_mask = (~torch.isnan(nf_loss.detach())) & (~torch.isinf(nf_loss.detach())) & mask
        # all_losses.append(weight * nf_loss[final_mask].mean())
        all_losses.append(1.0 * nf_loss[final_mask].mean())   # 将 weight 去掉后, 网络倾向于去掉静态的背景, 而非去掉 un-overlap 区域
    return sum(all_losses)


class loss_func(nn.Module):

    def __init__(self, config):
        super(loss_func, self).__init__()
        crop_size = config['data']['crop_size']
        self.max_flow = (crop_size[0] * crop_size[1]) ** 0.5    # 训练时限制最大光流幅值
        self.flow_sequence_loss = sequence_laplace_loss if config['model']['conf_forward'] else sequence_loss

    def forward(self, data_batch, training_output):
        # target: B1HW, (min_disp, max_disp)
        # output: a dict containing multi-scale disp map

        # exclude invalid pixels and extremely large displacements
        flow_true, valid = data_batch["gt1"], data_batch["mask"]
        mag = (flow_true ** 2).sum(dim=1, keepdim=True).sqrt()
        mask = (mag < self.max_flow) & valid.bool()
        mask.detach_()
        loss = 0.

        # 标准的 optical flow loss
        flow_list = training_output['flow_predictions']
        info_list = training_output['info_predictions']
        loss += self.flow_sequence_loss(flow_list, info_list, flow_true, mask, gamma=0.8)

        # 语义一致性, 重建一致性, 不使用这些损失
        loss += 0. * training_output['auxiliary']

        # 跨视角一致性
        feat1, feat2 = training_output['corr']
        loss += 1. * infoNCE_lossfunc(feat1, feat2, flow_true, mask, format="flow", topk=16)     # 光流下最多出现4个峰值(x,y各两个), 每个峰值最多占用4格(x,y各两格)

        # croco中暂时不使用的损失
        # loss += 0. * training_output['conf_predictions'][0].mean()

        return loss
