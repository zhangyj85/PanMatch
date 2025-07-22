"""
Ref: 
https://github.com/Parskatt/RoMa/blob/main/romatch/benchmarks/scannet_benchmark.py
https://github.com/Parskatt/DKM/blob/main/docs/benchmarks.md
Required:
(1) Download the meta data from: https://github.com/zju3dv/LoFTR/tree/master/assets/scannet_test_1500
(2) Download the ScanNet-1500-test (SuperGlue Split) from: https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc
"""
import os.path as osp
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from utils.matcher_utiles import forward_backward_consistency_check
from .utils import *
from .kde import kde


class ScanNetBenchmark:
    def __init__(self, data_root="data/scannet") -> None:
        self.data_root = data_root
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.sample_mode = "threshold_balanced"     # 当置信度大于最大阈值, 则视为完全可靠; balanced 用于根据密度进行采样
        self.sample_thresh = 0.999

    def sample(self, matches, certainty, num=10000):
        # ref: https://github.com/Parskatt/RoMa/blob/main/romatch/models/matcher.py
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

    def benchmark(self, model, tile_size):
        model.train(False)
        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(osp.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            for pairind in tqdm(pair_inds, smoothing=0.9):
                scene = pairs[pairind]                  # list, e.g., [707, 0, 45, 765]
                scene_name = f"scene0{scene[0]}_00"
                im_A_path = osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[2]}.jpg",
                    )
                im_A = Image.open(im_A_path)
                im_B_path = osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[3]}.jpg",
                    )
                im_B = Image.open(im_B_path)
                T_gt = rel_pose[pairind].reshape(3, 4)
                R, t = T_gt[:3, :3], T_gt[:3, 3]
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in open(
                            osp.join(
                                self.data_root,
                                "scans_test",
                                scene_name,
                                "intrinsic",
                                "intrinsic_color.txt",
                            ),
                            "r",
                        )
                        .read()
                        .split("\n")
                        if r
                    ]
                )
                w1, h1 = im_A.size
                w2, h2 = im_B.size
                K1 = K.copy()
                K2 = K.copy()

                # ### RoMa 的运行代码
                # dense_matches, dense_certainty = model.module.backbone.match(im_A_path, im_B_path)
                # sparse_matches, sparse_certainty = model.module.backbone.sample(
                #     dense_matches, dense_certainty, 5000
                # )

                ### 数据预处理
                device = next(model.parameters()).device
                imA = TF.normalize(TF.to_tensor(im_A), self.mean, self.std, inplace=True).unsqueeze(0).to(device)
                imB = TF.normalize(TF.to_tensor(im_B), self.mean, self.std, inplace=True).unsqueeze(0).to(device)
                # 由于 feature matching 的 motion 很大, 基于 tiling 的策略无法解决高分辨率图像的大 motion 估计, 因此这里对图像的大小进行限制
                imA = F.interpolate(imA, size=tile_size, mode="bilinear", align_corners=True)
                imB = F.interpolate(imB, size=tile_size, mode="bilinear", align_corners=True)

                ### 前馈并筛选有效点
                output_AtoB = model(imA, imB, tile_size)     # 取消并行, 保证不会显存溢出
                output_BtoA = model(imB, imA, tile_size)
                flow_AtoB, conf_AtoB = output_AtoB['flow'], output_AtoB['conf']
                flow_BtoA, conf_BtoA = output_BtoA['flow'], output_BtoA['conf']
                diff_Eu = forward_backward_consistency_check(flow_AtoB, flow_BtoA, imA, imB)
                conf_check = 1. - F.tanh(0.1 * diff_Eu)     # 0.1 是缩放系数, 减缓在 diff 很小时置信度快速衰减的速率
                final_certainty = conf_AtoB * conf_check

                ### flow(B,2,H,W) -> matches(B,H,W,4),  pixel range -> grid range(-1,1)
                dense_matches, dense_certainty = self.flow2match(flow_AtoB, final_certainty)
                try:
                    sparse_matches, sparse_certainty = self.sample(
                        dense_matches, dense_certainty, 5000
                    )
                except:
                    # continue    # 极为不负责任的做法, 将无法预测的场景直接剔除, 仅用于查看异常样本的对整体结果的影响
                    # 经测试, 跳过与否并不会对最终结果产生显著影响. AUC@5=23.1/23.0/23.2, AUC@10=42.3/42.2/42.5, AUC@20=59.3/59.2/59.3
                    sparse_matches = torch.empty(0, 4, device=dense_matches.device)  # 稀疏匹配结果为空, 表示匹配失败

                ### 将图像缩放到 (480,640), ref: RoMa, DKM, LoFTR, SuperGlue
                scale1 = 480 / min(w1, h1)
                scale2 = 480 / min(w2, h2)
                w1, h1 = scale1 * w1, scale1 * h1
                w2, h2 = scale2 * w2, scale2 * h2
                K1 = K1 * scale1
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
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as e:
                        print(repr(e))
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
