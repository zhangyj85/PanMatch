import math
import torch

__TRAIN_SIZE__=[448, 448]


def compute_grid_indices(image_shape, patch_size=__TRAIN_SIZE__, min_overlap=20):
    if isinstance(min_overlap, (int, float)):
        min_overlap = (min_overlap, min_overlap)
    if min_overlap[0] >= patch_size[0] or min_overlap[1] >= patch_size[1]:
        # raise ValueError("!!")
        return [(0,0)]  # 无需 tile
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap[0]))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap[1]))
    # remove those patches out of range
    while len(hs) > 0 and hs[-1] + patch_size[0] >= image_shape[0]:
        hs.pop()
    while len(ws) > 0 and ws[-1] + patch_size[1] >= image_shape[1]:
        ws.pop()
    # Make sure the final patch is flush with the image boundary
    hs.append(image_shape[0] - patch_size[0])
    ws.append(image_shape[1] - patch_size[1])
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=__TRAIN_SIZE__, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        try:
            # hw 可能会超出范围, 直接跳过并不会影响结果, 因为超出范围的结果本身可替代
            weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
        except:
            continue
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])

    return patch_weights