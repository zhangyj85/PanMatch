import torch
import torch.nn as nn
import torch.nn.functional as F


def match(img_source, img_target, flow_s2t, certainty_s2t):
    # img_source is tensor img in the shape of (b,3,h,w), value of (0., 255.)
    # certainty_s2t is bool tensor in the shape of (b,1,h,w)
    b, _, hs, ws = img_source.shape
    device = img_source.device
    coords_source = torch.meshgrid(
        (
            torch.arange(0, hs, device=device, dtype=img_source.dtype),
            torch.arange(0, ws, device=device, dtype=img_source.dtype),
        ),
        indexing = 'ij'
    )
    coords_source = torch.stack((coords_source[1], coords_source[0]))   # in (w,h) order
    coords_source = coords_source[None].expand(b, 2, hs, ws)

    coords_target = coords_source + flow_s2t
    grid = coords_target / torch.tensor([ws, hs], device=device).view(1, -1, 1, 1)
    grid = grid * 2 - 1                 # norm to (-1, 1)
    grid = grid.permute(0, 2, 3, 1)     # (b,h,w,2)

    # mask the out-of-vision regions
    if (grid.abs() > 1).any():
        oov = (grid.abs() > 1).sum(dim=-1) > 0  # (b,h,w)
        certainty_s2t[oov[:,None]] = 0          # (b,1,h,w)
    grid = torch.clamp(grid, -1, 1)

    warp_target = F.grid_sample(img_target, grid, mode="bilinear", align_corners=False)
    white_target = torch.ones_like(img_target) * 255.
    final_target = certainty_s2t * warp_target + (1 - certainty_s2t) * white_target
    return warp_target, final_target
