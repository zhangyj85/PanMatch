import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MemoryEncoder
from .decoder import MemoryDecoder, MemoryDecoderwithProb
from .thingsconfig import get_cfg


class FlowFormer(nn.Module):
    def __init__(self, pretrained=False, conf_forward=False):
        super(FlowFormer, self).__init__()
        all_cfg = get_cfg()
        cfg = all_cfg.latentcostformer
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoderwithProb(cfg) if conf_forward else MemoryDecoder(cfg)

        if pretrained:
            path = "Models/FlowAnything/CorrTransformer/things.pth"
            state_dict = torch.load(path, map_location="cpu")
            weights = state_dict
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weights.items():
                name = k[7:] if 'module' in k else k
                if 'feat_encoder' in name or 'context_encoder' in name or 'channel_convertor' in name or 'memory_decoder.att.pos_emb' in name:
                    # 跳过 cnet 和 fnet 的参数加载
                    continue
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict=True)

    def forward(self, feature1, feature2, context, flow_init=None):
        data = {}
        cost_memory = self.memory_encoder(feature1, feature2, data, context)
        flow_predictions, info_predictions = self.memory_decoder(cost_memory, context, data, flow_init)
        output = {}
        output['flow_predictions'] = flow_predictions
        output['info_predictions'] = info_predictions
        return output
