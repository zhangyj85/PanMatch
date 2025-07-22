# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by zhangyj85 (e-mail: zhangyj85@mail2.sysu.edu.cn)
# date: 2024.07.17
# radio without any modification

dependencies = ["torch", "timm", "einops"]

import os
import types
from typing import Dict, Any, Optional, Union, List
import warnings

import torch
import torch.nn as nn

from timm.models import clean_state_dict

from .radio.adaptor_registry import adaptor_registry
from .radio.common import DEFAULT_VERSION, RadioResource, RESOURCE_MAP
from .radio.enable_spectral_reparam import disable_spectral_reparam
from .radio.radio_model import RADIOModel, create_model_from_args
from .radio.input_conditioner import get_default_conditioner


activations = {}    # 全局变量, 用来给 get_activation() 存放 hock 输出


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}      # 全局变量, 用来给 get_attention() 存放 hock 输出


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


# vit 前向推理, 并将中间层结果转化为图像, 然后输出
def forward_radio(pretrained, x):
    """
    pretrained: including model & post-process
    x: input in image shape: b, 3, h, w
    """
    b, c, h, w = x.shape

    # ViT 前向推理, glob 为x对应的最终特征, (B,N,C)
    pretrained.model.eval()     # 冻结 dropout & drop path
    with torch.no_grad():
        # 逆归一化到(0,1), 然后再进行归一化
        assert c == 3, "Only support Image with 3 channels."
        old_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(x.device)
        old_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(x.device)
        x = x * old_std + old_mean

        # 使用官方提供的归一化方案
        x = pretrained.conditioner(x)

        # 直接使用官方的函数, 写得真好，真好用
        final, intermediates = pretrained.model.forward_intermediates(
            x,
            indices=pretrained.hooks,
            return_prefix_tokens=False,             # 丢弃所有层的 [cls] token
            norm=False,                             # 保留中间特征，不进行 norm 操作
            output_fmt="NCHW",                      # 直接导出为 patch 块的图像
            intermediates_only=False,               # 输出中间累积结果和最终结果
            aggregation=pretrained.aggregation,     # 可以改为 dense 看看性能提升
        )

        # final 是序列输出, intermediates 是图像 list
        final = final[:, pretrained.model.patch_generator.num_skip:]
        H = h // pretrained.model.patch_generator.patch_size
        W = w // pretrained.model.patch_generator.patch_size
        final = final.reshape(b, H, W, -1).permute(0, 3, 1, 2).contiguous()

    return final, intermediates


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return mod_state_dict


def _mask_pretrained_radio_vith16_432(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse"
):
    # 加载模型
    RADIO_checkpoint = "foundation_model_weights/radio_v2.1_bf16.pth.tar"
    chk = torch.load(RADIO_checkpoint, map_location="cpu")

    if "state_dict_ema" in chk:
        state_dict = chk["state_dict_ema"]
        chk['args'].spectral_reparam = False
    else:
        state_dict = chk["state_dict"]

    # 默认方式加载 vit_huge_patch16_224 model
    model = create_model_from_args(chk["args"])     # create model backbone, 已具备 forward_intermediates 方法

    state_dict = clean_state_dict(state_dict)       # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training

    # 加载 ViT-H/16 的参数
    key_warn = model.load_state_dict(get_prefix_state_dict(state_dict, "base_model."), strict=False)
    if key_warn.missing_keys:
        warnings.warn(f'Missing keys in state dict: {key_warn.missing_keys}')
    if key_warn.unexpected_keys:
        warnings.warn(f'Unexpected keys in state dict: {key_warn.unexpected_keys}')

    if chk['args'].spectral_reparam:
        # Spectral reparametrization uses PyTorch's "parametrizations" API. The idea behind
        # the method is that instead of there being a `weight` tensor for certain Linear layers
        # in the model, we make it a dynamically computed function. During training, this
        # helps stabilize the model. However, for downstream use cases, it shouldn't be necessary.
        # Disabling it in this context means that instead of having `w' = f(w)`, we just compute `w' = f(w)`
        # once, during this function call, and replace the parametrization with the realized weights.
        # This makes the model run faster, and also use less memory.
        disable_spectral_reparam(model)
        chk['args'].spectral_reparam = False

    # radio 使用全新的图像归一化方案, 因此记得对输出进行预处理
    conditioner = get_default_conditioner()
    conditioner.load_state_dict(get_prefix_state_dict(state_dict, "input_conditioner."), strict=True)

    pretrained = nn.Module()
    pretrained.model = model
    pretrained.conditioner = conditioner    # 不可学习的模块, 因此放在这里没有影响
    pretrained.hooks = hooks
    pretrained.vit_features = model.embed_dim
    pretrained.patch_size = model.patch_generator.patch_size
    pretrained.aggregation = aggregation

    # 添加钩子，可视化中间层特征
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        # 这个 hock 应该是用来可视化看网络关注了那些地方?
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    return pretrained
