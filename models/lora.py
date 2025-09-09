import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from timm.models.vision_transformer import Block as timm_Block
from torch import Tensor
from torch.nn.parameter import Parameter


class _LoRA_qkv_timm(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
    ):
        super().__init__()
        self.r = r
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        # self.w_identity = torch.eye(qkv.in_features)

        # self.dropoutq = torch.nn.Dropout(p=0.2)
        # self.dropoutv = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class LoRA(nn.Module):
    def __init__(self, vit_model: timm_ViT, r=16, lora_cnt=None, additional_head_cls=-1):
        super(LoRA, self).__init__()
        self.lora_layer = list(range(len(vit_model.blocks)))
        self.lora_cnt = lora_cnt
        self.additional_head_cls = additional_head_cls
        if self.lora_cnt:
            self.lora_layer = self.lora_layer[-self.lora_cnt:]

        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False
        vit_model.head = nn.Identity()

        if self.additional_head_cls > 0:
            d = vit_model.embed_dim
            self.additional_head = nn.Linear(d, self.additional_head_cls)
        else:
            self.additional_head = nn.Identity()

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            vit_model.blocks[t_layer_i].attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
            )
        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        vit_out = self.lora_vit(x)
        if self.additional_head_cls > 0:
            additional_out = self.additional_head(vit_out)
            return vit_out, additional_out
        return vit_out
    

class MOELoRA(nn.Module):
    def __init__(self, vit_model: timm_ViT, r=16, 
                 lora_cnt=None, additional_head_cls=-1,
                 moe_num=3):
        super(LoRA, self).__init__()
        self.lora_layer = list(range(len(vit_model.blocks)))
        self.lora_cnt = lora_cnt
        self.additional_head_cls = additional_head_cls
        if self.lora_cnt:
            self.lora_layer = self.lora_layer[-self.lora_cnt:]

        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False
        vit_model.head = nn.Identity()

        if self.additional_head_cls > 0:
            d = vit_model.embed_dim
            self.additional_head = nn.Linear(d, self.additional_head_cls)
        else:
            self.additional_head = nn.Identity()

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            vit_model.blocks[t_layer_i].attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
            )
        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        vit_out = self.lora_vit(x)
        if self.additional_head_cls > 0:
            additional_out = self.additional_head(vit_out)
            return vit_out, additional_out
        return vit_out


class _LoRA_block_timm(nn.Module):
    def __init__(
        self,
        block: timm_Block,
        linear_a: nn.Module,
        linear_b: nn.Module,
        r: int,
    ):
        super().__init__()
        self.r = r
        self.block = block
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = block.mlp.fc1.in_features

    def forward(self, x):
        block_out = self.block(x)
        block_offset = self.linear_b(self.linear_a(block_out))
        block_out = block_out + block_offset
        return block_out
    

class Block_LoRA(nn.Module):
    def __init__(self, vit_model: timm_ViT, r=64, lora_cnt=None, additional_head_cls=-1):
        super(Block_LoRA, self).__init__()
        self.lora_layer = list(range(len(vit_model.blocks)))
        self.lora_cnt = lora_cnt
        self.additional_head_cls = additional_head_cls
        if self.lora_cnt:
            self.lora_layer = self.lora_layer[-self.lora_cnt:]
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False
        vit_model.head = nn.Identity()

        if self.additional_head_cls > 0:
            d = vit_model.embed_dim
            self.additional_head = nn.Linear(d, self.additional_head_cls)
        else:
            self.additional_head = nn.Identity()

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            self.dim = blk.mlp.fc1.in_features
            w_a_linear = nn.Linear(self.dim, r, bias=False)
            w_b_linear = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear)
            self.w_Bs.append(w_b_linear)
            vit_model.blocks[t_layer_i] = _LoRA_block_timm(
                blk,
                w_a_linear,
                w_b_linear,
                r,
            )
        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        vit_out = self.lora_vit(x)
        if self.additional_head_cls > 0:
            additional_out = self.additional_head(vit_out)
            return vit_out, additional_out
        return vit_out


class WrapUp_helper(nn.Module):
    def __init__(self, vit_model: timm_ViT, additional_head_cls=-1):
        super(WrapUp_helper, self).__init__()
        self.additional_head_cls = additional_head_cls
        if self.additional_head_cls > 0:
            d = vit_model.embed_dim
            self.additional_head = nn.Linear(d, self.additional_head_cls)
        else:
            self.additional_head = nn.Identity()
        self.vit = vit_model

    def forward(self, x: Tensor) -> Tensor:
        vit_out = self.vit(x)
        if self.additional_head_cls > 0:
            additional_out = self.additional_head(vit_out)
            return vit_out, additional_out
        return vit_out