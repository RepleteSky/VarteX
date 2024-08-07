
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from torch import Tensor
from torch.nn.parameter import Parameter

from timm.models.layers import Mlp, DropPath

class Linear3d(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, extra_dim: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_dim = extra_dim
        self.weight = Parameter(torch.empty((input_dim, output_dim, extra_dim)))
        if bias:
            self.bias = Parameter(torch.empty(output_dim, extra_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input = torch.einsum("bncr, cdr->bndr",  (input, self.weight))
        if self.bias is not None:
            input = input + self.bias.view(1, *self.bias.shape)
        return input

    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias is not None}'



class MyAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            num_representatives: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.num_representatives = num_representatives
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = None

        self.qkv = Linear3d(dim, dim * 3, num_representatives, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear3d(dim, dim, num_representatives)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R, B, N, C = x.shape
        qkv = self.qkv(x.permute(1, 2, 3, 0)).reshape(B, N, 3, self.num_heads, self.head_dim, self.num_representatives).permute(2, 0, 3, 1, 4, 5) #3, B, num_heads, N, head_dim, R
        q, k, v = qkv.unbind(0)
        q, k, v = q.permute(4, 0, 1, 2, 3), k.permute(4, 0, 1, 2, 3), v.permute(4, 0, 1, 2, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # R, B, num_heads, N, head_dim

        x = x.permute(1, 2, 3, 4, 0).reshape(B, N, -1, self.num_representatives) # B, N, num_heads, head_dim, R -> B, N, (num_heads, head_dim), R
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(3, 0, 1, 2)
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MyBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_representatives: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_representatives = num_representatives

        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            num_heads=self.num_heads,
            num_representatives=self.num_representatives,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # R, B, L, D
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
