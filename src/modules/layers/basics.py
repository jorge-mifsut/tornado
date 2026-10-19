import typing

import torch
import torch.nn as nn
from torch.nn import functional as F


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    # basically a layer norm with out bias
    def __init__(self, dim, force_fp32=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
        self.force_fp32 = force_fp32

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            if self.force_fp32:
                x = x.float()
            x = F.layer_norm(x, [self.dim])
        return x * self.weight[None, None, :]


class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, expansion_ratio=4, dropout=0.0):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.fc1 = nn.Linear(dim, int(dim * expansion_ratio))
        self.fc2 = nn.Linear(int(dim * expansion_ratio), out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)
