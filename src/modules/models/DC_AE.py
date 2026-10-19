from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Encoder", "Decoder"]


#################################################################################
#                               Helpers                                         #
#################################################################################


def get_same_padding(kernel_size: int) -> int:
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def val2tuple(x: list | tuple | Any, min_len: int = 1) -> tuple:
    """Turn a scalar or sequence into a tuple, repeating the last element up to min_len."""
    x = list(x) if isinstance(x, (list, tuple)) else [x]
    if len(x) > 0:
        x[-1:-1] = [x[-1] for _ in range(min_len - len(x))]
    return tuple(x)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def build_norm(name: Optional[str], num_features=None) -> Optional[nn.Module]:
    if name == "bn2d":
        return nn.BatchNorm2d(num_features)
    if name == "ln":
        return nn.LayerNorm(num_features)
    if name == "ln2d":
        return LayerNorm2d(num_features)
    return None


ACTIVATIONS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: Optional[str]) -> Optional[nn.Module]:
    return ACTIVATIONS[name]() if name in ACTIVATIONS else None


#################################################################################
#                               Layers                                          #
#################################################################################


class ConvLayer(nn.Module):
    """Conv-norm-act, with an optional spherical padding mode.

    `padding_mode="sphere"` is for global lat/lon fields (ERA5): the longitude axis
    wraps, so it is padded circularly, while latitude does not and is padded by
    replication at the poles. Zero padding on a global field puts a ring of
    artificial zeros around the map, which shows up as a seam at the 0/360 meridian
    -- exactly where a reconstruction figure with coastlines makes it obvious.

    The default "zeros" reproduces the previous behaviour bit for bit, and no
    padding mode changes the parameter shapes, so existing checkpoints load either
    way.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm="bn2d",
        act_func="relu",
        padding_mode="zeros",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)

        if padding_mode not in ("zeros", "sphere"):
            raise ValueError(
                f"padding_mode must be 'zeros' or 'sphere', got {padding_mode!r}"
            )
        self.padding_mode = padding_mode
        self.pad = int(padding) if padding_mode == "sphere" else 0
        conv_padding = 0 if padding_mode == "sphere" else padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=conv_padding,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        if self.padding_mode == "sphere" and self.pad > 0:
            # Longitude (last axis) wraps; latitude (second to last) does not.
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        cond_channels = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            padding_mode=padding_mode,
        )

        if cond_channels > 0:
            self.cond_proj = nn.Linear(cond_channels, mid_channels)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        x = self.conv1(x)
        if cond is not None:
            # `nn.GELU(...)` here would *construct a module* with the tensor as its
            # `approximate` argument and then fail on `.unsqueeze`; the functional
            # form is what was meant. This is the only path that feeds conditioning
            # into the DC-AE, so it has to work for the date-conditioned VAE.
            cond_emb = F.gelu(self.cond_proj(cond)).unsqueeze(-1).unsqueeze(-1)  # b c 1 1
            x = x + cond_emb
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Module, shortcut: nn.Module):
        super(ResidualBlock, self).__init__()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        return self.main(x, cond) + self.shortcut(x)


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        for op in self.op_list:
            if isinstance(op, (ResidualBlock, OpSequential)):
                x = op(x, cond)
            else:
                x = op(x)
        return x


#################################################################################
#                               Builders                                        #
#################################################################################


def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], cond_channels,
    padding_mode: str = "zeros",
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            cond_channels=cond_channels,
            padding_mode=padding_mode,
        )
        block = ResidualBlock(main_block, IdentityLayer())
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int, cond_channels=0,
    padding_mode: str = "zeros",
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
            cond_channels=cond_channels,
            padding_mode=padding_mode,
        )
        stage.append(block)
    return stage


def build_downsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str],
                           padding_mode: str = "zeros") -> nn.Module:
    if block_type == "Conv":
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_bias=True,
            norm=None,
            act_func=None,
            padding_mode=padding_mode,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str],
                         padding_mode: str = "zeros") -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str,
                                   padding_mode: str = "zeros"):
    if factor == 1:
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            padding_mode=padding_mode,
        )
    elif factor == 2:
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    return block


def build_encoder_project_out_block(
    in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], shortcut: Optional[str],
    padding_mode: str = "zeros",
):
    block = OpSequential(
        [
            build_norm(norm),
            build_act(act),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                padding_mode=padding_mode,
            ),
        ]
    )
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    return block


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str],
                                   padding_mode: str = "zeros"):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
        padding_mode=padding_mode,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(
    in_channels: int, out_channels: int, factor: int, upsample_block_type: str, norm: Optional[str], act: Optional[str],
    padding_mode: str = "zeros",
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
    ]
    if factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                padding_mode=padding_mode,
            )
        )
    elif factor == 2:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None,
                padding_mode=padding_mode,
            )
        )
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)


class Encoder(nn.Module):
    def __init__(self, 
                 in_channels,
                 latent_channels,
                 width_list, 
                 depth_list, 
                 block_type,
                 downsample_block_type="ConvPixelUnshuffle",
                 norm="bn2d",
                 act="silu",
                 downsample_match_channel=True,
                 downsample_shortcut="averaging",
                 out_norm=None,
                 out_act=None,
                 out_shortcut="averaging",
                 kl_weight=None,
                 cond_dim=0,
                 padding_mode="zeros"):

        super().__init__()
        num_stages = len(width_list)
        self.num_stages = num_stages
        assert len(depth_list) == num_stages
        assert len(width_list) == num_stages
        assert isinstance(block_type, str) or (
            isinstance(block_type, list) and len(block_type) == num_stages
        )
        self.padding_mode = padding_mode

        self.project_in = build_encoder_project_in_block(
            in_channels=in_channels,
            out_channels=width_list[0] if depth_list[0] > 0 else width_list[1],
            factor=1 if depth_list[0] > 0 else 2,
            downsample_block_type=downsample_block_type,
            padding_mode=padding_mode,
        )

        if cond_dim > 0: 
            self.cond_proj = nn.Linear(cond_dim, width_list[0])
            self.cond_channels = width_list[0]
        else:
            self.cond_channels = 0

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(width_list, depth_list)):
            block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage = build_stage_main(
                width=width, depth=depth, block_type=block_type, norm=norm, act=act, input_width=width, cond_channels=self.cond_channels,
                padding_mode=padding_mode,
            )

            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=downsample_block_type,
                    in_channels=width,
                    out_channels=width_list[stage_id + 1] if downsample_match_channel else width,
                    shortcut=downsample_shortcut,
                    padding_mode=padding_mode,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        if kl_weight is not None:
            self.project_out = build_encoder_project_out_block(
                in_channels=width_list[-1],
                out_channels=latent_channels * 2,
                norm=out_norm,
                act=out_act,
                shortcut=out_shortcut,
                padding_mode=padding_mode,
            )
        else:
            self.project_out = build_encoder_project_out_block(
                in_channels=width_list[-1],
                out_channels=latent_channels,
                norm=out_norm,
                act=out_act,
            shortcut=out_shortcut,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        if cond is not None and self.cond_channels > 0:
            cond = self.cond_proj(cond)
        else:
            # Drop cond entirely when this model has no conditioning: the blocks
            # only test `cond is not None` and would call a cond_proj that was
            # never built. Lets a conditioned dataset feed an unconditioned AE.
            cond = None

        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x, cond)

        x = self.project_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_channels,
                 width_list, 
                 depth_list, 
                 block_type,
                 upsample_block_type="ConvPixelShuffle",
                 norm="bn2d",
                 act="silu",
                 in_shortcut="duplicating",
                 upsample_match_channel=True,
                 upsample_shortcut="duplicating",
                 out_norm="bn2d",
                 out_act="relu",
                 kl_weight=None,
                 cond_dim=0,
                 padding_mode="zeros"):
        super().__init__()
        num_stages = len(width_list)
        self.num_stages = num_stages
        self.padding_mode = padding_mode
        assert len(depth_list) == num_stages
        assert len(width_list) == num_stages
        assert isinstance(block_type, str) or (
            isinstance(block_type, list) and len(block_type) == num_stages
        )
        assert isinstance(norm, str) or (isinstance(norm, list) and len(norm) == num_stages)
        assert isinstance(act, str) or (isinstance(act, list) and len(act) == num_stages)

        self.project_in = build_decoder_project_in_block(
            in_channels=latent_channels,
            out_channels=width_list[-1],
            shortcut=in_shortcut,
            padding_mode=padding_mode,
        )

        if cond_dim > 0: 
            self.cond_proj = nn.Linear(cond_dim, width_list[0])
            self.cond_channels = width_list[0]
        else:
            self.cond_channels = 0

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(width_list, depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=width_list[stage_id + 1],
                    out_channels=width if upsample_match_channel else width_list[stage_id + 1],
                    shortcut=upsample_shortcut,
                    padding_mode=padding_mode,
                )
                stage.append(upsample_block)

            block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            norm = norm[stage_id] if isinstance(norm, list) else norm
            act = act[stage_id] if isinstance(act, list) else act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=block_type,
                    norm=norm,
                    act=act,
                    input_width=(
                        width if upsample_match_channel else width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                    cond_channels=self.cond_channels,
                    padding_mode=padding_mode,
                )
            )
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
                in_channels=width_list[0] if depth_list[0] > 0 else width_list[1],
                out_channels=in_channels,
                factor=1 if depth_list[0] > 0 else 2,
                upsample_block_type=upsample_block_type,
                norm=out_norm,
                act=out_act,
                padding_mode=padding_mode,
            )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        x = self.project_in(x)

        if cond is not None and self.cond_channels > 0:
            cond = self.cond_proj(cond)
        else:
            cond = None  # see Encoder.forward

        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x, cond)
            
        x = self.project_out(x)
        return x