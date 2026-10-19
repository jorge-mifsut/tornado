import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.basics import (
    MLP,
    LayerNorm,
    modulate_fused,
)
from modules.layers.embedding import RotaryEmbedding, TimestepEmbedder
from modules.layers.patch import PatchEmbed, Unpatchify


class DiTBlockDropout(nn.Module):
    """DiT block with explicit dropout on attention and MLP residual paths.
    Dropout is NOT applied to the conditioning pathway (adaLN)."""

    def __init__(
        self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1, rotary_emb=None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, expansion_ratio=mlp_ratio, dropout=dropout)

        # Explicit dropout layers for residual connections
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

        if cond_dim > 0:
            # Conditioning pathway — NO dropout here
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

        if rotary_emb is not None:
            self.rotary_emb = rotary_emb

    def forward(self, x, cond=None):
        batch_size = x.shape[0]
        x_skip = x

        if cond is not None:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(cond)[:, None].chunk(6, dim=2)
            x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        else:
            x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv,
            "b s (three h d) -> b h three s d",
            three=3,
            h=self.n_heads,
        )
        qk, v = qkv[:, :, :2], qkv[:, :, 2]
        q, k = qk[:, :, 0], qk[:, :, 1]

        if getattr(self, "rotary_emb", None) is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)

        if cond is not None:
            # Attention residual with dropout + gating
            attn_out = self.attn_dropout(self.attn_out(x))
            x = x_skip + gate_msa * attn_out

            # MLP residual with dropout + gating
            mlp_out = self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp))
            mlp_out = self.mlp_dropout(mlp_out)
            x = x + gate_mlp * mlp_out
        else:
            x = x_skip + self.attn_dropout(self.attn_out(x))
            x = x + self.mlp_dropout(self.mlp(self.norm2(x)))

        return x


class DiTDropout(nn.Module):
    """DiT backbone for MC Dropout inference.

    Key differences from DIT:
    - No scale_by_sigma / sigma_map (not a diffusion model)
    - No diffusion head / tornado — single output only
    - Explicit dropout on all DiT blocks for MC Dropout at inference
    - Dropout is NOT applied to conditioning pathways
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dim,
        num_heads,
        num_layers,
        patch_size=[2, 2],
        input_size=[64, 64],
        cond_dim=0,
        num_cond=0,
        dropout=0.1,
        # Accept and ignore DIT-specific kwargs for config compatibility
        scale_by_sigma=None,
        detach=None,
        block=None,
        tornado=None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_cond = num_cond
        self.patch_size = patch_size
        self.input_size = input_size
        self.dropout = dropout
        self.tornado = False  # never tornado

        self.grid_size = [
            input_size[i] // patch_size[i] for i in range(len(input_size))
        ]

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_dim,
            hidden_size=self.dim,
        )

        self.pe_embed = RotaryEmbedding(dim=self.dim // self.num_heads)

        if self.num_cond > 0:
            # Conditioning pathway — no dropout
            self.cond_map = TimestepEmbedder(self.cond_dim, num_conds=self.num_cond)

        sa_blocks = []
        for _ in range(self.num_layers):
            sa_blocks.append(
                DiTBlockDropout(
                    self.dim,
                    self.num_heads,
                    self.cond_dim,
                    dropout=self.dropout,
                    rotary_emb=self.pe_embed,
                )
            )

        self.sa_blocks = nn.ModuleList(sa_blocks)

        # Single output head (drift only, no diffusion head)
        self.head = Unpatchify(
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            in_dim=self.dim,
            out_dim=self.out_dim,
            cond_dim=self.cond_dim,
        )

        print(f"DiTDropout: dropout={self.dropout}, layers={self.num_layers}")

    def forward(self, u, cond=None):
        u = self.patch_embed(u)

        c = None

        if self.num_cond > 0 and cond is not None:
            c = self.cond_map(cond)

        # flatten
        u = rearrange(u, "b ny nx c -> b (ny nx) c")

        # trunk with dropout blocks
        for block in self.sa_blocks:
            u = block(u, c)

        # single output
        return self.head(u, c)
