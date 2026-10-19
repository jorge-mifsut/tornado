import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.basics import (
    MLP,
    LayerNorm,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)
from modules.layers.embedding import RotaryEmbedding, TimestepEmbedder
from modules.layers.patch import PatchEmbed, Unpatchify


class DiTBlock(nn.Module):
    def __init__(
        self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.0, rotary_emb=None
    ):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, expansion_ratio=mlp_ratio)
        self.dropout = dropout

        if cond_dim > 0:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

        if rotary_emb is not None:
            self.rotary_emb = rotary_emb

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, cond=None):
        batch_size = x.shape[0]
        x_skip = x

        if cond is not None:
            bias_dropout_scale_fn = self._get_bias_dropout_scale()
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

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)

        if cond is not None:
            x = bias_dropout_scale_fn(
                self.attn_out(x),
                None,
                gate_msa,
                x_skip,
                self.dropout,
            )

            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None,
                gate_mlp,
                x,
                self.dropout,
            )
        else:
            x = self.attn_out(x) + x_skip
            x = self.norm2(x)
            x = self.mlp(x) + x

        return x


class DIT(nn.Module):
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
        scale_by_sigma=True,
        detach=True,
        block = True,
        tornado = False
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
        self.grid_size = [
            input_size[i] // patch_size[i] for i in range(len(input_size))
        ]

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_dim,
            hidden_size=self.dim,
        )

        self.detach = detach
        print("detach: ", self.detach)
        self.block = block
        print("block: ", self.block)
        self.tornado = tornado
        print("tornado: ", self.tornado)

        self.pe_embed = RotaryEmbedding(dim=self.dim // self.num_heads)

        if self.num_cond > 0:
            self.cond_map = TimestepEmbedder(self.cond_dim, num_conds=self.num_cond)

        sa_blocks = []
        for _ in range(self.num_layers):
            sa_blocks.append(
                DiTBlock(
                    self.dim,
                    self.num_heads,
                    self.cond_dim,
                    rotary_emb=self.pe_embed,
                )
            )

        self.sa_blocks = nn.ModuleList(sa_blocks)

        self.scale_by_sigma = scale_by_sigma

        if self.scale_by_sigma:
            self.sigma_map = TimestepEmbedder(self.cond_dim)

        self.head_drift = Unpatchify(
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            in_dim=self.dim,
            out_dim=self.out_dim,
            cond_dim=self.cond_dim,
        )
        # --- A Satellite Diffusion Trunk ---
      
        self.diffusion_block = DiTBlock(
            self.dim,
            self.num_heads,
            self.cond_dim,
            rotary_emb=self.pe_embed,
        )
        
        # Standard Unpatchify (keep as is)
        self.head_diffusion = Unpatchify(
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            in_dim=self.dim,
            out_dim=self.out_dim,
            cond_dim=self.cond_dim,
        )   
            

    def forward(self, u, sigma_t=None, cond=None):
        u = self.patch_embed(u)

        c = None
        c_t = None

        if self.scale_by_sigma:
            assert sigma_t is not None, "scale_by_sigma=True requires sigma_t"
            if sigma_t.ndim == 1:
                sigma_t = sigma_t.unsqueeze(-1)
            c_t = F.silu(self.sigma_map(sigma_t))
            c = c_t  # base conditioning

        if self.num_cond > 0 and cond is not None:
            c_cond = self.cond_map(cond)
            c = c_cond if c is None else (c + c_cond)

        # flatten
        u = rearrange(u, "b ny nx c -> b (ny nx) c")

        # trunk
        for block in self.sa_blocks:
            u = block(u, c)

        # drift always
        drift = self.head_drift(u, c)

        diffusion = None
        if self.tornado:
            u_g = u.detach() if self.detach else u
            if self.block:
                u_g = self.diffusion_block(u_g, c)
            diffusion = self.head_diffusion(u_g, c)

        return drift, diffusion
