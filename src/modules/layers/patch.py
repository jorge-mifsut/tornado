import torch
from torch import nn
from einops import rearrange

from modules.layers.basics import modulate_fused


class PatchEmbed(nn.Module):
    """Image to patch embedding: [b, ny, nx, c] -> [b, ny/p, nx/p, hidden]."""

    def __init__(self, patch_size=[2, 2], in_chans=3, hidden_size=768, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x):
        x = rearrange(x, "b ny nx c -> b c ny nx")
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # BCHW -> BHWC


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, 
                 hidden_size,
                 cond_dim,
                 patch_size, 
                 out_channels):
        super().__init__()
        self.cond_dim = cond_dim

        patch_dim = patch_size ** 2

        if cond_dim == 0:
            self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_dim * out_channels, bias=True))
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.linear = nn.Linear(hidden_size, patch_dim * out_channels, bias=True)
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(cond_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        
        self.init_params()

    def forward(self, x, c=None):
        if c is None:
            if hasattr(self, 'output_layer'):
                x = self.output_layer(x)
            else:
                x = self.norm_final(x)
                x = self.linear(x)
            return x
        else:
            z = self.adaLN_modulation(c) # b, 2*hidden_size
            z = z.unsqueeze(1) # b, 1, 2*hidden_size
            shift, scale = z.chunk(2, dim=-1) # b, 1, hidden_size or b, 1, 1, hidden_size
            x = modulate_fused(self.norm_final(x), shift, scale)
            x = self.linear(x)
            return x
        
    def init_params(self):
        if self.cond_dim > 0:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.adaLN_modulation[0].weight, 0)
            nn.init.constant_(self.adaLN_modulation[0].bias, 0)
            nn.init.constant_(self.linear.weight, 0)
            nn.init.constant_(self.linear.bias, 0)

        else:
            nn.init.constant_(self.output_layer[1].weight, 0)
            nn.init.constant_(self.output_layer[1].bias, 0)


class Unpatchify(nn.Module):
    """
    Unpatchify a tensor.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, grid_size, patch_size, in_dim, out_dim, cond_dim=0):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.out_layer = FinalLayer(hidden_size=in_dim,
                                    cond_dim=cond_dim,
                                    patch_size=patch_size[0],
                                    out_channels=out_dim)
    
    def forward(self, x, cond=None):
        # x in shape [b, nlat//p * nlon//p, ndim] 
        x = self.out_layer(x, cond) # [batch_size, nlat//p * nlon//p, patch_size * patch_size * out_dim]
        c = self.out_dim

        h, w = self.grid_size
        assert h * w == x.shape[1]
        ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw)) # [b, c, nlat, nlon]
        imgs = imgs.permute(0, 2, 3, 1) # [b, nlat, nlon, c]

        return imgs
