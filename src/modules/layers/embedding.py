import math

import torch
from torch import nn
from einops import rearrange, repeat

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, num_conds=1, cache_freqs=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size*num_conds, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size
        self.num_conds = num_conds
        self.cache_freqs = cache_freqs

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, freqs=None):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if freqs is None:
            half = dim // 2
            freqs = torch.exp(
                - math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half).to(device=t.device)

        args = torch.einsum('bc,d->bcd', t.float(), freqs) # [b, num_conds, dim//2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding, freqs

    def forward(self, t):
        assert t.shape[-1] == self.num_conds
        if self.cache_freqs and hasattr(self, 'freqs'):
            t_emb, _ = self.timestep_embedding(t, self.frequency_embedding_size, freqs=self.freqs)
        else:
            t_emb, freqs = self.timestep_embedding(t, self.frequency_embedding_size)
            if self.cache_freqs:
                self.freqs = freqs
        t_emb = rearrange(t_emb, 'b nc d -> b (nc d)')
        t_emb = self.mlp(t_emb)
        return t_emb


#################################################################################
#                       Rotary Position Embedding                               #
#################################################################################


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryEmbedding(nn.Module):
    """1D rotary embedding over the flattened token sequence.

    Trimmed from lucidrains' rotary-embedding-torch to the one path the DiTs use
    (`freqs_for="lang"`, no xpos, no learned frequencies). `freqs` stays a frozen
    nn.Parameter, as there, so existing checkpoints and optimizer states load.
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self._cached_freqs = None  # [seq, head_dim], rebuilt when seq or device changes

    @torch.amp.autocast("cuda", enabled=False)
    def rotate_queries_or_keys(self, t):
        # t: [b, heads, seq, head_dim]. Positions are built in the frequencies'
        # dtype, not t's: in bf16, integer positions above 256 would be rounded.
        seq_len = t.shape[-2]
        freqs = self._cached_freqs
        if freqs is None or freqs.shape[0] != seq_len or freqs.device != t.device:
            seq = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
            freqs = torch.einsum("..., f -> ... f", seq, self.freqs.detach())
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)
            self._cached_freqs = freqs
        out = t * freqs.cos() + rotate_half(t) * freqs.sin()
        return out.type(t.dtype)
