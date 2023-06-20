import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.param

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1
        self.prob = prob

    def forward(self, x, force_keep_all=False):
        if not self.training or self.prob==0 or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device
        batch_indices = torch.arange(b, device=device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1- self.prob)))
        patch_indices_keep = torch.randn(b, n, device=device).topk(num_patches_keep, dim=-1).indices

        return x[batch_indices, patch_indices_keep]

