from functools import partial, wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum
from torch.utils.checkpoint import checkpoint


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def make_checkpoint(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([[isinstance(el, torch.Tensor) and el.requires_grad for el in args]])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

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

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        pos = 1. / (10000 ** (torch.range(0, dim, 2).float()/dim))
        self.register_buffer('pos', pos)

    def forward(self, seq_len, device):
        pos = self.pos
        t = torch.erange(seq_len, device=device).type_as(pos)
        freqs = torch.einsum('i, j -> i j', t, pos)
        return torch.cat((freqs, freqs), dim=-1)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x*F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=False, dropout=0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transforemr(nn.Module):
    def __init__(self, dim, *, depth, dim_head=64, heads=8,
                 causal=False, attn_dropout=0, ff_mult=4,
                 checkpoint_during_training=False):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim=dim, mult=ff_mult))
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(self, x, rotary_pos_emb=None, mask=None):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpint_fn = make_checkpoint if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)

class TextTransformer(nn.Module):
    def __init__(self, dim, *, num_tokens, max_seq_len,
                 dim_head, rotary_pos_dim=None, causal=False, **kwargs):

        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_dim else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_dim else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transforemr(dim, dim_head=dim_head, causal=causal, **kwargs)

    def forward(self, x, mask=None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device=device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        if exists(self.rotary_pos_emb):
            rotary_pos_dim = self.rotary_pos_emb(n + 1, device=device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)

        out = self.transformer(x, mask=mask, rotary_pos_dim=rotary_pos_dim)
        return out

    class VisionTransformer(nn.Module):
        def __init__(self, dim, x, image_size, patch_size, channels,
                     patch_dropout=0.5,
                     **kwargs):
            super().__init__()
            num_patches = (image_size // patch_size) **2
            patch_dim = channels * patch_size **2

            self.to_tokens = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim)
            )

            self.pos_emb = nn.Embedding(num_patches, dim)
            self.patch_dropout = PatchDropout(patch_dropout)

            self.transformer = Transforemr(dim, **kwargs)

            self.to_cls_tokens = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim, dim , bias=False),
                Rearrange('b d -> b 1 d')
            )

        def forward(self, x, keep_all_patches=False):

            device = x.device

            x = self.to_tokens(x)
            b, n, _ = x.shape

            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

            x = self.patch_dropout(x, force_keep_all=keep_all_patches)

            out = self.transformer(x)

            cls_tokens = self.to_cls_tokens(out)
            return torch.cat((cls_tokens, out), dim=1)