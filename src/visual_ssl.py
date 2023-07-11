import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from einops import rearrange

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def get_default_aug(image_size, channels=3):
    is_rgb = channels == 3
    is_greyscale = channels == 1
    rgb_or_greyscale = is_rgb or is_greyscale

    return torch.nn.Sequential(
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p = 0.3
        ) if rgb_or_greyscale else nn.Identity(),
        T.RandomGrayscale(p = 0.2) if is_rgb else nn.Identity(),
        T.RandomHorizontalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p = 0.2
        ),
        T.RandomResizedCrop((image_size, image_size)),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ) if is_rgb else nn.Identity(),
    )

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)