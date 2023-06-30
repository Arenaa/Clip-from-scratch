import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum
from torch.utils.checkpoint import checkpoint

from utils import *
from utils import TextTransformer, VisionTransformer


class CLIP(nn.Module):
    """CLIP

    A reimplementation of this paper:
    Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July).
    Learning transferable visual models from natural language supervision.
    In International conference on machine learning (pp. 8748-8763). PMLR.

    """
    def __init__(
            self, *, image_encoder=None, text_encoder=None,
                 dim_text=512, dim_image=512, dim_latent=512, num_text_tokens=10000,
                 text_enc_depth=6, text_seq_len=256, text_heads=8, text_dim_head=64,
                 text_has_cls_token=True, text_pad_id=0, text_rotary_pos_emb=False,
                 text_causal_mask=False,  text_eos_id = None, text_encode_without_mask = False, visual_enc_depth=6, visual_heads=8, visual_dim_head=64,
                 visual_image_head=64, visual_image_size = 256, visual_patch_size = 32,
                 visual_patch_dropout = 0.5, visual_has_cls_token = True, channels = 3,
                 use_all_token_embeds = False, downsample_image_embeds = False, decoupled_contrastive_learning = False,
                 extra_latent_projection = False, use_mlm = False, text_ssl_loss_weight = 0.05,
                 use_visual_ssl = False, visual_ssl = None, visual_ssl_type = 'simsiam',
                 visual_ssl_hidden_layer = -1, simclr_temperature = 0.1, image_ssl_loss_weight = 0.05,
                 multiview_loss_weight = 0.1, checkpoint_during_training = False,
                **kwargs
                ):
        super().__init__()

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_encode_without_mask = text_encode_without_mask

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens= num_text_tokens + (1 if use_mlm else 0),
                max_seq_len= text_seq_len,
                depth = text_enc_depth,
                heads = text_heads,
                causal = text_causal_mask,
                dim_heads = text_dim_head,
                rotary_pos_dim= text_rotary_pos_emb,
                checkpoint_during_training= checkpoint_during_training
            )

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size= visual_image_size,
                patch_size= visual_patch_size,
                channels= channels,
                depth = visual_enc_depth,
                heads = visual_heads,
                dim_head = visual_dim_head,
                patch_dropout= visual_patch_dropout,
                checkpoint_during_training = checkpoint_during_training
            )

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        #TODO: text mlm, image ssl

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        if downsample_image_embeds:

            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv2d(dim_image, dim_image, 4, stride=2, padding=1, bias=False, groups=dim_image),
                nn.Conv2d(dim_image, dim_latent, 1),
                Rearrange('b c h w -> (h w) c')
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)

        self.temperature = nn.Parameter(torch.tensor(1.))

        self.use_all_token_embeds = use_all_token_embeds

        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.extra_latent_projection = extra_latent_projection

        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        self.multiview_loss_weight = multiview_loss_weight

    def forward(self,
                text,
                image,
                return_loss=False,
                return_encodings=False,
                return_latents=False,
                freeze_image_encoder=False,
                text_to_image=True,
                aug_text=None,
                aug_image=None
            ):
        pass