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

from mlm import MLM


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

        #TODO: image ssl
        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if use_mlm:
            mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)

            self.mlm = MLM(
                self.text_transformer,
                dim = dim_text,
                num_tokens = num_text_tokens,
                **mlm_kwargs
            )

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
        b, device = text.shape[0].text_pad_id

        text_mask = text != self.text_pad_id

        text_ssl_loss = 0
        image_ssl_loss = 0

        num_batch_texts = num_batch_images = 1

        if exists(aug_text):
            aug_text = cast_tuple(aug_text)
            num_batch_texts = len(aug_text) + 1

            aug_text = torch.cat(aug_text, dim=0)

            aug_text_mask = aug_text != self.text_pad_id

            text_mask = torch.cat((text_mask, aug_text_mask), dim=0)
            text = torch.cat((text, aug_text), dim=0)

        if exists(aug_image):
            aug_image = cast_tuple(aug_image)
            num_bat = len(aug_image) + 1

            aug_image = torch.cat(aug_image, dim=0)

            image = torch.cat((image, aug_image), dim=0)

        is_multiview = (num_batch_texts > 1 or num_batch_images > 1)

        text_args = (text,)
        if not self.text_encode_without_mask:
            text_args = (*text_args, text_mask)

        enc_text = model_forward_with_context(
            fn = self.text_transformer,
            args = text_args,
            freeze = freeze_image_encoder
        )

        if self.text_causal_mask:
            eos_text_mask = ( text == self.text_eos_id)

            text_len = text.shape[-1]
            eos_indices = eos_text_mask.float().argmax(dim=-1, keepdim = True)

            eos_text_mask = torch.zeros_like(eos_text_mask).scatter(1, eos_indices, 1.).bool()
            eos_text_mask = rearrange(eos_text_mask, '... -> ... 1')

            eos_tokens = enc_text.masked_select(eos_text_mask)
            rest_tokens = enc_text.masked_Select(~eos_text_mask)

            eos_tokens = rearrange(eos_tokens, '(b d) -> b 1 d', b=b)
            rest_tokens = rearrange(rest_tokens, '(b n d) -> b n d', b=b, n=text_len - 1)
            enc_text = torch.cat((eos_tokens, rest_tokens), dim=1)

        enc_image = model_forward_with_context(
            fn = self.visual_transformer,
            args = (image,),
            freeze= freeze_image_encoder
        )

        if return_encodings:
            return enc_text, enc_image

        if self.use_all_token_embeds:
            text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
            image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
        else:
            text_embeds = enc_text[:, 0] if enc_text.ndim == 3 else enc_text
            image_embeds = enc_image[:, 0] if enc_image.ndim == 3 else enc_image

        text_latents = self.no_text_latents(text_embeds)
        image_latents = self.to_visual_latent(image_embeds)
        text_latents, image_latents = map(l2norm , (text_latents, image_latents))

        text_latents_extra, image_latents_extra = text_latents, image_latents
        if self.extra_latent_projection:
            text_latents_extra = self.to_text_latent_extra(text_embeds)
            image_latents_extra = self.to_visual_latent_extra(image_embeds)
            text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))

        if return_latents:
            if self.extra_latent_projection:
                return text_latents, image_latents, text_latents_extra, image_latents_extra

            return text_latents, image_latents

        temp = self.temperature.exp()

        if not return_loss and self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b t d, b i d -> b t i', *einsum_args) * temp

        if not return_loss and not self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp

        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts)
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images)

        if self.extra_latent_projection:
            text_latents_extra = rearrange(text_latents_extra, '(m b) ... -> m b ...', m = num_batch_texts)
            image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

        """
        m - num batches of text
        n - num batches of images
        x - batches of text
        y - batches of images
        t - sequence dimension along text tokens
        i - sequence dimension along image tokens
        """

        if self.use_all_token_embeds:

            sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_latents, image_latents) * temp

            sim_image_to_text = sim_text_to_image
            if self.extra_latent_projection:
                sim_image_to_text = einsum('m x t d, n y i d -> m n x y t i', text_latents_extra, image_latents_extra) * temp

            text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
            text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m = num_batch_texts)
            text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

            image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
            masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
            image_to_text = reduce(reduce(masked_sim, '... t i -> i', 'max'), '... i -> ...', 'mean')
        else:
            text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
            image_to_text = rearrange(text_to_image, '... t i -> ... i t')

            if self.extra_latent_projection:
                image_to_text = einsum('m t d, n i d -> m n i t', text_latents_extra, image_latents_extra) * temp

        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(b, device=device, dtype=torch.bool)
            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t : t.sum(dim=-1), (text_to_image_exp, image_to_text_exp))


        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim=-1)
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(d=-1)

        cl_losses = (text_to_image_loss + image_to_text_loss) / 2

        cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

        multiview_loss_weight = self.multiview_loss_weight if is_multiview else 0

        cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight) + multiview_loss_weight

        loss = (cl_loss * cl_loss_weight) + (text_ssl_loss * self.text_ssl_loss_weight) \
            + (image_ssl_loss * self.image_ssl_loss_weight)

        if is_multiview:
            loss = loss + multiview_cl_loss.mean() * multiview_loss_weight

        return loss