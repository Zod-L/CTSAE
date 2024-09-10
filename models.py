import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg


from model.multi_branch_conformer import auto_encoder_multi_branch
from model.cnn import auto_encoder_cnn
from model.vit_share import auto_encoder_vit_share
from model.multi_cnn_attn_conformer_share import auto_encoder_multi_cnn_attn_share
from model.conformer_no_communicate import auto_encoder_no_comm
from timm.models.registry import register_model




@register_model
def cnn(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_cnn(patch_size=16, channel_ratio_encoder=2, channel_ratio_decoder=2, embed_dim=384, decode_embed=192, depth=12,
                       im_size=224, first_up=2, use_vae=use_vae, num_branch=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model




@register_model
def vit(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_vit_share(patch_size=16, channel_ratio_encoder=2, channel_ratio_decoder=2, embed_dim=384, decode_embed=192, depth=12,
                       im_size=224, first_up=2, use_vae=use_vae, num_branch=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model




@register_model
def cnn_concat_attn(pretrained=False, **kwargs):
    model = auto_encoder_multi_branch(patch_size=16, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, num_branch=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model



@register_model
def cnn_share_attn(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_multi_cnn_attn_share(patch_size=16, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, num_branch=4, use_vae=use_vae, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def cnn_nofuse_attn(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_no_comm(patch_size=16, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, num_branch=4, use_vae=use_vae, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model




