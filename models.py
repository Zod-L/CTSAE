import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg

from model.vision_transformer import VisionTransformer, _cfg
from model.conformer import Conformer, auto_encoder
from model.multi_branch_conformer import auto_encoder_multi_branch
from model.cnn import auto_encoder_cnn
from model.vit import auto_encoder_vit
from model.multi_cnn_attn_conformer import auto_encoder_multi_cnn_attn
from timm.models.registry import register_model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_med_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model













@register_model
def cnn_split224_4branch(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_cnn(patch_size=16, channel_ratio_encoder=4, channel_ratio_decoder=2, embed_dim=384, decode_embed=192, depth=12,
                       im_size=224, first_up=2, use_vae=use_vae, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def attn_split224_4branch(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_vit(patch_size=16, channel_ratio_encoder=4, channel_ratio_decoder=2, embed_dim=384, decode_embed=192, depth=12,
                       im_size=224, first_up=2, use_vae=use_vae, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model














@register_model
def cls_attn_cnn_split224(pretrained=False, **kwargs):
    # model = auto_encoder(patch_size=16, channel_ratio=4, embed_dim=768, decode_embed=384, depth=12,
    #                   num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model = auto_encoder(patch_size=16, channel_ratio=4, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def cls_attn_cnn_split512(pretrained=False, **kwargs):
    model = auto_encoder(patch_size=32, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=512, first_up=2, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def cls_attn_cnn_split224_4branch(pretrained=False, **kwargs):
    model = auto_encoder_multi_branch(patch_size=16, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, num_branch=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model



@register_model
def cls_attn_cnn_split224_4cnn_attn(pretrained=False, use_vae=False, **kwargs):
    model = auto_encoder_multi_cnn_attn(patch_size=16, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, num_branch=4, use_vae=use_vae, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

