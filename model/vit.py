import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), branch=4):
        super().__init__()
        for i in range(branch):
            setattr(self, f"norm1_{i}", norm_layer(dim))
        self.branch = branch
        for i in range(branch):
            setattr(self, f"attn_{i}", Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        for i in range(branch):
            setattr(self, f"norm2_{i}", norm_layer(dim))
        mlp_hidden_dim = int(dim * mlp_ratio)
        for i in range(branch):
            setattr(self, f"mlp_{i}", Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop))

    def forward(self, x):
        xs = []
        spb = (x.shape[1] - 1) // self.branch
        for i in range(self.branch):
            _x = torch.cat([x[:, 0:1, :], x[:, 1+spb*i:1+spb*i+spb, :]], 1)
            _x = _x + self.drop_path(getattr(self, f"attn_{i}")(getattr(self, f"norm1_{i}")(_x)))
            _x = _x + self.drop_path(getattr(self, f"mlp_{i}")(getattr(self, f"norm2_{i}")(_x)))
            xs.append(_x[:, 1:, :])
        x = torch.cat([x[:, 0:1, :]] + xs, 1)
        return x














class encoder(nn.Module):

    def __init__(self, patch_size=16, decode_embed=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., im_size=224, use_vae=True):

        # Transformer
        super().__init__()
        self.decode_embed = decode_embed
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        num_patches = ((im_size // patch_size) ** 2) * 4
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Latent output13
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.mean_head = nn.Linear(embed_dim, decode_embed)
        self.var_head = nn.Linear(embed_dim, decode_embed) if use_vae else None

        for i in range(4):
            setattr(self, f"trans_patch_conv_{i}", nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0))


        self.pos_drop = nn.Dropout(p=drop_rate)

        # 1~12 stage
        init_stage = 1
        fin_stage = init_stage + depth
        for i in range(init_stage, fin_stage):
            self.add_module('trans_' + str(i),
                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[i-1])
            )



        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        pos_embed = self.pos_embed.repeat(B, 1, 1)
        

        # 1 stage
        cpb = x.shape[1] // 4
        x_t = []
        for i in range(4):
            x_t.append(getattr(self, f"trans_patch_conv_{i}")(x[:, i*cpb : i*cpb + cpb, :, :]).flatten(2).transpose(1, 2))
        x_t = torch.cat([cls_tokens] + x_t, dim=1)

        x_t = x_t + pos_embed
        x_t = self.pos_drop(x_t)
    
        # 1 ~ final 
        for i in range(1, 13):
            x_t = eval('self.trans_' + str(i))(x_t)




        # trans classification
        x_t = self.trans_norm(x_t)
        mu = self.mean_head(x_t)
        var = self.var_head(x_t) if self.var_head else None
        return mu, var
    




class decoder(nn.Module):

    def __init__(self, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., im_size=224):

        # Transformer
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        num_patches = ((im_size // patch_size) ** 2) * 4
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.patch_size = patch_size
        self.im_size = im_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        for i in range(1, 1+depth):
            self.add_module('trans_' + str(i),
                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[i-1])
            )



        self.decoder_norm = nn.LayerNorm(embed_dim)
        for i in range(4):
            setattr(self, f"last_fc_{i}", nn.Linear(embed_dim, patch_size**2 * 3, bias=True))



        self.apply(self._init_weights)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x):
        B, _, _ = x.shape
        pos_embed = self.pos_embed.repeat(B, 1, 1)
        x = x + pos_embed
        x = self.pos_drop(x)


        # 1 ~ final 
        for i in range(1, 13):
            x = eval('self.trans_' + str(i))(x)
        x = self.decoder_norm(x)

        
        x = x[:, 1:]
        im = []
        cpb = (x.shape[1]) // 4
        for i in range(4):
            _x = getattr(self, f"last_fc_{i}")(x[:, i*cpb:i*cpb+cpb, :])
            im.append(self.unpatchify(_x))
        im = torch.cat(im, 1)
        return im


    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1]**.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    



class auto_encoder_vit(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, decode_embed=384,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., im_size=224, use_vae=True, **kwargs):
        
        super().__init__()
        
        self.encoder = encoder(patch_size=patch_size, decode_embed=decode_embed,  
                               embed_dim=embed_dim, depth=depth, 
                               num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, 
                               attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, im_size=im_size, use_vae=use_vae)
        self.decoder = decoder(patch_size=patch_size, 
                              embed_dim=decode_embed, depth=depth, 
                               num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, 
                               attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, im_size=im_size)

    def sample(self, mu, log_var):
        if log_var is not None:
            var = torch.exp(0.5 * log_var)
            z = torch.randn_like(mu)
            z = var * z + mu
        else:
            z = mu
        return z


    def forward(self, x):
        mu, var  = self.encoder(x)
        latent = self.sample(mu, var)
        pred = self.decoder(latent)
        return pred, mu, var
        
