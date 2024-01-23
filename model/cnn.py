import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_




class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.LeakyReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion if outplanes > expansion else outplanes

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x



class ConvBlockDecode(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.LeakyReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlockDecode, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion if outplanes > expansion else outplanes

        self.conv1 = nn.ConvTranspose2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.ConvTranspose2d(med_planes, med_planes, kernel_size=3, stride=stride, output_padding=stride-1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.ConvTranspose2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size=1, stride=stride, output_padding=stride-1, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x






class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.LeakyReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer()

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride,
                 last_fusion=False, num_med_block=0, groups=1, decode=False):

        super(ConvTransBlock, self).__init__()
        if decode:
            self.cnn_block = ConvBlockDecode(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        else:
            self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if decode:
            if last_fusion:
                self.fusion_block = ConvBlockDecode(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
            else:
                self.fusion_block = ConvBlockDecode(inplanes=outplanes, outplanes=outplanes, groups=groups)
        else:
            if last_fusion:
                self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
            else:
                self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)


        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x):
        x, _ = self.cnn_block(x)



        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x = self.fusion_block(x, return_x_2=False)

        return x

















class encoder(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, decode_embed=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, im_size=224, use_vae=True):

        # Transformer
        super().__init__()
        self.decode_embed = decode_embed
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0


        # Latent output13
        self.last_pool = 4
        self.pooling = nn.AdaptiveAvgPool2d(self.last_pool)
        self.mean_head = nn.Linear(int(base_channel * channel_ratio * 4 * self.last_pool * self.last_pool), decode_embed)
        self.var_head = nn.Linear(int(base_channel * channel_ratio * 4 * self.last_pool * self.last_pool), decode_embed)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 2
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)


        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, res_conv, s,
                        num_med_block=num_med_block
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s,
                        num_med_block=num_med_block
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s,
                        num_med_block=num_med_block, last_fusion=last_fusion
                    )
            )
        self.fin_stage = fin_stage


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
        

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.act1(self.bn1(self.conv1(x)))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x = eval('self.conv_trans_' + str(i))(x)


        x_p = self.pooling(x).flatten(1)
        mu = self.mean_head(x_p)
        var = self.mean_head(var)


        return mu, var
    




class decoder(nn.Module):

    def __init__(self, patch_size=16, embed_dim=768, base_channel=64, channel_ratio=4, num_med_block=0,
                 depth=12, im_size=224, first_up=2):

        # Transformer
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0


        self.patch_size = patch_size
        self.im_size = im_size


        # Latent to map
        self.first_fc = nn.Linear(embed_dim, int(base_channel * (im_size // (16 * first_up)) * (im_size // (16 * first_up))))
        self.first_cnn = nn.Conv2d(base_channel, int(base_channel * channel_ratio * 4), kernel_size=1)
        self.frist_up = nn.Upsample(scale_factor=first_up)
        self.first_up_scale = first_up
        trans_dw_stride = patch_size // 16
        


        # 0~3 stage
        stage_1_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = 0  # 0
        fin_stage = init_stage
        fin_stage = fin_stage + depth // 3  # 4
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_1_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_1_channel, res_conv, s,
                        num_med_block=num_med_block, decode=True
                    )
            )

        # 3~7 stage
        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage # 4
        fin_stage = fin_stage + depth // 3 # 8
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, 
                        num_med_block=num_med_block, decode=True
                    )
            )


        # 8~11 stage
        stage_3_channel = int(base_channel * channel_ratio)
        init_stage = fin_stage
        fin_stage = init_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            last_fusion = True if i == fin_stage - 1 else False
            res_conv = True if i == init_stage or last_fusion else False
            channel = stage_3_channel // 2 if i == fin_stage - 1 else stage_3_channel
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, channel, res_conv, s,
                        num_med_block=num_med_block, last_fusion=last_fusion, decode=True
                    )
            )
        
        self.fin_stage = fin_stage
        self.dw_stride = trans_dw_stride * 2 * 2 * 2 * 2




        self.conv_last = ConvBlock(inplanes=stage_3_channel // 2, outplanes=3, res_conv=True, stride=1, act_layer=nn.Tanh)





        


        


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
        B, _ = x.shape
        x = self.first_fc(x).reshape(B, -1, (self.im_size // (16 * self.first_up_scale)), (self.im_size // (16 * self.first_up_scale)))
        x = self.first_cnn(x)
        x = self.frist_up(x)

        # 1 ~ final 
        for i in range(self.fin_stage):
            x = eval('self.conv_trans_' + str(i))(x)

        x = self.conv_last(x, return_x_2=False)
        return x


    
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
        p = 14
        channel = self.last_channel
        h = w = int(x.shape[1]**.5)

        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    



class auto_encoder_cnn(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, decode_embed=384, base_channel=64, channel_ratio_encoder=4, channel_ratio_decoder=4,
                  num_med_block=0, embed_dim=768, depth=12, im_size=224, first_up=2, use_vae=True, **kwargs):
        
        super().__init__()
        

        self.encoder = [encoder(patch_size=patch_size, in_chans=in_chans, decode_embed=decode_embed, base_channel=base_channel, 
                               channel_ratio=channel_ratio_encoder, num_med_block=num_med_block,embed_dim=embed_dim, depth=depth, 
                                im_size=im_size) for _ in range(4)]
        self.decoder = [decoder(patch_size=patch_size, base_channel=base_channel, 
                               channel_ratio=channel_ratio_decoder, num_med_block=num_med_block,embed_dim=decode_embed, depth=depth, 
                                im_size=im_size, first_up=first_up) for _ in range(4)]
        self.mlp_mean = nn.Linear(decode_embed * 4, decode_embed)
        self.mlp_var = nn.Linear(decode_embed * 4, decode_embed) if use_vae else None


    def sample(self, mu, log_var):
        if log_var is not None:
            var = torch.exp(0.5 * log_var)
            z = torch.randn_like(mu)
            z = var * z + mu
        else:
            z = mu
        return z



    def forward(self, x):
        mus = []
        vars = []
        for i in range(4):
            mu, var = self.encoder[i](x[:, i*3 : i*3+3, :, :])
            mus.append(mu)
            vars.append(var)
        mu = torch.cat(mus, 1)
        var = torch.cat(vars, 1)
        mu = self.mlp_mean(mu)
        var = self.mlp_var(var) if self.mlp_var else None
        latent = self.sample(mu, var)


        pred = []
        for i in range(4):
            pred.append(self.decoder[i](latent))
        pred = torch.cat(pred, 1)
        return pred, mu, var
        
