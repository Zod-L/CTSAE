from conformer import auto_encoder
from multi_branch_conformer import auto_encoder_multi_branch
import torch
model = auto_encoder_multi_branch(patch_size=16, channel_ratio=2, embed_dim=384, decode_embed=192, depth=12,
                      num_heads=6, mlp_ratio=2, qkv_bias=True, im_size=224, first_up=2, num_branch=4)

# total = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)


# part = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

# # part = sum(p.numel() for p in model.encoder.trans_patch_conv_1.parameters() if p.requires_grad)
# # part = 0
# # for i in range(2, 13):
# #     layer = getattr(getattr(model.encoder, f"conv_trans_{i}"), "trans_block")
# #     part += sum(p.numel() for p in layer.parameters() if p.requires_grad)
# print(total)
# print(part)
# print(part / total * 100)

x = torch.zeros((8, 12, 224, 224))
latent, pred = model(x)
print(latent.shape)
print(pred.shape)