from models import auto_encoder_multi_cnn_attn
import torch
model = auto_encoder_multi_cnn_attn()

# # total = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)


# # part = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

# # # part = sum(p.numel() for p in model.encoder.trans_patch_conv_1.parameters() if p.requires_grad)
# # # part = 0
# # # for i in range(2, 13):
# # #     layer = getattr(getattr(model.encoder, f"conv_trans_{i}"), "trans_block")
# # #     part += sum(p.numel() for p in layer.parameters() if p.requires_grad)
# # print(total)
# # print(part)
# # print(part / total * 100)



x = torch.zeros((8, 12, 224, 224))
latent, pred = model(x)
print(latent.shape)
print(pred.shape)


# print(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
# print(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad))


# print(sum(p.numel() for e in model.encoder for p in e.parameters() if p.requires_grad))
# print(sum(p.numel() for e in model.decoder for p in e.parameters() if p.requires_grad))