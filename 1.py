from models import cls_attn_cnn_split224_4cnn_attn
import torch
import os
import shutil
import random
import pickle
from data import four_scale_dataset
import numpy as np
model = cls_attn_cnn_split224_4cnn_attn()

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

# dataset = four_scale_dataset("../gravityspy/train/", 0, 223)
# print(len(dataset))


# dataset = four_scale_dataset("../gravityspy/val/", 0, 223)
# print(len(dataset))

# dataset = four_scale_dataset("../gravityspy/test/", 0, 223)
# print(len(dataset))

x = torch.zeros((8, 12, 224, 224))
pred, latent, var = model(x)
print(latent.shape)
print(pred.shape)


# print(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
# print(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad))


# print(sum(p.numel() for e in model.encoder for p in e.parameters() if p.requires_grad))
# print(sum(p.numel() for e in model.decoder for p in e.parameters() if p.requires_grad))

# fnames = {}
# for c in os.listdir("../gravityspy/processed/sub_0.5"):
#     fnames[c] = [fname for fname in os.listdir("../gravityspy/processed/sub_0.5/" + c)]
#     random.shuffle(fnames[c])


# for s in os.listdir("../gravityspy/processed/"):
#     if not os.path.exists(f"../gravityspy/train/{s}"):
#         os.makedirs(f"../gravityspy/train/{s}")
#     if not os.path.exists(f"../gravityspy/val/{s}"):
#         os.makedirs(f"../gravityspy/val/{s}")
#     if not os.path.exists(f"../gravityspy/test/{s}"):
#         os.makedirs(f"../gravityspy/test/{s}")


#     for c in os.listdir("../gravityspy/processed/" + s):
#         if not os.path.exists(f"../gravityspy/train/{s}/{c}"):
#             os.makedirs(f"../gravityspy/train/{s}/{c}")
#         if not os.path.exists(f"../gravityspy/val/{s}/{c}"):
#             os.makedirs(f"../gravityspy/val/{s}/{c}")
#         if not os.path.exists(f"../gravityspy/test/{s}/{c}"):
#             os.makedirs(f"../gravityspy/test/{s}/{c}")


# for c in os.listdir("../gravityspy/processed/sub_0.5"):



        
#     N = len(fnames[c])
#     print("=================")
#     print("total:", c, N)
#     for f in fnames[c][:int(0.7*N)]:
#         for scale in [0.5, 1.0, 2.0, 4.0]:
#             shutil.copy(f"../gravityspy/processed/sub_{scale}/{c}/{f.replace('0.5', str(scale))}", f"../gravityspy/train/sub_{scale}/{c}/{f.replace('0.5', str(scale))}")
#     print("train:", c, len(list(os.listdir(f"../gravityspy/train/sub_{0.5}/{c}"))))

    
#     for f in fnames[c][int(0.7*N):int(0.8*N)]:
#         for scale in [0.5, 1.0, 2.0, 4.0]:
#             shutil.copy(f"../gravityspy/processed/sub_{scale}/{c}/{f.replace('0.5', str(scale))}", f"../gravityspy/val/sub_{scale}/{c}/{f.replace('0.5', str(scale))}")
#     print("val:", c, len(list(os.listdir(f"../gravityspy/val/sub_{0.5}/{c}"))))



#     for f in fnames[c][int(0.8*N):]:
#         for scale in [0.5, 1.0, 2.0, 4.0]:
#             shutil.copy(f"../gravityspy/processed/sub_{scale}/{c}/{f.replace('0.5', str(scale))}", f"../gravityspy/test/sub_{scale}/{c}/{f.replace('0.5', str(scale))}")
#     print("test:", c, len(list(os.listdir(f"../gravityspy/test/sub_{0.5}/{c}"))))

# train_fnames = {}
# test_fnames = {}
# val_fnames = {}
# for c in os.listdir("../gravityspy/processed/sub_0.5"):
#     train_fnames[c] = list(os.listdir(f"../gravityspy/train/sub_0.5/{c}"))
#     test_fnames[c] = list(os.listdir(f"../gravityspy/test/sub_0.5/{c}"))
#     val_fnames[c] = list(os.listdir(f"../gravityspy/val/sub_0.5/{c}"))


# with open('fnames.pkl', 'wb') as fp:
#     pickle.dump(dict(train_fnames=train_fnames, test_fnames=test_fnames, val_fnames=val_fnames), fp)

a = np.load("test/1080Lines/H1_1j35rdGqhj_spectrogram.png.npy")
b = np.load("H1_1j35rdGqhj_spectrogram.png.npy")
print(a == b)
