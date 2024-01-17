import os
import torch
import numpy as np
from data import *
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from timm.models import create_model
from models import all_attn_cnn_split

device = "cuda:4"
model = all_attn_cnn_split()
model.load_state_dict(torch.load("output/cls_attn_cnn_concat/checkpoint_280.pth")["model"])
model = model.to(device)

dataset = four_scale_dataset_with_fname("../gravityspy/processed/", 0)
C = dataset[0][0].shape[0]
batch = 16
dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, num_workers=8)
with torch.no_grad():
    for im, ori_im, msk, fnames in tqdm(dataloader):

        


        latent, pred = model(im.to(device))
        





        for i, fname in enumerate(fnames):
            if not os.path.exists(os.path.join("test", fname.split("/")[-2])):
                os.makedirs(os.path.join("test", fname.split("/")[-2]))

            if not os.path.exists(os.path.join("test_im", fname.split("/")[-2])):
                os.makedirs(os.path.join("test_im", fname.split("/")[-2]))
            _im = torch.cat([im[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
            _ori_im = torch.cat([ori_im[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
            _pred = torch.cat([pred[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
            res = torch.cat([_im.cpu(), _pred.cpu()], dim=0)
            np.save(os.path.join("test", fname.split("/")[-2],fname.split("/")[-1]), latent[i:i+1, ...].detach().cpu().numpy())
            save_image(res, os.path.join("test_im", fname.split("/")[-2],fname.split("/")[-1]), normalize=True, value_range=(-1, 1), nrow=4)
    
    