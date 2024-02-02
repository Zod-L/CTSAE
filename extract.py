import os
import torch
import numpy as np
from data import *
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models import cls_attn_cnn_split224_4branch
import shutil


def main(i):
    device = "cuda:4"
    model = cls_attn_cnn_split224_4branch(use_vae=False)
    model.load_state_dict(torch.load(f"output/4cnn_1attn_split/checkpoint_{i}.pth", map_location=device)["model"])
    model = model.to(device)
    model.eval()
    dataset = four_scale_dataset_with_fname("../gravityspy/split/test/", 0)
    C = dataset[0][0].shape[0]
    batch = 16
    dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=8)

    latent_dir = f"./test_out/test_{i}"
    im_dir = f"./test_out/test_im_{i}"




    with torch.no_grad():
        for im, ori_im, msk, fnames in tqdm(dataloader):



            pred, latent, var = model(im.to(device))
            





            for i, fname in enumerate(fnames):
                if not os.path.exists(os.path.join(latent_dir, fname.split("/")[-2])):
                    os.makedirs(os.path.join(latent_dir, fname.split("/")[-2]))

                if not os.path.exists(os.path.join(im_dir, fname.split("/")[-2])):
                    os.makedirs(os.path.join(im_dir, fname.split("/")[-2]))
                _im = torch.cat([im[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
                _ori_im = torch.cat([ori_im[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
                _pred = torch.cat([pred[i:i+1, 3*j : 3*(j+1), ...] for j in range(C // 3)], 0)
                res = torch.cat([_im.cpu(), _pred.cpu()], dim=0)

                # comp = np.load(os.path.join("_test", fname.split("/")[-2],fname.split("/")[-1])+".npy")
                # if not (comp == latent[i:i+1, ...].detach().cpu().numpy()).all().item():
                #     print(fname)
                #     print(comp)
                #     print(latent[i:i+1, ...].detach().cpu().numpy())
                
                np.save(os.path.join(latent_dir, fname.split("/")[-2],fname.split("/")[-1]), latent[i:i+1, ...].detach().cpu().numpy())
                #save_image(res, os.path.join(im_dir, fname.split("/")[-2],fname.split("/")[-1]), normalize=True, value_range=(-1, 1), nrow=4)
    
if os.path.exists("./test_out"):
    shutil.rmtree("./test_out")
for i in range(20, 201, 20):
    main(i)