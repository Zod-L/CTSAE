import os
import torch
import numpy as np
from data import *
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models import cnn_share_attn, cnn_split_attn, cnn, vit_share, conformer
import shutil



def main(i, dir, out_dir, model_class):
    latent_dir = os.path.join(out_dir, f"{split}/test_{i}")
    im_dir = os.path.join(out_dir, f"{split}/test_im_{i}")

    if os.path.exists(latent_dir):
        return

    model = model_class(use_vae=False)
    model.load_state_dict(torch.load(f"{dir}/checkpoint_{i}.pth", map_location=device)["model"])
    model = model.to(device)
    model.eval()
    dataset = four_scale_dataset_with_fname(f"../gravityspy/mixed_split/{split}/", 0)
    C = dataset[0][0].shape[0]
    batch = 16
    dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=4)



    
    with torch.no_grad():
        for im, ori_im, msk, fnames in tqdm(dataloader):



            pred, latent, var = model(im.to(device))
            latent = latent[:, :]





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
    
# if os.path.exists("./test_out"):
#     shutil.rmtree("./test_out")
split = "test"                
indir = "mix_output/conformer"
outdir = "latent_code/conformer"
device = "cuda"
ckpt_num = [int(fname.split(".")[0]) for fname in os.listdir(indir) if fname.endswith(".png")]
ckpt_num.sort()
for i in ckpt_num:
    if i % 10 == 0:
        main(i, indir, outdir, conformer)