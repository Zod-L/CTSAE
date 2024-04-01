import os
import torch
import numpy as np
from data import *
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models import cnn_share_attn, cnn_split_attn, cnn, vit_share, conformer, cnn_concat_attn, cnn_nofuse_attn
import shutil
import json
import torch.nn.functional as F


def main(idx, dir, out_dir, model_class, split):
    latent_dir = os.path.join(out_dir, f"{split}/test_{idx}")
    im_dir = os.path.join(out_dir, f"{split}/test_im_{idx}")

    if os.path.exists(latent_dir):
        return

    model = model_class(use_vae=False)
    # if hasattr(model, "encoder"):
    #     print(f"Number of encoder parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}")
    # else:
    #     print(f"Number of encoder parameters: {sum(p.numel() for i in range(model.num_branch) for p in getattr(model, f'encoder_{i}').parameters()  if p.requires_grad)}")
    # if hasattr(model, "decoder"):
    #     print(f"Number of decoder parameters: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}")
    # else:
    #     print(f"Number of encoder parameters: {sum(p.numel() for i in range(model.num_branch) for p in getattr(model, f'decoder_{i}').parameters()  if p.requires_grad)}") 

    # exit()
    model.load_state_dict(torch.load(f"{dir}/checkpoint_{idx}.pth", map_location=device)["model"])
    model = model.to(device)
    model.eval()
    dataset = four_scale_dataset_with_fname(f"../gravityspy/mixed_split/{split}/", 0)
    C = dataset[0][0].shape[0]
    dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=4)
    l2 = 0
    l1 = 0


    
    with torch.no_grad():
        for im, ori_im, msk, fnames in tqdm(dataloader):




            pred, latent, var = model(im.to(device))

            l1 += F.l1_loss(pred, im.to(device)) / len(dataloader)
            l2 += F.mse_loss(pred, im.to(device)) / len(dataloader)
            if latent.ndim == 3:
                latent = latent[:, 0, :]





            for i, fname in enumerate(fnames):
                if not os.path.exists(os.path.join(latent_dir, fname.split("/")[-2])):
                    os.makedirs(os.path.join(latent_dir, fname.split("/")[-2]))

                # if not os.path.exists(os.path.join(im_dir, fname.split("/")[-2])):
                #     os.makedirs(os.path.join(im_dir, fname.split("/")[-2]))
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
            
        res = {}
        if os.path.exists(os.path.join(out_dir.split("/")[0], "l1_l2.json")):
            with open(os.path.join(out_dir.split("/")[0], "l1_l2.json")) as fh:
                res = json.load(fh)
        
        res[f"{out_dir.split('/')[1]}_{split}_{idx}"] = f"l1 : {l1.item()} l2 : {l2.item()}"


        with open(os.path.join(out_dir.split("/")[0], "l1_l2.json"), "w") as fh: 
            json.dump(res, fh)
    
# if os.path.exists("./test_out"):
#     shutil.rmtree("./test_out")

batch = 32
for name in ["cnn_split_attn"]:
    indir = f"mix_output/{name}"
    outdir = f"latent_code/{name}"
    for split in ["val", "test"]:
        device = "cuda"
        ckpt_num = [int(fname.split(".")[0]) for fname in os.listdir(indir) if fname.endswith(".png")]
        ckpt_num.sort()
        for i in ckpt_num:
            if i % 20 == 0:
                main(i, indir, outdir, eval(name), split)