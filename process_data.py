
import torch
import cv2
import os


for scale in ["sub_0.5", "sub_1.0", "sub_2.0", "sub_4.0"]:
    indir = f"../gravityspy/test_sidd_H1/"
    outdir = f"./data"
    
    for dir in os.listdir(os.path.join(indir, scale)):
        for fname in os.listdir(os.path.join(indir, scale, dir)):
            im = cv2.imread(os.path.join(indir, scale, dir, fname))
            im = im[61 : 540, 101 : 674, :]
            im = cv2.resize(im, (512, 512))
            if not os.path.exists(os.path.join(f"{outdir}/{scale}", dir)):
                os.makedirs(os.path.join(f"{outdir}/{scale}", dir))
            cv2.imwrite(os.path.join(f"{outdir}/{scale}", dir, fname), im)
