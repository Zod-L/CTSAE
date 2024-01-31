
import torch
import cv2
import os


for scale in ["sub_0.5", "sub_1.0", "sub_2.0", "sub_4.0"]:
    root = f"../gravityspy/test_sidd_H1/"
    
    for dir in os.listdir(os.path.join(root, scale)):
        for fname in os.listdir(os.path.join(root, scale, dir)):
            im = cv2.imread(os.path.join(root, scale, dir, fname))
            im = im[61 : 540, 101 : 674, :]
            im = cv2.resize(im, (512, 512))
            if not os.path.exists(os.path.join(f"../gravityspy/test_sidd/{scale}", dir)):
                os.makedirs(os.path.join(f"../gravityspy/test_sidd/{scale}", dir))
            cv2.imwrite(os.path.join(f"../gravityspy/test_sidd/{scale}", dir, fname), im)
