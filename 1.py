from models import auto_encoder_no_comm_224
import torch
import os
import shutil
import random
import pickle
from data import four_scale_dataset
import numpy as np



for root, _, fnames in os.walk("/data/otn7723/gs_2.0/"):
    for fname in fnames:
        if ".gz" in fname:
            print(os.path.join(root, fname))
            print(os.path.join(root, fname.replace(".gz", "")))
            print()
            shutil.move(os.path.join(root, fname), os.path.join(root, fname.replace(".gz", "")))
            
