import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import random

# class folder_dataset(Dataset):
#     def __init__(self, path, threshold):
#         super().__init__()
#         self.path = path
#         self.fnames = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]
#         self.to_tensor = transforms.Compose([transforms.ToTensor()])
#         self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                              transforms.Resize(224, antialias=True)])
#         self.threshold = threshold
#     def __len__(self):
#         return len(self.fnames)
    
#     def __getitem__(self, idx):
#         im = Image.open(self.fnames[idx])
#         im = self.to_tensor(im)
#         msk = (im > self.threshold).any(0, keepdim=True)

#         return self.normalize(msk * im), self.normalize(im), msk 


class four_scale_dataset(Dataset):
    def __init__(self, path, threshold, im_size=224):
        super().__init__()
        self.path = path
        self.scale = ["0.5", "1.0", "2.0", "4.0"]
        # self.scale = ["4.0"]
        self.fnames = [os.path.join(dir, f.replace("_4.0", "")) for dir in os.listdir(f"{path}/sub_4.0/") for f in os.listdir(os.path.join(f"{path}/sub_4.0/", dir))]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                             transforms.Resize(im_size, antialias=True)])
        self.threshold = threshold
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        msk_ims = []
        ims = []
        msks = []
        for scale in self.scale:
            im = Image.open(os.path.join(self.path, f"sub_{scale}", self.fnames[idx].replace(".png", f"_{scale}.png")))
            im = self.to_tensor(im)
            msk = (im > self.threshold).any(0, keepdim=True)
            msk_ims.append(self.normalize(msk * im))
            ims.append(self.normalize(im))
            msks.append(msk)
        return torch.concat(msk_ims, 0), torch.concat(ims, 0), torch.concat(msks, 0) 



# class folder_dataset_with_fname(Dataset):
#     def __init__(self, path, threshold):
#         super().__init__()
#         self.path = path
#         self.fnames = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]
#         self.to_tensor = transforms.Compose([transforms.ToTensor()])
#         self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                              transforms.Resize(224, antialias=True)])
#         self.threshold = threshold
#     def __len__(self):
#         return len(self.fnames)
    
#     def __getitem__(self, idx):
#         im = Image.open(self.fnames[idx])
#         im = self.to_tensor(im)
#         msk = (im > self.threshold).any(0, keepdim=True)
#         return self.normalize(msk * im), self.normalize(im), msk, self.fnames[idx] 





class four_scale_dataset_with_fname(Dataset):
    def __init__(self, path, threshold, im_size=224):
        super().__init__()
        self.path = path
        self.scale = ["0.5", "1.0", "2.0", "4.0"]
        # self.scale = ["4.0"]
        self.fnames = [os.path.join(dir, f.replace("_0.5", "")) for dir in os.listdir(f"{path}/sub_0.5/") for f in os.listdir(os.path.join(f"{path}/sub_0.5/", dir))]
        
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                             transforms.Resize(im_size, antialias=True)])
        self.threshold = threshold
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        msk_ims = []
        ims = []
        msks = []
        for scale in self.scale:
            im = Image.open(os.path.join(self.path, f"sub_{scale}", self.fnames[idx].replace(".png", f"_{scale}.png")))
            im = self.to_tensor(im)
            msk = (im > self.threshold).any(0, keepdim=True)
            msk_ims.append(self.normalize(msk * im))
            ims.append(self.normalize(im))
            msks.append(msk)

        return torch.concat(msk_ims, 0), torch.concat(ims, 0), torch.concat(msks, 0), self.fnames[idx] 






class gs2_dataset(Dataset):
    def __init__(self, path, threshold, im_size=224):
        super().__init__()
        self.path = path
        self.scale = ["0.5", "1.0", "2.0", "4.0"]
        self.fnames = [os.path.join(root, fname) for root, _, fnames in os.walk(path) for fname in fnames if "0.5" in fname]
        random.shuffle(self.fnames)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                             transforms.Resize(im_size, antialias=True)])
        self.threshold = threshold
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        msk_ims = []
        ims = []
        msks = []
        for scale in self.scale:
            im = Image.open(self.fnames[idx].replace("_0.5.png", f"_{scale}.png"))
            im = self.to_tensor(im)
            msk = (im > self.threshold).any(0, keepdim=True)
            msk_ims.append(self.normalize(msk * im))
            ims.append(self.normalize(im))
            msks.append(msk)

        return torch.concat(msk_ims, 0), torch.concat(ims, 0), torch.concat(msks, 0), self.fnames[idx] 
