import os
import pickle
import shutil
fnames = {}
with open("fnames.pkl", "rb") as fh:
    fnames = pickle.load(fh)


for s in os.listdir("../gravityspy/processed/"):
    if not os.path.exists(f"../gravityspy/train/{s}"):
        os.makedirs(f"../gravityspy/train/{s}")
    if not os.path.exists(f"../gravityspy/val/{s}"):
        os.makedirs(f"../gravityspy/val/{s}")
    if not os.path.exists(f"../gravityspy/test/{s}"):
        os.makedirs(f"../gravityspy/test/{s}")


    for c in os.listdir("../gravityspy/processed/" + s):
        if not os.path.exists(f"../gravityspy/train/{s}/{c}"):
            os.makedirs(f"../gravityspy/train/{s}/{c}")
        if not os.path.exists(f"../gravityspy/val/{s}/{c}"):
            os.makedirs(f"../gravityspy/val/{s}/{c}")
        if not os.path.exists(f"../gravityspy/test/{s}/{c}"):
            os.makedirs(f"../gravityspy/test/{s}/{c}")


for c in os.listdir("../gravityspy/processed/sub_0.5"):



        
    N = sum((len(fnames["train_fnames"][c]), len(fnames["val_fnames"][c]), len(fnames["test_fnames"][c])))
    print("=================")
    print("total:", c, N)
    for f in fnames["train_fnames"][c]:
        for scale in [0.5, 1.0, 2.0, 4.0]:
            shutil.copy(f"../gravityspy/processed/sub_{scale}/{c}/{f.replace('0.5', str(scale))}", f"../gravityspy/train/sub_{scale}/{c}/{f.replace('0.5', str(scale))}")
    print("train:", c, len(list(os.listdir(f"../gravityspy/train/sub_{0.5}/{c}"))))

    
    for f in fnames["val_fnames"][c]:
        for scale in [0.5, 1.0, 2.0, 4.0]:
            shutil.copy(f"../gravityspy/processed/sub_{scale}/{c}/{f.replace('0.5', str(scale))}", f"../gravityspy/val/sub_{scale}/{c}/{f.replace('0.5', str(scale))}")
    print("val:", c, len(list(os.listdir(f"../gravityspy/val/sub_{0.5}/{c}"))))



    for f in fnames["test_fnames"][c]:
        for scale in [0.5, 1.0, 2.0, 4.0]:
            shutil.copy(f"../gravityspy/processed/sub_{scale}/{c}/{f.replace('0.5', str(scale))}", f"../gravityspy/test/sub_{scale}/{c}/{f.replace('0.5', str(scale))}")
    print("test:", c, len(list(os.listdir(f"../gravityspy/test/sub_{0.5}/{c}"))))

train_fnames = {}
test_fnames = {}
val_fnames = {}
for c in os.listdir("../gravityspy/processed/sub_0.5"):
    train_fnames[c] = list(os.listdir(f"../gravityspy/train/sub_0.5/{c}"))
    test_fnames[c] = list(os.listdir(f"../gravityspy/test/sub_0.5/{c}"))
    val_fnames[c] = list(os.listdir(f"../gravityspy/val/sub_0.5/{c}"))


with open('fnames.pkl', 'wb') as fp:
    pickle.dump(dict(train_fnames=train_fnames, test_fnames=test_fnames, val_fnames=val_fnames), fp)