import os
import shutil

BASE = "results"
PDD = os.path.join(BASE, "pdd")
STAGES = ["stage1", "stage2", "stage3"]

os.makedirs(PDD, exist_ok=True)
for s in STAGES:
    dst = os.path.join(PDD, s)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(os.path.join(PDD, "X_"+("stage1" if s=="stage1" else s)+".npy"), os.path.join(dst, "X_train.npy"))
    shutil.copy2(os.path.join(PDD, "y_"+("stage1" if s=="stage1" else s)+".npy"), os.path.join(dst, "y_train.npy"))
    shutil.copy2(os.path.join(BASE, "baseline", "X_val.npy"), os.path.join(dst, "X_val.npy"))
    shutil.copy2(os.path.join(BASE, "baseline", "y_val.npy"), os.path.join(dst, "y_val.npy"))
    shutil.copy2(os.path.join(BASE, "baseline", "X_test.npy"), os.path.join(dst, "X_test.npy"))
    shutil.copy2(os.path.join(BASE, "baseline", "y_test.npy"), os.path.join(dst, "y_test.npy"))
print("PDD folders created.")
