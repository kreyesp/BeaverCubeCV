import numpy as np
import os

SAVE_DIR = "./cloudsen12_local"
VALID = {0, 1, 2, 3, 99}

bad_files = []
for fname in os.listdir(f"{SAVE_DIR}/labels"):
    path = os.path.join(SAVE_DIR, "labels", fname)
    label = np.load(path)
    unique = set(np.unique(label).tolist())
    bad = unique - VALID
    if bad:
        bad_files.append((fname, bad))

print(f"Bad files: {len(bad_files)} / 2000")
for fname, bad in bad_files[:20]:
    print(f"  {fname} | unexpected values: {bad}")