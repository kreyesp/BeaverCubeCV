import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import SAVE_DIR, SEED


class CloudSEN12Dataset(Dataset):
    def __init__(self, save_dir=SAVE_DIR, max_samples=None, seed=SEED):
        self.img_dir = os.path.join(save_dir, "images")
        self.lbl_dir = os.path.join(save_dir, "labels")

        available = [
            int(f.replace(".npy", ""))
            for f in os.listdir(self.img_dir)
            if f.endswith(".npy")
        ]
        available.sort()

        if max_samples and max_samples < len(available):
            random.seed(seed)
            self.indices = random.sample(available, max_samples)
        else:
            self.indices = available

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        rgb   = np.load(f"{self.img_dir}/{i}.npy")  # uint8  (480, 752, 3)
        label = np.load(f"{self.lbl_dir}/{i}.npy")  # int8   (480, 752)

        # Normalize to [0, 1] float32 for PyTorch
        rgb = rgb.astype(np.float32) / 255.0

        # Cast label back to int64 for CrossEntropyLoss
        label = label.astype(np.int64)

        # Remap no-data value: 99 stored as int8 overflows, use -1 instead
        label[label == 99] = -1
        label[label == -27] = -1  # int8 overflow of 99

        # Temporary debug — remove once fixed
        unique = np.unique(label)
        if not all(v in [-1, 0, 1, 2, 3] for v in unique):
            print(f"WARNING: unexpected label values: {unique}")

        # (H, W, 3) → (3, H, W) for PyTorch
        rgb = rgb.transpose(2, 0, 1)

        return torch.tensor(rgb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)