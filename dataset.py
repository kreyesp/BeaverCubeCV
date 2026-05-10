import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import SAVE_DIR, SEED


def add_sensor_noise(rgb_01, read_noise_std=0.005, shot_noise_scale=0.002):
    read = np.random.normal(0, read_noise_std,  rgb_01.shape).astype(np.float32)
    shot = np.random.normal(0, shot_noise_scale, rgb_01.shape).astype(np.float32) \
           * np.sqrt(np.clip(rgb_01, 0, None))
    return np.clip(rgb_01 + read + shot, 0, 1)


class CloudSEN12Dataset(Dataset):
    def __init__(self, save_dir=SAVE_DIR, max_samples=None, seed=SEED, add_noise=True):
        self.img_dir   = os.path.join(save_dir, "images")
        self.lbl_dir   = os.path.join(save_dir, "labels")
        self.add_noise = add_noise

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

        rgb   = np.load(f"{self.img_dir}/{i}.npy")  # float32 (33, 33, 3) already in [0, 1]
        label = np.load(f"{self.lbl_dir}/{i}.npy")  # int8    (33, 33)

        # Add noise at load time so each epoch sees different noise
        if self.add_noise:
            rgb = add_sensor_noise(rgb)

        # Remap no-data labels
        # Remap no-data labels
        label = label.astype(np.int64)
        label[label == 99]  = -1
        label[label == -27] = -1   # int8 overflow of 99
        label[~np.isin(label, [0, 1, 2, 3])] = -1   # catches 4, 5, 6 and anything unexpected

        # Sanity check -- remove once confirmed clean
        unique = np.unique(label)
        if not all(v in [-1, 0, 1, 2, 3] for v in unique):
            print(f"WARNING: unexpected label values at index {i}: {unique}")

        # (H, W, 3) → (3, H, W) for PyTorch
        rgb = rgb.transpose(2, 0, 1)

        return torch.tensor(rgb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)