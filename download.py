import os
import random
import time
import numpy as np
import rasterio as rio
from tqdm import tqdm
import tacoreader.v1 as tacoreader
from config import SAVE_DIR, MAX_SAMPLES, SEED
from transforms import rescale_to_iss_resolution, random_crop
from datasets import load_dataset, concatenate_datasets


def download_dataset(taco_dataset, max_samples=MAX_SAMPLES, seed=SEED, retries=3, wait=5):
    os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/labels", exist_ok=True)

    # Load metadata to filter by label type
    metadata = concatenate_datasets([
        load_dataset("csaybar/CloudSEN12-high", split="train"),
        load_dataset("csaybar/CloudSEN12-high", split="validation"),
        load_dataset("csaybar/CloudSEN12-high", split="test"),
    ])

    high_quality_indices = set(
        row["index"] for row in metadata
        if row["label_type"] == "high"
    )
    print(f"High quality samples available: {len(high_quality_indices)}")
    # Then filter your indices in download_dataset
    indices = [i for i in range(len(taco_dataset)) if i in high_quality_indices]
    print(f"Filtered to {len(indices)} high quality samples")

   
    if max_samples and max_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, max_samples)

    skipped = 0
    for i in tqdm(indices, desc="Downloading patches"):
        img_path = f"{SAVE_DIR}/images/{i}.npy"
        lbl_path = f"{SAVE_DIR}/labels/{i}.npy"

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            continue

        for attempt in range(retries):
            try:
                s2_uri  = taco_dataset.read(i).read(0)
                lbl_uri = taco_dataset.read(i).read(1)

                with rio.open(s2_uri) as src, rio.open(lbl_uri) as dst:
                    rgb   = src.read([4, 3, 2]).astype(np.float32)  # (3, H, W)
                    label = dst.read(1).astype(np.int8)              # (H, W) — 0,1,2,3,99 fits in int8

                # (3, H, W) → (H, W, 3) for transforms
                rgb = rgb.transpose(1, 2, 0)

                # Normalize to [0, 1] then convert to uint8 [0, 255] — matches camera output
                rgb = (rgb / 3000.0).clip(0, 1)
                rgb = (rgb * 255).astype(np.uint8)

                # Rescale to ISS GSD (~1006x1006) — use nearest for uint8 to avoid artifacts
                rgb   = rescale_to_iss_resolution(rgb,   is_label=False)
                # Cast to uint8 before resizing so cv2 INTER_NEAREST works correctly
                label = label.astype(np.uint8)
                label = rescale_to_iss_resolution(label, is_label=True)
                label = label.astype(np.int8)

                # Random crop to camera resolution (752x480)
                rgb, label = random_crop(rgb, label)

                # Save
                np.save(img_path, rgb)    # uint8  (480, 752, 3)
                np.save(lbl_path, label)  # int8   (480, 752)
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    print(f"Failed on index {i} after {retries} attempts: {e}")
                    skipped += 1

    print(f"Done. Skipped {skipped} samples.")
    return indices


if __name__ == "__main__":
    dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
    print(f"Total samples: {len(dataset)}")
    download_dataset(dataset)