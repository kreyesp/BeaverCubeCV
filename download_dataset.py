import os
import random
import time
import numpy as np
import rasterio as rio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cv2
import tacoreader.v1 as tacoreader
from datasets import load_dataset, concatenate_datasets

# ── Config ─────────────────────────────────────────────────────────────────────
SAVE_DIR    = "data/bluefox"
MAX_SAMPLES = None
SEED        = 42
RETRIES     = 3
WAIT        = 5

# ── Camera specs ───────────────────────────────────────────────────────────────
SPECS = {
    "s2":  {"pixel_m": 7.5e-6, "focal_m": 0.600, "alt_m": 786_000.0},
    "bc2": {"pixel_m": 6.0e-6, "focal_m": 0.016, "alt_m": 410_000.0,
            "f_number": 2.8},
}

def gsd(cam):
    return cam["pixel_m"] * cam["alt_m"] / cam["focal_m"]

GSD_S2  = gsd(SPECS["s2"])
GSD_BC2 = gsd(SPECS["bc2"])
SCALE   = GSD_BC2 / GSD_S2   # ~15.65

def psf_sigma_s2px(cam_target, cam_source, wavelength_m=550e-9):
    D             = cam_target["focal_m"] / cam_target["f_number"]
    r_airy_rad    = 1.22 * wavelength_m / D
    r_airy_ground = r_airy_rad * cam_target["alt_m"]
    r_airy_src_px = r_airy_ground / gsd(cam_source)
    return max(r_airy_src_px / 1.22, 0.5)

PSF_SIGMA = psf_sigma_s2px(SPECS["bc2"], SPECS["s2"])
print(f"GSD S2: {GSD_S2:.1f} m | GSD BC2: {GSD_BC2:.1f} m | "
      f"scale: {SCALE:.2f}x | PSF sigma: {PSF_SIGMA:.2f} S2-px")

# ── Transform (no noise -- applied at train time instead) ──────────────────────
def s2_to_bluefox(rgb_s2, label_s2, scale=SCALE, psf_sigma=PSF_SIGMA):
    """
    rgb_s2:   (3, H, W) uint16
    label_s2: (H, W)    int
    Returns:
        rgb_bf:   (H', W', 3) float32 [0, 1]   -- saved as float32
        label_bf: (H', W')    int8              -- saved as int8
    """
    _, H, W = rgb_s2.shape
    new_h = max(1, int(round(H / scale)))
    new_w = max(1, int(round(W / scale)))

    # 1. Normalize
    rgb_f = rgb_s2.astype(np.float32) / 10_000.0

    # 2. PSF blur (C, H, W)
    rgb_f = gaussian_filter(rgb_f, sigma=(0, psf_sigma, psf_sigma))

    # 3. INTER_AREA downsample -- expects (H, W, C)
    rgb_hwc = rgb_f.transpose(1, 2, 0)
    rgb_ds  = cv2.resize(rgb_hwc, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Labels: nearest-neighbour, no interpolation
    label_ds = cv2.resize(label_s2.astype(np.uint8), (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST).astype(np.int8)

    return rgb_ds, label_ds   # (H', W', 3) float32,  (H', W') int8


# ── Download ───────────────────────────────────────────────────────────────────
def download_dataset(taco_dataset, max_samples=MAX_SAMPLES, seed=SEED):
    os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/labels", exist_ok=True)

    # Filter to high-quality labels only
    metadata = load_dataset("csaybar/CloudSEN12-high", split="train+validation+test")
    high_quality_indices = set(
        row["index"] for row in metadata if row["label_type"] == "high"
    )
    print(f"High quality samples available: {len(high_quality_indices)}")

    indices = [i for i in range(len(taco_dataset)) if i in high_quality_indices]
    print(f"Filtered to {len(indices)} high quality samples")

    if max_samples and max_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, max_samples)

    skipped = 0
    for i in tqdm(indices, desc="Downloading and transforming patches"):
        img_path = f"{SAVE_DIR}/images/{i}.npy"
        lbl_path = f"{SAVE_DIR}/labels/{i}.npy"

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            continue

        for attempt in range(RETRIES):
            try:
                s2_uri  = taco_dataset.read(i).read(0)
                lbl_uri = taco_dataset.read(i).read(1)

                with rio.open(s2_uri) as src, rio.open(lbl_uri) as dst:
                    rgb   = src.read([4, 3, 2])          # (3, H, W) uint16
                    label = dst.read(1).astype(np.int8)  # (H, W)

                rgb_bf, label_bf = s2_to_bluefox(rgb, label)

                # rgb_bf:   (H', W', 3) float32  e.g. (33, 33, 3)
                # label_bf: (H', W')    int8      e.g. (33, 33)
                np.save(img_path, rgb_bf)
                np.save(lbl_path, label_bf)
                break

            except Exception as e:
                if attempt < RETRIES - 1:
                    time.sleep(WAIT)
                else:
                    print(f"Failed on index {i} after {RETRIES} attempts: {e}")
                    skipped += 1

    print(f"Done. Skipped {skipped} samples.")
    print(f"Saved to {SAVE_DIR}/images and {SAVE_DIR}/labels")
    print(f"Image shape: (33, 33, 3) float32 | Label shape: (33, 33) int8")


if __name__ == "__main__":
    dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
    print(f"Total samples: {len(dataset)}")
    download_dataset(dataset)