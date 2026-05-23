import torch
import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
from model import SmallCloudNet
from config import CHECKPOINT, NUM_CLASSES

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Config -- change these ─────────────────────────────────────────────────────
REPO_NAME  = "beavercube-cloud-segmentation"   # will appear as your-username/beavercube-cloud-segmentation
PRIVATE    = False                              # set True if you don't want it public

# ── Load model + checkpoint ────────────────────────────────────────────────────
device     = torch.device("cpu")
model      = SmallCloudNet(in_ch=3, num_classes=NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f}, "
      f"val_mean_iou={checkpoint.get('val_mean_iou', 'N/A'):.4f})")

# ── Save a clean weights-only copy for upload ──────────────────────────────────
upload_path = "model_upload.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "epoch":            checkpoint["epoch"],
    "val_loss":         checkpoint["val_loss"],
    "val_mean_iou":     checkpoint.get("val_mean_iou"),
    "val_mean_f1":      checkpoint.get("val_mean_f1"),
    "num_classes":      NUM_CLASSES,
    "architecture":     "SmallCloudNet",
    "in_channels":      3,
    "description":      "U-Net trained on CloudSEN12-L1C, simulated to BlueFOX IGC-200w GSD (153.75m)",
}, upload_path)

# ── Create repo and upload ─────────────────────────────────────────────────────
api = HfApi(token=HF_TOKEN)

# Get your username
user = api.whoami()["name"]
repo_id = f"{user}/{REPO_NAME}"
print(f"Uploading to: https://huggingface.co/{repo_id}")

create_repo(repo_id, token=HF_TOKEN, private=PRIVATE, exist_ok=True)

# Upload model weights
api.upload_file(
    path_or_fileobj=upload_path,
    path_in_repo="best_model.pth",
    repo_id=repo_id,
    token=HF_TOKEN,
    commit_message=f"Upload SmallCloudNet checkpoint (epoch {checkpoint['epoch']}, "
                   f"val_loss={checkpoint['val_loss']:.4f})",
)

# Upload model definition
api.upload_file(
    path_or_fileobj="model.py",
    path_in_repo="model.py",
    repo_id=repo_id,
    token=HF_TOKEN,
    commit_message="Upload model definition",
)

# Upload config
api.upload_file(
    path_or_fileobj="config.py",
    path_in_repo="config.py",
    repo_id=repo_id,
    token=HF_TOKEN,
    commit_message="Upload config",
)

# Write and upload a model card
model_card = f"""---
license: other
tags:
  - image-segmentation
  - satellite-imagery
  - cloud-detection
  - cubesat
  - beavercube
---

# BeaverCube Cloud Segmentation — SmallCloudNet

U-Net trained to detect clouds in imagery simulated from the Matrix Vision mvBlueFOX-IGC-200w camera
on the BeaverCube 2 CubeSat (MIT), using the CloudSEN12-L1C Sentinel-2 dataset.

## Model details

| Property | Value |
|---|---|
| Architecture | U-Net (SmallCloudNet) |
| Parameters | 1.86M |
| Input size (training) | 33×33 px |
| Input size (inference) | any (fully convolutional) |
| Classes | clear, thick cloud, thin cloud, shadow |

## Performance

| Metric | Value |
|---|---|
| Mean IoU | 0.38 |
| Mean F1 | 0.54 |
| Accuracy | 64% |
| Clear IoU | 0.58 |
| Thick cloud IoU | 0.45 |
| Shadow IoU | 0.27 |
| Thin cloud IoU | 0.22 |

## Usage

```python
import torch
from model import SmallCloudNet

model = SmallCloudNet(in_ch=3, num_classes=4)
checkpoint = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# img: float32 tensor (1, 3, H, W) normalised to [0, 1]
with torch.no_grad():
    logits = model(img)               # (1, 4, H, W)
    mask   = logits.argmax(dim=1)     # (1, H, W)
```

## Training data

CloudSEN12-L1C (Sentinel-2 L1C), preprocessed to simulate BlueFOX GSD (153.75 m) via:
1. Gaussian PSF blur (σ = 1.6 px, derived from Kowa 16mm lens Airy disk)
2. 15.65× INTER_AREA downsample (512×512 → 33×33)
3. Read + shot noise augmentation applied each epoch
"""

with open("model_card.md", "w", encoding="utf-8") as f:
    f.write(model_card)

api.upload_file(
    path_or_fileobj="model_card.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    token=HF_TOKEN,
    commit_message="Add model card",
)

# Cleanup temp files
os.remove(upload_path)
os.remove("model_card.md")

print(f"\nDone. Model uploaded to: https://huggingface.co/{repo_id}")