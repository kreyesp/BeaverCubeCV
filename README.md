# BeaverCube CV — Cloud Segmentation

Lightweight cloud segmentation model trained to run on the BeaverCube 2 satellite's onboard vision payload (Matrix Vision mvBlueFOX-IGC-200w + 16mm Kowa lens, Raspberry Pi Compute Module 3+).

The model is trained on CloudSEN12 Sentinel-2 imagery that has been preprocessed to physically simulate what the BlueFOX camera will actually see in orbit.

---

## Why preprocessing matters

Sentinel-2 imagery has roughly **10 m ground sampling distance (GSD)**. The BlueFOX on BeaverCube 2 will see **~154 m GSD** — over 15× coarser per pixel — because of its shorter focal length and different pixel pitch.

A naive altitude-ratio rescaling misses this entirely. The correct quantity is the pinhole-derived GSD:

```
GSD = pixel_pitch × altitude / focal_length
```

| Camera | Pixel pitch | Focal length | Altitude | GSD |
|---|---|---|---|---|
| Sentinel-2 MSI | 7.5 μm | 600 mm | 786 km | 9.83 m |
| BlueFOX + Kowa 16mm | 6.0 μm | 16 mm | 410 km | 153.75 m |

Scale factor: **15.65×**

---

## Preprocessing pipeline

Implemented in `download_dataset.py`. Each Sentinel-2 chip is transformed as follows:

1. **Normalize** — uint16 [0, 10000] → float32 [0, 1]
2. **Gaussian PSF blur** — σ = 1.6 source pixels, derived from the Kowa lens Airy disk (r = 1.22λ/D)
3. **INTER_AREA downsample** — 15.65× downsample (512×512 → 33×33); pixel-area averaging physically matches how a sensor integrates light over its ground footprint
4. **Sensor noise** — applied at training time in the DataLoader, not at preprocessing. Read noise (fixed Gaussian) + shot noise (signal-dependent, ∝ √signal). Re-randomized each epoch as free data augmentation.

The PSF blur and downsample are deterministic and saved to disk. Noise is stochastic and applied on-the-fly during training.

---

## Model

**SmallCloudNet** (`model.py`) — a custom U-Net designed for small inputs:

- 3 encoder levels, channels 64 → 128 → 256
- Skip connections with size matching (cropping when shapes mismatch)
- **GroupNorm** instead of BatchNorm (more stable at small spatial sizes; batch statistics get noisy when feature maps are only a few pixels)
- 1.86M parameters, ~7.4 MB on disk as float32

### Why U-Net?

Fully convolutional — no hardcoded spatial dimensions, so the model trained on 33×33 patches can run inference on full 752×480 BlueFOX frames with no code changes. Skip connections preserve spatial detail through the encoder/decoder bottleneck, important for clean cloud boundaries.

---

## Training

`train.py` — standard PyTorch training loop with:

- **Class-weighted cross-entropy** to upweight rarer classes (thin cloud, shadow)
- **`ignore_index=-1`** for no-data pixels
- **Gradient clipping** at norm 1.0 for stability
- **Early stopping** on validation loss with configurable patience
- **AdamW** optimizer with weight decay
- Tracks mean IoU, mean F1, per-class IoU, and accuracy with torchmetrics

Splits: 70% train / 15% val / 15% test, seeded for reproducibility.

---

## Trained model

The best checkpoint is available on Hugging Face:
**[kreyesp/beavercube-cloud-segmentation](https://huggingface.co/kreyesp/beavercube-cloud-segmentation)**

```python
import torch
from model import SmallCloudNet

model = SmallCloudNet(in_ch=3, num_classes=4)
checkpoint = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## Results

Best run: 1.86M parameter SmallCloudNet, ~8,500 high-quality CloudSEN12-L1C samples.

| Metric | Value |
|---|---|
| Mean IoU | 0.38 |
| Mean F1 | 0.54 |
| Accuracy | 64% |
| Clear IoU | 0.58 |
| Thick cloud IoU | 0.45 |
| Shadow IoU | 0.27 |
| Thin cloud IoU | 0.22 |

Thin cloud and shadow underperform because at 154 m GSD they are near the physical limit of detectability — the resolution itself is the bottleneck, not the model.

A 4-level encoder version with 7M parameters was tried and overfit (worse on test), confirming that 1.86M is the right capacity for this dataset size.

---

## File structure

```
.
├── config.py             # Single source of truth for camera specs, paths, hyperparams
├── download_dataset.py   # Downloads CloudSEN12 + applies S2 → BlueFOX preprocessing
├── dataset.py            # PyTorch Dataset wrapper; applies noise at load time
├── model.py              # SmallCloudNet U-Net
├── train.py              # Training loop with early stopping + metrics
├── visualize.py          # Inference on random test samples
├── best_model.pth        # Best checkpoint by val loss
├── requirements.txt
└── data/bluefox/         # Preprocessed dataset (created by download_dataset.py)
    ├── images/           # 33×33 float32 .npy files
    └── labels/           # 33×33 int8   .npy files
```

---

## Setup

### Hugging Face token

The dataset is hosted on Hugging Face and requires authentication. Create a free account at [huggingface.co](https://huggingface.co), then generate a read token at **Settings → Access Tokens**.

Create a `.env` file in the repo root:

```
HF_TOKEN=your_token_here
```

This file is loaded automatically by `config.py` via `python-dotenv`. Never commit it — it is already listed in `.gitignore`.

---

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your HF token to .env (see Setup above)

# 3. Download and preprocess dataset (~118 MB, runs once)
python download_dataset.py

# 4. Train
python train.py

# 5. Visualize predictions on random test samples
python visualize.py
```

---

## Design decisions worth knowing

- **GSD ratio, not altitude ratio.** The scale factor accounts for pixel pitch and focal length, not just orbital altitude.
- **Save preprocessed, augment on load.** Deterministic steps (PSF, downsample) live on disk; stochastic steps (noise) run in the DataLoader so each epoch sees variation.
- **GroupNorm over BatchNorm.** Feature maps at the bottleneck are only ~4×4; BatchNorm statistics would be unreliable at that size.
- **Cropping skip connections in forward.** Odd input sizes (33) don't divide cleanly through pooling. Cropping the encoder feature map to match the upsampled decoder shape avoids needing to pad inputs to powers of 2.
- **Float32 weights, no quantization.** Model is 7 MB on disk; the Raspberry Pi has 1 GB RAM. Quantization would only matter if inference speed becomes a problem.
- **Trained on 33×33, deploys on 752×480.** Because the U-Net is fully convolutional, the same weights handle both. The tradeoff is that the model never saw long-range spatial context during training.

---

## Camera and orbit references

- BeaverCube 2: eoPortal mission page
- mvBlueFOX-IGC datasheet (Matrix Vision)
- Sentinel-2 MSI technical guide (ESA Sentinel Online)
- CloudSEN12: Aybar et al., 2022