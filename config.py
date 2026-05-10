from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ── Camera & optics (single source of truth) ───────────────────────────────────
SPECS = {
    "s2":  {"pixel_m": 7.5e-6, "focal_m": 0.600, "alt_m": 786_000.0},
    "bc2": {"pixel_m": 6.0e-6, "focal_m": 0.016, "alt_m": 410_000.0,
            "f_number": 2.8},
}

def gsd(cam):
    return cam["pixel_m"] * cam["alt_m"] / cam["focal_m"]

GSD_S2      = gsd(SPECS["s2"])           # 9.83 m
GSD_BC2     = gsd(SPECS["bc2"])          # 153.75 m
SCALE_FACTOR = GSD_BC2 / GSD_S2          # ~15.65  (replaces old altitude-only ratio)

# ── Camera sensor dimensions ───────────────────────────────────────────────────
CAMERA_W      = 752    # physical BlueFOX sensor width  (inference frame size)
CAMERA_H      = 480    # physical BlueFOX sensor height (inference frame size)
PATCH_SIZE    = 33     # output size of each training patch after GSD downsampling
                       # = floor(512 / SCALE_FACTOR)

# ── Paths ──────────────────────────────────────────────────────────────────────
SAVE_DIR   = "data/bluefox"
CHECKPOINT = "best_model.pth"

# ── Dataset ────────────────────────────────────────────────────────────────────
MAX_SAMPLES = 8490
SEED        = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Model ──────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 4
CLASS_NAMES  = ["clear", "thick_cloud", "thin_cloud", "shadow"]

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 512     # 33×33 patches are tiny -- can afford larger batches
NUM_WORKERS   = 0
NUM_EPOCHS    = 50
PATIENCE      = 8
LEARNING_RATE = 2e-4   # bumped up slightly from 2e-5 -- fine for a small CNN
WEIGHT_DECAY  = 0.01