from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Altitudes
SENTINEL2_ALT_KM = 786
ISS_ALT_KM       = 400
SCALE_FACTOR     = SENTINEL2_ALT_KM / ISS_ALT_KM  # ≈ 1.965

# Camera
CAMERA_W = 752
CAMERA_H = 480

# Paths
SAVE_DIR = "./cloudsen12_local"

# Training
BATCH_SIZE  = 8
NUM_WORKERS = 1
MAX_SAMPLES = 7324
SEED        = 42
NUM_EPOCHS    = 50   # higher ceiling since early stopping will kick in
PATIENCE      = 5    # stop if val loss doesn't improve for 5 epochs
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
CHECKPOINT    = "best_segformer.pth"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

NUM_CLASSES = 4
CLASS_NAMES = ["clear", "thick_cloud", "thin_cloud", "shadow"]