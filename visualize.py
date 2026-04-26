import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import random
from config import SAVE_DIR, CLASS_NAMES, SEED

# Label colour map
COLORS = {
    -1: [0.2, 0.2, 0.2],   # no-data — dark grey
     0: [0.2, 0.6, 0.2],   # clear — green
     1: [0.9, 0.9, 0.9],   # thick cloud — white
     2: [0.6, 0.8, 1.0],   # thin cloud — light blue
     3: [0.4, 0.4, 0.4],   # shadow — dark grey
}


def label_to_rgb(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for val, color in COLORS.items():
        mask = label == val
        rgb[mask] = color
    return rgb


def visualize(n_samples=6, seed=SEED):
    random.seed(seed)

    img_dir = os.path.join(SAVE_DIR, "images")
    lbl_dir = os.path.join(SAVE_DIR, "labels")

    files = os.listdir(img_dir)
    chosen = random.sample(files, n_samples)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    fig.suptitle("CloudSEN12 — Image vs Label", fontsize=14, y=1.01)

    for i, fname in enumerate(chosen):
        img   = np.load(os.path.join(img_dir, fname))           # uint8 (H, W, 3)
        label = np.load(os.path.join(lbl_dir, fname)).astype(np.int64)
        label[label == 99] = -1

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{fname} — RGB")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_to_rgb(label))
        axes[i, 1].set_title("Label")
        axes[i, 1].axis("off")

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS[-1], label="No-data"),
        mpatches.Patch(color=COLORS[0],  label="Clear"),
        mpatches.Patch(color=COLORS[1],  label="Thick cloud"),
        mpatches.Patch(color=COLORS[2],  label="Thin cloud"),
        mpatches.Patch(color=COLORS[3],  label="Shadow"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    plt.tight_layout()
    plt.savefig("sample_visualization.png", bbox_inches="tight", dpi=150)
    print("Saved to sample_visualization.png")
    plt.show()


if __name__ == "__main__":
    visualize()