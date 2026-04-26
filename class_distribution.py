import numpy as np
import os
from tqdm import tqdm

SAVE_DIR = "./cloudsen12_local"
counts = {0: 0, 1: 0, 2: 0, 3: 0, 99: 0}

for fname in tqdm(os.listdir(f"{SAVE_DIR}/labels")):
    path = os.path.join(SAVE_DIR, "labels", fname)
    label = np.load(path).astype(np.int64)
    for v in counts:
        counts[v] += (label == v).sum()

total = sum(v for k, v in counts.items() if k != 99)
print("\nClass distribution:")
for k, v in counts.items():
    pct = 100 * v / (total + counts[99])
    print(f"  {k}: {v:>12,} pixels ({pct:.1f}%)")

print("\nSuggested inverse-frequency weights:")
for k in [0, 1, 2, 3]:
    w = total / (4 * counts[k]) if counts[k] > 0 else 0
    print(f"  class {k}: {w:.4f}")