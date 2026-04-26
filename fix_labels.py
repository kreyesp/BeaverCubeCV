import numpy as np
import os
import cv2

SAVE_DIR = "./cloudsen12_local"
VALID = {0, 1, 2, 3, 99}


def nearest_valid(label):
    """Replace invalid values with the nearest valid neighbour."""
    invalid_mask = ~np.isin(label, list(VALID))
    if not invalid_mask.any():
        return label

    # For each invalid pixel, find the nearest valid pixel
    from scipy.ndimage import distance_transform_edt
    valid_mask = ~invalid_mask
    # Get indices of nearest valid pixel for each invalid pixel
    _, indices = distance_transform_edt(invalid_mask, return_indices=True)
    fixed = label.copy()
    fixed[invalid_mask] = label[indices[0][invalid_mask], indices[1][invalid_mask]]
    return fixed


fixed_count = 0
for fname in os.listdir(f"{SAVE_DIR}/labels"):
    path = os.path.join(SAVE_DIR, "labels", fname)
    label = np.load(path)
    unique = set(np.unique(label).tolist())
    bad = unique - VALID

    if bad:
        fixed = nearest_valid(label)
        # Verify fix worked
        remaining = set(np.unique(fixed).tolist()) - VALID
        if remaining:
            print(f"  FAILED to fix {fname} — remaining bad values: {remaining}")
        else:
            np.save(path, fixed.astype(np.int8))
            fixed_count += 1

print(f"\nFixed {fixed_count} / 252 bad label files.")