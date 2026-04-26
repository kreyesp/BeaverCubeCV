from transformers import SegformerForSemanticSegmentation
from config import CAMERA_H, CAMERA_W
import torch
import torch.nn as nn

# 4 classes: 0=clear, 1=thick cloud, 2=thin cloud, 3=shadow
# 99 (no-data) is ignored in the loss function
NUM_CLASSES = 4

def get_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NUM_CLASSES,
        )
    return model