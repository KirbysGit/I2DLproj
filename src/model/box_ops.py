# model / box_ops.py

# -----

# Computes IoU Between Two Sets of Boxes.
# IoU Measures Overlap Between Our Predicted Boxes & Ground Truth Boxes.

# -----

# Imports.
import torch

# Box IoU Function.
def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    
    # Compute Areas.
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute Intersection.
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # Compute Width and Height.
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    # Compute Union.
    union = area1[:, None] + area2 - intersection
    
    # Compute IoU.
    return intersection / union 