# model / box_ops.py

# -----
# Computes IoU Between Two Sets of Boxes.
# IoU Measures Overlap Between Predicted Boxes & Ground Truth Boxes.
# -----

# Imports
import torch

# Box IoU Function.
def box_iou(boxes1, boxes2):
    """
    Compute Intersection-over-Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Tensor of shape [N, 4] representing N boxes (x1, y1, x2, y2).
        boxes2 (torch.Tensor): Tensor of shape [M, 4] representing M boxes (x1, y1, x2, y2).

    Returns:
        torch.Tensor: IoU matrix of shape [N, M] where IoU[i, j] is the IoU of boxes1[i] and boxes2[j].
    """

    # Handle empty input case to prevent runtime errors
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # Compute Box Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Compute Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # Ensure non-negative width/height
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute Union
    union = area1[:, None] + area2 - intersection  # [N, M]

    # Compute IoU (Prevent division by zero)
    return intersection / (union + 1e-6)  # Adding small epsilon to avoid NaN
