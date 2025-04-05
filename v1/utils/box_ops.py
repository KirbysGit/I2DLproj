# model / box_ops.py

# -----
# Computes IoU Between Two Sets of Boxes.
# IoU Measures Overlap Between Predicted Boxes & Ground Truth Boxes.
# -----

# Imports
import torch

# Box IoU Function.
def test_box_iou(boxes1, boxes2):
    """
    Compute Intersection-over-Union (IoU) between two sets of bounding boxes.
    Works with both normalized [0,1] coordinates and pixel coordinates.

    Args:
        boxes1 (torch.Tensor): Tensor of shape [N, 4] representing N boxes (x1, y1, x2, y2).
        boxes2 (torch.Tensor): Tensor of shape [M, 4] representing M boxes (x1, y1, x2, y2).

    Returns:
        torch.Tensor: IoU matrix of shape [N, M] where IoU[i, j] is the IoU of boxes1[i] and boxes2[j].
    """
    # Handle empty input case to prevent runtime errors
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # Validate box coordinates (x2 > x1, y2 > y1)
    assert torch.all(boxes1[:, 2] > boxes1[:, 0]), "boxes1: x2 must be greater than x1"
    assert torch.all(boxes1[:, 3] > boxes1[:, 1]), "boxes1: y2 must be greater than y1"
    assert torch.all(boxes2[:, 2] > boxes2[:, 0]), "boxes2: x2 must be greater than x1"
    assert torch.all(boxes2[:, 3] > boxes2[:, 1]), "boxes2: y2 must be greater than y1"

    # Compute Box Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Compute Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # Ensure non-negative width/height
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # Compute Union
    union = area1[:, None] + area2 - intersection  # [N, M]

    # Compute IoU (Prevent division by zero)
    iou = intersection / (union + 1e-6)  # Adding small epsilon to avoid NaN

    # Validate IoU values
    assert torch.all((iou >= 0) & (iou <= 1)), "IoU values must be between [0,1]"

    return iou


def box_iou(boxes1, boxes2, chunk_size=10000):
    """
    Compute IoU between two sets of bounding boxes in a memory-efficient way using chunking.

    Args:
        boxes1 (torch.Tensor): [N, 4]
        boxes2 (torch.Tensor): [M, 4]
        chunk_size (int): Max number of boxes1 rows to process at once

    Returns:
        torch.Tensor: [N, M] IoU matrix
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    assert torch.all(boxes1[:, 2] > boxes1[:, 0]), "boxes1: x2 must be > x1"
    assert torch.all(boxes1[:, 3] > boxes1[:, 1]), "boxes1: y2 must be > y1"
    assert torch.all(boxes2[:, 2] > boxes2[:, 0]), "boxes2: x2 must be > x1"
    assert torch.all(boxes2[:, 3] > boxes2[:, 1]), "boxes2: y2 must be > y1"

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    iou_matrix = []

    for i in range(0, boxes1.size(0), chunk_size):
        b1 = boxes1[i:i+chunk_size]
        a1 = area1[i:i+chunk_size]

        lt = torch.max(b1[:, None, :2], boxes2[:, :2])  # [chunk, M, 2]
        rb = torch.min(b1[:, None, 2:], boxes2[:, 2:])  # [chunk, M, 2]

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = a1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)

        iou_matrix.append(iou)

    return torch.cat(iou_matrix, dim=0)  # [N, M]

