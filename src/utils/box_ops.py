import torch

def box_iou(boxes1, boxes2):
    """Memory-efficient IoU computation."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Handle empty boxes
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - intersection
    return intersection / (union + 1e-6)  # Add epsilon to avoid division by zero 