import torch

def normalize_boxes(boxes, orig_size, resized_size):
    """
    Normalize boxes from original image space to 0-1 normalized space.
    
    Args:
        boxes: tensor of shape [N, 4] containing bounding boxes in (x1, y1, x2, y2) format
        orig_size: tuple of (height, width) of original image
        resized_size: tuple of (height, width) of resized image
    
    Returns:
        Normalized boxes in range [0, 1]
    """
    if len(boxes) == 0:
        return boxes
        
    orig_h, orig_w = orig_size
    res_h, res_w = resized_size
    
    # First scale boxes to resized image space
    scale_x = res_w / orig_w
    scale_y = res_h / orig_h
    
    boxes = boxes.clone()  # Create a copy to avoid modifying original
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    # Then normalize to [0, 1] range
    boxes[:, [0, 2]] /= res_w
    boxes[:, [1, 3]] /= res_h
    
    return boxes


def unnormalize_boxes(boxes, orig_size, resized_size):
    """
    Map boxes from normalized [0-1] space back to original image space.
    
    Args:
        boxes: tensor of shape [N, 4] containing normalized boxes in (x1, y1, x2, y2) format
        orig_size: tuple of (height, width) of original image
        resized_size: tuple of (height, width) of resized image
    
    Returns:
        Boxes in original image pixel coordinates
    """
    if len(boxes) == 0:
        return boxes
        
    orig_h, orig_w = orig_size
    res_h, res_w = resized_size
    
    boxes = boxes.clone()  # Create a copy to avoid modifying original
    
    # First map from [0, 1] to resized image space
    boxes[:, [0, 2]] *= res_w
    boxes[:, [1, 3]] *= res_h
    
    # Then scale to original image space
    scale_x = orig_w / res_w
    scale_y = orig_h / res_h
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    return boxes


def box_iou(boxes1, boxes2):
    # Computes IoU between 2 sets of boxes
    # boxes1: [N, 4], boxes2: [M, 4]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou
