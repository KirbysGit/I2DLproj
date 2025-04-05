import torch

def normalize_boxes(boxes, orig_size, resized_size):
    """
    Normalize boxes from original image space to resized image space.
    """
    orig_h, orig_w = orig_size
    res_h, res_w = resized_size
    scale_x = res_w / orig_w
    scale_y = res_h / orig_h

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


def unnormalize_boxes(boxes, orig_size, resized_size):
    """
    Map boxes from resized space back to original image space.
    """
    orig_h, orig_w = orig_size
    res_h, res_w = resized_size
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
