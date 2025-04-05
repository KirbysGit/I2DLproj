# restart/test/visualize_detections.py

import cv2
import torch
import matplotlib.pyplot as plt
from restart.utils.box_ops import unnormalize_boxes
from torchvision.transforms.functional import to_pil_image
import numpy as np

def visualize_detections(image_tensor, detections, ground_truths=None, orig_size=None, resize_size=None, save_path=None, title="Detections"):
    """
    Draws predicted (green) and ground truth (red) boxes on an image tensor and displays with matplotlib.

    Args:
        image_tensor (Tensor): Image tensor in CHW format, range [0, 1].
        detections (dict): Dictionary with 'boxes', 'scores', and 'labels' for predicted outputs.
        ground_truths (dict, optional): Dictionary with 'boxes' and 'labels' for ground truth boxes.
        title (str): Title for the displayed image.
    """

    # Unnormalize boxes if they are normalized.
    if ground_truths is not None and orig_size is not None and resize_size is not None:
        ground_truths['boxes'] = unnormalize_boxes(ground_truths['boxes'].clone(), orig_size, resize_size)



    image = image_tensor.cpu().clone()
    image = to_pil_image(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw predicted boxes (green)
    boxes = detections['boxes'].cpu()
    scores = detections['scores'].cpu()
    labels = detections['labels'].cpu()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"P: {label.item()} | {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw ground truth boxes (red)
    if ground_truths is not None:
        gt_boxes = ground_truths['boxes'].cpu()
        gt_labels = ground_truths['labels'].cpu()

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label_text = f"GT: {label.item()}"
            cv2.putText(image, label_text, (x1, y2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Convert back to RGB for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

