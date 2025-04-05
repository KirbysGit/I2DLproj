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
        orig_size (tuple): Original image size (H, W)
        resize_size (tuple): Resized image size (H, W)
        title (str): Title for the displayed image.
    """
    # Convert image to numpy and scale to 0-255
    image = image_tensor.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    H, W = image.shape[:2]

    #print(f"\nVisualization Debug:")
    #print(f"Image shape: {image.shape}")
    #print(f"Original size: {orig_size}, Resize size: {resize_size}")

    # Create figure and axis
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # Draw predicted boxes (green)
    if detections is not None:
        pred_boxes = detections['boxes'].cpu().clone()
        scores = detections['scores'].cpu()
        labels = detections['labels'].cpu()

        #print(f"Pred boxes before scaling: {pred_boxes[:5]}")  # Print first 5 boxes
        # Scale normalized coordinates to pixel coordinates
        pred_boxes[:, [0, 2]] *= W  # scale x coordinates
        pred_boxes[:, [1, 3]] *= H  # scale y coordinates
        pred_boxes = pred_boxes.int()
        #print(f"Pred boxes after scaling: {pred_boxes[:5]}")

        for box, score, label in zip(pred_boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f'P:{score:.2f}', color='g', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw ground truth boxes (blue)
    if ground_truths is not None:
        gt_boxes = ground_truths['boxes'].cpu().clone()
        gt_labels = ground_truths['labels'].cpu()

        #print(f"GT boxes before scaling: {gt_boxes[:5]}")
        # First unnormalize using original and resize dimensions
        if orig_size is not None and resize_size is not None:
            gt_boxes = unnormalize_boxes(gt_boxes, orig_size, resize_size)
            #print(f"GT boxes after unnormalize: {gt_boxes[:5]}")
        
        # Then scale to current image dimensions
        scale_x = W / resize_size[1]
        scale_y = H / resize_size[0]
        gt_boxes[:, [0, 2]] *= scale_x
        gt_boxes[:, [1, 3]] *= scale_y
        gt_boxes = gt_boxes.int()
        #print(f"GT boxes after final scaling: {gt_boxes[:5]}")

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='b', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y2+10, f'GT', color='b', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f"{title}\nImage size: {W}x{H}, Orig: {orig_size}, Resize: {resize_size}")
    plt.axis('on')  # Show axes to debug coordinate issues

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    plt.close()

