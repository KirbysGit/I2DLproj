# utils / visualization.py

# -----
# Visualization Tools for Object Detection.
# Displays Predictions, Ground Truth, and Matched Anchors.
# -----

# Imports.
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple
from v1.utils.box_ops import box_iou
from v1.utils.metrics import DetectionMetrics

# Detection Visualizer Class.
class DetectionVisualizer:
    """Visualization Tools for Object Detection."""

    # Visualize Batch of Images.
    @staticmethod
    def visualize_batch(images: torch.Tensor, 
                       predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor] = None,
                       max_images: int = 4,
                       score_threshold: float = 0.5) -> None:
        """
        Visualize Predictions and Ground Truth Boxes for a Batch of Images.
        
        Args:
            images: Batch of images [B, C, H, W]
            predictions: Dict containing 'boxes', 'scores', 'labels'
            targets: Optional dict containing ground truth 'boxes', 'labels'
            max_images: Maximum number of images to display
            score_threshold: Minimum confidence score for predictions
        """

        # Get Batch Size.
        batch_size = min(len(images), max_images)

        # Create Figure.
        fig, axes = plt.subplots(batch_size, 2 if targets else 1, 
                                figsize=(10 * (2 if targets else 1), 5 * batch_size))
        if batch_size == 1:
            axes = np.array([axes])
        
        # Iterate Over Batch.
        for i in range(batch_size):
            # Get Image.
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
            
            # Plot Predictions.
            ax = axes[i, 0] if targets else axes[i]
            ax.imshow(img)
            ax.set_title('Predictions')
            
            # Draw Predicted Boxes.
            pred_boxes = predictions['boxes'][i].cpu().numpy()
            pred_scores = predictions['scores'][i].squeeze(-1).cpu().numpy()
            
            # Iterate Over Predictions.
            for box, score in zip(pred_boxes, pred_scores):
                if score > score_threshold:
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor='r',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(box[0], box[1], f'{score:.2f}', 
                           color='white', fontsize=8,
                           bbox=dict(facecolor='red', alpha=0.5))
            
            # Plot Ground Truth if Available.
            if targets:
                ax = axes[i, 1]
                ax.imshow(img)
                ax.set_title('Ground Truth')
                
                gt_boxes = targets['boxes'][i].cpu().numpy()
                for box in gt_boxes:
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor='g',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
        
        # Tight Layout.
        plt.tight_layout()

        # Show Figure.
        plt.show()
    
    # Visualize Feature Maps.
    @staticmethod
    def visualize_feature_maps(feature_maps: Dict[str, torch.Tensor], 
                             max_features: int = 4) -> None:
        """
        Visualize Feature Maps from Different Layers.
        
        Args:
            feature_maps: Dict of Feature Tensors from Backbone/FPN
            max_features: Maximum number of feature channels to display per layer
        """

        # Get Number of Layers.
        num_layers = len(feature_maps)

        # Create Figure.
        fig, axes = plt.subplots(num_layers, max_features, 
                                figsize=(3 * max_features, 3 * num_layers))

        # Iterate Over Layers.
        for i, (layer_name, features) in enumerate(feature_maps.items()):
            features = features[0].cpu()  # Take First Image in Batch
            
            # Select Features to Display.
            num_channels = min(features.size(0), max_features)
            for j in range(num_channels):
                feature = features[j]
                
                # Normalize Feature Map.
                feature = (feature - feature.min()) / (feature.max() - feature.min())
                
                # Plot Feature.
                axes[i, j].imshow(feature, cmap='viridis')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'{layer_name}')

        # Tight Layout.
        plt.tight_layout()

        # Show Figure.
        plt.show()

    # Visualize Anchors.
    def visualize_anchors(self, image: torch.Tensor, 
                         anchors_per_level: List[torch.Tensor],
                         max_anchors_per_level: int = 100,
                         colors: List[str] = ['r', 'g', 'b', 'y'],
                         base_sizes: List[int] = [32, 64, 128, 256]):
        """Visualize Anchor Boxes from Different FPN Levels."""

        # Create Figure.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show Image with All Anchors.
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax1.imshow(img)
        ax1.set_title('All Anchors')
        
        # Plot Anchors from Each Level.
        for level_id, anchors in enumerate(anchors_per_level):
            # Sample Random Anchors.
            if len(anchors) > max_anchors_per_level:
                indices = torch.randperm(len(anchors))[:max_anchors_per_level]
                anchors = anchors[indices]
            
            # Plot Boxes.
            for box in anchors.cpu().numpy():
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=1,
                    edgecolor=colors[level_id],
                    facecolor='none',
                    alpha=0.5
                )
                ax1.add_patch(rect)
        
        # Show Image w/ Center Points.
        ax2.imshow(img)
        ax2.set_title('Anchor Centers')
        
        # Plot Anchor Centers.
        for level_id, anchors in enumerate(anchors_per_level):
            centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
            centers_y = (anchors[:, 1] + anchors[:, 3]) / 2
            ax2.scatter(centers_x.cpu(), centers_y.cpu(), 
                       c=colors[level_id], alpha=0.5, s=1)
        
        # Add Legend.
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor=color, 
                         label=f'P{i+2} (stride={base_sizes[i]})')
            for i, color in enumerate(colors[:len(anchors_per_level)])
        ]
        ax1.legend(handles=legend_elements)
        ax2.legend(handles=legend_elements)
        
        plt.show()

    # Visualize Matched Anchors.
    def visualize_matched_anchors(self, image: torch.Tensor,
                                anchors: List[torch.Tensor],
                                target_boxes: torch.Tensor,
                                matched_labels: torch.Tensor,
                                matched_boxes: torch.Tensor,
                                ious: torch.Tensor):
        """Enhanced Visualization with Quality Metrics and Better Layout."""

        # Create Figure with GridSpec for Flexible Layout.
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Main Image w/ Ground Truth (Larger).
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_title('Ground Truth Boxes', fontsize=12, pad=10)
        
        # Show Image.
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax1.imshow(img)
        
        # Plot Ground Truth Boxes.
        for box in target_boxes.cpu().numpy():
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='lime',
                facecolor='none',
                alpha=0.8
            )
            ax1.add_patch(rect)
        ax1.axis('off')
        
        # Matched Anchors Visualization.
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.set_title('Matched Anchors (with IoU scores)', fontsize=12, pad=10)
        ax2.imshow(img)
        
        # Plot Positive Matches w/ IoU Scores.
        all_anchors = torch.cat(anchors, dim=0).cpu().numpy()
        matched_labels = matched_labels.cpu().numpy()
        max_ious, _ = ious.max(dim=1)
        max_ious = max_ious.cpu().numpy()
        
        positive_indices = matched_labels == 1
        positive_anchors = all_anchors[positive_indices]
        positive_ious = max_ious[positive_indices]
        
        # Color Map Based on IoU Scores.
        colors = plt.cm.RdYlGn(positive_ious)  # Red to Green colormap
        
        # Plot Positive Matches w/ IoU Scores.
        for box, iou, color in zip(positive_anchors, positive_ious, colors):
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            ax2.add_patch(rect)
            ax2.text(box[0], box[1], f'{iou:.2f}', 
                    color='white', fontsize=8,
                    bbox=dict(facecolor=color, alpha=0.7))
        ax2.axis('off')
        
        # IoU Distribution.
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('IoU Distribution', fontsize=12, pad=10)
        ax3.hist(max_ious, bins=50, color='skyblue', edgecolor='black')
        ax3.set_xlabel('IoU')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # Metrics Summary.
        ax4 = fig.add_subplot(gs[1, 2])
        metrics = DetectionMetrics.compute_matching_quality(matched_labels, ious)
        ax4.set_title('Detection Metrics', fontsize=12, pad=10)
        
        # Format Metric Names.
        metric_names = {
            'mean_iou': 'Mean IoU',
            'num_positive': 'Positive Anchors',
            'positive_ratio': 'Positive Ratio',
            'max_iou': 'Max IoU',
            'min_iou': 'Min IoU'
        }
        
        # Plot Metrics.
        y_pos = np.arange(len(metrics))
        ax4.barh(y_pos, list(metrics.values()), color='skyblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([metric_names[k] for k in metrics.keys()])
        
        # Add Value Labels on Bars.
        for i, v in enumerate(metrics.values()):
            ax4.text(v, i, f'{v:.3f}', va='center')
        
        # Tight Layout.
        plt.tight_layout()

        # Show Figure.
        plt.show()

    def visualize_predictions(self, image, pred_boxes, pred_scores, gt_boxes, output_path, title=""):
        """Visualize predictions using the proven visualization function."""
        from v1.evaluate import visualize_detections
        
        # Get dimensions from the transformed image
        _, H, W = image.shape
        resize_size = (H, W)
        
        # Call the proven visualization function
        visualize_detections(
            image=image,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            output_path=output_path,
            resize_size=resize_size
        )

def visualize_anchors_and_gt(image, gt_boxes, anchors_by_level, output_path, max_anchors_per_level=100):
    """
    Visualize anchors and ground truth boxes on an image.
    
    Args:
        image (torch.Tensor): Image tensor [C, H, W]
        gt_boxes (torch.Tensor): Ground truth boxes in [x1, y1, x2, y2] format, normalized
        anchors_by_level (list): List of anchor tensors for each FPN level
        output_path (str): Path to save visualization
        max_anchors_per_level (int): Maximum number of anchors to plot per level for clarity
    """
    # Convert image for plotting
    img = image.cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5  # Denormalize
    img = np.clip(img, 0, 1)
    
    H, W = img.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original image with GT boxes
    ax[0].imshow(img)
    ax[0].set_title("Ground Truth Boxes")
    
    # Plot GT boxes in red
    for box in gt_boxes:
        x1, y1, x2, y2 = box.cpu().numpy() * [W, H, W, H]  # Denormalize
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
    
    # Plot image with anchors
    ax[1].imshow(img)
    ax[1].set_title("Anchors by Level")
    
    # Colors for different FPN levels
    colors = ['b', 'g', 'c', 'm', 'y']
    
    # Plot anchors for each level
    for level_idx, anchors in enumerate(anchors_by_level):
        # Subsample anchors if too many
        if len(anchors) > max_anchors_per_level:
            indices = torch.randperm(len(anchors))[:max_anchors_per_level]
            anchors = anchors[indices]
        
        # Plot anchors
        for anchor in anchors:
            x1, y1, x2, y2 = anchor.cpu().numpy() * [W, H, W, H]  # Denormalize
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height,
                                  linewidth=1, edgecolor=colors[level_idx % len(colors)],
                                  facecolor='none', alpha=0.5)
            ax[1].add_patch(rect)
    
    # Add legend
    legend_elements = [patches.Patch(facecolor='none', edgecolor='r', label='Ground Truth')]
    for i in range(len(anchors_by_level)):
        legend_elements.append(
            patches.Patch(facecolor='none', edgecolor=colors[i % len(colors)],
                         label=f'Level {i} Anchors')
        )
    ax[1].legend(handles=legend_elements, loc='upper right')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_anchor_coverage(gt_boxes, anchors_by_level):
    """
    Analyze how well anchors cover ground truth boxes.
    
    Args:
        gt_boxes (torch.Tensor): Ground truth boxes
        anchors_by_level (list): List of anchor tensors for each FPN level
    
    Returns:
        dict: Statistics about anchor coverage
    """
    stats = {
        'gt_stats': {},
        'anchor_stats': {},
        'matching_stats': {}
    }
    
    # Ground truth statistics
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_areas = gt_widths * gt_heights
    gt_aspects = gt_widths / gt_heights
    
    stats['gt_stats'] = {
        'min_size': float(min(gt_widths.min(), gt_heights.min())),
        'max_size': float(max(gt_widths.max(), gt_heights.max())),
        'mean_aspect': float(gt_aspects.mean()),
        'area_range': [float(gt_areas.min()), float(gt_areas.max())]
    }
    
    # Analyze each FPN level
    for level_idx, anchors in enumerate(anchors_by_level):
        # Anchor statistics
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_areas = anchor_widths * anchor_heights
        anchor_aspects = anchor_widths / anchor_heights
        
        stats['anchor_stats'][f'level_{level_idx}'] = {
            'min_size': float(min(anchor_widths.min(), anchor_heights.min())),
            'max_size': float(max(anchor_widths.max(), anchor_heights.max())),
            'mean_aspect': float(anchor_aspects.mean()),
            'area_range': [float(anchor_areas.min()), float(anchor_areas.max())]
        }
        
        # Calculate IoU between anchors and GT boxes
        ious = box_iou(anchors, gt_boxes)
        max_ious, _ = ious.max(dim=1)
        
        # Matching statistics
        stats['matching_stats'][f'level_{level_idx}'] = {
            'mean_iou': float(max_ious.mean()),
            'max_iou': float(max_ious.max()),
            'matched_anchors': int((max_ious > 0.5).sum()),
            'total_anchors': len(anchors)
        }
    
    return stats

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1, boxes2: Tensors of shape (N, 4) and (M, 4)
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-6) 