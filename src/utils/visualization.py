import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple

class DetectionVisualizer:
    """Visualization tools for object detection."""
    
    @staticmethod
    def visualize_batch(images: torch.Tensor, 
                       predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor] = None,
                       max_images: int = 4,
                       score_threshold: float = 0.5) -> None:
        """
        Visualize predictions and ground truth boxes for a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
            predictions: Dict containing 'boxes', 'scores', 'labels'
            targets: Optional dict containing ground truth 'boxes', 'labels'
            max_images: Maximum number of images to display
            score_threshold: Minimum confidence score for predictions
        """
        batch_size = min(len(images), max_images)
        fig, axes = plt.subplots(batch_size, 2 if targets else 1, 
                                figsize=(10 * (2 if targets else 1), 5 * batch_size))
        if batch_size == 1:
            axes = np.array([axes])
        
        for i in range(batch_size):
            # Get image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
            
            # Plot predictions
            ax = axes[i, 0] if targets else axes[i]
            ax.imshow(img)
            ax.set_title('Predictions')
            
            # Draw predicted boxes
            pred_boxes = predictions['boxes'][i].cpu().numpy()
            pred_scores = predictions['scores'][i].squeeze(-1).cpu().numpy()
            
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
            
            # Plot ground truth if available
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
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_feature_maps(feature_maps: Dict[str, torch.Tensor], 
                             max_features: int = 4) -> None:
        """
        Visualize feature maps from different layers.
        
        Args:
            feature_maps: Dict of feature tensors from backbone/FPN
            max_features: Maximum number of feature channels to display per layer
        """
        num_layers = len(feature_maps)
        fig, axes = plt.subplots(num_layers, max_features, 
                                figsize=(3 * max_features, 3 * num_layers))
        
        for i, (layer_name, features) in enumerate(feature_maps.items()):
            features = features[0].cpu()  # Take first image in batch
            
            # Select features to display
            num_channels = min(features.size(0), max_features)
            for j in range(num_channels):
                feature = features[j]
                
                # Normalize feature map
                feature = (feature - feature.min()) / (feature.max() - feature.min())
                
                axes[i, j].imshow(feature, cmap='viridis')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'{layer_name}')
        
        plt.tight_layout()
        plt.show()

    def visualize_anchors(self, image: torch.Tensor, 
                         anchors_per_level: List[torch.Tensor],
                         max_anchors_per_level: int = 100,
                         colors: List[str] = ['r', 'g', 'b', 'y'],
                         base_sizes: List[int] = [32, 64, 128, 256]):
        """Visualize anchor boxes from different FPN levels."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show image with all anchors
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax1.imshow(img)
        ax1.set_title('All Anchors')
        
        # Plot anchors from each level
        for level_id, anchors in enumerate(anchors_per_level):
            # Sample random anchors
            if len(anchors) > max_anchors_per_level:
                indices = torch.randperm(len(anchors))[:max_anchors_per_level]
                anchors = anchors[indices]
            
            # Plot boxes
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
        
        # Show image with center points
        ax2.imshow(img)
        ax2.set_title('Anchor Centers')
        
        # Plot anchor centers
        for level_id, anchors in enumerate(anchors_per_level):
            centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
            centers_y = (anchors[:, 1] + anchors[:, 3]) / 2
            ax2.scatter(centers_x.cpu(), centers_y.cpu(), 
                       c=colors[level_id], alpha=0.5, s=1)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor=color, 
                         label=f'P{i+2} (stride={base_sizes[i]})')
            for i, color in enumerate(colors[:len(anchors_per_level)])
        ]
        ax1.legend(handles=legend_elements)
        ax2.legend(handles=legend_elements)
        
        plt.show()

    def visualize_matched_anchors(self, image: torch.Tensor,
                                anchors: List[torch.Tensor],
                                target_boxes: torch.Tensor,
                                matched_labels: torch.Tensor,
                                matched_boxes: torch.Tensor):
        """Visualize anchor matching results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show image
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        # Plot all anchors and ground truth
        ax1.imshow(img)
        ax1.set_title('All Anchors and Ground Truth')
        
        # Plot ground truth boxes in green
        for box in target_boxes.cpu().numpy():
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='g',
                facecolor='none'
            )
            ax1.add_patch(rect)
        
        # Plot matched anchors
        ax2.imshow(img)
        ax2.set_title('Matched Anchors')
        
        all_anchors = torch.cat(anchors, dim=0).cpu().numpy()
        matched_labels = matched_labels.cpu().numpy()
        
        # Plot positive matches in red
        positive_anchors = all_anchors[matched_labels == 1]
        for box in positive_anchors:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor='r',
                facecolor='none',
                alpha=0.5
            )
            ax2.add_patch(rect)
        
        plt.show() 