# utils / metrics.py

# -----
# Computes Matching Quality Metrics for Anchors.
# -----

# Imports.
import torch
from typing import Dict

# Detection Metrics Class.
class DetectionMetrics:

    # Compute Matching Quality Metrics.
    @staticmethod
    def compute_matching_quality(matched_labels: torch.Tensor, 
                               ious: torch.Tensor) -> Dict[str, float]:
        """
        Compute anchor matching quality metrics.
        
        Args:
            matched_labels:     Binary tensor indicating positive matches.
            ious:               IoU matrix between anchors and targets.
            
        Returns:
            dict: Dictionary of quality metrics.
        """

        # Convert numpy arrays to tensors if needed.
        if not isinstance(matched_labels, torch.Tensor):
            matched_labels = torch.from_numpy(matched_labels)
        if not isinstance(ious, torch.Tensor):
            ious = torch.from_numpy(ious)
            
        # Move to Same Device as ious.
        matched_labels = matched_labels.to(ious.device)
        
        # Create Binary Mask.
        positive_mask = matched_labels == 1
        
        # Handle Empty Case.
        if not positive_mask.any():
            return {
                'mean_iou': 0.0,
                'num_positive': 0,
                'positive_ratio': 0.0,
                'max_iou': 0.0,
                'min_iou': 0.0
            }
        
        # Compute Metrics Using Tensor Operations.
        positive_ious = ious[positive_mask]
        metrics = {
            'mean_iou': positive_ious.mean().item(),
            'num_positive': positive_mask.sum().item(),
            'positive_ratio': (positive_mask.sum().float() / positive_mask.numel()).item(),
            'max_iou': positive_ious.max().item(),
            'min_iou': positive_ious.min().item()
        }
        
        return metrics

    # Compute Matching Statistics.
    @staticmethod
    def compute_matching_statistics(matched_labels, ious):
        """Compute detailed matching statistics."""
        stats = {
            'total_anchors': len(matched_labels),                        # Total Anchors.
            'positive_anchors': (matched_labels > 0).sum().item(),       # Positive Anchors.
            'mean_iou': ious[matched_labels > 0].mean().item(),          # Mean IoU.
            'max_iou': ious.max().item(),                                # Max IoU.
            'min_positive_iou': ious[matched_labels > 0].min().item(),   # Min Positive IoU.
            'std_iou': ious[matched_labels > 0].std().item()             # Std IoU.
        }
        return stats