import torch
from typing import Dict

class DetectionMetrics:
    @staticmethod
    def compute_matching_quality(matched_labels: torch.Tensor, 
                               ious: torch.Tensor) -> Dict[str, float]:
        """
        Compute anchor matching quality metrics.
        
        Args:
            matched_labels: Binary tensor indicating positive matches
            ious: IoU matrix between anchors and targets
            
        Returns:
            dict: Dictionary of quality metrics
        """
        # Convert numpy arrays to tensors if needed
        if not isinstance(matched_labels, torch.Tensor):
            matched_labels = torch.from_numpy(matched_labels)
        if not isinstance(ious, torch.Tensor):
            ious = torch.from_numpy(ious)
            
        # Move to same device as ious
        matched_labels = matched_labels.to(ious.device)
        
        # Create binary mask
        positive_mask = matched_labels == 1
        
        # Handle empty case
        if not positive_mask.any():
            return {
                'mean_iou': 0.0,
                'num_positive': 0,
                'positive_ratio': 0.0,
                'max_iou': 0.0,
                'min_iou': 0.0
            }
        
        # Compute metrics using tensor operations
        positive_ious = ious[positive_mask]
        metrics = {
            'mean_iou': positive_ious.mean().item(),
            'num_positive': positive_mask.sum().item(),
            'positive_ratio': (positive_mask.sum().float() / positive_mask.numel()).item(),
            'max_iou': positive_ious.max().item(),
            'min_iou': positive_ious.min().item()
        }
        
        return metrics

    @staticmethod
    def compute_matching_statistics(matched_labels, ious):
        """Compute detailed matching statistics."""
        stats = {
            'total_anchors': len(matched_labels),
            'positive_anchors': (matched_labels > 0).sum().item(),
            'mean_iou': ious[matched_labels > 0].mean().item(),
            'max_iou': ious.max().item(),
            'min_positive_iou': ious[matched_labels > 0].min().item(),
            'std_iou': ious[matched_labels > 0].std().item()
        }
        return stats