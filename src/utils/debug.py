import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DetectionDebugger:
    """Helper class for debugging object detection training."""
    
    def __init__(self, save_dir="debug_output"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_boxes(self, boxes, name="boxes"):
        """Analyze box statistics and potential issues."""
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
            
        # Compute basic statistics
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        areas = widths * heights
        aspect_ratios = widths / (heights + 1e-6)
        
        stats = {
            "width": {"min": widths.min(), "max": widths.max(), "mean": widths.mean()},
            "height": {"min": heights.min(), "max": heights.max(), "mean": heights.mean()},
            "area": {"min": areas.min(), "max": areas.max(), "mean": areas.mean()},
            "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max(), "mean": aspect_ratios.mean()}
        }
        
        # Check for potential issues
        issues = []
        if (widths <= 0).any():
            issues.append("Found boxes with zero or negative width")
        if (heights <= 0).any():
            issues.append("Found boxes with zero or negative height")
        if (areas <= 0).any():
            issues.append("Found boxes with zero or negative area")
        if not np.isfinite(boxes).all():
            issues.append("Found boxes with non-finite coordinates")
            
        return stats, issues
    
    def analyze_loss_components(self, loss_dict):
        """Analyze individual loss components."""
        analysis = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            analysis[key] = {
                "value": value,
                "is_zero": abs(value) < 1e-6,
                "is_nan": np.isnan(value),
                "is_inf": np.isinf(value)
            }
        return analysis
    
    def analyze_gradients(self, model):
        """Analyze gradient statistics for model parameters."""
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                grad_stats[name] = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "min": grad.min().item(),
                    "max": grad.max().item(),
                    "has_nan": torch.isnan(grad).any().item(),
                    "has_inf": torch.isinf(grad).any().item()
                }
        return grad_stats
    
    def visualize_batch(self, images, boxes, predictions=None, epoch=0, batch_idx=0):
        """Visualize a batch of images with their boxes and predictions."""
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Show image
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for visualization
            ax.imshow(img)
            
            # Show ground truth boxes in green
            if boxes is not None:
                for box in boxes[i]:
                    x1, y1, x2, y2 = box.detach().cpu().numpy()
                    width = x2 - x1
                    height = y2 - y1
                    ax.add_patch(plt.Rectangle((x1, y1), width, height,
                                             fill=False, color='g', linewidth=2))
            
            # Show predictions in red if available
            if predictions is not None:
                for box in predictions[i]:
                    x1, y1, x2, y2 = box.detach().cpu().numpy()
                    width = x2 - x1
                    height = y2 - y1
                    ax.add_patch(plt.Rectangle((x1, y1), width, height,
                                             fill=False, color='r', linewidth=2))
            
            ax.set_title(f'Epoch {epoch}, Batch {batch_idx}, Image {i}')
            plt.savefig(self.save_dir / f'epoch_{epoch}_batch_{batch_idx}_img_{i}.png')
            plt.close()
    
    def save_stats(self, stats, name, epoch):
        """Save statistics to a file."""
        with open(self.save_dir / f'{name}_epoch_{epoch}.txt', 'w') as f:
            f.write(f"=== {name} Statistics for Epoch {epoch} ===\n\n")
            for key, value in stats.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n") 