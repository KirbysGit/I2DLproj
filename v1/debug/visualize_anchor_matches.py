import torch
import matplotlib.pyplot as plt
import numpy as np
from v1.data.dataset import SKU110KDataset
from v1.model.detector import ObjectDetector
from v1.utils.box_ops import box_iou
import seaborn as sns
from torchvision.ops import nms
import os

def denormalize_image(image):
    """Convert normalized tensor to numpy array for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    return image

def plot_boxes(ax, boxes, color='g', alpha=0.5, linewidth=2):
    """Plot boxes on axes."""
    for box in boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, 
                           fill=False, color=color, 
                           alpha=alpha, linewidth=linewidth)
        ax.add_patch(rect)

def visualize_matches(image, gt_boxes, anchors, matched_labels, iou_matrix, output_dir, 
                     max_anchors_to_plot=1000):
    """Visualize anchor matches and statistics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup the figure
    plt.figure(figsize=(20, 6))
    
    # 1. Original Image with GT Boxes
    ax1 = plt.subplot(131)
    image_np = denormalize_image(image)
    ax1.imshow(image_np)
    plot_boxes(ax1, gt_boxes, color='g', linewidth=2)
    ax1.set_title('Ground Truth Boxes')
    ax1.axis('off')
    
    # 2. Image with Matched Positive Anchors
    ax2 = plt.subplot(132)
    ax2.imshow(image_np)
    
    # Get positive matches
    pos_mask = matched_labels > 0
    pos_anchors = anchors[pos_mask]
    
    # Get IoUs for coloring
    max_ious, _ = iou_matrix.max(dim=1)
    pos_ious = max_ious[pos_mask]
    
    # Subsample anchors if too many
    if len(pos_anchors) > max_anchors_to_plot:
        indices = torch.randperm(len(pos_anchors))[:max_anchors_to_plot]
        pos_anchors = pos_anchors[indices]
        pos_ious = pos_ious[indices]
    
    # Plot anchors colored by IoU
    for anchor, iou in zip(pos_anchors, pos_ious):
        color = plt.cm.viridis(iou.item())
        plot_boxes(ax2, [anchor], color=color, alpha=0.3)
    
    ax2.set_title(f'Positive Anchors (colored by IoU)\nShowing {len(pos_anchors)} of {pos_mask.sum().item()} matches')
    ax2.axis('off')
    
    # 3. Statistics
    ax3 = plt.subplot(133)
    
    # IoU histogram
    sns.histplot(max_ious[pos_mask].cpu().numpy(), bins=50, ax=ax3)
    ax3.set_title('IoU Distribution for Positive Matches')
    ax3.set_xlabel('IoU')
    ax3.set_ylabel('Count')
    
    # Add statistics as text
    stats_text = f"""
    Matching Statistics:
    -------------------
    Total GT Boxes: {len(gt_boxes)}
    Total Anchors: {len(anchors)}
    Positive Matches: {pos_mask.sum().item()}
    Mean IoU: {max_ious[pos_mask].mean():.3f}
    Max IoU: {max_ious[pos_mask].max():.3f}
    Min IoU: {max_ious[pos_mask].min():.3f}
    Matches per GT: {pos_mask.sum().item() / len(gt_boxes):.1f}
    """
    
    plt.figtext(0.98, 0.98, stats_text,
                fontsize=10, family='monospace',
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                horizontalalignment='right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anchor_matches.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_anchor_matches(model, dataset, device='cuda', num_images=5, output_dir='anchor_analysis'):
    """Analyze anchor matching for multiple images."""
    model.eval()
    
    for idx in range(min(num_images, len(dataset))):
        print(f"\nAnalyzing image {idx + 1}/{num_images}")
        
        # Get image and targets
        image, target = dataset[idx]
        image = image.unsqueeze(0).to(device)  # [1, C, H, W]
        gt_boxes = target['boxes'].to(device)
        gt_labels = target['labels'].to(device)
        
        # Run backbone and FPN
        with torch.no_grad():
            backbone_features = model.backbone(image)
            fpn_input_features = {
                k: v for k, v in backbone_features.items() 
                if k in ['layer1', 'layer2', 'layer3', 'layer4']
            }
            fpn_features = model.fpn(fpn_input_features)
        
        # Generate anchors
        anchors = []
        for level_idx, (level_name, feature) in enumerate(fpn_features.items()):
            H, W = feature.shape[2:]
            level_anchors = model.detection_head.anchor_generator.generate_anchors_for_level(
                feature_map_size=(H, W),
                stride=model.detection_head.strides[level_idx],
                device=device
            )
            anchors.append(level_anchors)
        
        # Combine all anchors
        all_anchors = torch.cat(anchors, dim=0)
        
        # Match anchors to targets
        matched_labels, matched_boxes = model.detection_head.match_anchors_to_targets(
            all_anchors,
            gt_boxes,
            gt_labels
        )
        
        # Compute IoU matrix for visualization
        iou_matrix = box_iou(all_anchors, gt_boxes)
        
        # Create output directory for this image
        image_output_dir = os.path.join(output_dir, f'image_{idx}')
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Visualize matches
        visualize_matches(
            image=image[0],  # Remove batch dimension
            gt_boxes=gt_boxes,
            anchors=all_anchors,
            matched_labels=matched_labels,
            iou_matrix=iou_matrix,
            output_dir=image_output_dir
        )
        
        # Save matching statistics to file
        pos_mask = matched_labels > 0
        max_ious, _ = iou_matrix.max(dim=1)
        
        stats = {
            'num_gt_boxes': len(gt_boxes),
            'num_anchors': len(all_anchors),
            'num_positive_matches': pos_mask.sum().item(),
            'mean_iou': max_ious[pos_mask].mean().item(),
            'max_iou': max_ious[pos_mask].max().item(),
            'min_iou': max_ious[pos_mask].min().item(),
            'matches_per_gt': pos_mask.sum().item() / len(gt_boxes),
            'anchor_stats_by_level': [
                {
                    'level': i,
                    'num_anchors': len(anch),
                    'anchor_sizes': {
                        'min': (anch[:, 2:] - anch[:, :2]).min().item(),
                        'max': (anch[:, 2:] - anch[:, :2]).max().item(),
                        'mean': (anch[:, 2:] - anch[:, :2]).mean().item()
                    }
                }
                for i, anch in enumerate(anchors)
            ]
        }
        
        # Save statistics
        import json
        with open(os.path.join(image_output_dir, 'matching_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"Analysis complete for image {idx + 1}. Results saved to {image_output_dir}")

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset with correct parameters matching train.py
    dataset = SKU110KDataset(
        data_dir='datasets/SKU-110K',
        split='val',
        transform=None,
        resize_dims=(512, 512)  # Using standard image size, adjust if needed
    )
    
    # Load model
    model = ObjectDetector(
        pretrained_backbone=True,
        fpn_out_channels=256,
        num_classes=1,
        num_anchors=3,
        debug=True
    ).to(device)
    
    # Load checkpoint if available
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        try:
            # First try loading with weights_only=True
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            print("Successfully loaded checkpoint")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {str(e)}")
            print("Proceeding with initialized weights")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Proceeding with initialized weights")
    
    # Run analysis
    analyze_anchor_matches(
        model=model,
        dataset=dataset,
        device=device,
        num_images=5,
        output_dir='debug/anchor_analysis'
    )
