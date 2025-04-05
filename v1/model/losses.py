# model / losses.py

# -----

# Defines Loss Functions for Object Detection.
# Focal Loss -> Handles Class Imbalance During Classification.
# IoU Loss -> Improves Bounding Box Regression by Optimizing IoU Metric.
# Detection Loss -> Combines Focal Loss & IoU Loss for Object Detection.
# IoU Weighted Box Loss -> Improves Training Stability by Weighting Loss by IoU.

# -----

# Imports.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss Class.
class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection as Described in RetinaNet Paper.
    Focuses Training on Hard Examples by Down-Weighting Easy Examples.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting Factor for Rare Class.
            gamma (float): Focusing Parameter.
            reduction (str): 'none', 'mean', 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted Class Scores [B, N, C].
            target (torch.Tensor): Target Classes [B, N].
            
        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Ensure Inputs Require Gradients.
        pred = pred.requires_grad_(True)
        
        # Ensure Target is Long Type for One-Hot Encoding.
        target = target.long()
        
        # Convert Targets to One-Hot Encoding.
        num_classes = pred.size(-1)
        target_one_hot = F.one_hot(target, num_classes).float()
        
        # Compute Focal Weight.
        probs = torch.sigmoid(pred)
        pt = target_one_hot * probs + (1 - target_one_hot) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute Weighted Cross Entropy.
        alpha_weight = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
        loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, 
            reduction='none'
        )
        loss = alpha_weight * focal_weight * loss
        
        # Apply Reduction.
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# IoU Loss Class.
class IoULoss(nn.Module):
    """
    IoU Loss for Bounding Box Regression.
    Directly Optimizes the IoU Metric.
    """
    
    def __init__(self, reduction='mean', eps=1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted Boxes [B, N, 4] in (x1, y1, x2, y2) Format.
            target (torch.Tensor): Target Boxes [B, N, 4] in (x1, y1, x2, y2) Format.
            
        Returns:
            torch.Tensor: Computed IoU Loss.
        """
        # Ensure inputs require gradients.
        pred = pred.requires_grad_(True)
        
        # Compute Box Areas.
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        
        # Compute Intersection.
        left_top = torch.max(pred[..., :2], target[..., :2])
        right_bottom = torch.min(pred[..., 2:], target[..., 2:])
        wh = (right_bottom - left_top).clamp(min=0)
        intersection = wh[..., 0] * wh[..., 1]
        
        # Compute IoU.
        union = pred_area + target_area - intersection
        iou = intersection / (union + self.eps)
        
        # Compute Loss.
        loss = 1 - iou
        
        # Apply Reduction.
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# Detection Loss Class. 
class DetectionLoss(nn.Module):
    """
    Combined Loss for Object Detection:
    - Focal Loss for Classification
    - IoU Loss for Box Regression
    """
    
    def __init__(self, 
                 num_classes=1,
                 cls_loss_weight=1.0,
                 box_loss_weight=1.0,
                 l2_reg_weight=0.0001,
                 box_loss_clip=10.0):
        super().__init__()
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = IoULoss()
        self.cls_weight = cls_loss_weight
        self.box_weight = box_loss_weight
        self.l2_reg_weight = l2_reg_weight
        self.box_loss_clip = box_loss_clip
    
    def forward(self, predictions, targets):
        """Forward pass with gradient clipping and normalized coordinates."""
        # Initialize Losses.
        cls_losses = []
        box_losses = []
        
        # Compute Loss for Each Feature Level.
        for cls_pred, box_pred, cls_target, box_target in zip(
            predictions['cls_scores'],
            predictions['bbox_preds'],
            targets['cls_targets'],
            targets['box_targets']
        ):
            # Reshape Predictions.
            B, A, C, H, W = cls_pred.shape
            cls_pred = cls_pred.view(B, -1, C)
            box_pred = box_pred.view(B, -1, 4)
            
            # Reshape Targets.
            cls_target = cls_target.view(B, -1)
            box_target = box_target.view(B, -1, 4)
            
            # Normalize box coordinates to [0, 1] range
            box_pred_norm = box_pred.clone()
            box_target_norm = box_target.clone()
            
            # Compute Losses.
            cls_loss = self.cls_loss(cls_pred, cls_target)
            box_loss = self.box_loss(box_pred_norm, box_target_norm)
            
            # Clip Box Loss Gradient.
            if self.box_loss_clip > 0:
                box_loss = torch.clamp(box_loss, max=self.box_loss_clip)
            
            cls_losses.append(cls_loss)
            box_losses.append(box_loss)
        
        # Combine Losses.
        cls_loss = torch.stack(cls_losses).mean()
        box_loss = torch.stack(box_losses).mean()
        
        # Add L2 Regularization.
        l2_reg_loss = 0
        for param in predictions.get('model_params', []):
            l2_reg_loss += torch.norm(param) ** 2
        l2_reg_loss *= self.l2_reg_weight
        
        # Compute Total Loss.
        total_loss = (
            self.cls_weight * cls_loss + 
            self.box_weight * box_loss + 
            l2_reg_loss
        )
        
        # Return Losses.
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'reg_loss': l2_reg_loss
        }

# IoU Weighted Box Loss Class.
class IoUWeightedBoxLoss(nn.Module):
    def forward(self, pred_boxes, target_boxes, ious):
        """IoU-weighted box regression loss."""
        # Ensure inputs require gradients
        pred_boxes = pred_boxes.requires_grad_(True)
        
        # Basic Regression Loss.
        loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none')
        
        # Weight by IoU.
        iou_weights = ious.detach()
        weighted_loss = (loss * iou_weights.unsqueeze(-1)).mean()
        
        # Return Weighted Loss.
        return weighted_loss
