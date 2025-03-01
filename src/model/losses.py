import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection as described in RetinaNet paper.
    Focuses training on hard examples by down-weighting easy examples.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
            reduction (str): 'none', 'mean', 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted class scores [B, N, C]
            target (torch.Tensor): Target classes [B, N]
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Ensure target is long type for one_hot
        target = target.long()
        
        # Convert targets to one-hot
        num_classes = pred.size(-1)
        target_one_hot = F.one_hot(target, num_classes).float()
        
        # Compute focal weight
        probs = torch.sigmoid(pred)
        pt = target_one_hot * probs + (1 - target_one_hot) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute weighted cross entropy
        alpha_weight = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
        loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, 
            reduction='none'
        )
        loss = alpha_weight * focal_weight * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class IoULoss(nn.Module):
    """
    IoU Loss for bounding box regression.
    Directly optimizes the IoU metric.
    """
    
    def __init__(self, reduction='mean', eps=1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted boxes [B, N, 4] in (x1, y1, x2, y2) format
            target (torch.Tensor): Target boxes [B, N, 4] in (x1, y1, x2, y2) format
            
        Returns:
            torch.Tensor: Computed IoU loss
        """
        # Compute box areas
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        
        # Compute intersection
        left_top = torch.max(pred[..., :2], target[..., :2])
        right_bottom = torch.min(pred[..., 2:], target[..., 2:])
        wh = (right_bottom - left_top).clamp(min=0)
        intersection = wh[..., 0] * wh[..., 1]
        
        # Compute IoU
        union = pred_area + target_area - intersection
        iou = intersection / (union + self.eps)
        
        # Compute loss
        loss = 1 - iou
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class DetectionLoss(nn.Module):
    """
    Combined loss for object detection:
    - Focal Loss for classification
    - IoU Loss for box regression
    """
    
    def __init__(self, 
                 num_classes=1,
                 cls_loss_weight=1.0,
                 box_loss_weight=1.0):
        super().__init__()
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = IoULoss()
        self.cls_weight = cls_loss_weight
        self.box_weight = box_loss_weight
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions (dict): Dictionary containing:
                - cls_scores (list): List of classification scores
                - bbox_preds (list): List of box predictions
            targets (dict): Dictionary containing:
                - cls_targets (list): List of classification targets
                - box_targets (list): List of box targets
                
        Returns:
            dict: Dictionary containing:
                - loss: Total loss
                - cls_loss: Classification loss
                - box_loss: Box regression loss
        """
        cls_losses = []
        box_losses = []
        
        # Compute loss for each feature level
        for cls_pred, box_pred, cls_target, box_target in zip(
            predictions['cls_scores'],
            predictions['bbox_preds'],
            targets['cls_targets'],
            targets['box_targets']
        ):
            # Reshape predictions
            B, A, C, H, W = cls_pred.shape
            cls_pred = cls_pred.view(B, -1, C)
            box_pred = box_pred.view(B, -1, 4)
            
            # Reshape targets
            cls_target = cls_target.view(B, -1)
            box_target = box_target.view(B, -1, 4)
            
            # Compute losses
            cls_loss = self.cls_loss(cls_pred, cls_target)
            box_loss = self.box_loss(box_pred, box_target)
            
            cls_losses.append(cls_loss)
            box_losses.append(box_loss)
        
        # Combine losses
        cls_loss = torch.stack(cls_losses).mean()
        box_loss = torch.stack(box_losses).mean()
        total_loss = self.cls_weight * cls_loss + self.box_weight * box_loss
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss
        }

class IoUWeightedBoxLoss(nn.Module):
    def forward(self, pred_boxes, target_boxes, ious):
        """IoU-weighted box regression loss."""
        # Basic regression loss
        loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none')
        
        # Weight by IoU
        iou_weights = ious.detach()  # Don't backprop through weights
        weighted_loss = (loss * iou_weights.unsqueeze(-1)).mean()
        
        return weighted_loss 