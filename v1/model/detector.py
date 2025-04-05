# model / detector.py

# -----

# Defines Complete Object Detection Model.
# Combines Backbone, Feature Pyramid Network, & Detection Head.

# ----- 

# Imports.
import torch
import torch.nn as nn
from torchvision.ops import nms
from .backbone import ResNetBackbone
from .fpn import FeaturePyramidNetwork
from .detection_head import DetectionHead

# Object Detector Class.
class ObjectDetector(nn.Module):
    """Complete object detector with backbone, FPN, and detection head."""
    
    # Initialize Object Detector.
    def __init__(self, 
                 pretrained_backbone=True,
                 fpn_out_channels=256,
                 num_classes=1,
                 num_anchors=3,
                 debug=False):
        super().__init__()
        
        # Backbone.
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        
        # FPN
        in_channels_list = [
            self.backbone.out_channels['layer1'],  # 256
            self.backbone.out_channels['layer2'],  # 512
            self.backbone.out_channels['layer3'],  # 1024
            self.backbone.out_channels['layer4'],  # 2048
        ]
        self.fpn = FeaturePyramidNetwork(in_channels_list, fpn_out_channels)
        
        # Detection Head.
        self.detection_head = DetectionHead(
            in_channels=fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            debug=debug
        )
        
        # Set Thresholds Values - More permissive for dense objects
        self.confidence_thresholds = [0.1, 0.1, 0.1, 0.1]   
        
        # Adjust Size Ranges for SKU-110K - objects are relatively small and dense
        self.min_sizes = [0.005, 0.01, 0.02, 0.04]  # Smaller minimum sizes
        self.max_sizes = [0.03, 0.06, 0.12, 0.24]   # Larger maximum sizes
        
        # More permissive NMS for dense objects
        self.nms_thresholds = [0.3, 0.3, 0.3, 0.3]  # Lower NMS threshold to keep more boxes
        
        # Box Regression Parameters - Adjusted for better localization
        self.box_scale = [2, 4, 8, 16]  # Reduced scales for finer control
        self.box_delta_means = [0.0, 0.0, 0.0, 0.0]
        self.box_delta_stds = [0.1, 0.1, 0.1, 0.1]  # Reduced std for more precise localization
        
        # Size Scaling Factors - Adjusted for SKU-110K objects
        self.width_scales = [0.5, 0.75, 1.0, 1.25]   # More variation in width
        self.height_scales = [0.5, 0.75, 1.0, 1.25]  # More variation in height
    
    # Load Model from Checkpoint.
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cpu'):
        """Load model from checkpoint."""
        # Create Model Instance.
        model = cls()
        
        # Load Checkpoint.
        try:
            # Try Loading with Weights Only First.
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Warning: Could not load with weights_only=True, attempting legacy loading: {str(e)}")
            try:
                # Try Loading Full Checkpoint.
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                raise
        
        # Load State Dict.
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Error loading state dict: {str(e)}")
            # Try Loading with Strict=False.
            model.load_state_dict(state_dict, strict=False)
            print("Loaded checkpoint with some missing or unexpected keys")
        
        # Move Model to Device.
        model = model.to(device)
        
        # Set Model to Evaluation Mode.
        model.eval()
        
        # Return Model.
        return model

    # Forward Pass.
    def forward(self, images, boxes=None, labels=None):
        """Forward Pass Through the Complete Detector."""
        # Get Backbone Features.
        backbone_features = self.backbone(images)

        img_h, img_w = images.shape[2], images.shape[3]
        image_size = torch.tensor([img_w, img_h, img_w, img_h], device=images.device).float()

        
        # Get Features for FPN (C2 to C5).
        fpn_input_features = {
            k: v for k, v in backbone_features.items() 
            if k in ['layer1', 'layer2', 'layer3', 'layer4']
        }
        
        # Generate FPN Features (P2 to P5).
        fpn_features = self.fpn(fpn_input_features)
        
        # Get Predictions and Anchors from Detection Head.
        head_outputs = self.detection_head(fpn_features, image_size=image_size)
        
        # Initialize Lists to Store Predictions from All Levels.
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Process Each FPN Level.
        for level_idx, (level_scores, level_boxes, level_anchors) in enumerate(zip(
            head_outputs['cls_scores'], 
            head_outputs['bbox_preds'],
            head_outputs['anchors']
        )):
            #print(f"\nLevel {level_idx} Shapes:")
            #print(f"Scores: {level_scores.shape}")
            #print(f"Boxes: {level_boxes.shape}")
            #print(f"Anchors: {level_anchors.shape}")
            
            # Move Tensors to CPU for Processing.
            level_scores = level_scores.cpu()
            level_boxes = level_boxes.cpu()
            level_anchors = level_anchors.cpu()
            
            B, num_anchors, num_classes, H, W = level_scores.shape
            
            # Reshape Predictions.
            level_scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)
            level_boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            
            # Apply Sigmoid to Get Confidence Scores.
            level_scores = torch.sigmoid(level_scores)
            
            # Convert Box Deltas to Actual Boxes.
            anchor_widths = level_anchors[:, 2] - level_anchors[:, 0]
            anchor_heights = level_anchors[:, 3] - level_anchors[:, 1]
            anchor_ctr_x = level_anchors[:, 0] + 0.5 * anchor_widths
            anchor_ctr_y = level_anchors[:, 1] + 0.5 * anchor_heights
            
            # Get Image Size for Normalization.
            image_size = float(images.shape[-1])

            # Normalize Anchor Coordinates to [0, 1].
            anchor_widths = anchor_widths / image_size
            anchor_heights = anchor_heights / image_size
            anchor_ctr_x = anchor_ctr_x / image_size
            anchor_ctr_y = anchor_ctr_y / image_size

            # Scale Deltas Based on Feature Level
            scale = self.box_scale[level_idx]
            feature_scale = 1.0 / scale

            # Center Deltas are Relative to Feature Scale and Anchor Size
            dx = level_boxes[..., 0] * self.box_delta_stds[0] + self.box_delta_means[0]
            dy = level_boxes[..., 1] * self.box_delta_stds[1] + self.box_delta_means[1]
            dw = level_boxes[..., 2] * self.box_delta_stds[2] + self.box_delta_means[2]
            dh = level_boxes[..., 3] * self.box_delta_stds[3] + self.box_delta_means[3]

            # Stricter clamping for center shifts
            dx = dx.clamp(-0.5, 0.5) * feature_scale  # Reduced range
            dy = dy.clamp(-0.5, 0.5) * feature_scale  # Reduced range

            # Calculate Predicted Centers with Anchor-Relative Constraints
            pred_ctr_x = anchor_ctr_x.unsqueeze(0) + dx * torch.min(anchor_widths, anchor_heights).unsqueeze(0)
            pred_ctr_y = anchor_ctr_y.unsqueeze(0) + dy * torch.min(anchor_widths, anchor_heights).unsqueeze(0)

            # Enforce Center Point Distribution
            # Add gradient-friendly center point constraints
            center_weight = torch.exp(-((pred_ctr_y - 0.5).pow(2) / 0.08))  # Bias towards vertical center
            pred_ctr_y = pred_ctr_y * center_weight + anchor_ctr_y.unsqueeze(0) * (1 - center_weight)

            # Clamp Centers with Dynamic Margins
            margin_x = 0.1 * (1.0 - torch.abs(pred_ctr_x - 0.5))  # Tighter margin near edges
            margin_y = 0.1 * (1.0 - torch.abs(pred_ctr_y - 0.5))  # Tighter margin near edges
            pred_ctr_x = pred_ctr_x.clamp(margin_x, 1.0 - margin_x)
            pred_ctr_y = pred_ctr_y.clamp(margin_y, 1.0 - margin_y)

            # More Conservative Size Deltas
            dw = torch.clamp(dw, -0.5, 0.5)  # Reduced range
            dh = torch.clamp(dh, -0.5, 0.5)  # Reduced range

            # Calculate Base Sizes from Anchors with Size Constraints
            base_w = anchor_widths.unsqueeze(0)
            base_h = anchor_heights.unsqueeze(0)

            # Apply Size Deltas with Conservative Scaling
            width_scale = height_scale = 0.1  # Reduced scale factor

            # Calculate Predicted Widths and Heights with Aspect Ratio Preservation
            pred_w = base_w * torch.exp(dw * width_scale)
            pred_h = base_h * torch.exp(dh * height_scale)

            # Enforce Minimum and Maximum Size Constraints
            min_size = torch.min(base_w, base_h) * 0.5  # Half the anchor size
            max_size = torch.max(base_w, base_h) * 1.5  # 1.5x the anchor size
            
            pred_w = torch.clamp(pred_w, min_size, max_size)
            pred_h = torch.clamp(pred_h, min_size, max_size)

            # Convert to [x1, y1, x2, y2] Format with Careful Handling of Small Boxes.
            pred_boxes = torch.stack([
                (pred_ctr_x - 0.5 * pred_w).clamp(0.0, 1.0),
                (pred_ctr_y - 0.5 * pred_h).clamp(0.0, 1.0),
                (pred_ctr_x + 0.5 * pred_w).clamp(0.0, 1.0),
                (pred_ctr_y + 0.5 * pred_h).clamp(0.0, 1.0)
            ], dim=-1)
            
            #print(f"\nBox regression statistics for level {level_idx}:")
            #print(f"Center ranges: x=[{pred_ctr_x.min():.4f}, {pred_ctr_x.max():.4f}], y=[{pred_ctr_y.min():.4f}, {pred_ctr_y.max():.4f}]")
            #print(f"Size ranges: w=[{pred_w.min():.4f}, {pred_w.max():.4f}], h=[{pred_h.min():.4f}, {pred_h.max():.4f}]")

            # 🔒 Clamp predicted boxes to valid pixel range
            image_h, image_w = images.shape[-2:]  # Get height and width of image
            pred_boxes[..., [0, 2]] = pred_boxes[..., [0, 2]].clamp(0, image_w)
            pred_boxes[..., [1, 3]] = pred_boxes[..., [1, 3]].clamp(0, image_h)

            #print(f"Delta stats - dx: {dx.mean():.4f}, dw: {dw.mean():.4f}")
            #print(f"Pred box range x1: [{pred_boxes[...,0].min()}, {pred_boxes[...,0].max()}]")

            # Get Confidence Scores and Filter by Level-Specific Threshold.
            max_scores, max_classes = level_scores.max(dim=-1)
            confidence_mask = max_scores > self.confidence_thresholds[level_idx]
            
            #print(f"\nChunk statistics:")
            #print(f"Max score: {max_scores.max().item():.4f}")
            #print(f"Mean score: {max_scores.mean().item():.4f}")
            #print(f"Predictions above threshold: {confidence_mask.sum().item()}")
            
            # Initialize Lists for This Level.
            level_filtered_boxes = []
            level_filtered_scores = []
            level_filtered_labels = []
            
            if confidence_mask.any():
                # Process Each Batch Item.
                for b in range(B):
                    b_mask = confidence_mask[b]
                    if not b_mask.any():
                        continue
                        
                    b_boxes = pred_boxes[b][b_mask]
                    b_scores = max_scores[b][b_mask]
                    b_labels = max_classes[b][b_mask]
                    
                    # Ensure boxes and scores have the same dtype
                    b_boxes = b_boxes.to(dtype=torch.float32)
                    b_scores = b_scores.to(dtype=torch.float32)
                    
                    # Filter by Box Size with Level-Specific Thresholds.
                    widths = b_boxes[:, 2] - b_boxes[:, 0]
                    heights = b_boxes[:, 3] - b_boxes[:, 1]
                    
                    # Get Minimum and Maximum Size Thresholds for This Level.
                    min_size = self.min_sizes[level_idx]
                    max_size = self.max_sizes[level_idx]
                    
                    # More Permissive Size Filtering.
                    size_mask = (widths > min_size * 0.5) & (heights > min_size * 0.5) & \
                               (widths < max_size * 2.0) & (heights < max_size * 2.0)
                    
                    if not size_mask.any():
                        print("No Boxes Passed Size Filtering!")
                        continue
                        
                    # Filter Boxes by Size.
                    b_boxes = b_boxes[size_mask]
                    b_scores = b_scores[size_mask]
                    b_labels = b_labels[size_mask]
                    
                    # Apply NMS with More Permissive Threshold.
                    keep = nms(b_boxes, b_scores, iou_threshold=self.nms_thresholds[level_idx])
                    # print(f"Boxes after NMS: {len(keep)}")
                    
                    # Add Valid Predictions to Lists.
                    if len(keep) > 0:
                        level_filtered_boxes.append(b_boxes[keep])
                        level_filtered_scores.append(b_scores[keep])
                        level_filtered_labels.append(b_labels[keep])
                        # print(f"Added {len(keep)} boxes from level {level_idx}")
            
            # Add Level Predictions to Overall Lists if We Found Any.
            if level_filtered_boxes:
                # print(f"\nAdding Predictions from Level {level_idx}:")
                # print(f"Number of Predictions: {sum(len(boxes) for boxes in level_filtered_boxes)}")
                all_boxes.extend(level_filtered_boxes)
                all_scores.extend(level_filtered_scores)
                all_labels.extend(level_filtered_labels)
        
        # If We Have Any Valid Predictions.
        if all_boxes:
            # Stack Predictions from All Levels.
            boxes_out = torch.cat(all_boxes, dim=0)
            scores_out = torch.cat(all_scores, dim=0)
            labels_out = torch.cat(all_labels, dim=0)
            
            # Move Back to Original Device.
            boxes_out = boxes_out.to(images.device)
            scores_out = scores_out.to(images.device)
            labels_out = labels_out.to(images.device)
            
            # Create detections list with one dict per batch item
            detections = []
            for b in range(len(boxes_out)):
                # Filter predictions for this batch
                batch_mask = (boxes_out[:, 0] >= 0)  # All valid boxes
                batch_boxes = boxes_out[batch_mask]
                batch_scores = scores_out[batch_mask]
                batch_labels = labels_out[batch_mask]
                
                detections.append({
                    'boxes': batch_boxes,
                    'scores': batch_scores,
                    'labels': batch_labels + 1  # Convert to 1-based indexing
                })
            
            outputs = {
                'detections': detections,  # Add detections key
                'cls_scores': head_outputs['cls_scores'],  # Keep original scores for loss computation
                'bbox_preds': head_outputs['bbox_preds']  # Keep original boxes for loss computation
            }
        else:
            # Return Empty Predictions.
            empty_detections = [{
                'boxes': torch.zeros((0, 4), device=images.device),
                'scores': torch.zeros(0, device=images.device),
                'labels': torch.zeros(0, dtype=torch.long, device=images.device)
            }]
            
            outputs = {
                'detections': empty_detections,  # Add detections key
                'cls_scores': head_outputs['cls_scores'],  # Keep original scores for loss computation
                'bbox_preds': head_outputs['bbox_preds']  # Keep original boxes for loss computation
            }
        
        # Compute losses if in training mode (boxes and labels provided)
        if boxes is not None and labels is not None:
            # Compute classification loss
            cls_loss = self.detection_head.cls_loss(
                head_outputs['cls_scores'],
                boxes,
                labels
            )
            
            # Compute box regression loss
            box_loss = self.detection_head.box_loss(
                head_outputs['bbox_preds'],
                boxes,
                labels
            )
            
            # Add losses to outputs
            outputs['cls_loss'] = cls_loss
            outputs['box_loss'] = box_loss
        
        return outputs

# ----------------------------------------------------------------------------

# Build Detector.   
def build_detector(config):
    """Build the complete detection model."""
    # Create Backbone.
    backbone = ResNetBackbone(
        pretrained=config['pretrained_backbone']
    )
    
    # Create FPN with Proper Channel List.
    in_channels_list = [
        backbone.out_channels['layer1'],  # 256
        backbone.out_channels['layer2'],  # 512
        backbone.out_channels['layer3'],  # 1024
        backbone.out_channels['layer4']   # 2048
    ]
    
    # Create FPN.
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,  # Pass the list instead of the dict
        out_channels=config['fpn_out_channels']
    )
    
    # Create Detector.
    detector = ObjectDetector(
        pretrained_backbone=config['pretrained_backbone'],
        fpn_out_channels=config['fpn_out_channels'],
        num_classes=config['num_classes'],
        num_anchors=config['num_anchors'],
        debug=config.get('debug', False)  # Get debug setting from config
    )
    
    return detector

# ----------------------------------------------------------------------------

# Detection Model Class.
class DetectionModel(nn.Module):
    """Complete detection model."""
    
    # Constructor.
    def __init__(self, backbone, fpn, num_classes, num_anchors):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        # ... might come back to this 

    # Forward Pass.
    def forward(self, x, boxes=None, labels=None, debug=False):
        """Forward Pass of the Detector.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            boxes (torch.Tensor, optional): Ground truth boxes of shape [B, N, 4]
            labels (torch.Tensor, optional): Ground truth labels of shape [B, N]
            debug (bool, optional): Whether to print debug information
        Returns:
            dict: Dictionary containing the model outputs
        """
        # Extract Features Using the Backbone.
        features = self.backbone(x)
        
        # Generate FPN Features.
        fpn_features = self.fpn(features)
        
        # Initialize Lists to Store Predictions from Each Level.
        all_scores = []
        all_boxes = []
        all_anchors = []
        
        # Process Each FPN Level.
        for level, feature in enumerate(fpn_features):
            if debug:
                print(f"\nLevel {level} Shapes:")
                print(f"Scores: {feature.shape}")
                print(f"Boxes: {feature.shape}")
                print(f"Anchors: {self.anchor_generator.anchors[level].shape}")
            
            # Get Predictions for This Level.
            level_scores, level_boxes = self.detection_head(feature)
            level_anchors = self.anchor_generator.anchors[level]
            
            # Reshape Predictions for Processing.
            B, _, H, W = feature.shape
            level_scores = level_scores.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            level_boxes = level_boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)
            
            # Apply Sigmoid to Get Confidence Scores.
            level_scores = torch.sigmoid(level_scores)
            
            # Convert Box Deltas to Actual Boxes.
            level_boxes = self.apply_deltas_to_anchors(level_boxes, level_anchors)
            
            # Clip Boxes to Valid Range [0, 1].
            level_boxes = torch.clamp(level_boxes, 0, 1)
            
            all_scores.append(level_scores)
            all_boxes.append(level_boxes)
            all_anchors.append(level_anchors)
        
        # Concatenate Predictions from All Levels.
        scores = torch.cat(all_scores, dim=1)  # [B, N, num_classes]
        boxes = torch.cat(all_boxes, dim=1)    # [B, N, 4]
        anchors = torch.cat(all_anchors, dim=0)  # [N, 4]
        
        #if debug:
        #    print(f"\nTotal predictions: {len(boxes[0])}")
        #    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # During Training, Return the Raw Predictions.
        if boxes is not None and labels is not None:
            return {
                'pred_boxes': boxes,
                'pred_scores': scores,
                'gt_boxes': boxes,
                'gt_labels': labels
            }
        
        # During Inference, Filter and Process Predictions.
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        
        for b in range(len(boxes)):
            # Get Maximum Score and Corresponding Class for Each Prediction.
            max_scores, max_classes = scores[b].max(dim=1)
            
            # Filter by Confidence Threshold.
            mask = max_scores > self.confidence_threshold
            if mask.sum() == 0:
                if debug:
                    print("\nNo Predictions Above Confidence Threshold!")
                continue
            
            # Get Boxes, Scores, and Labels for This Batch Item.
            b_boxes = boxes[b][mask]
            b_scores = max_scores[mask]
            b_labels = max_classes[mask]
            
            # Filter by Box Size.
            widths = b_boxes[:, 2] - b_boxes[:, 0]
            heights = b_boxes[:, 3] - b_boxes[:, 1]
            size_mask = (widths > 0.01) & (heights > 0.01) & (widths < 0.99) & (heights < 0.99)
            
            # If No Valid Box Sizes, Skip.
            if size_mask.sum() == 0:
                if debug:
                    print("\nNo Valid Box Sizes!")
                continue
            
            # Filter Boxes by Size.
            b_boxes = b_boxes[size_mask]
            b_scores = b_scores[size_mask]
            b_labels = b_labels[size_mask]
            
            # Print Box Statistics.
            if debug:
                print(f"\nBox Statistics:")
                print(f"Total Boxes: {len(b_boxes)}")
                print(f"Width range: [{widths[size_mask].min():.4f}, {widths[size_mask].max():.4f}]")
                print(f"Height range: [{heights[size_mask].min():.4f}, {heights[size_mask].max():.4f}]")
            
            # Apply NMS per Class.
            keep_indices = nms(b_boxes, b_scores, self.nms_threshold)
            
            # Add Valid Predictions to Lists.
            filtered_boxes.append(b_boxes[keep_indices])
            filtered_scores.append(b_scores[keep_indices])
            filtered_labels.append(b_labels[keep_indices])
        
        # If No Predictions Survived Filtering.
        if not filtered_boxes:
            return {
                'boxes': torch.zeros((0, 4), device=x.device),
                'scores': torch.zeros(0, device=x.device),
                'labels': torch.zeros(0, dtype=torch.long, device=x.device),
                'all_scores': scores  # Keep all class scores for debugging [B, N, num_classes]
            }
        
        return {
            'boxes': torch.stack(filtered_boxes),
            'scores': torch.stack(filtered_scores),
            'labels': torch.stack(filtered_labels),
            'all_scores': scores  # Keep all class scores for debugging [B, N, num_classes]
        }

    def apply_deltas_to_anchors(self, deltas, anchors):
        """Convert Box Deltas to Actual Boxes using Anchor Boxes.
        Args:
            deltas (torch.Tensor): Box Deltas of Shape [B, N, 4]
            anchors (torch.Tensor): Anchor Boxes of Shape [N, 4]
        Returns:
            torch.Tensor: Predicted boxes of shape [B, N, 4]
        """
        # Extract Anchor Coordinates.
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
        
        # Extract Predicted Deltas.
        dx = deltas[..., 0]
        dy = deltas[..., 1]
        dw = deltas[..., 2]
        dh = deltas[..., 3]
        
        # Prevent Extreme Deltas.
        dw = torch.clamp(dw, max=4.0)
        dh = torch.clamp(dh, max=4.0)
        
        # Apply Deltas.
        pred_ctr_x = dx * anchor_widths + anchor_ctr_x
        pred_ctr_y = dy * anchor_heights + anchor_ctr_y
        pred_w = torch.exp(dw) * anchor_widths
        pred_h = torch.exp(dh) * anchor_heights
        
        # Convert to [x1, y1, x2, y2] Format.
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[..., 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[..., 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[..., 3] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes 