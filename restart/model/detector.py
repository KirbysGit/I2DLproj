# model / detector.py

# -----

# Defines Complete Object Detection Model.
# Combines Backbone, Feature Pyramid Network, & Detection Head.

# ----- 

# Imports.
import torch
import torch.nn as nn
from torchvision.ops import nms
from restart.model.backbone import ResNetBackbone
from restart.model.fpn import FeaturePyramidNetwork
from restart.model.detection_head import DetectionHead
from restart.model.anchor_generator import AnchorGenerator
debug = False

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
        
        # Create Anchor Generator w/ Improved Configuration.
        self.anchor_generator = AnchorGenerator(
            base_sizes=[64, 128, 256, 512],   
            scales=[0.5, 1.0, 1.5],
            aspect_ratios=[0.5, 1.0, 2.0],
        )

        self.num_anchors = len(self.anchor_generator.aspect_ratios) * len(self.anchor_generator.scales)

        # Detection Head.
        self.detection_head = DetectionHead(
            in_channels=fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            anchor_generator=self.anchor_generator,
            debug=debug
        )
        
        # More permissive thresholds for early training
        self.conf_thresh = 0.5
        self.nms_thresh = 0.3

        
        # Much more permissive size ranges
        self.min_sizes = [0.001, 0.005, 0.01, 0.02]  # More permissive minimum sizes
        self.max_sizes = [0.1, 0.2, 0.3, 0.4]       # More permissive maximum sizes
        
        
        # Debug flag
        self.debug = debug

    # Forward Pass.
    def forward(self, images, boxes=None, labels=None):
        batch_size = images.shape[0]
        device = images.device
        
        # Get Backbone Features
        backbone_features = self.backbone(images)

        img_h, img_w = images.shape[2], images.shape[3]
        image_size = torch.tensor([img_w, img_h, img_w, img_h], device=device).float()
        
        fpn_input_features = {
            k: v for k, v in backbone_features.items() 
            if k in ['layer1', 'layer2', 'layer3', 'layer4']
        }
        
        fpn_features = self.fpn(fpn_input_features)
        head_outputs = self.detection_head(fpn_features, image_size=image_size)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for level_idx, (level_scores, level_boxes, level_anchors) in enumerate(zip(
            head_outputs['cls_scores'], 
            head_outputs['bbox_preds'],
            head_outputs['anchors']
        )):
            level_scores = level_scores.to(device)
            level_boxes = level_boxes.to(device)
            level_anchors = level_anchors.to(device)

            B, A, C, H, W = level_scores.shape  # C should be 1 for binary
            
            # Reshape: [B, A, H, W, C] -> [B, N]
            level_scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1)
            level_boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            anchors_expanded = level_anchors.unsqueeze(0).expand(B, -1, 4)  # [B, N, 4]
            
            # Apply sigmoid confidence
            level_scores = torch.sigmoid(level_scores)
            max_scores = level_scores
            max_classes = torch.zeros_like(max_scores, dtype=torch.long)  # class=0

            # Flatten for delta application
            level_boxes_flat = level_boxes.reshape(-1, 4)
            anchors_flat = anchors_expanded.reshape(-1, 4)

            # Apply deltas
            pred_boxes_flat, valid_mask_flat = apply_deltas_to_anchors(
                level_boxes_flat,
                anchors_flat,
                max_delta=2.0,
                min_size=self.min_sizes[level_idx],
                max_size=self.max_sizes[level_idx],
                image_size=(img_w, img_h)
            )

            # Reshape back
            pred_boxes = pred_boxes_flat.view(B, -1, 4)
            valid_mask = valid_mask_flat.view(B, -1)

            # Confidence filtering
            confidence_mask = (max_scores > self.conf_thresh) & valid_mask
            assert confidence_mask.shape == max_scores.shape, "Shape mismatch"

            # Filtered lists
            level_filtered_boxes = []
            level_filtered_scores = []
            level_filtered_labels = []

            for b in range(B):
                b_mask = confidence_mask[b]
                if not b_mask.any():
                    continue

                b_boxes = pred_boxes[b][b_mask]
                b_scores = max_scores[b][b_mask]
                b_labels = max_classes[b][b_mask]

                # NMS
                keep = nms(b_boxes, b_scores, iou_threshold=self.nms_thresh)
                keep = keep[:10]

                if len(keep) > 0:
                    level_filtered_boxes.append(b_boxes[keep])
                    level_filtered_scores.append(b_scores[keep])
                    level_filtered_labels.append(b_labels[keep])

            # Add to all outputs
            if level_filtered_boxes:
                all_boxes.extend(level_filtered_boxes)
                all_scores.extend(level_filtered_scores)
                all_labels.extend(level_filtered_labels)

        # Create detections output per image
        detections = [{'boxes': [], 'scores': [], 'labels': []} for _ in range(batch_size)]

        for i in range(len(all_boxes)):
            img_idx = i % batch_size  # safer indexing
            detections[img_idx]['boxes'].append(all_boxes[i])
            detections[img_idx]['scores'].append(all_scores[i])
            detections[img_idx]['labels'].append(all_labels[i])
            
        for i in range(batch_size):
            if detections[i]['boxes']:
                detections[i]['boxes'] = torch.cat(detections[i]['boxes'], dim=0)
                detections[i]['scores'] = torch.cat(detections[i]['scores'], dim=0)
                detections[i]['labels'] = torch.ones_like(detections[i]['scores'], dtype=torch.long)
            else:
                detections[i]['boxes'] = torch.zeros((0, 4), device=device)
                detections[i]['scores'] = torch.zeros((0,), device=device)
                detections[i]['labels'] = torch.zeros((0,), dtype=torch.long, device=device)
            
        outputs = {
            'detections': detections,
            'cls_scores': head_outputs['cls_scores'],
            'bbox_preds': head_outputs['bbox_preds']
        }

        if boxes is not None and labels is not None:
            all_anchors = torch.cat(head_outputs['anchors'], dim=0).to(device)  # [total_anchors, 4]

            cls_loss = self.detection_head.cls_loss(
                head_outputs['cls_scores'],
                boxes,
                labels,
                all_anchors
            )

            box_loss = self.detection_head.box_loss(
                head_outputs['bbox_preds'],
                boxes,
                labels,
                all_anchors
            )

            outputs['cls_loss'] = cls_loss
            outputs['box_loss'] = box_loss


        return outputs

# ----------------------------------------------------------------------------

def apply_deltas_to_anchors(deltas, anchors, max_delta=2.0, min_size=0.01, max_size=0.4, image_size=None):
    """Applies predicted deltas to anchors and clamps boxes to valid values."""
    
    # Compute widths and heights of anchors
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    if debug:
        print(f"Width range: {widths.min()} - {widths.max()}")
        print(f"Height range: {heights.min()} - {heights.max()}")

    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    # Clip the deltas
    dx = deltas[:, 0].clamp(-max_delta, max_delta)
    dy = deltas[:, 1].clamp(-max_delta, max_delta)
    dw = deltas[:, 2].clamp(-max_delta, max_delta)
    dh = deltas[:, 3].clamp(-max_delta, max_delta)

    # Apply deltas
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h


    # ✅ Clamp to image boundaries if available
    if image_size is not None:
        img_w, img_h = image_size
        x1 = x1.clamp(0, img_w)
        y1 = y1.clamp(0, img_h)
        x2 = x2.clamp(0, img_w)
        y2 = y2.clamp(0, img_h)
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    pred_w = x2 - x1
    pred_h = y2 - y1



    # Filter out boxes with extreme sizes
    valid = (pred_w > 1) & (pred_h > 1) & (pred_w < 2000) & (pred_h < 2000)
    if debug:
        print("DW/DH stats:", dw.mean(), dh.mean())

    return boxes, valid
