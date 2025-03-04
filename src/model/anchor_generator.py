# model / anchor_generator.py

# -----

# Generates Anchors for Object Detection.
# Generates Boxes at Diff Scales & Aspect Ratios for FPN-Style Detection.

# -----

# Imports.
import torch
import math
from typing import List, Tuple

# Anchor Generator Class.
class AnchorGenerator:
    """Generate Anchors for Object Detection."""
    
    # Initialize Anchor Generator.
    def __init__(self,
                 base_sizes: List[int] = [32, 64, 128, 256],  # Base anchor sizes for each FPN level
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],  # Width/height ratios
                 scales: List[float] = [1.0]):  # Additional scaling factors
        """
        Initialize Anchor Generator.
        
        Args:
            base_sizes:         Base anchor sizes for each feature pyramid level.
            aspect_ratios:      Width/height ratios for anchors.
            scales:             Additional scaling factors for base sizes.
        """

        # Initialize Anchor Generator.
        self.base_sizes = base_sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_anchors = len(aspect_ratios) * len(scales)
        
        # Pre-compute Base Anchors.
        self.base_anchors = self._generate_base_anchors()
        
    # Generate Base Anchors.
    def _generate_base_anchors(self) -> List[torch.Tensor]:
        """Generate Base Anchors for Each FPN Level."""
        base_anchors = []
        
        # Iterate Over Each Base Size.
        for base_size in self.base_sizes:
            level_anchors = []
            
            # Iterate Over Each Aspect Ratio.
            for ratio in self.aspect_ratios:
                # Iterate Over Each Scale.
                for scale in self.scales:
                    # Calculate Width and Height.
                    size = base_size * scale
                    w = size * math.sqrt(ratio)
                    h = size / math.sqrt(ratio)
                    
                    # Validate Dimensions.
                    if w <= 0 or h <= 0:
                        print(f"[WARNING] Invalid anchor dimensions at base_size={base_size}, ratio={ratio}, scale={scale}")
                        print(f"  w={w}, h={h}")
                        continue
                    
                    # Create Anchor Box [x1, y1, x2, y2] Centered at Origin.
                    anchor = torch.tensor([
                        -w/2, -h/2, w/2, h/2
                    ], dtype=torch.float32)
                    
                    # Validate Anchor Coordinates.
                    if anchor[0] >= anchor[2] or anchor[1] >= anchor[3]:
                        print(f"[WARNING] Invalid anchor coordinates: {anchor}")
                        continue
                    
                    level_anchors.append(anchor)
            
            # Stack Anchors for This Level.
            if level_anchors:
                base_anchors.append(torch.stack(level_anchors))
            else:
                raise ValueError(f"[ERROR] No valid anchors generated for base_size={base_size}")
        
        return base_anchors
    
    # Generate Anchors.
    def generate_anchors(self, 
                        feature_maps: List[torch.Tensor]
                        ) -> List[torch.Tensor]:
        """
        Generate Anchors for Each Location in Feature Maps.
        
        Args:
            feature_maps: List of Feature Maps from FPN [B, C, H, W].
            
        Returns:
            List of Anchor Tensors for Each Level [num_anchors*H*W, 4].
        """
        anchors_per_level = []
        
        # Iterate Over Each Feature Map.
        for level_id, feature_map in enumerate(feature_maps):
            # Get Feature Map Size.
            _, _, height, width = feature_map.shape
            
            # Generate Grid Coordinates.
            grid_x = torch.arange(0, width, device=feature_map.device)
            grid_y = torch.arange(0, height, device=feature_map.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            
            # Calculate Stride Based on Feature Level.
            stride = self.base_sizes[level_id]
            
            # Shift Coordinates to Input Image Scale.
            grid_x = grid_x * stride + stride // 2
            grid_y = grid_y * stride + stride // 2
            
            
            # Get Base Anchors for This Level.
            base_anchors = self.base_anchors[level_id].to(feature_map.device)
            
            
            # Broadcast Base Anchors to All Positions.
            anchors = torch.zeros(height, width, self.num_anchors, 4, 
                                device=feature_map.device)
            
            # Properly Expand Grid Coordinates & Apply Base Anchors.
            grid_x_expanded = grid_x.unsqueeze(-1).expand(-1, -1, self.num_anchors)
            grid_y_expanded = grid_y.unsqueeze(-1).expand(-1, -1, self.num_anchors)
            
            # Apply Base Anchors to Grid Centers.
            anchors[..., 0] = grid_x_expanded + base_anchors[:, 0]  # x1
            anchors[..., 1] = grid_y_expanded + base_anchors[:, 1]  # y1
            anchors[..., 2] = grid_x_expanded + base_anchors[:, 2]  # x2
            anchors[..., 3] = grid_y_expanded + base_anchors[:, 3]  # y2
            
            # Validate Generated Anchors.
            valid_mask = (anchors[..., 2] > anchors[..., 0]) & (anchors[..., 3] > anchors[..., 1])
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum().item()
                print(f"[WARNING] {invalid_count} invalid anchors at level {level_id+2}")
            
            # Reshape to [H*W*num_anchors, 4].
            anchors = anchors.view(-1, 4)
            
            # Append to List.
            anchors_per_level.append(anchors)
        
        return anchors_per_level
    
    # Generate Anchors for a Single Feature Level.
    def generate_anchors_for_level(self, feature_map_size, stride, device):
        """Generate anchors for a single feature level."""
        # Get Feature Map Size.
        H, W = feature_map_size
        
        # Generate Grid Coordinates.
        grid_x = torch.arange(0, W, device=device)
        grid_y = torch.arange(0, H, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # Convert to Center Coordinates and Scale by Stride.
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride
        
        # Generate Base Anchors.
        base_anchors = []
        for scale in self.scales:
            scale_size = stride * scale
            for ratio in self.aspect_ratios:
                w = scale_size * math.sqrt(ratio)
                h = scale_size / math.sqrt(ratio)
                
                # Validate Dimensions.
                if w <= 0 or h <= 0:
                    print(f"[WARNING] Invalid anchor dimensions at scale={scale}, ratio={ratio}")
                    print(f"  w={w}, h={h}")
                    continue
                
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        # Convert to Tensor.
        base_anchors = torch.tensor(base_anchors, device=device)
        num_base_anchors = len(base_anchors)
        
        # Combine Grid Centers with Base Anchors.
        centers = torch.stack([
            centers_x.reshape(-1),
            centers_y.reshape(-1),
            centers_x.reshape(-1),
            centers_y.reshape(-1)
        ], dim=1)  # [H*W, 4]
        
        # Add Centers to Base Anchors for Each Position.
        anchors = (centers.unsqueeze(1) + base_anchors.unsqueeze(0))  # [H*W, A, 4]
        
        # Validate Generated Anchors.
        valid_mask = (anchors[..., 2] > anchors[..., 0]) & (anchors[..., 3] > anchors[..., 1])
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"[WARNING] {invalid_count} invalid anchors")
        
        # Reshape to [H*W*A, 4].
        return anchors.reshape(-1, 4)  # [H*W*A, 4]
    
    # String Representation.
    def __repr__(self):
        """String Representation."""
        return (f"AnchorGenerator(\n"
                f"  base_sizes={self.base_sizes},\n"
                f"  aspect_ratios={self.aspect_ratios},\n"
                f"  scales={self.scales},\n"
                f"  num_anchors_per_location={self.num_anchors}\n"
                f")") 