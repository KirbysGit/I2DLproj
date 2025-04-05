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
    """Generate Anchors for Object Detection with improved filtering and center sampling."""
    
    # Initialize Anchor Generator.
    def __init__(self,
                 base_sizes: List[int] = [32, 64, 128, 256],  # Increased base sizes
                 aspect_ratios: List[float] = [0.7, 1.0, 1.4],  # Focused ratios
                 scales: List[float] = [0.75, 1.0],  # Reduced scales
                 min_size: float = 0.02,  # Minimum relative size (2% of image)
                 max_size: float = 0.3,   # Maximum relative size (30% of image)
                 center_sampling_radius: float = 0.5):  # Radius for center sampling
        """
        Initialize Anchor Generator with appropriate constraints.
        
        Args:
            base_sizes: Base anchor sizes for each feature pyramid level
            aspect_ratios: Reduced set of width/height ratios
            scales: Reduced set of scaling factors
            min_size: Minimum allowed size relative to image (2%)
            max_size: Maximum allowed size relative to image (30%)
            center_sampling_radius: Radius for center sampling (relative to stride)
        """
        self.base_sizes = base_sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.min_size = min_size
        self.max_size = max_size
        self.center_sampling_radius = center_sampling_radius
        self.num_anchors = len(aspect_ratios) * len(scales)
        
        # Pre-compute base anchors
        self.base_anchors = self._generate_base_anchors()
        
    # Generate Base Anchors.
    def _generate_base_anchors(self) -> List[torch.Tensor]:
        """Generate base anchors with appropriate size validation."""
        base_anchors = []
        
        for base_size in self.base_sizes:
            level_anchors = []
            # More lenient minimum size - use 12.5% of base size or 8 pixels, whichever is larger
            min_pixels = max(8, base_size * 0.125)  
            # Maximum size is 4x the base size
            max_pixels = base_size * 4.0
            
            for ratio in self.aspect_ratios:
                for scale in self.scales:
                    # Calculate initial size
                    size = base_size * scale
                    
                    # Calculate width and height
                    w = size * math.sqrt(ratio)
                    h = size / math.sqrt(ratio)
                    
                    # Skip if dimensions are invalid
                    if w < min_pixels or h < min_pixels or w > max_pixels or h > max_pixels:
                        continue
                    
                    # Create anchor box [x1, y1, x2, y2] centered at origin
                    anchor = torch.tensor([
                        -w/2, -h/2, w/2, h/2
                    ], dtype=torch.float32)
                    
                    level_anchors.append(anchor)
            
            # If no anchors were valid for this level, create a fallback anchor
            if not level_anchors:
                # Create a single square anchor with size equal to base_size
                fallback_size = base_size * 0.75  # Use 75% of base size as fallback
                anchor = torch.tensor([
                    -fallback_size/2, -fallback_size/2, 
                    fallback_size/2, fallback_size/2
                ], dtype=torch.float32)
                level_anchors.append(anchor)
                print(f"Warning: Using fallback anchor for base_size={base_size}")
            
            base_anchors.append(torch.stack(level_anchors))
        
        return base_anchors
    
    # Generate Anchors.
    def generate_anchors(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate anchors with center sampling and strict filtering."""
        anchors_per_level = []
        image_size = feature_maps[0].shape[-2:]  # Get image size from first feature map
        
        for level_id, feature_map in enumerate(feature_maps):
            _, _, height, width = feature_map.shape
            stride = self.base_sizes[level_id]
            
            # Generate grid coordinates
            grid_x = torch.arange(0, width, device=feature_map.device)
            grid_y = torch.arange(0, height, device=feature_map.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            
            # Shift coordinates to input image scale
            grid_x = grid_x * stride + stride // 2
            grid_y = grid_y * stride + stride // 2
            
            # Get base anchors for this level
            base_anchors = self.base_anchors[level_id].to(feature_map.device)
            
            # Generate anchors for all positions
            anchors = torch.zeros(height, width, self.num_anchors, 4, device=feature_map.device)
            
            # Expand grid coordinates
            grid_x_expanded = grid_x.unsqueeze(-1).expand(-1, -1, self.num_anchors)
            grid_y_expanded = grid_y.unsqueeze(-1).expand(-1, -1, self.num_anchors)
            
            # Apply base anchors to grid centers
            anchors[..., 0] = grid_x_expanded + base_anchors[:, 0]  # x1
            anchors[..., 1] = grid_y_expanded + base_anchors[:, 1]  # y1
            anchors[..., 2] = grid_x_expanded + base_anchors[:, 2]  # x2
            anchors[..., 3] = grid_y_expanded + base_anchors[:, 3]  # y2
            
            # Normalize coordinates to [0, 1]
            anchors[..., [0, 2]] /= image_size[1]  # x coordinates
            anchors[..., [1, 3]] /= image_size[0]  # y coordinates
            
            # Calculate anchor centers for center sampling
            anchor_centers_x = (anchors[..., 0] + anchors[..., 2]) / 2
            anchor_centers_y = (anchors[..., 1] + anchors[..., 3]) / 2
            
            # Calculate anchor dimensions
            widths = anchors[..., 2] - anchors[..., 0]
            heights = anchors[..., 3] - anchors[..., 1]
            
            # Create validity masks
            valid_sizes = (widths >= self.min_size) & (heights >= self.min_size) & \
                        (widths <= self.max_size) & (heights <= self.max_size)
            
            valid_centers = (anchor_centers_x >= self.center_sampling_radius) & \
                          (anchor_centers_x <= 1 - self.center_sampling_radius) & \
                          (anchor_centers_y >= self.center_sampling_radius) & \
                          (anchor_centers_y <= 1 - self.center_sampling_radius)
            
            valid_coords = (anchors[..., 0] >= 0) & (anchors[..., 1] >= 0) & \
                         (anchors[..., 2] <= 1) & (anchors[..., 3] <= 1)
            
            # Combine all validity masks
            valid_mask = valid_sizes & valid_centers & valid_coords
            
            # Filter anchors
            anchors = anchors[valid_mask]
            
            if len(anchors) == 0:
                print(f"[WARNING] No valid anchors at level {level_id+2}")
                continue
            
            # Add level-specific anchor statistics
            #if self.training:
            #   print(f"\nLevel {level_id+2} anchor statistics:")
            #   print(f"  Total anchors: {len(anchors)}")
            #    print(f"  Size range: [{widths[valid_mask].min():.3f}, {widths[valid_mask].max():.3f}] x "
            #          f"[{heights[valid_mask].min():.3f}, {heights[valid_mask].max():.3f}]")
            #    print(f"  Center range X: [{anchor_centers_x[valid_mask].min():.3f}, {anchor_centers_x[valid_mask].max():.3f}]")
            #    print(f"  Center range Y: [{anchor_centers_y[valid_mask].min():.3f}, {anchor_centers_y[valid_mask].max():.3f}]")
            #    print(f"  Rejected by size: {(~valid_sizes).sum().item()}")
            #    print(f"  Rejected by center: {(~valid_centers).sum().item()}")
            #    print(f"  Rejected by coords: {(~valid_coords).sum().item()}")
            
            anchors_per_level.append(anchors)
        
        return anchors_per_level
    
    # Generate Anchors for a Single Feature Level.
    def generate_anchors_for_level(self, feature_map_size, stride, device):
        """Generate anchors for a single feature level with center sampling."""
        H, W = feature_map_size
        
        # Generate grid coordinates
        grid_x = torch.arange(0, W, device=device)
        grid_y = torch.arange(0, H, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # Convert to center coordinates and scale by stride
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride
        
        # Generate base anchors with size constraints
        base_anchors = []
        min_pixels = max(16, stride * 0.25)  # Minimum 16 pixels or 25% of stride
        max_pixels = stride * 2.5  # Maximum 2.5x the stride
        
        for scale in self.scales:
            scale_size = stride * scale
            for ratio in self.aspect_ratios:
                w = scale_size * math.sqrt(ratio)
                h = scale_size / math.sqrt(ratio)
                
                # Skip invalid dimensions
                if w < min_pixels or h < min_pixels or w > max_pixels or h > max_pixels:
                    continue
                
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        if not base_anchors:
            raise ValueError(f"No valid anchors generated for stride {stride}")
        
        # Convert to tensor
        base_anchors = torch.tensor(base_anchors, device=device)
        
        # Combine grid centers with base anchors
        centers = torch.stack([
            centers_x.reshape(-1),
            centers_y.reshape(-1),
            centers_x.reshape(-1),
            centers_y.reshape(-1)
        ], dim=1)
        
        # Add centers to base anchors
        anchors = (centers.unsqueeze(1) + base_anchors.unsqueeze(0))
        
        return anchors.reshape(-1, 4)
    
    # String Representation.
    def __repr__(self):
        """String representation with anchor statistics."""
        return (f"AnchorGenerator(\n"
                f"  base_sizes={self.base_sizes},\n"
                f"  aspect_ratios={self.aspect_ratios},\n"
                f"  scales={self.scales},\n"
                f"  min_size={self.min_size:.3f},\n"
                f"  max_size={self.max_size:.3f},\n"
                f"  center_sampling_radius={self.center_sampling_radius:.3f},\n"
                f"  anchors_per_location={self.num_anchors}\n"
                f")") 