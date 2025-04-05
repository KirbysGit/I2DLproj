# model / anchor_generator.py

# -----

# Generates Anchors for Object Detection.
# Generates Boxes at Diff Scales & Aspect Ratios for FPN-Style Detection.

# -----

# Imports.
import torch
import math
from typing import List

# Anchor Generator Class.
class AnchorGenerator:
    """Generate Anchors for Object Detection with improved filtering and center sampling."""
    
    # Initialize Anchor Generator.
    def __init__(self,
                 base_sizes: List[int] = [32, 64, 128, 256, 512],  # Added larger base size
                 aspect_ratios: List[float] = [0.5, 0.7, 1.0, 1.4, 2.0],  # More diverse ratios
                 scales: List[float] = [0.5, 0.75, 1.0, 1.25]):  # More diverse scales
        """
        Initialize Anchor Generator with appropriate constraints.
        
        Args:
            base_sizes: Base anchor sizes for each feature pyramid level
            aspect_ratios: Wider range of width/height ratios
            scales: More diverse set of scaling factors
            min_size: Minimum allowed size relative to image (2%)
            max_size: Maximum allowed size relative to image (30%)
            center_sampling_radius: Radius for center sampling (relative to stride)
        """

        # Initialize Anchor Generator.
        self.base_sizes = base_sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_anchors = len(aspect_ratios) * len(scales)
    
    # Generate Anchors for a Single Feature Level.
    def generate_anchors_for_level(self, feature_map_size, stride, device):
        """Generate anchors for a single feature level with center sampling."""
        H, W = feature_map_size
        
        # Generate Grid Coordinates.
        grid_x = torch.arange(0, W, device=device)
        grid_y = torch.arange(0, H, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # Convert to Center Coordinates and Scale by Stride.
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride
        
        # Generate Base Anchors with Size Constraints.
        base_anchors = []
        min_pixels = max(16, stride * 0.25)  # Minimum 16 Pixels or 25% of Stride.
        max_pixels = stride * 2.5  # Maximum 2.5x the Stride.
        
        # Generate Base Anchors.
        for scale in self.scales:
            # Calculate Scale Size.
            scale_size = stride * scale

            # Generate Anchors for Each Aspect Ratio.
            for ratio in self.aspect_ratios:
                # Calculate Anchor Dimensions.
                w = scale_size * math.sqrt(ratio)
                h = scale_size / math.sqrt(ratio)
                
                # Skip Invalid Dimensions.
                #if w < min_pixels or h < min_pixels or w > max_pixels or h > max_pixels:
                #    continue
                
                # Append Anchor.
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        if not base_anchors:
            raise ValueError(f"No Valid Anchors Generated for Stride {stride}")
        
        # Convert to Tensor.
        base_anchors = torch.tensor(base_anchors, device=device)
        
        # Combine Grid Centers with Base Anchors.
        centers = torch.stack([
            centers_x.reshape(-1),
            centers_y.reshape(-1),
            centers_x.reshape(-1),
            centers_y.reshape(-1)
        ], dim=1)
        
        # Add Centers to Base Anchors.
        anchors = (centers.unsqueeze(1) + base_anchors.unsqueeze(0))
        
        # Return Anchors.
        return anchors.reshape(-1, 4)