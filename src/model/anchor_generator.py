import torch
import math
from typing import List, Tuple

class AnchorGenerator:
    """Generate anchors for object detection."""
    
    def __init__(self,
                 base_sizes: List[int] = [32, 64, 128, 256],  # Base anchor sizes for each FPN level
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],  # Width/height ratios
                 scales: List[float] = [1.0]):  # Additional scaling factors
        """
        Initialize anchor generator.
        
        Args:
            base_sizes: Base anchor sizes for each feature pyramid level
            aspect_ratios: Width/height ratios for anchors
            scales: Additional scaling factors for base sizes
        """
        self.base_sizes = base_sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_anchors = len(aspect_ratios) * len(scales)
        
        # Pre-compute base anchors for each combination
        self.base_anchors = self._generate_base_anchors()
    
    def _generate_base_anchors(self) -> List[torch.Tensor]:
        """Generate base anchors for each FPN level."""
        base_anchors = []
        
        for base_size in self.base_sizes:
            level_anchors = []
            for ratio in self.aspect_ratios:
                for scale in self.scales:
                    # Calculate width and height
                    size = base_size * scale
                    w = size * math.sqrt(ratio)
                    h = size / math.sqrt(ratio)
                    
                    # Create anchor box [x1, y1, x2, y2] centered at origin
                    anchor = torch.tensor([
                        -w/2, -h/2, w/2, h/2
                    ], dtype=torch.float32)
                    
                    level_anchors.append(anchor)
            
            # Stack anchors for this level
            base_anchors.append(torch.stack(level_anchors))
        
        return base_anchors
    
    def generate_anchors(self, 
                        feature_maps: List[torch.Tensor]
                        ) -> List[torch.Tensor]:
        """
        Generate anchors for each location in feature maps.
        
        Args:
            feature_maps: List of feature maps from FPN [B, C, H, W]
            
        Returns:
            List of anchor tensors for each level [num_anchors*H*W, 4]
        """
        anchors_per_level = []
        
        for level_id, feature_map in enumerate(feature_maps):
            # Get feature map size
            _, _, height, width = feature_map.shape
            
            # Generate grid coordinates
            grid_x = torch.arange(0, width, device=feature_map.device)
            grid_y = torch.arange(0, height, device=feature_map.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            
            # Calculate stride based on feature level
            stride = self.base_sizes[level_id]
            
            # Shift coordinates to input image scale
            grid_x = grid_x * stride + stride // 2
            grid_y = grid_y * stride + stride // 2
            
            # Repeat grid for each anchor
            grid_x = grid_x.unsqueeze(-1).repeat(1, 1, self.num_anchors)
            grid_y = grid_y.unsqueeze(-1).repeat(1, 1, self.num_anchors)
            
            # Get base anchors for this level
            base_anchors = self.base_anchors[level_id]
            
            # Broadcast base anchors to all positions
            anchors = torch.zeros(height, width, self.num_anchors, 4, 
                                device=feature_map.device)
            anchors[..., 0] = grid_x - base_anchors[:, 0]  # x1
            anchors[..., 1] = grid_y - base_anchors[:, 1]  # y1
            anchors[..., 2] = grid_x + base_anchors[:, 2]  # x2
            anchors[..., 3] = grid_y + base_anchors[:, 3]  # y2
            
            # Reshape to [H*W*num_anchors, 4]
            anchors = anchors.view(-1, 4)
            anchors_per_level.append(anchors)
        
        return anchors_per_level
    
    def generate_anchors_for_level(self, feature_map_size, stride, device):
        """Generate anchors for a single feature level."""
        H, W = feature_map_size
        
        # Generate grid coordinates
        grid_x = torch.arange(0, W, device=device)
        grid_y = torch.arange(0, H, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # Convert to center coordinates and scale by stride
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride
        
        # Generate base anchors
        base_anchors = []
        for scale in self.scales:
            scale_size = stride * scale
            for ratio in self.aspect_ratios:
                w = scale_size * math.sqrt(ratio)
                h = scale_size / math.sqrt(ratio)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(base_anchors, device=device)
        num_base_anchors = len(base_anchors)
        
        # Combine grid centers with base anchors
        centers = torch.stack([
            centers_x.reshape(-1),
            centers_y.reshape(-1),
            centers_x.reshape(-1),
            centers_y.reshape(-1)
        ], dim=1)  # [H*W, 4]
        
        # Add centers to base anchors for each position
        anchors = (centers.unsqueeze(1) + base_anchors.unsqueeze(0))  # [H*W, A, 4]
        return anchors.reshape(-1, 4)  # [H*W*A, 4]
    
    def __repr__(self):
        """String representation."""
        return (f"AnchorGenerator(\n"
                f"  base_sizes={self.base_sizes},\n"
                f"  aspect_ratios={self.aspect_ratios},\n"
                f"  scales={self.scales},\n"
                f"  num_anchors_per_location={self.num_anchors}\n"
                f")") 