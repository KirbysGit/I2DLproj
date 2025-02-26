import torch
import numpy as np

class AnchorGenerator:
    """
    Generate anchors for each feature level of the FPN.
    Creates a set of anchor boxes with different scales and aspect ratios.
    """
    
    def __init__(self, 
                 feature_map_sizes=None,  # List of feature map sizes [(H1,W1), (H2,W2), ...]
                 strides=[4, 8, 16, 32],
                 # Adjust scales per stride level
                 anchor_scales={
                     4: [32, 64],
                     8: [64, 128],
                     16: [128, 256],
                     32: [256, 512]
                 },
                 anchor_ratios=[0.5, 1.0, 2.0]):
        """
        Initialize anchor generator with stride-specific scales.
        
        Args:
            feature_map_sizes: List of (height, width) tuples for each feature level
            strides: Feature map strides relative to input image
            anchor_scales: Dictionary of anchor box sizes per stride
            anchor_ratios: List of anchor aspect ratios (height/width)
        """
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.strides = strides
        self.feature_map_sizes = feature_map_sizes
        
        # Generate base anchors for each feature level
        self.base_anchors = self._generate_base_anchors()
        
    def _generate_base_anchors(self):
        """Generate base anchor boxes for all scales and ratios"""
        base_anchors = {}
        
        for stride in self.strides:
            anchors = []
            # Use stride-specific scales
            for scale in self.anchor_scales[stride]:
                for ratio in self.anchor_ratios:
                    h = scale * np.sqrt(ratio)
                    w = scale * np.sqrt(1.0 / ratio)
                    
                    # Center-based anchor box [x_center, y_center, width, height]
                    anchors.append([-w/2, -h/2, w/2, h/2])
            
            base_anchors[stride] = torch.tensor(anchors, dtype=torch.float32)
        
        return base_anchors
    
    def _shift_anchors(self, base_anchors, stride, height, width):
        """
        Shift base anchors to all positions on the feature map.
        
        Args:
            base_anchors: Base anchor boxes [N, 4]
            stride: Feature map stride
            height, width: Feature map size
            
        Returns:
            shifted_anchors: [height * width * N, 4]
        """
        # Generate grid coordinates
        shift_x = torch.arange(0, width * stride, stride, dtype=torch.float32)
        shift_y = torch.arange(0, height * stride, stride, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        # Reshape shifts to [height * width, 4]
        shifts = torch.stack([
            shift_x.reshape(-1),
            shift_y.reshape(-1),
            shift_x.reshape(-1),
            shift_y.reshape(-1)
        ], dim=1)
        
        # Add dimension for broadcasting
        shifts = shifts.view(-1, 1, 4)
        base_anchors = base_anchors.view(1, -1, 4)
        
        # Generate all anchors
        anchors = shifts + base_anchors
        return anchors.reshape(-1, 4)
    
    def generate_anchors(self, image_size):
        """
        Generate anchors for all feature levels.
        
        Args:
            image_size: (height, width) of input image
            
        Returns:
            dict: Anchors for each feature level {stride: anchors_tensor}
        """
        if self.feature_map_sizes is None:
            # Calculate feature map sizes based on image size and strides
            self.feature_map_sizes = [
                (image_size[0] // stride, image_size[1] // stride)
                for stride in self.strides
            ]
        
        anchors = {}
        for stride, (height, width) in zip(self.strides, self.feature_map_sizes):
            base_anchors = self.base_anchors[stride]
            anchors[stride] = self._shift_anchors(base_anchors, stride, height, width)
            
        return anchors 