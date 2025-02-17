import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange  # Library for tensor reshaping operations
import os

class ViTBlock(nn.Module):
    """
    Vision Transformer Block implementing the standard transformer architecture:
    - Multi-head self-attention
    - Layer normalization
    - MLP feed-forward network
    - Residual connections
    """
    def __init__(self, config):
        super().__init__()
        # Extract transformer configuration from config file
        hidden_dim = config['model']['vit']['hidden_dim']  # Dimension of feature vectors
        num_heads = config['model']['vit']['num_heads']    # Number of attention heads
        mlp_dim = config['model']['vit']['mlp_dim']       # Dimension of MLP layer
        dropout = config['model']['vit']['dropout']        # Dropout rate
        
        # Layer normalization before attention (normalize each feature independently)
        self.norm1 = nn.LayerNorm(hidden_dim)
        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(hidden_dim)
        # MLP block with GELU activation and dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Apply attention block with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)  # Self-attention
        x = x + attn_out  # Residual connection
        
        # Apply MLP block with residual connection
        normed = self.norm2(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out  # Residual connection
        
        return x

class CNNViTHybrid(nn.Module):
    """
    Hybrid CNN-ViT model for object detection:
    1. CNN backbone (ResNet50) for feature extraction
    2. Vision Transformer for global context modeling
    3. Detection head for bounding box prediction
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load ResNet but remove the final layers
        backbone = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2]  # Remove avg pool and fc layers
        )
        
        # Try to load weights locally if available
        weights_path = 'models/resnet50_weights.pth'
        if os.path.exists(weights_path):
            self.backbone.load_state_dict(torch.load(weights_path))
        else:
            try:
                # Load pretrained weights
                pretrained = models.resnet50(pretrained=True)
                # Remove the last two layers from pretrained weights
                pretrained_dict = {k: v for k, v in pretrained.state_dict().items() 
                                 if not k.startswith('fc.') and not k.startswith('avgpool.')}
                self.backbone.load_state_dict(pretrained_dict, strict=False)
            except:
                print("Warning: Could not load pretrained weights. Using random initialization.")
        
        # Calculate feature map size after ResNet
        # Input: 640x640 -> After ResNet: 20x20 (32x downsampling)
        self.feature_size = 640 // 32  # = 20
        
        # Configure Vision Transformer components
        vit_config = config['model']['vit']
        self.patch_size = vit_config['patch_size']
        self.hidden_dim = vit_config['hidden_dim']
        self.num_layers = vit_config['num_layers']
        
        # 1x1 convolution to project CNN features to transformer dimension
        self.projection = nn.Conv2d(2048, self.hidden_dim, kernel_size=1)
        
        # Calculate number of patches
        num_patches = (self.feature_size ** 2)  # 20x20 = 400 patches
        
        # Learnable position embeddings for transformer
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, self.hidden_dim)
        )
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ViTBlock(config) for _ in range(self.num_layers)
        ])
        
        # Detection head: predicts bounding box coordinates and objectness score
        self.detection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 5)  # [x1, y1, x2, y2, objectness]
        )

    def forward(self, x):
        # Extract features using CNN backbone (now outputs proper feature maps)
        features = self.backbone(x)  # Shape: [B, 2048, 20, 20]
        
        # Project features to transformer dimension
        projected = self.projection(features)  # Shape: [B, hidden_dim, 20, 20]
        
        # Reshape to sequence of patches
        patches = rearrange(projected, 'b c h w -> b (h w) c')
        
        # Add positional embeddings (now matches in size)
        patches = patches + self.pos_embedding
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            patches = block(patches)
        
        # Apply detection head
        detections = self.detection_head(patches)
        
        # Reshape output to spatial grid format using stored feature_size
        output = rearrange(detections, 'b (h w) c -> b h w c', 
                          h=self.feature_size, w=self.feature_size)
        
        return output
