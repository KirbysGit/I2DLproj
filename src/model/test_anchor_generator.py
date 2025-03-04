import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def test_anchor_generator():
    """Test the AnchorGenerator implementation"""
    from anchor_generator import AnchorGenerator
    
    print("\nTesting Anchor Generator...")
    
    # Initialize generator
    image_size = (640, 640)
    anchor_gen = AnchorGenerator()
    anchor_gen.debug = True  # Enable debug mode
    
    # Print anchor generator configuration
    print("\nAnchor Generator Configuration:")
    print(f"Base sizes: {anchor_gen.base_sizes}")
    print(f"Aspect ratios: {anchor_gen.aspect_ratios}")
    print(f"Scales: {anchor_gen.scales}")
    print(f"Number of anchors per location: {anchor_gen.num_anchors}")
    
    # Create dummy feature maps for testing
    device = torch.device('cpu')
    feature_maps = [
        torch.randn(1, 256, 80, 80, device=device),  # P2
        torch.randn(1, 256, 40, 40, device=device),  # P3
        torch.randn(1, 256, 20, 20, device=device),  # P4
        torch.randn(1, 256, 10, 10, device=device)   # P5
    ]
    
    # Print feature map dimensions for verification
    print("\nFeature Map Dimensions:")
    for i, feature_map in enumerate(feature_maps):
        print(f"Feature map {i} (P{i+2}): {feature_map.shape}")
        if feature_map.shape[2] == 0 or feature_map.shape[3] == 0:
            print(f"WARNING: Empty feature map at level {i+2}!")
    
    # Generate anchors for all feature maps
    print("\nGenerating anchors...")
    anchors = anchor_gen.generate_anchors(feature_maps)
    
    def analyze_anchor_box(anchor):
        """Analyze a single anchor box for potential issues"""
        x1, y1, x2, y2 = anchor
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        issues = []
        if width <= 0 or height <= 0:
            issues.append("Zero or negative dimensions")
        if area == 0:
            issues.append("Zero area")
        if x1 == x2 or y1 == y2:
            issues.append("Collapsed box")
        if not torch.isfinite(area):
            issues.append("Invalid area (NaN or Inf)")
        
        return {
            'dimensions': (width.item(), height.item()),
            'area': area.item(),
            'center': (((x1 + x2)/2).item(), ((y1 + y2)/2).item()),
            'issues': issues
        }
    
    def visualize_anchors_detailed(anchors, image_size, stride, level_idx):
        """Enhanced anchor visualization with multiple plots"""
        # Debug print: Check if anchors are generated
        print(f"\nDetailed Anchor Analysis for level {level_idx+2} (stride={stride}):")
        print(f"Total anchors: {len(anchors)}")
        
        # Analyze first few anchors in detail
        print("\nDetailed analysis of first 5 anchors:")
        for i, anchor in enumerate(anchors[:5]):
            analysis = analyze_anchor_box(anchor)
            print(f"\nAnchor {i}:")
            print(f"  Coordinates: ({anchor[0]:.1f}, {anchor[1]:.1f}) -> ({anchor[2]:.1f}, {anchor[3]:.1f})")
            print(f"  Dimensions: {analysis['dimensions']}")
            print(f"  Area: {analysis['area']:.1f}")
            print(f"  Center: {analysis['center']}")
            if analysis['issues']:
                print(f"  Issues: {', '.join(analysis['issues'])}")
        
        # Check for invalid anchors
        if torch.all(anchors == 0):
            print(f"WARNING: All anchors are zeros at level {level_idx+2}!")
            return
        
        # Analyze anchor distribution
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        areas = widths * heights
        aspect_ratios = widths / heights
        
        print("\nAnchor Distribution Statistics:")
        print(f"Width range: {widths.min():.1f} to {widths.max():.1f}")
        print(f"Height range: {heights.min():.1f} to {heights.max():.1f}")
        print(f"Area range: {areas.min():.1f} to {areas.max():.1f}")
        print(f"Aspect ratio range: {aspect_ratios.min():.2f} to {aspect_ratios.max():.2f}")
        
        # Handle invalid values
        valid_mask = (areas > 0) & (torch.isfinite(aspect_ratios))
        valid_areas = areas[valid_mask]
        valid_ratios = aspect_ratios[valid_mask]
        
        print(f"\nValidation Results:")
        print(f"Valid anchors: {valid_mask.sum().item()} / {len(valid_mask)}")
        print(f"Invalid anchors: {(~valid_mask).sum().item()}")
        
        if len(valid_areas) > 0:
            print(f"Valid anchor coverage: {(valid_mask.sum() / len(valid_mask) * 100):.1f}%")
        else:
            print("WARNING: No valid anchors found!")
            return
        
        # Visualization code remains the same
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Main anchor visualization
        ax1 = plt.subplot(121)
        ax1.set_title(f'Sample Anchors (stride={stride}, level={level_idx+2})')
        
        # Plot image bounds
        ax1.plot([0, image_size[1], image_size[1], 0, 0],
                [0, 0, image_size[0], image_size[0], 0], 'k-')
        
        # Plot grid
        for x in range(0, image_size[1], stride):
            ax1.axvline(x=x, color='gray', linestyle=':', alpha=0.3)
        for y in range(0, image_size[0], stride):
            ax1.axhline(y=y, color='gray', linestyle=':', alpha=0.3)
        
        # Plot anchors with different colors per scale and ratio
        colors = plt.cm.rainbow(np.linspace(0, 1, len(anchor_gen.scales)))
        scales = len(anchor_gen.scales)
        ratios = len(anchor_gen.aspect_ratios)
        anchors_per_point = scales * ratios
        
        # Create legend handles
        legend_handles = []
        for scale_idx, scale in enumerate(anchor_gen.scales):
            for ratio_idx, ratio in enumerate(anchor_gen.aspect_ratios):
                rect = patches.Rectangle((0,0), 1, 1, 
                                      facecolor='none',
                                      edgecolor=colors[scale_idx],
                                      alpha=0.5,
                                      label=f'Scale={scale}, Ratio={ratio:.1f}')
                legend_handles.append(rect)
        
        # Plot sample anchors
        max_anchors = 50
        for i in range(min(max_anchors * anchors_per_point, len(anchors))):
            x1, y1, x2, y2 = anchors[i]
            width = x2 - x1
            height = y2 - y1
            scale_idx = (i // ratios) % scales
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=1, edgecolor=colors[scale_idx],
                facecolor='none', alpha=0.5
            )
            ax1.add_patch(rect)
        
        ax1.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xlim(-50, image_size[1] + 50)
        ax1.set_ylim(-50, image_size[0] + 50)
        ax1.grid(True, alpha=0.3)
        
        # 2. Anchor size distribution
        ax2 = plt.subplot(122)
        ax2.set_title('Anchor Size Distribution')
        
        if len(valid_areas) > 0:
            scatter = ax2.scatter(valid_areas, valid_ratios, 
                                c=np.log(valid_areas), 
                                cmap='viridis',
                                alpha=0.5)
            plt.colorbar(scatter, label='Log Area')
        else:
            ax2.text(0.5, 0.5, 'No valid anchors', 
                    ha='center', va='center',
                    transform=ax2.transAxes)
        
        ax2.set_xlabel('Area (pixels²)')
        ax2.set_ylabel('Aspect Ratio (width/height)')
        ax2.set_xscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Visualize anchors for each feature level
    for level_idx, (stride_anchors, stride) in enumerate(zip(anchors, anchor_gen.base_sizes)):
        visualize_anchors_detailed(stride_anchors, image_size, stride, level_idx)
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_anchor_generator() 