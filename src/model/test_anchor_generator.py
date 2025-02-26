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
    anchors = anchor_gen.generate_anchors(image_size)
    
    def visualize_anchors_detailed(anchors, image_size, stride):
        """Enhanced anchor visualization with multiple plots"""
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Main anchor visualization
        ax1 = plt.subplot(121)
        ax1.set_title(f'Sample Anchors (stride={stride})')
        
        # Plot image bounds
        ax1.plot([0, image_size[1], image_size[1], 0, 0],
                [0, 0, image_size[0], image_size[0], 0], 'k-')
        
        # Plot grid
        for x in range(0, image_size[1], stride):
            ax1.axvline(x=x, color='gray', linestyle=':', alpha=0.3)
        for y in range(0, image_size[0], stride):
            ax1.axhline(y=y, color='gray', linestyle=':', alpha=0.3)
        
        # Plot anchors with different colors per scale and ratio
        colors = plt.cm.rainbow(np.linspace(0, 1, len(anchor_gen.anchor_scales[stride])))
        scales = len(anchor_gen.anchor_scales[stride])
        ratios = len(anchor_gen.anchor_ratios)
        anchors_per_point = scales * ratios
        
        # Create legend handles
        legend_handles = []
        for scale_idx, scale in enumerate(anchor_gen.anchor_scales[stride]):
            for ratio_idx, ratio in enumerate(anchor_gen.anchor_ratios):
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
        
        # Calculate areas and aspect ratios
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        areas = widths * heights
        aspect_ratios = widths / heights
        
        # Create scatter plot
        scatter = ax2.scatter(areas, aspect_ratios, 
                            c=np.log(areas), 
                            cmap='viridis',
                            alpha=0.5)
        plt.colorbar(scatter, label='Log Area')
        
        ax2.set_xlabel('Area (pixels²)')
        ax2.set_ylabel('Aspect Ratio (width/height)')
        ax2.set_xscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print(f"\nDetailed Statistics for stride {stride}:")
        print(f"Area range: {areas.min():.1f} to {areas.max():.1f} pixels²")
        print(f"Aspect ratio range: {aspect_ratios.min():.2f} to {aspect_ratios.max():.2f}")
        print(f"Coverage: {(areas > 0).sum() / len(areas) * 100:.1f}% of anchors are valid")
    
    # Visualize anchors for each stride
    for stride, stride_anchors in anchors.items():
        visualize_anchors_detailed(stride_anchors, image_size, stride)
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_anchor_generator() 