import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def generate_overlay(original_image, binary_mask, boundaries, results_dict, output_path="result_overlay.png"):
    """
    Generates the final visualization overlay.
    
    Args:
        original_image (numpy.ndarray): C, H, W (RGB preferred for visualization).
        binary_mask (numpy.ndarray): H, W mask.
        boundaries (shapely.geometry.MultiPolygon): Flood boundaries.
        results_dict (dict): Dictionary with area_km2, etc.
        output_path (str): Where to save the image.
    """
    if original_image.shape[0] >= 3:
        # Assuming first 3 channels are RGB
        img_vis = np.transpose(original_image[:3, :, :], (1, 2, 0))
    else:
        # Fallback to grayscale if not enough channels
        img_vis = np.transpose(original_image, (1, 2, 0))
        img_vis = np.repeat(img_vis, 3, axis=2)
        
    # Minmax scale for display
    img_vis = (img_vis - np.min(img_vis)) / (np.max(img_vis) - np.min(img_vis) + 1e-6)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_vis)
    
    # Overlay the flood mask in semi-transparent blue
    mask_rgba = np.zeros((*binary_mask.shape, 4))
    mask_rgba[binary_mask == 1] = [0, 0, 1, 0.4] # Blue, alpha 0.4
    ax.imshow(mask_rgba)
    
    # Plot boundaries
    from shapely.geometry import Polygon
    for geom in boundaries.geoms:
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color='red', linewidth=1.5)
            
    # Add text overlay
    textstr = '\n'.join((
        f"Flood detected: {'YES' if results_dict['flood_pixels'] > 0 else 'NO'}",
        f"Flood area: {results_dict['area_km2']:.4f} km²",
        f"Flood percentage: {results_dict['flood_percentage']:.2f}%"
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
