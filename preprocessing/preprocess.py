import rasterio
import numpy as np

def load_geotiff(filepath):
    """
    Loads a GeoTIFF file.
    
    Args:
        filepath (str): Path to the GeoTIFF file.
        
    Returns:
        tuple: (numpy.ndarray, dict, rasterio.Affine) 
               The image data (C, H, W), the metadata profile, and the transform.
    """
    with rasterio.open(filepath) as src:
        data = src.read()
        profile = src.profile
        transform = src.transform
    return data, profile, transform

def normalize_image(image_array, method="minmax"):
    """
    Normalizes an image array (C, H, W).
    
    Args:
        image_array (numpy.ndarray): The input image array.
        method (str): Normalization method ('minmax', 'percentile', 'imagenet').
        
    Returns:
        numpy.ndarray: Normalized image array as float32.
    """
    img = image_array.astype(np.float32)
    
    if method == "minmax":
        min_val = np.nanmin(img)
        max_val = np.nanmax(img)
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
    elif method == "percentile":
        # 2nd and 98th percentile stretch per channel
        for c in range(img.shape[0]):
            c_data = img[c]
            p2, p98 = np.nanpercentile(c_data, (2, 98))
            c_data = np.clip((c_data - p2) / max(p98 - p2, 1e-6), 0, 1)
            img[c] = c_data
    elif method == "imagenet":
        # Assumes input is already RGB and scaled 0-1
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = (img - mean) / std
        
    return img

def stack_bands(band_paths):
    """
    Reads and stacks multiple single-band GeoTIFFs into a multi-channel array.
    
    Args:
        band_paths (list): List of paths to single-band GeoTIFFs.
        
    Returns:
        tuple: (stacked_array (C, H, W), profile, transform)
    """
    bands = []
    profile = None
    transform = None
    
    for i, path in enumerate(band_paths):
        with rasterio.open(path) as src:
            if i == 0:
                profile = src.profile
                transform = src.transform
            bands.append(src.read(1))
            
    stacked = np.stack(bands, axis=0)
    return stacked, profile, transform
