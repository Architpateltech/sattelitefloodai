import torch
import numpy as np
from preprocessing.preprocess import load_geotiff, normalize_image
from .segmentation import FloodSegmenter
from .boundary import extract_flood_boundaries
from .area import estimate_flood_area

def run_prediction(image_path, model_path, device="cpu"):
    """
    End-to-end inference for a single satellite image.
    
    Args:
        image_path (str): Path to GeoTIFF image.
        model_path (str): Path to trained model weights.
        device (str): Device to run on.
        
    Returns:
        dict: Results containing mask, area, etc.
    """
    # 1. Load image and metadata
    data, profile, transform = load_geotiff(image_path)
    crs = profile['crs']
    
    # 2. Preprocess (normalize)
    norm_data = normalize_image(data, method="percentile")
    input_tensor = torch.from_numpy(norm_data).unsqueeze(0).to(device)
    
    # 3. Segmentation
    segmenter = FloodSegmenter(model_path, device)
    binary_mask, prob_mask = segmenter.predict_mask(input_tensor)
    
    # 4. Boundary extraction
    multi_poly, geojson = extract_flood_boundaries(binary_mask, transform)
    
    # 5. Area calculation
    area_km2, flood_pixels, percentage = estimate_flood_area(binary_mask, transform, crs)
    
    return {
        "original_image": norm_data,
        "binary_mask": binary_mask,
        "prob_mask": prob_mask,
        "boundaries": multi_poly,
        "geojson": geojson,
        "area_km2": area_km2,
        "flood_pixels": flood_pixels,
        "flood_percentage": percentage,
        "profile": profile
    }
