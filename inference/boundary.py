import rasterio.features
from shapely.geometry import shape, MultiPolygon
import numpy as np
import cv2

def extract_flood_boundaries(binary_mask, transform):
    """
    Extracts flood boundaries as Shapely polygons from a binary mask.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask (H, W) where 1 is flood.
        transform (rasterio.Affine): Geotransform for the mask.
        
    Returns:
        shapely.geometry.MultiPolygon: Polygons representing the flooded areas.
        list: GeoJSON features.
    """
    # Morphological closing to clean up small holes and noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Extract shapes using rasterio features
    shapes = rasterio.features.shapes(cleaned_mask, transform=transform)
    
    polygons = []
    geojson_features = []
    
    for geom, value in shapes:
        if value == 1: # Flood pixel
            poly = shape(geom)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
                geojson_features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"class": "flood"}
                })
                
    multi_poly = MultiPolygon(polygons)
    return multi_poly, geojson_features
