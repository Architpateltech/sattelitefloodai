from preprocessing.geospatial import calculate_flood_area_km2

def estimate_flood_area(binary_mask, transform, crs):
    """
    Estimates the flood area in square kilometers.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask where 1 is flood.
        transform (rasterio.Affine): Geotransform of the mask.
        crs (rasterio.crs.CRS): CRS of the mask.
        
    Returns:
        float: Area in km2.
        int: Total flood pixels.
        float: Percentage of image flooded.
    """
    flood_pixels = int(binary_mask.sum())
    total_pixels = binary_mask.size
    percentage = (flood_pixels / total_pixels) * 100.0
    
    area_km2 = calculate_flood_area_km2(flood_pixels, transform, crs)
    
    return area_km2, flood_pixels, percentage
