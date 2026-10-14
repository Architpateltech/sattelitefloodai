import logging
from typing import Optional

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class AreaCalculator:

    def __init__(self, default_pixel_size: float = 10.0):
        """
        Args:
            default_pixel_size: Ground resolution in metres per pixel if not available.
        """
        self.default_pixel_size = default_pixel_size

    def calculate(self, mask: Optional[np.ndarray], image_path: Optional[str] = None) -> float:
        """
        Convert an affected-pixel mask to an area in km².

        Args:
            mask: Binary mask (uint8) where 1 = affected pixels.
            image_path: Optional path to the satellite image to extract real pixel size.

        Returns:
            Affected area in km², rounded to 2 decimal places.
        """
        if mask is None:
            logger.warning("AreaCalculator.calculate received a None mask — returning 0.0")
            return 0.0

        pixel_size = self.default_pixel_size

        if image_path and image_path.lower().endswith(('.tif', '.tiff')):
            try:
                with rasterio.open(image_path) as src:
                    # src.transform[0] is pixel width, src.transform[4] is pixel height (usually negative)
                    pixel_size = abs(src.transform[0])
                    
                    # If the pixel size is extremely small (e.g., < 1.0), the CRS is likely Geographic (degrees)
                    # We can convert degrees to approximate meters at the equator (1 degree ≈ 111,320 meters)
                    if pixel_size < 1.0:
                        pixel_size = pixel_size * 111320.0
                        logger.info("Converted geographic pixel size to approximate meters: %.2fm", pixel_size)
                        
            except Exception as e:
                logger.error("Failed to read pixel size from %s: %s", image_path, e)

        affected_pixels = int(np.sum(mask == 1))
        area_m2 = affected_pixels * (pixel_size ** 2)
        area_km2 = area_m2 / 1_000_000
        return round(area_km2, 2)