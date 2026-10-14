import logging
from typing import Optional

import cv2
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

def load_rgb_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image for display, handling multi-band .tif files."""
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(image_path) as src:
                num_bands = src.count
                if num_bands >= 4:
                    # S2: B4, B3, B2
                    r = src.read(4)
                    g = src.read(3)
                    b = src.read(2)
                    rgb = np.stack([b, g, r], axis=-1).astype(np.float32) # BGR for cv2
                elif num_bands == 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    rgb = np.stack([b, g, r], axis=-1).astype(np.float32)
                elif num_bands == 2:
                    # S1 dual pol
                    vv = src.read(1)
                    vh = src.read(2)
                    rgb = np.stack([vv, vh, vv], axis=-1).astype(np.float32)
                else:
                    band = src.read(1)
                    rgb = np.stack([band, band, band], axis=-1).astype(np.float32)
                
                # Simple percentile stretch for display
                p2, p98 = np.percentile(rgb, (2, 98))
                if p98 > p2:
                    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                else:
                    rgb = np.clip(rgb, 0, 1)
                
                return (rgb * 255).astype(np.uint8)
        else:
            return cv2.imread(image_path)
    except Exception as e:
        logger.error("load_rgb_image: failed to load %s: %s", image_path, e)
        return None



def create_overlay(image_path: str, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Blend a red highlight over the affected region defined by the mask.

    Args:
        image_path: Path to the original satellite image.
        mask: Binary mask (uint8) where 1 = affected pixels.

    Returns:
        Blended overlay image (BGR), or None if the image cannot be read.
    """
    image = load_rgb_image(image_path)
    if image is None:
        logger.error("create_overlay: cannot read image at '%s'", image_path)
        return None

    # Resize mask to match image dimensions if they differ
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    # Highlight water with cyan color (BGR: 255, 255, 0)
    overlay[mask == 1] = [255, 255, 0]

    # Blend the overlay with the original image
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # Find boundaries (contours) and draw them
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw red boundaries around the water (BGR: 0, 0, 255)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

    return result


def create_side_by_side(
    original_path: str,
    analyzed_image: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Stack the original and the analysed overlay image horizontally.

    Args:
        original_path: Path to the original satellite image.
        analyzed_image: Analysed/overlay image as a NumPy array (BGR).

    Returns:
        Horizontally stacked comparison image, or None on failure.
    """
    original = load_rgb_image(original_path)
    if original is None:
        logger.error("create_side_by_side: cannot read image at '%s'", original_path)
        return analyzed_image  # fall back to just returning the overlay

    if analyzed_image is None:
        return original

    # Normalise both images to the same dimensions
    height = min(original.shape[0], analyzed_image.shape[0])
    width = min(original.shape[1], analyzed_image.shape[1])

    original = cv2.resize(original, (width, height))
    analyzed_image = cv2.resize(analyzed_image, (width, height))

    comparison = np.hstack((original, analyzed_image))
    return comparison