import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def create_heatmap(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Convert a binary mask into a coloured heatmap using the JET colormap.

    Args:
        mask: Binary mask (uint8) where 1 = affected pixels.

    Returns:
        BGR heatmap image, or None if mask is None.
    """
    if mask is None:
        logger.warning("create_heatmap: received None mask")
        return None

    heatmap = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap