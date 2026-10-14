import logging
from typing import Optional

import cv2
import numpy as np
import rasterio
import torch
import os
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class DisasterSegmentation:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Load the model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "best_flood_segmentation_model.pth")
        if os.path.exists(model_path):
            try:
                # We used resnet50, 13 channels for Sentinel-2, 1 class (flood)
                self.model = smp.Unet(
                    encoder_name="resnet50",
                    encoder_weights=None,
                    in_channels=13,
                    classes=1,
                )
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info("Segmentation model loaded successfully on device: %s", self.device)
            except Exception as e:
                logger.exception("Failed to load segmentation model: %s", e)
                self.model = None
        else:
            logger.warning("No trained segmentation model found at '%s'. Using empty mask fallback.", model_path)

    def create_mask(self, image_path: str) -> Optional[np.ndarray]:
        """
        Create a binary segmentation mask for the given satellite image using U-Net.

        Args:
            image_path: Path to the satellite image (.tif).

        Returns:
            Binary mask (uint8) where 1 = affected pixels, or None on failure.
        """
        try:
            # 1. Read the image with rasterio
            with rasterio.open(image_path) as src:
                image = src.read() # (C, H, W)
                
                if image.shape[0] != 13:
                    logger.error("Expected a 13-band Sentinel-2 image, got %d bands.", image.shape[0])
                    return np.zeros((src.height, src.width), dtype=np.uint8)

                height, width = src.height, src.width
                
            # 2. Preprocess (normalize)
            image = image.astype(np.float32)
            for c in range(image.shape[0]):
                p2 = np.percentile(image[c], 2)
                p98 = np.percentile(image[c], 98)
                image[c] = np.clip((image[c] - p2) / (p98 - p2 + 1e-8), 0, 1)

            # 3. Predict if model is loaded
            if self.model is not None:
                tensor = torch.from_numpy(image).unsqueeze(0).to(self.device) # (1, 13, H, W)
                with torch.no_grad():
                    output = self.model(tensor)
                    prob = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
                
                # Threshold at 0.5
                mask = (prob > 0.5).astype(np.uint8)
                return mask
            else:
                # Fallback empty mask
                return np.zeros((height, width), dtype=np.uint8)
                
        except Exception as exc:
            logger.exception("create_mask failed for image '%s': %s", image_path, exc)
            return None