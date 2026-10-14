import os
import logging

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

CLASSES = ["Flood", "No Flood"]


class DisasterPredictor:

    def __init__(self):
        logger.info("Initializing Disaster AI predictor...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        model_path = os.path.join(os.path.dirname(__file__), "flood_model.pth")

        if os.path.exists(model_path):
            logger.info("Loading trained model from '%s'...", model_path)
            try:
                self.model = models.resnet18(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully on device: %s", self.device)
            except Exception as exc:
                logger.exception("Failed to load model weights — running in demo mode.")
                self.model = None
        else:
            logger.warning("No trained model found at '%s'. Running in demo mode.", model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path: str) -> dict:
        """
        Predict the disaster type for a satellite image.

        Args:
            image_path: Path to the image file.

        Returns:
            dict with keys: 'disaster', 'confidence', 'area', 'severity'
        """
        if self.model is None:
            # Demo / stub mode
            logger.info("Demo mode: returning stub prediction.")
            return {
                "disaster": "Flood",
                "confidence": 95.5,
                "area": "Not calculated",
                "severity": "Pending",
            }

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.exception("Cannot open image '%s'", image_path)
            raise

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probability = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probability, dim=1)

        return {
            "disaster": CLASSES[prediction.item()],
            "confidence": round(confidence.item() * 100, 2),
            "area": "Not calculated",
            "severity": "Pending",
        }