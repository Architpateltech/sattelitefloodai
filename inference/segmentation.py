import torch
import numpy as np
import cv2

class FloodSegmenter:
    """
    Handles loading the trained segmentation model and running inference on preprocessed tensors.
    """
    def __init__(self, model_path, device="cpu"):
        self.device = device
        # Import inside to avoid circular dependency
        from models.unet import get_flood_segmentation_model
        
        self.model = get_flood_segmentation_model()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict_mask(self, input_tensor, threshold=0.5):
        """
        Predicts the binary flood mask.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W).
            threshold (float): Threshold for turning probabilities into binary mask.
            
        Returns:
            numpy.ndarray: Binary mask (H, W) where 1 is flood, 0 is no flood.
            numpy.ndarray: Probability mask (H, W) for confidence mapping.
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        binary_mask = (probs > threshold).astype(np.uint8)
        return binary_mask, probs
