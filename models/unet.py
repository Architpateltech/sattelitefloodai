import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_flood_segmentation_model(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1):
    """
    Creates a U-Net model with the specified encoder for flood segmentation.

    Args:
        encoder_name (str): Name of the encoder (e.g., 'resnet50').
        encoder_weights (str): Pre-trained weights to use (e.g., 'imagenet').
        in_channels (int): Number of input channels (e.g., 3 for RGB, 5 for RGB+SAR).
        classes (int): Number of output classes (1 for binary flood mask).

    Returns:
        nn.Module: The segmentation model.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model

if __name__ == "__main__":
    # Test model instantiation
    model = get_flood_segmentation_model()
    print("Model created successfully.")
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
