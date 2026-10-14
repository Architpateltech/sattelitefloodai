import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Useful for segmentation tasks with class imbalance.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss (takes logits directly)
        bce_loss = self.bce(logits, targets)

        # Dice Loss (requires probabilities)
        probs = torch.sigmoid(logits)
        
        # Flatten predictions and targets
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculates segmentation metrics.
    
    Args:
        preds (torch.Tensor): Logits or probabilities (N, C, H, W).
        targets (torch.Tensor): Ground truth masks (N, C, H, W).
        threshold (float): Threshold for binary classification.
        
    Returns:
        dict: Dictionary containing IoU, Dice, Precision, Recall, F1, Pixel Accuracy.
    """
    # Convert logits to binary predictions if necessary
    if preds.max() > 1.0 or preds.min() < 0.0:
        probs = torch.sigmoid(preds)
    else:
        probs = preds
        
    preds_bin = (probs > threshold).float()
    
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)
    
    tp = (preds_bin * targets).sum()
    fp = ((preds_bin == 1) & (targets == 0)).sum()
    fn = ((preds_bin == 0) & (targets == 1)).sum()
    tn = ((preds_bin == 0) & (targets == 0)).sum()
    
    smooth = 1e-6
    
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + smooth)
    
    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "pixel_accuracy": pixel_acc.item()
    }
