import torch
from tqdm import tqdm
from .losses import calculate_metrics

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """
    Validates the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    metrics_sum = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "f1": 0, "pixel_accuracy": 0}
    
    pbar = tqdm(dataloader, desc="Validation")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item()
        
        # Calculate metrics for the batch
        batch_metrics = calculate_metrics(outputs, masks)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
            
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{batch_metrics['iou']:.4f}"})
        
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    epoch_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    
    return epoch_loss, epoch_metrics
