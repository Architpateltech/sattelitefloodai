import os
import torch
from tqdm import tqdm
from .losses import calculate_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    metrics_sum = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "f1": 0, "pixel_accuracy": 0}
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, patience=5, save_dir="checkpoints"):
    """
    Full training loop with validation and early stopping.
    """
    os.makedirs(save_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device.type else None
    
    best_val_iou = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_metrics['iou']:.4f} | Train Dice: {train_metrics['dice']:.4f}")
        
        # Validate (Imported dynamically to avoid circular dependency if validate.py imports train_model)
        from .validate import validate_epoch
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        
        # Step scheduler
        if scheduler:
            scheduler.step()
            
        # Early Stopping & Checkpointing
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            epochs_no_improve = 0
            save_path = os.path.join(save_dir, "best_flood_segmentation_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to {save_path} (IoU: {best_val_iou:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    return model
