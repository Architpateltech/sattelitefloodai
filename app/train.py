import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add the project root to the python path so it can find 'data', 'models', etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import Sen1Floods11Dataset
from models.unet import get_flood_segmentation_model
from training.losses import BCEDiceLoss
from training.train import train_model
from models.unet import get_flood_segmentation_model
from training.losses import BCEDiceLoss
from training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Flood Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="data/Sen1Floods11_HandLabeled", help="Path to Sen1Floods11 HandLabeled data")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory for models")
    parser.add_argument("--use_s1", action="store_true", help="Use Sentinel-1 SAR instead of Sentinel-2 Optical")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Datasets
    print("Loading datasets...")
    train_dataset = Sen1Floods11Dataset(args.data_dir, split="train", use_s1=args.use_s1)
    val_dataset = Sen1Floods11Dataset(args.data_dir, split="val", use_s1=args.use_s1)
    
    # Num workers logic
    import platform
    nw = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    
    # 2. Model
    print("Initializing U-Net ResNet50 model...")
    # S1 has 2 bands (VV, VH), S2 has 13 bands in Sen1Floods11
    in_channels = 2 if args.use_s1 else 13 
    model = get_flood_segmentation_model(encoder_name="resnet50", encoder_weights="imagenet", in_channels=in_channels, classes=1)
    model.to(device)
    
    # 3. Loss, Optimizer, Scheduler
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Train
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        save_dir=args.out_dir
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
