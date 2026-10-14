import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

class Sen1Floods11Dataset(Dataset):
    """
    Dataset loader for Sen1Floods11 HandLabeled subset.
    """
    def __init__(self, data_dir, split="train", transform=None, use_s1=False):
        """
        Args:
            data_dir (str): Path to the Sen1Floods11_HandLabeled directory.
            split (str): 'train', 'val', or 'test'.
            transform: Albumentations transforms to apply.
            use_s1 (bool): If True, load Sentinel-1. Otherwise, load Sentinel-2.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.use_s1 = use_s1
        
        # In the HandLabeled subset, files are typically paired like:
        # S2Hand/India_12345_S2Hand.tif
        # LabelHand/India_12345_LabelHand.tif
        
        # Let's find all label files to use as our base index
        label_dir = os.path.join(data_dir, "LabelHand")
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*_LabelHand.tif")))
        
        # Very simple split logic based on index (in a real scenario, use official CSV splits)
        # 70% train, 15% val, 15% test
        n_total = len(self.label_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        if split == "train":
            self.label_files = self.label_files[:n_train]
        elif split == "val":
            self.label_files = self.label_files[n_train:n_train + n_val]
        else:
            self.label_files = self.label_files[n_train + n_val:]
            
        print(f"[{split.upper()}] Found {len(self.label_files)} images.")

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_path = self.label_files[idx]
        base_name = os.path.basename(label_path).replace("_LabelHand.tif", "")
        
        if self.use_s1:
            img_path = os.path.join(self.data_dir, "S1Hand", f"{base_name}_S1Hand.tif")
        else:
            img_path = os.path.join(self.data_dir, "S2Hand", f"{base_name}_S2Hand.tif")
            
        # 1. Read Image
        with rasterio.open(img_path) as src:
            image = src.read() # (C, H, W)
                
        # Normalize image to 0-1 range (simplified percentile stretch)
        image = image.astype(np.float32)
        for c in range(image.shape[0]):
            p2, p98 = np.percentile(image[c], (2, 98))
            image[c] = np.clip((image[c] - p2) / max(p98 - p2, 1e-6), 0, 1)
            
        # Move channel to last dimension for transforms
        image = np.transpose(image, (1, 2, 0))
            
        # 2. Read Label
        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.float32) # (H, W)
            
        # Sen1Floods11 labels: 1 = flood, 0 = non-flood, -1 = valid non-water, etc.
        # We binarize: anything strictly equal to 1 is flood, rest is background
        binary_mask = (mask == 1).astype(np.float32)
        
        # 3. Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
            
        # Ensure correct shapes for PyTorch: image (C, H, W), mask (1, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
        if isinstance(binary_mask, np.ndarray):
            binary_mask = torch.from_numpy(binary_mask).unsqueeze(0)
            
        return image, binary_mask
