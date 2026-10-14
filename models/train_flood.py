"""
================================================================================
SEN12-FLOOD TRAINING PIPELINE (Sentinel-2 RGB, flood / no-flood classification)
================================================================================
REAL DATASET STRUCTURE (as discovered by probing your extracted folder):

    sen12flood/
        S2list.json          <- SINGLE root-level JSON keyed by sequence ID
        S1list.json          <- S1 metadata (not used here)
        0/                   <- sequence folder (numeric or zero-padded)
        0001/
            S2_2018-12-18_B02.tif
            S2_2018-12-18_B03.tif
            S2_2018-12-18_B04.tif   <- filename: S2_{YYYY-MM-DD}_{band}.tif
            S2_2018-12-23_B04.tif
            ...                      (9 dates * 12 S2 bands = 108 S2 tifs)
        0004/
        ...

    S2list.json structure:
        {
          "0001": {
            "1": {"date": "2018-12-18", "FLOODING": false, "filename": "S2_2018-12-18"},
            "2": {"date": "2018-12-23", "FLOODING": false, ...},
            ...
          },
          "0004": {...},
          ...
        }

Pipeline:
  1. Load root S2list.json -> seq_id -> {date -> FLOODING label, filename}
  2. For each (seq_id, date), build exact path: {seq_dir}/{filename}_{band}.tif
  3. Percentile-stretch reflectance so ImageNet backbone can use it.
  4. Cache each sample as .npy once; subsequent epochs just np.load().
  5. Split train/val/test by SEQUENCE ID — prevents location leakage.

FIXES vs. original script:
  - Reads the real single-root S2list.json (not per-seq files that don't exist)
  - Uses exact filename prefix from JSON (no glob guessing)
  - num_workers=0 on Windows (rasterio + spawn workers = crash/freeze)
  - NpyCache: 3 rasterio.open() calls -> 1 np.load() per epoch (3-10x speedup)
  - Normalized class weights (prevents gradient explosion on severe imbalance)
  - T_max=0 guard in CosineAnnealingLR (prevents ZeroDivisionError at unfreeze)
  - cudnn.benchmark=True (~15-25% free GPU throughput for fixed 224x224)
  - Best model saved to disk, not deepcopy (~100 MB RAM per improvement epoch)
  - drop_last conditionalized (prevents discarding rare flood samples)
  - Separated train/eval epoch functions

INSTALL:
    pip install torch torchvision rasterio scikit-learn tqdm numpy

RUN (from satellitedisasterai/):
    python models/train_flood.py --data_dir sen12flood --epochs 30
================================================================================"""

import argparse
import glob
import json
import os
import platform
import random
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# ------------------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# 1. INDEX THE DATASET
#
# Real structure (probed from your extracted folder):
#   sen12flood/
#       S2list.json          <- single root-level JSON, keyed by seq_id
#       0001/
#           S2_2018-12-18_B04.tif   <- S2_{YYYY-MM-DD}_{band}.tif
#           S2_2018-12-18_B03.tif
#           S2_2018-12-18_B02.tif
#           ...
# ------------------------------------------------------------------------------
RGB_BANDS = ("B04", "B03", "B02")  # red, green, blue


def find_band_file_exact(seq_dir: str, filename_prefix: str, band: str):
    """
    Build the exact path using the 'filename' field from S2list.json.
    Real filename format: S2_{YYYY-MM-DD}_{band}.tif
    e.g. filename_prefix='S2_2018-12-18'  band='B04'
         -> sen12flood/0001/S2_2018-12-18_B04.tif

    Falls back to a glob only if the exact path doesn't exist (handles
    edge cases where the filename key is absent from older JSON entries).
    """
    # 1. Exact path using the JSON 'filename' field
    exact = os.path.join(seq_dir, f"{filename_prefix}_{band}.tif")
    if os.path.exists(exact):
        return exact

    # 2. Glob fallback using the date portion (extracted from prefix)
    #    prefix is 'S2_YYYY-MM-DD' -> date part is after 'S2_'
    date_part = filename_prefix.replace("S2_", "")  # e.g. '2018-12-18'
    pattern = os.path.join(seq_dir, f"S2_{date_part}_{band}.tif")
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[find_band_file_exact] Ambiguous: {len(matches)} matches "
              f"for {band} prefix={filename_prefix} in {seq_dir}")
    return None


def index_dataset(data_dir: str):
    """
    Reads the single root-level S2list.json from data_dir, then for each
    (seq_id, acquisition) builds a sample dict:
        {
          'seq_id': '0001',
          'date':   '2018-12-18',
          'label':  0,
          'paths':  {'B04': '...tif', 'B03': '...tif', 'B02': '...tif'}
        }

    Skips acquisitions that:
      - Have FULL-DATA-COVERAGE == False (cloud/no-data mask)
      - Are missing any of the three RGB band files on disk
    """
    json_path = os.path.join(data_dir, "S2list.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"S2list.json not found at {json_path}\n"
            f"Expected a single root-level S2list.json in your data_dir."
        )

    with open(json_path, "r") as f:
        meta = json.load(f)

    samples, skipped = [], []

    for seq_id, acquisitions in sorted(meta.items()):
        if not isinstance(acquisitions, dict):
            continue
        seq_dir = os.path.join(data_dir, seq_id)
        if not os.path.isdir(seq_dir):
            skipped.append((seq_id, "sequence directory not found on disk"))
            continue

        for acq_key, acq in sorted(acquisitions.items()):
            if not isinstance(acq, dict):
                continue

            date  = acq.get("date") or acq.get("Date")
            label = acq.get("FLOODING")
            full_coverage = acq.get("FULL-DATA-COVERAGE", True)  # skip cloud-covered

            if date is None or label is None:
                continue
            if not full_coverage:          # skip acquisitions with data gaps
                continue

            # Prefer the exact filename from JSON; fall back to constructed name
            filename_prefix = acq.get("filename") or f"S2_{date}"

            paths, missing = {}, False
            for band in RGB_BANDS:
                p = find_band_file_exact(seq_dir, filename_prefix, band)
                if p is None:
                    missing = True
                    skipped.append((seq_id, f"date {date}: {band} not found "
                                    f"(prefix={filename_prefix})"))
                    break
                paths[band] = p

            if not missing:
                samples.append({
                    "seq_id": seq_id,
                    "date":   date,
                    "label":  int(bool(label)),
                    "paths":  paths,
                })

    if skipped:
        print(f"[index_dataset] Skipped {len(skipped)} entries (first 5):")
        for s in skipped[:5]:
            print("   ", s)

    return samples


def train_test_split_by_sequence(samples: list, test_frac: float = 0.15,
                                  seed: int = 42):
    """
    Hold out a fraction of sequences as a test set.
    Unlike a random image split, this ensures no geographic leakage.
    """
    seq_ids = sorted(set(s["seq_id"] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(seq_ids)
    n_test = max(1, int(len(seq_ids) * test_frac))
    test_ids = set(seq_ids[:n_test])
    train_all = [s for s in samples if s["seq_id"] not in test_ids]
    test_s    = [s for s in samples if s["seq_id"] in test_ids]
    return train_all, test_s


# ------------------------------------------------------------------------------
# 2. NpyCache — pre-process GeoTIFFs once, then load fast .npy on every epoch
# ------------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def _percentile_stretch(arr: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def build_npy_cache(samples: list, cache_dir: str) -> list:
    """
    For each sample, reads the three GeoTIFF bands, percentile-stretches them,
    and saves a (3, H, W) uint8 .npy file to cache_dir.
    Returns an updated samples list with a 'npy_path' key added.
    On subsequent runs the existing .npy is reused — no re-reading GeoTIFFs.
    This eliminates the largest I/O bottleneck: 3 rasterio.open() calls per
    sample per epoch become a single np.load() call.
    """
    os.makedirs(cache_dir, exist_ok=True)
    updated = []
    already_cached = sum(
        1 for s in samples
        if os.path.exists(os.path.join(cache_dir, f"{s['seq_id']}_{s['date']}.npy"))
    )
    print(f"  {already_cached}/{len(samples)} samples already cached in {cache_dir}")

    for s in tqdm(samples, desc="  Caching .npy"):
        npy_name = f"{s['seq_id']}_{s['date']}.npy"
        npy_path = os.path.join(cache_dir, npy_name)
        if not os.path.exists(npy_path):
            bands = []
            for band in RGB_BANDS:
                with rasterio.open(s["paths"][band]) as src:
                    arr = src.read(1).astype(np.float32)
                bands.append(_percentile_stretch(arr))
            np.save(npy_path, np.stack(bands, axis=0))  # (3, H, W) channel-first
        updated.append({**s, "npy_path": npy_path})
    return updated


# ------------------------------------------------------------------------------
# 3. DATASET — loads from .npy cache, applies torchvision transforms
# ------------------------------------------------------------------------------
class Sen12FloodDataset(Dataset):
    """
    Loads pre-cached (3, H, W) uint8 numpy arrays and applies augmentation.
    __getitem__ performs only np.load() + transform — no GeoTIFF I/O at runtime.
    """

    def __init__(self, samples: list, augment: bool):
        self.samples = samples
        if augment:
            self.tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),   # no canonical 'up' in satellite
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # (3, H, W) uint8 -> transpose to (H, W, 3) for ToPILImage
        rgb = np.load(s["npy_path"]).transpose(1, 2, 0)
        img = self.tfms(rgb)
        return img, s["label"]


# ------------------------------------------------------------------------------
# 3. SPLIT BY SEQUENCE ID (prevents location leakage between train/val)
# ------------------------------------------------------------------------------
def split_by_sequence(samples, val_frac=0.15, seed=42):
    seq_ids = sorted(set(s["seq_id"] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(seq_ids)
    n_val = max(1, int(len(seq_ids) * val_frac))
    val_ids = set(seq_ids[:n_val])
    train_samples = [s for s in samples if s["seq_id"] not in val_ids]
    val_samples = [s for s in samples if s["seq_id"] in val_ids]
    return train_samples, val_samples


# ------------------------------------------------------------------------------
# 4. MODEL: pretrained ResNet50, standard 3-channel input
# ------------------------------------------------------------------------------
def build_model(num_classes=2, freeze_backbone=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    if freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
    return model


# ------------------------------------------------------------------------------
# 5. TRAIN / EVAL LOOPS — separated so optimizer is never passed into eval
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                             enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="  val  "):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type,
                             enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, class_weights, device,
                epochs, lr, patience, unfreeze_at_epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, "flood_model.pth")

    model = model.to(device)

    # Normalized class weights: keeps loss magnitude stable regardless of
    # imbalance ratio (fix: raw 1/count weights can cause gradient explosions)
    w = torch.tensor(class_weights, dtype=torch.float32).to(device)
    w = w / w.sum() * len(w)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc, no_improve = 0.0, 0

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_at_epoch:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), lr=lr / 10, weight_decay=1e-4
            )
            # Guard against T_max=0 (ZeroDivisionError in PyTorch scheduler)
            remaining = max(1, epochs - epoch)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining
            )
            print(">> Unfroze backbone; lowered LR for fine-tuning.")

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            # Save directly to disk instead of deepcopy (~100 MB per improvement epoch)
            torch.save(model.state_dict(), best_ckpt)
            print(f"   checkpointed  (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Best val accuracy: {best_acc:.4f}")
    return modelel


@torch.no_grad()
def evaluate(model, loader, device, class_names=("no_flood", "flood")):
    model.eval()
    preds, labels = [], []
    for images, y in tqdm(loader, desc="Test eval"):
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    print("\n=== Classification Report (test set) ===")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(labels, preds))


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def _safe_num_workers() -> int:
    """
    On Windows, rasterio (GDAL C-extension) is not safely picklable across
    spawn-based multiprocessing workers. Use 0 on Windows, up to 4 on Linux.
    After NpyCache is built, workers only do np.load() — fast even at 0.
    """
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 1)


def main():
    ap = argparse.ArgumentParser(
        description="SEN12-FLOOD binary flood classifier (Sentinel-2 RGB)"
    )
    ap.add_argument(
        "--data_dir", required=True,
        help="Path to the flat sen12flood/ root containing S2list.json + sequence dirs",
    )
    ap.add_argument("--cache_dir",          type=str,   default=".npy_cache",
                    help="Directory to store preprocessed .npy files (created if absent)")
    ap.add_argument("--epochs",             type=int,   default=30)
    ap.add_argument("--batch_size",         type=int,   default=32)
    ap.add_argument("--lr",                 type=float, default=3e-4)
    ap.add_argument("--patience",           type=int,   default=5)
    ap.add_argument("--val_frac",           type=float, default=0.15,
                    help="Fraction of sequences used for validation")
    ap.add_argument("--test_frac",          type=float, default=0.15,
                    help="Fraction of sequences held out as test set")
    ap.add_argument("--unfreeze_at_epoch",  type=int,   default=5)
    ap.add_argument("--out_dir",            type=str,   default="checkpoints")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  (platform: {platform.system()})")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # ~15-25% free throughput for 224x224

    # ── Index the full flat dataset ────────────────────────────────────────
    print(f"\nIndexing dataset at: {args.data_dir}")
    all_samples = index_dataset(args.data_dir)
    n_seqs = len(set(s["seq_id"] for s in all_samples))
    n_flood = sum(s["label"] for s in all_samples)
    print(f"  {len(all_samples)} usable acquisitions across {n_seqs} sequences")
    print(f"  Flood={n_flood}  No-flood={len(all_samples)-n_flood}")

    if not all_samples:
        print(
            "\n[ERROR] No samples indexed. Possible causes:\n"
            "  - Wrong --data_dir (should point to the folder containing S2list.json)\n"
            "  - S2list.json has a different structure than expected\n"
            "  - All entries have FULL-DATA-COVERAGE=False\n"
            "Run: python models/preflight.py to diagnose."
        )
        return

    # ── Sequence-aware train / val / test split ────────────────────────────
    train_val, test_samps = train_test_split_by_sequence(
        all_samples, test_frac=args.test_frac
    )
    train_samps, val_samps = split_by_sequence(
        train_val, val_frac=args.val_frac
    )
    print(
        f"\nSplit (by sequence, no geographic leakage):\n"
        f"  Train : {len(train_samps)} images\n"
        f"  Val   : {len(val_samps)} images\n"
        f"  Test  : {len(test_samps)} images"
    )

    # ── Build .npy cache (one-time, reused on every subsequent run) ────────
    print(f"\nBuilding / verifying .npy cache -> {args.cache_dir}")
    train_samps = build_npy_cache(train_samps, os.path.join(args.cache_dir, "train"))
    val_samps   = build_npy_cache(val_samps,   os.path.join(args.cache_dir, "val"))
    test_samps  = build_npy_cache(test_samps,  os.path.join(args.cache_dir, "test"))

    # ── Normalized class weights ───────────────────────────────────────────
    label_arr = np.array([s["label"] for s in train_samps])
    counts    = np.bincount(label_arr, minlength=2)
    raw_w     = 1.0 / np.maximum(counts, 1)
    class_weights = raw_w / raw_w.sum() * len(raw_w)   # sum = num_classes
    print(f"\nClass counts  (no_flood={counts[0]}, flood={counts[1]})")
    print(f"Class weights {class_weights.tolist()}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    nw = _safe_num_workers()
    print(f"DataLoader num_workers={nw}")

    train_ds = Sen12FloodDataset(train_samps, augment=True)
    val_ds   = Sen12FloodDataset(val_samps,   augment=False)
    test_ds  = Sen12FloodDataset(test_samps,  augment=False)

    do_drop = len(train_ds) > 2 * args.batch_size

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        drop_last=do_drop, persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw,
    )

    # ── Build & train ──────────────────────────────────────────────────────
    model = build_model(num_classes=2, freeze_backbone=True)
    model = train_model(
        model, train_loader, val_loader, class_weights, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        unfreeze_at_epoch=args.unfreeze_at_epoch, out_dir=args.out_dir,
    )

    # ── Final evaluation ───────────────────────────────────────────────────
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()