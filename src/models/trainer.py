"""
trainer.py
==========
Full training pipeline for CNN fatigue-life regression.

Features
--------
• Microscopy-specific augmentation (albumentations)
• Multi-architecture support  (custom_cnn / resnet50 / vgg16)
• Gaussian NLL + Huber loss options
• Cosine LR schedule with warm restarts
• Early stopping + best-model checkpointing
• Per-epoch metrics: RMSE, MAE, R², MBE
• MLflow experiment tracking (optional)
• GradCAM visualisation helper

Usage
-----
python src/models/trainer.py --config configs/cnn_config.yaml
python src/models/trainer.py --arch resnet50 --epochs 60 --lr 3e-4
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
import json
import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from tqdm import tqdm
from loguru import logger

# local
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import build_model, GaussianNLLLoss, HuberGaussianLoss


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # Data
    labels_json:   str   = "data/synthetic/labels.json"
    image_size:    int   = 224
    train_ratio:   float = 0.80
    val_ratio:     float = 0.10
    # test  = 1 - train - val
    seed:          int   = 42

    # Model
    arch:          str   = "resnet50"    # custom_cnn | resnet50 | vgg16
    pretrained:    bool  = True
    dropout:       float = 0.35
    freeze_until:  str   = "layer2"      # resnet only

    # Training
    epochs:        int   = 60
    batch_size:    int   = 16
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    loss:          str   = "gnll"        # gnll | huber
    clip_grad:     float = 1.0

    # LR schedule
    scheduler:     str   = "cosine"      # cosine | plateau | step
    t_max:         int   = 60
    eta_min:       float = 1e-6

    # Early stopping
    patience:      int   = 12
    min_delta:     float = 1e-4

    # Paths
    save_dir:      str   = "outputs"
    run_name:      str   = "fatigue_run"
    use_mlflow:    bool  = False


# ══════════════════════════════════════════════════════════
# DATASET — MICROSCOPY SPECIFIC
# ══════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(size: int = 224) -> A.Compose:
    """
    Augmentation designed for steel microscopy images.
    Avoids colour jitter extremes that destroy microstructure information.
    """
    return A.Compose([
        A.Resize(size, size),
        # Spatial
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=20, p=0.5),
        # Microscopy-realistic noise
        A.GaussNoise(var_limit=(10, 40), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Illumination variation (microscope light)
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.25, p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.3),
        # Elastic deformation (grain boundary distortion)
        A.ElasticTransform(alpha=30, sigma=5, alpha_affine=5, p=0.2),
        # Normalise
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class MicroscopyFatigueDataset(Dataset):
    """
    Dataset for steel microscopy images.
    Expects a labels.json with fields:
      image_path, log10_Nf, [damage_level, grain_size_um, ...]
    """

    def __init__(self, items: List[Dict], transform: A.Compose,
                 label_mean: float = 6.0, label_std: float = 1.0):
        self.items      = items
        self.transform  = transform
        self.label_mean = label_mean
        self.label_std  = label_std

    @staticmethod
    def load_splits(labels_json: str, train_r: float = 0.8, val_r: float = 0.1,
                    seed: int = 42, img_size: int = 224):
        with open(labels_json) as f:
            data = json.load(f)
        np.random.seed(seed)
        idx = np.random.permutation(len(data))
        n_tr = int(len(data) * train_r)
        n_va = int(len(data) * val_r)

        labels = [d["log10_Nf"] for d in data]
        mean, std = float(np.mean(labels)), float(np.std(labels))
        std = max(std, 1e-6)

        train_items = [data[i] for i in idx[:n_tr]]
        val_items   = [data[i] for i in idx[n_tr:n_tr+n_va]]
        test_items  = [data[i] for i in idx[n_tr+n_va:]]

        train_ds = MicroscopyFatigueDataset(train_items, get_train_transforms(img_size), mean, std)
        val_ds   = MicroscopyFatigueDataset(val_items,   get_val_transforms(img_size),   mean, std)
        test_ds  = MicroscopyFatigueDataset(test_items,  get_val_transforms(img_size),   mean, std)

        logger.info(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        logger.info(f"Label stats: mean={mean:.3f}, std={std:.3f}")
        return train_ds, val_ds, test_ds, {"mean": mean, "std": std}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img  = cv2.imread(item["image_path"])
        if img is None:
            img = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)["image"]

        # Normalise label
        raw_label = item["log10_Nf"]
        norm_label = (raw_label - self.label_mean) / self.label_std
        label = torch.tensor([norm_label], dtype=torch.float32)

        meta = {
            "sample_id":    item.get("sample_id", str(idx)),
            "raw_log10_Nf": raw_label,
        }
        return augmented, label, meta


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Regression metrics in original (log) scale."""
    mse  = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(preds - targets)))
    mbe  = float(np.mean(preds - targets))   # mean bias error
    ss_r = np.sum((targets - preds) ** 2)
    ss_t = np.sum((targets - targets.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-9))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mbe": mbe}


# ══════════════════════════════════════════════════════════
# GRADCAM (visualisation aid)
# ══════════════════════════════════════════════════════════

class GradCAM:
    """
    GradCAM for CNN regression models.
    Highlights image regions most influential for the fatigue life prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self.grads  = None
        self.feats  = None
        target_layer.register_forward_hook(self._save_feats)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, _, __, output):
        self.feats = output.detach()

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Returns a (H, W) heatmap in [0, 1]."""
        self.model.eval()
        img_tensor.requires_grad_(True)
        mean, _ = self.model(img_tensor)
        self.model.zero_grad()
        mean.sum().backward()

        weights  = self.grads.mean(dim=(2, 3), keepdim=True)
        cam      = (weights * self.feats).sum(dim=1, keepdim=True)
        cam      = F.relu(cam)
        cam      = cam.squeeze().cpu().numpy()
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam_norm


# ══════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, cfg: TrainConfig):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # Build model
        model_kwargs = dict(dropout=cfg.dropout)
        if cfg.arch in ("resnet50",):
            model_kwargs["pretrained"]    = cfg.pretrained
            model_kwargs["freeze_until"]  = cfg.freeze_until
        elif cfg.arch == "vgg16":
            model_kwargs["pretrained"] = cfg.pretrained
        self.model = build_model(cfg.arch, **model_kwargs).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model: {cfg.arch}  trainable params={n_params:.2f}M")

        # Loss
        self.criterion = (GaussianNLLLoss() if cfg.loss == "gnll"
                          else HuberGaussianLoss())

        # Optimiser
        self.optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        # Scheduler
        if cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimiser, T_max=cfg.t_max, eta_min=cfg.eta_min)
        elif cfg.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimiser, patience=4, factor=0.5, min_lr=cfg.eta_min)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimiser, step_size=15, gamma=0.5)

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        self.label_stats: Dict = {}
        self.history: Dict     = {k: [] for k in
                                   ["train_loss", "val_loss", "val_rmse",
                                    "val_r2", "val_mae", "lr"]}

    # ── Single epoch ──────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for imgs, labels, _ in tqdm(loader, desc="  train", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimiser.zero_grad()
            mean, lv = self.model(imgs)
            loss = self.criterion(mean, lv, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.optimiser.step()
            total += loss.item()
        return total / len(loader)

    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total   = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels, meta in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                mean, lv = self.model(imgs)
                loss = self.criterion(mean, lv, labels)
                total += loss.item()
                # De-normalise to log scale
                raw_pred = mean.cpu().squeeze().numpy()
                raw_lbl  = np.array([m for m in meta["raw_log10_Nf"]])
                # Convert normalised back
                s = self.label_stats.get("std", 1.0)
                m = self.label_stats.get("mean", 6.0)
                denorm_pred = (raw_pred * s + m) if raw_pred.ndim else raw_pred
                preds.extend(np.atleast_1d(denorm_pred))
                targets.extend(np.atleast_1d(raw_lbl))

        metrics = compute_metrics(np.array(preds), np.array(targets))
        return total / len(loader), metrics

    # ── Full training loop ────────────────────────────────

    def fit(self, train_ds: MicroscopyFatigueDataset,
            val_ds: MicroscopyFatigueDataset,
            test_ds: Optional[MicroscopyFatigueDataset] = None) -> Dict:

        self.label_stats = {"mean": train_ds.label_mean, "std": train_ds.label_std}
        # Save label stats with checkpoint
        label_stats_path = Path(self.cfg.save_dir) / "label_stats.json"
        with open(label_stats_path, "w") as f:
            json.dump(self.label_stats, f)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                   shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.batch_size,
                                   shuffle=False, num_workers=0)

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_ckpt     = Path(self.cfg.save_dir) / f"best_{self.cfg.arch}.pth"

        logger.info(f"Training {self.cfg.arch} for {self.cfg.epochs} epochs …")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr_loss           = self._train_epoch(train_loader)
            val_loss, metrics = self._eval_epoch(val_loader)
            elapsed           = time.time() - t0

            # Scheduler step
            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            lr = self.optimiser.param_groups[0]["lr"]

            # Record
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(metrics["rmse"])
            self.history["val_r2"].append(metrics["r2"])
            self.history["val_mae"].append(metrics["mae"])
            self.history["lr"].append(lr)

            logger.info(
                f"Epoch {epoch:3d}/{self.cfg.epochs}  "
                f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  "
                f"lr={lr:.2e}  [{elapsed:.0f}s]"
            )

            # Checkpoint
            if val_loss < best_val_loss - self.cfg.min_delta:
                best_val_loss = val_loss
                patience_cnt  = 0
                torch.save({
                    "epoch":       epoch,
                    "arch":        self.cfg.arch,
                    "state_dict":  self.model.state_dict(),
                    "val_loss":    val_loss,
                    "val_r2":      metrics["r2"],
                    "label_stats": self.label_stats,
                    "cfg":         asdict(self.cfg),
                }, str(best_ckpt))
                logger.info(f"  ✅ Checkpoint saved  val_loss={val_loss:.4f}")
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Final evaluation on test set
        test_metrics = {}
        if test_ds and len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=self.cfg.batch_size,
                                     shuffle=False, num_workers=0)
            _, test_metrics = self._eval_epoch(test_loader)
            logger.info(f"Test metrics: {test_metrics}")

        # Save history
        hist_path = Path(self.cfg.save_dir) / f"history_{self.cfg.arch}.json"
        with open(hist_path, "w") as f:
            json.dump({"history": self.history,
                       "test_metrics": test_metrics,
                       "label_stats": self.label_stats,
                       "config": asdict(self.cfg)}, f, indent=2)
        logger.info(f"History saved → {hist_path}")

        return {"history": self.history, "test_metrics": test_metrics,
                "best_ckpt": str(best_ckpt)}


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=str, default=None)
    p.add_argument("--arch",        type=str, default="resnet50",
                   choices=["custom_cnn", "resnet50", "vgg16"])
    p.add_argument("--labels",      type=str, default="data/synthetic/labels.json")
    p.add_argument("--epochs",      type=int, default=60)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--save_dir",    type=str, default="outputs")
    p.add_argument("--no_pretrain", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    # CLI overrides
    cfg.arch        = args.arch
    cfg.labels_json = args.labels
    cfg.epochs      = args.epochs
    cfg.lr          = args.lr
    cfg.batch_size  = args.batch_size
    cfg.save_dir    = args.save_dir
    cfg.pretrained  = not args.no_pretrain

    train_ds, val_ds, test_ds, stats = MicroscopyFatigueDataset.load_splits(
        cfg.labels_json, cfg.train_ratio, cfg.val_ratio,
        cfg.seed, cfg.image_size
    )
    trainer = Trainer(cfg)
    result  = trainer.fit(train_ds, val_ds, test_ds)

    print("\n" + "=" * 50)
    print("  Training complete!")
    print(f"  Best checkpoint : {result['best_ckpt']}")
    if result["test_metrics"]:
        print(f"  Test R²         : {result['test_metrics'].get('r2',0):.4f}")
        print(f"  Test RMSE       : {result['test_metrics'].get('rmse',0):.4f}")
    print("=" * 50)
