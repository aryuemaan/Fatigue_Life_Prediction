"""
transfer_model.py
-----------------
ResNet50-based transfer learning model for fatigue life regression.
Predicts log10(N_f) — log cycles to failure.

Architecture:
  ResNet50 (ImageNet pretrained) → Custom regression head
  Output: [log_Nf, log_Nf_lower, log_Nf_upper] (point + CI)

Usage:
    from src.models.transfer_model import FatigueResNet, FatigueDataset
    model = FatigueResNet()
    trainer = ModelTrainer(model, config)
    trainer.train(train_dataset, val_dataset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
import json
import os
from loguru import logger
from tqdm import tqdm


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class FatigueDataset(Dataset):
    """
    PyTorch Dataset for steel fatigue images.
    Expects labels.json with 'image_path' and 'log10_Nf' keys.
    """

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, labels_path: str, split: str = "train",
                 train_ratio: float = 0.8, seed: int = 42):
        with open(labels_path) as f:
            all_data = json.load(f)

        np.random.seed(seed)
        idx = np.random.permutation(len(all_data))
        n_train = int(len(all_data) * train_ratio)

        if split == "train":
            self.data = [all_data[i] for i in idx[:n_train]]
            self.transform = self.TRAIN_TRANSFORMS
        else:
            self.data = [all_data[i] for i in idx[n_train:]]
            self.transform = self.VAL_TRANSFORMS

        logger.info(f"Dataset ({split}): {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(item["image_path"])
        if img is None:
            img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label = torch.tensor([item["log10_Nf"]], dtype=torch.float32)
        return img, label


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

class FatigueResNet(nn.Module):
    """
    ResNet50 with custom regression head for fatigue life prediction.
    Outputs: mean prediction + uncertainty (log variance).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3,
                 freeze_backbone: bool = False):
        super().__init__()

        # Load backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        if freeze_backbone:
            for param in list(self.backbone.parameters())[:-30]:
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original classifier

        # Regression head with uncertainty
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.mean_head = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)  # log variance for uncertainty

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        hidden = self.regressor(features)
        mean = self.mean_head(hidden)
        log_var = self.logvar_head(hidden)
        return mean, log_var

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns prediction dict with mean and 95% CI."""
        self.eval()
        with torch.no_grad():
            mean, log_var = self.forward(x)
            std = torch.exp(0.5 * log_var)
            return {
                "log_Nf": mean.squeeze(),
                "std": std.squeeze(),
                "ci_lower": (mean - 1.96 * std).squeeze(),
                "ci_upper": (mean + 1.96 * std).squeeze(),
            }


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log Likelihood Loss for uncertainty estimation."""
    def forward(self, mean, log_var, target):
        precision = torch.exp(-log_var)
        loss = 0.5 * (precision * (target - mean)**2 + log_var)
        return loss.mean()


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class ModelTrainer:
    def __init__(self, model: FatigueResNet, config: Dict, device: str = "auto"):
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get("epochs", 50)
        )
        self.criterion = GaussianNLLLoss()
        self.history = {"train_loss": [], "val_loss": [], "val_r2": []}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(loader, desc="Train", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            mean, log_var = self.model(imgs)
            loss = self.criterion(mean, log_var, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Val", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                mean, log_var = self.model(imgs)
                loss = self.criterion(mean, log_var, labels)
                total_loss += loss.item()
                all_preds.extend(mean.squeeze().cpu().numpy())
                all_targets.extend(labels.squeeze().cpu().numpy())

        preds = np.array(all_preds)
        targets = np.array(all_targets)
        ss_res = np.sum((targets - preds)**2)
        ss_tot = np.sum((targets - targets.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        rmse = np.sqrt(np.mean((targets - preds)**2))
        return total_loss / len(loader), r2

    def train(self, train_dataset, val_dataset, save_dir: str = "outputs"):
        train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size", 16),
                                   shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.get("batch_size", 16),
                                 shuffle=False, num_workers=0)

        Path(save_dir).mkdir(exist_ok=True)
        best_val_loss = float("inf")
        epochs = self.config.get("epochs", 50)

        logger.info(f"Training on {self.device} for {epochs} epochs")
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_r2 = self.val_epoch(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(val_r2)

            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                        f"Train: {train_loss:.4f} | "
                        f"Val: {val_loss:.4f} | "
                        f"R²: {val_r2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = Path(save_dir) / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "config": self.config,
                }, str(path))
                logger.info(f"  ✅ Saved best model (val_loss={val_loss:.4f})")

        return self.history


# ─────────────────────────────────────────────
# Training Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/synthetic/labels.json")
    parser.add_argument("--config", default="configs/resnet_config.yaml")
    parser.add_argument("--save_dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_ds = FatigueDataset(args.labels, split="train")
    val_ds = FatigueDataset(args.labels, split="val")

    model = FatigueResNet(
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3),
        freeze_backbone=config.get("freeze_backbone", False)
    )
    trainer = ModelTrainer(model, config)
    history = trainer.train(train_ds, val_ds, save_dir=args.save_dir)

    # Save training history
    history_path = Path(args.save_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training complete. History: {history_path}")
