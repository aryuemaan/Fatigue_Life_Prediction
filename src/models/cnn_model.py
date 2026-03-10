"""
cnn_model.py
============
CNN-based regression for fatigue life prediction from microscopy images.

Supports three architectures:
  1. FatigueCNN        — Custom lightweight CNN built from scratch
  2. FatigueResNet50   — ResNet50 fine-tuned for regression
  3. FatigueVGG16      — VGG16 fine-tuned for regression

All models output:
  • mean log₁₀(N_f)  — predicted fatigue life (log scale)
  • log_var           — epistemic uncertainty (learned)

Loss: Gaussian NLL → handles heteroscedastic uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict


# ══════════════════════════════════════════════════════════
#  1.  CUSTOM CNN  (no pretrained weights needed)
# ══════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional MaxPool"""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FatigueCNN(nn.Module):
    """
    Custom 5-layer CNN with SE attention and dual output head.
    Input: (B, 3, 224, 224)   Output: mean, log_var
    Designed for microscopy images — sensitive to fine crack structures.
    """
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3,   32,  pool=True)   # → 32 × 112 × 112
        self.enc2 = ConvBlock(32,  64,  pool=True)   # → 64 × 56 × 56
        self.enc3 = ConvBlock(64,  128, pool=True)   # → 128 × 28 × 28
        self.enc4 = ConvBlock(128, 256, pool=True)   # → 256 × 14 × 14
        self.enc5 = ConvBlock(256, 512, pool=True)   # → 512 × 7 × 7

        # Channel attention after each encoder
        self.se3 = SEBlock(128)
        self.se4 = SEBlock(256)
        self.se5 = SEBlock(512)

        self.gap = nn.AdaptiveAvgPool2d(1)    # → 512 × 1 × 1
        self.gmp = nn.AdaptiveMaxPool2d(1)    # → 512 × 1 × 1

        # Regression head — two parallel branches
        self.regressor = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 64),
            nn.GELU(),
        )
        self.mean_head   = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.se3(self.enc3(x))
        x = self.se4(self.enc4(x))
        x = self.se5(self.enc5(x))

        avg = self.gap(x).flatten(1)
        mx  = self.gmp(x).flatten(1)
        x   = torch.cat([avg, mx], dim=1)

        h = self.regressor(x)
        return self.mean_head(h), self.logvar_head(h)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            mean, lv = self.forward(x)
            std = torch.exp(0.5 * lv).clamp(max=2.0)
            return {
                "log_Nf":   mean.squeeze(-1),
                "std":      std.squeeze(-1),
                "ci_lower": (mean - 1.96 * std).squeeze(-1),
                "ci_upper": (mean + 1.96 * std).squeeze(-1),
            }


# ══════════════════════════════════════════════════════════
#  2.  RESNET-50  FINE-TUNED
# ══════════════════════════════════════════════════════════

class FatigueResNet50(nn.Module):
    """
    ResNet-50 with custom regression head.
    Strategy: freeze early layers, fine-tune layer3 + layer4 + head.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.35,
                 freeze_until: str = "layer2"):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Freeze layers up to freeze_until
        freeze = True
        for name, child in backbone.named_children():
            if name == freeze_until:
                freeze = False
            if freeze:
                for p in child.parameters():
                    p.requires_grad = False

        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_f = backbone.fc.in_features  # 2048

        self.head = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.mean_head   = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        h = self.head(x)
        return self.mean_head(h), self.logvar_head(h)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mean, lv = self.forward(x)
            std = torch.exp(0.5 * lv).clamp(max=2.0)
            return {
                "log_Nf":   mean.squeeze(-1),
                "std":      std.squeeze(-1),
                "ci_lower": (mean - 1.96 * std).squeeze(-1),
                "ci_upper": (mean + 1.96 * std).squeeze(-1),
            }


# ══════════════════════════════════════════════════════════
#  3.  VGG-16  FINE-TUNED
# ══════════════════════════════════════════════════════════

class FatigueVGG16(nn.Module):
    """
    VGG-16 with custom regression head.
    VGG's large receptive field is well-suited for texture-heavy microscopy.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        backbone = models.vgg16(weights=weights)

        # Freeze first 3 feature blocks
        for i, layer in enumerate(backbone.features):
            if i < 17:  # up to block3
                for p in layer.parameters():
                    p.requires_grad = False

        self.features = backbone.features      # spatial feature maps
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )
        self.mean_head   = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.features(x))
        h = self.head(x)
        return self.mean_head(h), self.logvar_head(h)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mean, lv = self.forward(x)
            std = torch.exp(0.5 * lv).clamp(max=2.0)
            return {
                "log_Nf":   mean.squeeze(-1),
                "std":      std.squeeze(-1),
                "ci_lower": (mean - 1.96 * std).squeeze(-1),
                "ci_upper": (mean + 1.96 * std).squeeze(-1),
            }


# ══════════════════════════════════════════════════════════
#  4.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood.
    Jointly optimises prediction accuracy and calibrated uncertainty.
    """
    def forward(self, mean, log_var, target):
        precision = torch.exp(-log_var)
        return (0.5 * (precision * (target - mean) ** 2 + log_var)).mean()


class HuberGaussianLoss(nn.Module):
    """
    Hybrid: Huber loss (robust to outliers) + uncertainty regulariser.
    Best for real datasets with noisy labels.
    """
    def __init__(self, delta: float = 0.5, lambda_unc: float = 0.3):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.lambda_unc = lambda_unc

    def forward(self, mean, log_var, target):
        main_loss = self.huber(mean, target)
        unc_reg   = self.lambda_unc * log_var.mean()
        return main_loss + unc_reg


# ══════════════════════════════════════════════════════════
#  5.  MODEL FACTORY
# ══════════════════════════════════════════════════════════

def build_model(arch: str = "resnet50", **kwargs) -> nn.Module:
    """
    Factory function.
    arch: "custom_cnn" | "resnet50" | "vgg16"
    """
    arch_map = {
        "custom_cnn": FatigueCNN,
        "resnet50":   FatigueResNet50,
        "vgg16":      FatigueVGG16,
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(arch_map)}")
    return arch_map[arch](**kwargs)


# ══════════════════════════════════════════════════════════
#  Quick sanity check
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    batch = torch.randn(4, 3, 224, 224)
    for arch in ["custom_cnn", "resnet50", "vgg16"]:
        model = build_model(arch, pretrained=False)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        mean, lv = model(batch)
        print(f"[{arch:12s}] params={n_params:.1f}M  "
              f"mean={mean.shape}  logvar={lv.shape}")
