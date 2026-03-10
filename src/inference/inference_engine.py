"""
inference_engine.py
===================
Production inference engine — loads a trained CNN checkpoint and runs
predictions with GradCAM visualisation on steel microscopy images.

Supports:
  • Single-image prediction
  • Batch prediction with CSV export
  • GradCAM heatmap overlay (shows *why* the model gave that prediction)
  • OpenCV-annotated output image
  • Uncertainty-aware output (95 % confidence interval)

Usage
-----
from src.inference.inference_engine import InferenceEngine
engine = InferenceEngine("outputs/best_resnet50.pth")
result = engine.predict("sample.png", return_gradcam=True)
engine.save_annotated(result, "out.png")
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import build_model
from preprocessing.microscopy_preprocessor import MicroscopyPreprocessor, MicroscopyConfig


# ══════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    sample_id:       str
    log10_Nf:        float
    Nf_cycles:       float
    log10_Nf_lower:  float
    log10_Nf_upper:  float
    std:             float
    risk_category:   str
    inference_ms:    float
    gradcam:         Optional[np.ndarray] = None   # (H,W) float32 [0,1]
    annotated_img:   Optional[np.ndarray] = None   # BGR annotated image

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("gradcam", "annotated_img")}
        return d

    @staticmethod
    def risk_from_log(log_nf: float) -> str:
        if log_nf < 4.5:   return "CRITICAL"
        if log_nf < 5.5:   return "HIGH"
        if log_nf < 6.5:   return "MEDIUM"
        return "LOW"


# ══════════════════════════════════════════════════════════
# GRADCAM
# ══════════════════════════════════════════════════════════

class GradCAM:
    """
    GradCAM for any CNN regression model.
    Hooks into the final convolutional feature map.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model      = model
        self._feats     = None
        self._grads     = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_feats", o))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0]))

    @torch.enable_grad()
    def generate(self, img_tensor: torch.Tensor,
                 img_size: Tuple[int,int] = (224, 224)) -> np.ndarray:
        """
        Returns a (H, W) CAM heatmap resized to img_size.
        """
        self.model.eval()
        img_tensor = img_tensor.detach().clone().requires_grad_(True)
        mean, _ = self.model(img_tensor)
        self.model.zero_grad()
        mean.sum().backward()

        grads = self._grads       # (B, C, H, W)
        feats = self._feats       # (B, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * feats).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().detach().numpy()
        cam_n   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_r   = cv2.resize(cam_n, img_size[::-1])  # cv2 uses (w,h)
        return cam_r.astype(np.float32)


def get_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the last conv layer for GradCAM hookup."""
    if arch == "resnet50":
        return model.features[-1][-1].conv3  # layer4 last bottleneck
    elif arch == "vgg16":
        # last conv in features
        for m in reversed(list(model.features.children())):
            if isinstance(m, nn.Conv2d):
                return m
    else:  # custom_cnn
        return model.enc5.block[-3]   # last conv before pool


def overlay_gradcam(img_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend JET colourmap CAM over image."""
    h, w = img_bgr.shape[:2]
    cam_r   = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


# ══════════════════════════════════════════════════════════
# ANNOTATOR
# ══════════════════════════════════════════════════════════

RISK_PALETTE = {
    "LOW":      ((20, 180, 20),   (10, 60, 10)),
    "MEDIUM":   ((20, 190, 255),  (10, 60, 80)),
    "HIGH":     ((20, 80, 255),   (10, 30, 80)),
    "CRITICAL": ((30, 30, 255),   (10, 10, 80)),
    "UNKNOWN":  ((150, 150, 150), (50, 50, 50)),
}

def annotate_image(img_bgr: np.ndarray, result: PredictionResult,
                   cam: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw a comprehensive annotation panel on the image.
    Includes: risk badge, log Nf value, CI bar, crack overlay (optional CAM).
    """
    if cam is not None:
        base = overlay_gradcam(img_bgr.copy(), cam, alpha=0.40)
    else:
        base = img_bgr.copy()

    h, w = base.shape[:2]
    risk           = result.risk_category
    color, bg_dark = RISK_PALETTE.get(risk, RISK_PALETTE["UNKNOWN"])

    # ── Top banner ────────────────────────────────────────
    banner = np.zeros((70, w, 3), dtype=np.uint8)
    banner[:, :] = bg_dark
    cv2.rectangle(banner, (0, 0), (w, 70), bg_dark, -1)
    # risk badge
    badge_txt = f" {risk} "
    (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(banner, (8, 8), (8 + tw + 6, 8 + th + 10), color, -1)
    cv2.putText(banner, badge_txt, (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 10), 1, cv2.LINE_AA)
    # values
    nf_str = f"{result.Nf_cycles:.3e}"
    cv2.putText(banner,
                f"log\u2081\u2080(N\u1099) = {result.log10_Nf:.3f}   N\u1099 = {nf_str} cycles",
                (8 + tw + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    ci_txt = f"95% CI  [{result.log10_Nf_lower:.2f}, {result.log10_Nf_upper:.2f}]  " \
             f"\u00b1{result.std:.3f}"
    cv2.putText(banner, ci_txt, (8 + tw + 20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1, cv2.LINE_AA)

    # ── Life bar across bottom ────────────────────────────
    bar_h    = 12
    bar_img  = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar_img[:, :] = (25, 25, 30)
    norm = np.clip((result.log10_Nf - 3.5) / 5.0, 0, 1)
    fill = int(norm * w)
    bar_img[:, :fill] = color
    # CI ticks
    lo_x = int(np.clip((result.log10_Nf_lower - 3.5) / 5.0, 0, 1) * w)
    hi_x = int(np.clip((result.log10_Nf_upper - 3.5) / 5.0, 0, 1) * w)
    bar_img[:, lo_x:hi_x] = tuple(min(c + 40, 255) for c in color)

    canvas = np.vstack([banner, base, bar_img])

    # ── Right-side info panel ─────────────────────────────
    panel_w = 170
    if w >= 300:
        panel = np.full((canvas.shape[0], panel_w, 3), (18, 20, 28), dtype=np.uint8)
        y = 30
        lines = [
            ("FATIGUE ANALYSIS", (0, 200, 150)),
            ("─" * 22,          (40, 42, 55)),
            (f"Risk:    {risk}",          color),
            (f"logNf:   {result.log10_Nf:.4f}",  (200,200,200)),
            (f"Nf:      {nf_str}",         (200,200,200)),
            (f"CI low:  {result.log10_Nf_lower:.3f}", (140,140,150)),
            (f"CI high: {result.log10_Nf_upper:.3f}", (140,140,150)),
            (f"Uncert:  {result.std:.3f}",        (140,140,150)),
            ("─" * 22, (40, 42, 55)),
            (f"Time:  {result.inference_ms:.0f}ms", (90,90,100)),
        ]
        for txt, col in lines:
            cv2.putText(panel, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, col, 1, cv2.LINE_AA)
            y += 22
        canvas = np.hstack([canvas, panel])

    return canvas


# ══════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def img_to_tensor(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """Preprocess BGR uint8 → normalised float tensor (1,3,H,W)."""
    img  = cv2.resize(img_bgr, (size, size))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img  = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


class InferenceEngine:
    """
    Load a trained CNN checkpoint and run predictions.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.arch         = "resnet50"
        self.label_stats  = {"mean": 6.0, "std": 1.0}
        self.model        = None
        self.gradcam_hook = None

        self._load_checkpoint(checkpoint_path)
        self.preprocessor = MicroscopyPreprocessor(
            MicroscopyConfig(target_size=(224, 224))
        )

    def _load_checkpoint(self, path: str):
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path}. Using mock mode.")
            self._mock_mode = True
            return
        self._mock_mode = False

        ckpt = torch.load(path, map_location=self.device)
        self.arch        = ckpt.get("arch", "resnet50")
        self.label_stats = ckpt.get("label_stats", {"mean": 6.0, "std": 1.0})

        self.model = build_model(self.arch, pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # GradCAM hook
        try:
            target_layer     = get_target_layer(self.model, self.arch)
            self.gradcam_hook = GradCAM(self.model, target_layer)
        except Exception as e:
            logger.warning(f"GradCAM hook failed: {e}")
            self.gradcam_hook = None

        logger.info(f"Loaded {self.arch} | val_r2={ckpt.get('val_r2', '?')}")

    def _mock_predict(self, img_array: np.ndarray) -> Tuple[float, float]:
        seed = int(np.mean(img_array)) % 9999
        np.random.seed(seed)
        log_nf = float(np.random.uniform(4.2, 8.2))
        std    = float(np.random.uniform(0.12, 0.25))
        return log_nf, std

    def _denorm(self, norm_val: float) -> float:
        return norm_val * self.label_stats["std"] + self.label_stats["mean"]

    def predict(self, input_data,
                return_gradcam: bool = True,
                annotate: bool = True,
                sample_id: Optional[str] = None) -> PredictionResult:
        """
        input_data: str (image path) or np.ndarray (BGR)
        """
        t0 = time.time()

        # Load + preprocess
        if isinstance(input_data, str):
            result_dict = self.preprocessor.process(str(input_data))
            processed   = result_dict["processed"]
            sid         = sample_id or Path(str(input_data)).stem
        else:
            processed = self.preprocessor.process_array(input_data)
            sid       = sample_id or "upload"

        # Inference
        if self._mock_mode:
            log_nf, std = self._mock_predict(processed)
        else:
            tensor = img_to_tensor(processed, size=224).to(self.device)
            with torch.no_grad():
                out = self.model.predict(tensor)
            log_nf = float(self._denorm(out["log_Nf"].item()))
            std    = float(out["std"].item() * self.label_stats["std"])

        Nf       = 10 ** log_nf
        ci_lo    = log_nf - 1.96 * std
        ci_hi    = log_nf + 1.96 * std

        # GradCAM
        cam = None
        if return_gradcam and self.gradcam_hook and not self._mock_mode:
            try:
                tensor_gc = img_to_tensor(processed, 224).to(self.device)
                cam       = self.gradcam_hook.generate(tensor_gc, (224, 224))
            except Exception as e:
                logger.warning(f"GradCAM failed: {e}")

        elapsed = (time.time() - t0) * 1000
        res = PredictionResult(
            sample_id       = sid,
            log10_Nf        = round(log_nf, 4),
            Nf_cycles       = float(Nf),
            log10_Nf_lower  = round(ci_lo, 4),
            log10_Nf_upper  = round(ci_hi, 4),
            std             = round(std, 4),
            risk_category   = PredictionResult.risk_from_log(log_nf),
            inference_ms    = round(elapsed, 1),
            gradcam         = cam,
        )

        if annotate:
            res.annotated_img = annotate_image(processed, res, cam)

        return res

    def predict_batch(self, paths: List[str],
                      return_gradcam: bool = False) -> List[PredictionResult]:
        results = []
        for p in paths:
            try:
                r = self.predict(p, return_gradcam=return_gradcam, annotate=False)
                results.append(r)
            except Exception as e:
                logger.error(f"Failed {p}: {e}")
        return results

    def save_annotated(self, result: PredictionResult, path: str):
        if result.annotated_img is None:
            logger.warning("No annotated image to save.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), result.annotated_img)
        logger.info(f"Saved annotated image → {path}")
