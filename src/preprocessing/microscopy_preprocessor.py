"""
microscopy_preprocessor.py
===========================
OpenCV preprocessing pipeline tuned for steel optical microscopy images.

Pipeline stages
---------------
1.  Load & validate           – check image quality, flag corruptions
2.  Auto-crop borders         – remove SEM/microscope frame artefacts
3.  Denoise                   – Non-local means (preserves grain edges)
4.  Illumination correction   – rolling-ball background subtraction + CLAHE
5.  Contrast stretch          – percentile-based
6.  Crack enhancement         – Scharr + Black-hat morphology
7.  Grain boundary sharpening – Unsharp mask
8.  Output normalisation      – uint8 [0, 255]

Also provides:
• QualityChecker  – SNR, blur score, exposure score
• CrackMapper     – binary crack map + statistics
• GrainAnalyser   – grain size from watershed
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, NamedTuple
from loguru import logger


# ══════════════════════════════════════════════════════════
#  QUALITY CHECKER
# ══════════════════════════════════════════════════════════

class ImageQuality(NamedTuple):
    blur_score:     float   # higher = sharper (Laplacian variance)
    snr_db:         float   # signal-to-noise ratio estimate
    exposure:       float   # 0=under, 1=good, 2=over
    is_acceptable:  bool

def check_quality(img: np.ndarray) -> ImageQuality:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Blur
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # SNR (mean/std of a mid-crop)
    h, w = gray.shape
    crop = gray[h//4:3*h//4, w//4:3*w//4]
    snr  = 20 * np.log10(crop.mean() / (crop.std() + 1e-6) + 1e-6)
    # Exposure (fraction of saturated pixels)
    over  = np.mean(gray > 245)
    under = np.mean(gray < 10)
    if over > 0.05:   exp = 2.0
    elif under > 0.1: exp = 0.0
    else:             exp = 1.0
    ok = blur > 20 and 5 < snr < 60 and exp == 1.0
    return ImageQuality(blur, snr, exp, ok)


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════

@dataclass
class MicroscopyConfig:
    target_size:       Tuple[int,int]  = (224, 224)
    # Denoising
    denoise_h:         int             = 8         # NLM filter strength
    denoise_template:  int             = 7
    denoise_search:    int             = 21
    # Illumination
    bg_ball_radius:    int             = 40        # rolling-ball radius
    clahe_clip:        float           = 3.0
    clahe_tile:        Tuple[int,int]  = (8, 8)
    # Contrast
    percentile_lo:     float           = 2.0
    percentile_hi:     float           = 98.0
    # Crack enhancement
    crack_bhat_ksize:  int             = 17        # black-hat kernel length
    crack_blend:       float           = 0.30
    # Sharpening
    sharpen_amount:    float           = 1.4
    sharpen_sigma:     float           = 1.0
    # Border crop (fraction to strip from each edge — removes scale bars)
    border_crop:       float           = 0.04
    # Output
    output_channels:   int             = 3         # 1 or 3


# ══════════════════════════════════════════════════════════
#  PREPROCESSOR
# ══════════════════════════════════════════════════════════

class MicroscopyPreprocessor:

    def __init__(self, cfg: Optional[MicroscopyConfig] = None):
        self.cfg   = cfg or MicroscopyConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=self.cfg.clahe_tile,
        )

    # ── 1. Load ──────────────────────────────────────────

    def load(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        return img

    # ── 2. Border crop ───────────────────────────────────

    def crop_borders(self, img: np.ndarray) -> np.ndarray:
        """Strip outer border fraction (removes scale bar, frame artefacts)."""
        h, w = img.shape[:2]
        p = self.cfg.border_crop
        y0, y1 = int(h*p), int(h*(1-p))
        x0, x1 = int(w*p), int(w*(1-p))
        return img[y0:y1, x0:x1]

    # ── 3. Denoise ───────────────────────────────────────

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            gray,
            h=self.cfg.denoise_h,
            templateWindowSize=self.cfg.denoise_template,
            searchWindowSize=self.cfg.denoise_search,
        )

    # ── 4. Illumination correction ───────────────────────

    def correct_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Rolling-ball background subtraction:
        estimate background (large Gaussian blur) and divide it out,
        then apply CLAHE.
        """
        r = self.cfg.bg_ball_radius
        # Kernel size must be odd and > 2*r
        k = 2 * r + 1
        background = cv2.GaussianBlur(gray.astype(np.float32), (k, k), r / 2)
        corrected  = gray.astype(np.float32) / (background + 1.0) * 128.0
        corrected  = np.clip(corrected, 0, 255).astype(np.uint8)
        return self.clahe.apply(corrected)

    # ── 5. Contrast stretch ──────────────────────────────

    def contrast_stretch(self, gray: np.ndarray) -> np.ndarray:
        lo = np.percentile(gray, self.cfg.percentile_lo)
        hi = np.percentile(gray, self.cfg.percentile_hi)
        stretched = np.clip((gray.astype(float) - lo) / (hi - lo + 1e-6) * 255,
                             0, 255).astype(np.uint8)
        return stretched

    # ── 6. Crack enhancement ─────────────────────────────

    def enhance_cracks(self, gray: np.ndarray) -> np.ndarray:
        """
        Highlight dark linear crack features using:
        • Scharr directional gradients
        • Black-hat morphology (horizontal, vertical, diagonal kernels)
        """
        # Scharr
        sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        grad = np.sqrt(sx**2 + sy**2)
        grad = cv2.normalize(grad, None, 0, 60, cv2.NORM_MINMAX).astype(np.uint8)

        # Black-hat in multiple orientations
        k  = self.cfg.crack_bhat_ksize
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k//2, k//2))

        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kh)
        bv = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kv)
        bd = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kd)
        crack_map = cv2.max(cv2.max(bh, bv), bd)

        combined = cv2.add(crack_map, grad)
        enhanced = cv2.addWeighted(
            gray, 1.0 - self.cfg.crack_blend,
            combined, self.cfg.crack_blend, 0
        )
        return enhanced

    # ── 7. Unsharp mask (grain boundary sharpening) ──────

    def sharpen(self, gray: np.ndarray) -> np.ndarray:
        sigma  = self.cfg.sharpen_sigma
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        a = self.cfg.sharpen_amount
        sharpened = cv2.addWeighted(gray, 1 + a, blurred, -a, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── 8. Resize ─────────────────────────────────────────

    def resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        th, tw = self.cfg.target_size
        scale  = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        pad_t = (th - nh) // 2;  pad_b = th - nh - pad_t
        pad_l = (tw - nw) // 2;  pad_r = tw - nw - pad_l
        return cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REFLECT_101)

    # ── Main pipeline ─────────────────────────────────────

    def process(self, path: str) -> Dict[str, np.ndarray]:
        """
        Returns:
          processed       – final output ready for CNN input
          original        – resized original (no processing)
          crack_enhanced  – intermediate crack-enhanced image
          quality         – ImageQuality namedtuple
        """
        raw  = self.load(path)
        qual = check_quality(raw)
        if not qual.is_acceptable:
            logger.warning(f"Low-quality image: blur={qual.blur_score:.1f} "
                           f"snr={qual.snr_db:.1f} exp={qual.exposure}")

        cropped = self.crop_borders(raw)
        gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Save original (after crop + resize)
        orig_resized = self.resize(gray)

        denoised    = self.denoise(gray)
        illum_corr  = self.correct_illumination(denoised)
        stretched   = self.contrast_stretch(illum_corr)
        crack_enh   = self.enhance_cracks(stretched)
        sharpened   = self.sharpen(crack_enh)
        final_gray  = self.resize(sharpened)

        def to_out(g):
            if self.cfg.output_channels == 3:
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            return g

        return {
            "processed":      to_out(final_gray),
            "original":       to_out(orig_resized),
            "crack_enhanced": to_out(self.resize(crack_enh)),
            "gray":           final_gray,
            "quality":        qual,
        }

    def process_array(self, img: np.ndarray) -> np.ndarray:
        """Process a BGR numpy array directly (for API / GUI use)."""
        qual = check_quality(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        denoised   = self.denoise(gray)
        illum_corr = self.correct_illumination(denoised)
        stretched  = self.contrast_stretch(illum_corr)
        crack_enh  = self.enhance_cracks(stretched)
        sharpened  = self.sharpen(crack_enh)
        final      = self.resize(sharpened)
        if self.cfg.output_channels == 3:
            return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        return final

    def pipeline_grid(self, path: str) -> np.ndarray:
        """Return a 1×5 grid showing each pipeline stage for inspection."""
        raw     = self.load(path)
        gray    = cv2.cvtColor(self.crop_borders(raw), cv2.COLOR_BGR2GRAY)
        stages  = [
            ("Original",      gray),
            ("Illumination",  self.correct_illumination(self.denoise(gray))),
            ("Contrast",      self.contrast_stretch(
                               self.correct_illumination(self.denoise(gray)))),
            ("Cracks",        self.enhance_cracks(
                               self.contrast_stretch(
                               self.correct_illumination(self.denoise(gray))))),
            ("Sharpened",     self.resize(self.sharpen(
                               self.enhance_cracks(
                               self.contrast_stretch(
                               self.correct_illumination(self.denoise(gray))))))),
        ]
        th, tw = self.cfg.target_size
        panel  = np.zeros((th + 28, tw * len(stages), 3), dtype=np.uint8)
        for i, (name, img) in enumerate(stages):
            resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            bgr     = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            panel[28:, i*tw:(i+1)*tw] = bgr
            cv2.putText(panel, name, (i*tw + 4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 150), 1, cv2.LINE_AA)
        return panel


# ══════════════════════════════════════════════════════════
#  CRACK MAPPER
# ══════════════════════════════════════════════════════════

class CrackMapper:
    """Extracts a binary crack map + statistics from a preprocessed image."""

    def __init__(self, threshold: int = 35, min_area: int = 20, min_aspect: float = 2.5):
        self.threshold  = threshold
        self.min_area   = min_area
        self.min_aspect = min_aspect

    def map(self, gray: np.ndarray) -> Dict:
        blurred = cv2.GaussianBlur(gray, (21, 21), 5)
        diff    = blurred.astype(int) - gray.astype(int)
        mask    = np.clip(diff, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(mask, self.threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        crack_data = []
        for i in range(1, n_lbl):
            a = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if a < self.min_area:
                continue
            asp = max(w, h) / max(min(w, h), 1)
            if asp < self.min_aspect:
                continue
            crack_data.append({"area": a, "width": w, "height": h,
                                "aspect": round(asp, 2)})

        total_area = gray.shape[0] * gray.shape[1]
        total_crack = sum(c["area"] for c in crack_data)
        return {
            "binary_mask":   mask,
            "n_cracks":      len(crack_data),
            "crack_density": round(total_crack / total_area, 5),
            "total_length":  sum(max(c["width"], c["height"]) for c in crack_data),
            "cracks":        crack_data,
        }
