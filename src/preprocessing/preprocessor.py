"""
preprocessor.py
---------------
OpenCV-based image preprocessing pipeline for steel microstructure images.
Handles: denoising, enhancement, normalization, crack enhancement.

Usage:
    from src.preprocessing.preprocessor import SteelImagePreprocessor
    preprocessor = SteelImagePreprocessor()
    result = preprocessor.process("image.png")
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from loguru import logger


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (224, 224)
    denoise_strength: int = 7
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    normalize: bool = True
    enhance_cracks: bool = True
    remove_artifacts: bool = True
    output_channels: int = 3  # 1 for grayscale, 3 for RGB


class SteelImagePreprocessor:
    """
    Full preprocessing pipeline for steel microstructure images.
    Steps:
    1. Load & validate
    2. Resize with aspect ratio preservation
    3. Denoise (Non-local means)
    4. Flatten illumination (CLAHE)
    5. Crack enhancement (Frangi vesselness / Scharr edges)
    6. Normalize to [0, 1] or standardize
    7. Return processed image + diagnostic map
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid
        )

    def load(self, path: str) -> np.ndarray:
        """Load image, convert to grayscale for processing."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        logger.debug(f"Loaded {path}: shape={img.shape}")
        return gray

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size maintaining aspect ratio with padding."""
        h, w = img.shape[:2]
        th, tw = self.config.target_size
        scale = min(tw / w, th / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exact target size
        pad_top = (th - new_h) // 2
        pad_bottom = th - new_h - pad_top
        pad_left = (tw - new_w) // 2
        pad_right = tw - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_REFLECT_101)
        return padded

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Non-local means denoising - preserves microstructure edges."""
        h = self.config.denoise_strength
        return cv2.fastNlMeansDenoising(img, h=h, templateWindowSize=7, searchWindowSize=21)

    def equalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """CLAHE to handle uneven microscope illumination."""
        return self.clahe.apply(img)

    def enhance_cracks(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance crack visibility using:
        - Scharr gradient for thin crack detection
        - Morphological bottom-hat for dark linear features
        """
        # Scharr edges (better than Sobel for fine cracks)
        scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        gradient_mag = np.sqrt(scharr_x**2 + scharr_y**2)
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 50, cv2.NORM_MINMAX).astype(np.uint8)

        # Bottom-hat: highlights dark features (cracks) on bright background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_diag = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
        bhat_h = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        bhat_v = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel.T)
        bhat_d = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_diag)
        crack_map = cv2.add(cv2.add(bhat_h, bhat_v), bhat_d)

        # Blend back into original
        enhanced = cv2.addWeighted(img, 0.75, crack_map, 0.25, 0)
        return enhanced

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize to [0, 255] uint8."""
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def to_output_channels(self, img: np.ndarray) -> np.ndarray:
        """Convert to desired output channel count."""
        if self.config.output_channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def process(self, path: str) -> Dict[str, np.ndarray]:
        """
        Full pipeline. Returns dict with:
        - 'processed': final image for model input
        - 'original_resized': original after resize only
        - 'enhanced': crack-enhanced intermediate
        """
        gray = self.load(path)

        # Step 1: Resize
        resized = self.resize(gray)
        original_resized = resized.copy()

        # Step 2: Denoise
        denoised = self.denoise(resized)

        # Step 3: Illuminate equalization
        equalized = self.equalize_illumination(denoised)

        # Step 4: Crack enhancement
        if self.config.enhance_cracks:
            enhanced = self.enhance_cracks(equalized)
        else:
            enhanced = equalized

        # Step 5: Normalize
        if self.config.normalize:
            final = self.normalize(enhanced)
        else:
            final = enhanced

        # Step 6: Output channels
        final_out = self.to_output_channels(final)
        original_out = self.to_output_channels(original_resized)

        return {
            "processed": final_out,
            "original_resized": original_out,
            "enhanced": self.to_output_channels(enhanced),
            "gray": final
        }

    def process_array(self, img_array: np.ndarray) -> np.ndarray:
        """Process from numpy array directly (for API use)."""
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array

        resized = self.resize(gray)
        denoised = self.denoise(resized)
        equalized = self.equalize_illumination(denoised)

        if self.config.enhance_cracks:
            enhanced = self.enhance_cracks(equalized)
        else:
            enhanced = equalized

        final = self.normalize(enhanced)
        return self.to_output_channels(final)

    def visualize_pipeline(self, path: str, save_path: Optional[str] = None):
        """Create a side-by-side visualization of all pipeline stages."""
        gray = self.load(path)
        resized = self.resize(gray)
        denoised = self.denoise(resized)
        equalized = self.equalize_illumination(denoised)
        enhanced = self.enhance_cracks(equalized)
        normalized = self.normalize(enhanced)

        stages = [
            ("1. Original", resized),
            ("2. Denoised", denoised),
            ("3. CLAHE", equalized),
            ("4. Crack Enhanced", enhanced),
            ("5. Normalized", normalized),
        ]

        h, w = resized.shape
        panel = np.zeros((h + 30, w * len(stages), 3), dtype=np.uint8)

        for i, (label, img) in enumerate(stages):
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            panel[30:, i*w:(i+1)*w] = bgr
            cv2.putText(panel, label, (i*w + 5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

        if save_path:
            cv2.imwrite(save_path, panel)
            logger.info(f"Pipeline visualization saved to {save_path}")
        return panel
