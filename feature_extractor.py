"""
feature_extractor.py
--------------------
Extracts physically meaningful features from steel microstructure images
using OpenCV and scikit-image.

Features:
- Crack: count, total length, average width, density, branching
- Grain: average size, size distribution, aspect ratio
- Texture: GLCM statistics, LBP histogram
- Shape: HOG descriptor
- Porosity: pore count, void fraction
- Fractal: crack complexity dimension
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from loguru import logger


@dataclass
class FeatureVector:
    # === Crack Features ===
    crack_count: float = 0.0
    crack_total_length: float = 0.0
    crack_avg_length: float = 0.0
    crack_max_length: float = 0.0
    crack_density: float = 0.0          # crack area / total area
    crack_avg_width: float = 0.0
    crack_branching_index: float = 0.0  # junctions / length

    # === Grain Features ===
    grain_count: float = 0.0
    grain_avg_size: float = 0.0
    grain_size_std: float = 0.0
    grain_avg_aspect_ratio: float = 0.0
    grain_size_cv: float = 0.0          # coefficient of variation

    # === Porosity Features ===
    pore_count: float = 0.0
    pore_void_fraction: float = 0.0
    pore_avg_size: float = 0.0
    pore_max_size: float = 0.0

    # === Texture Features (GLCM proxies via OpenCV) ===
    texture_contrast: float = 0.0
    texture_energy: float = 0.0
    texture_homogeneity: float = 0.0
    texture_entropy: float = 0.0
    texture_mean_intensity: float = 0.0
    texture_std_intensity: float = 0.0

    # === LBP Histogram (8 bins summarized) ===
    lbp_uniformity: float = 0.0
    lbp_entropy: float = 0.0

    # === Gradient / Edge Features ===
    edge_density: float = 0.0
    gradient_mean: float = 0.0
    gradient_std: float = 0.0

    # === Fractal Features ===
    fractal_dimension: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float32)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def feature_names() -> List[str]:
        return list(FeatureVector.__dataclass_fields__.keys())


class SteelFeatureExtractor:
    """
    Extracts a comprehensive feature vector from a preprocessed steel image.
    Input: grayscale uint8 image (H x W)
    Output: FeatureVector
    """

    def __init__(self,
                 crack_threshold: int = 40,
                 pore_max_area: int = 200,
                 grain_min_area: int = 100):
        self.crack_threshold = crack_threshold
        self.pore_max_area = pore_max_area
        self.grain_min_area = grain_min_area

    def extract(self, img: np.ndarray) -> FeatureVector:
        """Main extraction method."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        fv = FeatureVector()
        h, w = gray.shape
        total_area = h * w

        # Crack detection
        self._extract_cracks(gray, fv, total_area)

        # Grain structure
        self._extract_grains(gray, fv, total_area)

        # Porosity
        self._extract_pores(gray, fv, total_area)

        # Texture
        self._extract_texture(gray, fv)

        # LBP
        self._extract_lbp(gray, fv)

        # Gradient / edges
        self._extract_gradients(gray, fv, total_area)

        # Fractal dimension of cracks
        fv.fractal_dimension = self._box_counting_dimension(gray)

        return fv

    def _extract_cracks(self, gray: np.ndarray, fv: FeatureVector, total_area: int):
        """Detect dark linear features (cracks) using thresholding + morphology."""
        # Dark crack detection: pixels significantly below local mean
        blurred = cv2.GaussianBlur(gray, (21, 21), 5)
        diff = blurred.astype(int) - gray.astype(int)
        crack_mask = (diff > self.crack_threshold).astype(np.uint8) * 255

        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find connected crack components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_mask)

        crack_areas = []
        crack_lengths = []
        crack_widths = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 5:
                continue
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            length = max(bw, bh)
            width = area / max(length, 1)

            # Aspect ratio filter: cracks are elongated
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect < 2.0:
                continue

            crack_areas.append(area)
            crack_lengths.append(length)
            crack_widths.append(width)

        fv.crack_count = len(crack_lengths)
        fv.crack_total_length = sum(crack_lengths)
        fv.crack_avg_length = np.mean(crack_lengths) if crack_lengths else 0.0
        fv.crack_max_length = max(crack_lengths) if crack_lengths else 0.0
        fv.crack_density = sum(crack_areas) / total_area if crack_areas else 0.0
        fv.crack_avg_width = np.mean(crack_widths) if crack_widths else 0.0

        # Branching index (skeleton junction count / total crack length)
        if fv.crack_total_length > 0:
            skeleton = cv2.ximgproc.thinning(crack_mask) if hasattr(cv2, 'ximgproc') else crack_mask
            kernel_j = np.ones((3, 3), np.uint8)
            junction_map = cv2.filter2D(
                (skeleton > 0).astype(np.float32), -1,
                np.ones((3, 3), np.float32)
            )
            junctions = np.sum(junction_map[skeleton > 0] > 3)
            fv.crack_branching_index = junctions / max(fv.crack_total_length, 1)

    def _extract_grains(self, gray: np.ndarray, fv: FeatureVector, total_area: int):
        """Estimate grain size from watershed/contour segmentation."""
        # Laplacian sharpening to highlight grain boundaries
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        lap_abs = cv2.convertScaleAbs(lap)

        # Threshold to get grain boundary mask
        _, boundary_mask = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(boundary_mask)

        # Find grain regions
        contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        grain_areas = []
        grain_aspects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.grain_min_area or area > total_area * 0.3:
                continue
            grain_areas.append(area)

            # Aspect ratio from bounding rect
            _, _, gw, gh = cv2.boundingRect(cnt)
            aspect = max(gw, gh) / max(min(gw, gh), 1)
            grain_aspects.append(aspect)

        if grain_areas:
            fv.grain_count = len(grain_areas)
            fv.grain_avg_size = float(np.mean(grain_areas))
            fv.grain_size_std = float(np.std(grain_areas))
            fv.grain_avg_aspect_ratio = float(np.mean(grain_aspects))
            fv.grain_size_cv = fv.grain_size_std / (fv.grain_avg_size + 1e-6)

    def _extract_pores(self, gray: np.ndarray, fv: FeatureVector, total_area: int):
        """Detect circular dark features (pores/voids)."""
        # Very dark, compact regions = pores
        _, pore_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pore_mask = cv2.morphologyEx(pore_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pore_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3 < area < self.pore_max_area:
                # Circularity filter
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
                if circularity > 0.3:
                    pore_areas.append(area)

        fv.pore_count = len(pore_areas)
        fv.pore_void_fraction = sum(pore_areas) / total_area if pore_areas else 0.0
        fv.pore_avg_size = float(np.mean(pore_areas)) if pore_areas else 0.0
        fv.pore_max_size = float(max(pore_areas)) if pore_areas else 0.0

    def _extract_texture(self, gray: np.ndarray, fv: FeatureVector):
        """Compute GLCM-inspired statistics using OpenCV."""
        # Co-occurrence via shifted differences
        dx = np.abs(gray[:, :-1].astype(int) - gray[:, 1:].astype(int))
        dy = np.abs(gray[:-1, :].astype(int) - gray[1:, :].astype(int))

        fv.texture_contrast = float(np.mean(dx**2 + dy**2))
        fv.texture_energy = float(np.mean(gray.astype(float)**2))
        fv.texture_homogeneity = float(1.0 / (1.0 + np.mean(dx)))
        fv.texture_mean_intensity = float(np.mean(gray))
        fv.texture_std_intensity = float(np.std(gray))

        # Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        fv.texture_entropy = float(entropy)

    def _extract_lbp(self, gray: np.ndarray, fv: FeatureVector):
        """Local Binary Pattern features."""
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        center = gray[1:-1, 1:-1]

        neighbors = [
            gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
            gray[1:-1, 2:],   gray[2:,   2:],   gray[2:,   1:-1],
            gray[2:,   0:-2], gray[1:-1, 0:-2]
        ]
        for i, nb in enumerate(neighbors):
            lbp += ((nb >= center).astype(np.uint8)) << i

        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist_norm = hist / (hist.sum() + 1e-6)

        # Uniformity (fraction of uniform patterns)
        fv.lbp_uniformity = float(np.max(hist_norm))
        fv.lbp_entropy = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])))

    def _extract_gradients(self, gray: np.ndarray, fv: FeatureVector, total_area: int):
        """Sobel gradient statistics and edge density."""
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        fv.gradient_mean = float(np.mean(magnitude))
        fv.gradient_std = float(np.std(magnitude))

        edges = cv2.Canny(gray, 30, 80)
        fv.edge_density = float(np.sum(edges > 0) / total_area)

    def _box_counting_dimension(self, gray: np.ndarray) -> float:
        """Estimate fractal dimension of crack network via box counting."""
        # Threshold to binary
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        for s in sizes:
            resized = cv2.resize(binary, (256 // s, 256 // s),
                                  interpolation=cv2.INTER_AREA)
            count = np.sum(resized > 0)
            counts.append(count + 1)  # avoid log(0)

        # Linear regression on log-log
        log_s = np.log(sizes)
        log_c = np.log(counts)
        coeffs = np.polyfit(log_s, log_c, 1)
        return float(abs(coeffs[0]))  # slope = fractal dimension proxy
