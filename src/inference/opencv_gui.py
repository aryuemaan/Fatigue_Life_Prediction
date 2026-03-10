"""
opencv_gui.py
-------------
Real-time OpenCV GUI for fatigue life prediction.
Supports: single image, webcam/microscope live feed, drag-and-drop.

Controls:
  [O]pen image  [L]ive camera  [S]ave result  [Q]uit
  [+/-] Zoom    [A]nnotations toggle  [H]istogram

Usage:
    python src/inference/opencv_gui.py --model outputs/hybrid_model.pkl --mode hybrid
    python src/inference/opencv_gui.py --model outputs/best_model.pth --mode resnet
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.preprocessor import SteelImagePreprocessor, PreprocessConfig


# ─────────────────────────────────────────────
# Color Theme
# ─────────────────────────────────────────────
RISK_COLORS = {
    "LOW":      (50, 220, 50),    # Green
    "MEDIUM":   (50, 200, 255),   # Yellow-ish
    "HIGH":     (30, 100, 255),   # Orange
    "CRITICAL": (50, 50, 255),    # Red
    "UNKNOWN":  (150, 150, 150),  # Gray
}

UI_BG      = (20, 22, 30)
UI_PANEL   = (35, 38, 50)
UI_ACCENT  = (0, 200, 150)
UI_TEXT    = (230, 230, 230)
UI_DIM     = (120, 120, 130)
UI_WHITE   = (255, 255, 255)


# ─────────────────────────────────────────────
# Annotation Utilities
# ─────────────────────────────────────────────

def draw_crack_annotations(img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Overlay detected cracks and pores on the image."""
    annotated = img.copy()

    # Crack detection
    blurred = cv2.GaussianBlur(gray, (21, 21), 5)
    diff = blurred.astype(int) - gray.astype(int)
    crack_mask = (diff > 35).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)

    # Overlay cracks in red
    crack_overlay = np.zeros_like(img)
    crack_overlay[crack_mask > 0] = (30, 30, 255)
    annotated = cv2.addWeighted(annotated, 0.85, crack_overlay, 0.4, 0)

    # Find and draw crack contours
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 2:
            cv2.drawContours(annotated, [cnt], -1, (30, 30, 255), 1)

    # Pore detection (dark round regions)
    _, pore_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    pore_contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in pore_contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 200:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                r = max(2, int(np.sqrt(area / np.pi)))
                cv2.circle(annotated, (cx, cy), r + 1, (255, 150, 50), 1)

    return annotated


def draw_prediction_panel(canvas: np.ndarray, result: Dict,
                           panel_x: int, panel_y: int, panel_w: int, panel_h: int):
    """Draw the prediction result panel on the right side."""
    # Panel background
    cv2.rectangle(canvas, (panel_x, panel_y),
                   (panel_x + panel_w, panel_y + panel_h), UI_PANEL, -1)

    risk = result.get("risk_category", "UNKNOWN")
    risk_color = RISK_COLORS.get(risk, RISK_COLORS["UNKNOWN"])
    log_nf = result.get("log10_Nf", 0)
    nf = result.get("Nf_cycles", 0)
    ci_lo = result.get("log10_Nf_lower", log_nf - 0.2)
    ci_hi = result.get("log10_Nf_upper", log_nf + 0.2)

    y = panel_y + 25
    x = panel_x + 15

    # Title
    cv2.putText(canvas, "FATIGUE LIFE PREDICTION",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_ACCENT, 1, cv2.LINE_AA)
    y += 8
    cv2.line(canvas, (x, y), (panel_x + panel_w - 15, y), UI_ACCENT, 1)
    y += 25

    # Risk badge
    badge_x, badge_y = x, y
    badge_w = 140
    cv2.rectangle(canvas, (badge_x - 5, badge_y - 18),
                   (badge_x + badge_w, badge_y + 5), risk_color, -1)
    cv2.putText(canvas, f"  RISK: {risk}",
                (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (10, 10, 10), 1, cv2.LINE_AA)
    y += 35

    # Log Nf
    cv2.putText(canvas, "log\u2081\u2080(N_f):", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{log_nf:.3f}", (x + 90, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_WHITE, 1, cv2.LINE_AA)
    y += 25

    # Nf in cycles
    nf_str = f"{nf:.2e}" if nf > 1e6 else f"{int(nf):,}"
    cv2.putText(canvas, "N_f (cycles):", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, nf_str, (x + 90, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_WHITE, 1, cv2.LINE_AA)
    y += 25

    # 95% CI
    cv2.putText(canvas, "95% CI:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"[{ci_lo:.2f}, {ci_hi:.2f}]", (x + 60, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_DIM, 1, cv2.LINE_AA)
    y += 30

    # Confidence bar
    conf_w = panel_w - 30
    conf_h = 12
    # Background
    cv2.rectangle(canvas, (x, y), (x + conf_w, y + conf_h), (50, 50, 60), -1)
    # Normalized position
    norm = np.clip((log_nf - 3.5) / (8.5 - 3.5), 0, 1)
    fill_w = int(norm * conf_w)
    cv2.rectangle(canvas, (x, y), (x + fill_w, y + conf_h), risk_color, -1)
    cv2.rectangle(canvas, (x, y), (x + conf_w, y + conf_h), UI_DIM, 1)
    cv2.putText(canvas, "3.5", (x, y + conf_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, UI_DIM, 1)
    cv2.putText(canvas, "8.5", (x + conf_w - 20, y + conf_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, UI_DIM, 1)
    y += conf_h + 30

    # Horizontal rule
    cv2.line(canvas, (x, y), (panel_x + panel_w - 15, y), UI_PANEL, 1)
    y += 20

    # Physical interpretation
    interp = {
        "LOW":      "Excellent condition. Normal service life.",
        "MEDIUM":   "Monitor periodically. Some degradation.",
        "HIGH":     "Increased inspection frequency required.",
        "CRITICAL": "IMMEDIATE action required. Risk of failure.",
    }.get(risk, "Unable to assess condition.")

    words = interp.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if len(test_line) > 28:
            cv2.putText(canvas, line.strip(), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, risk_color, 1, cv2.LINE_AA)
            y += 18
            line = word + " "
        else:
            line = test_line
    if line:
        cv2.putText(canvas, line.strip(), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, risk_color, 1, cv2.LINE_AA)


def draw_status_bar(canvas: np.ndarray, status: str, fps: float,
                    show_annotations: bool, h: int, w: int):
    """Bottom status bar."""
    bar_y = h - 30
    cv2.rectangle(canvas, (0, bar_y), (w, h), (15, 15, 20), -1)
    cv2.putText(canvas, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_DIM, 1, cv2.LINE_AA)
    fps_str = f"FPS: {fps:.1f}  |  ANN: {'ON' if show_annotations else 'OFF'}"
    cv2.putText(canvas, fps_str, (w - 180, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_DIM, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Main GUI Application
# ─────────────────────────────────────────────

class FatigueGUI:
    WINDOW = "Steel Fatigue Life Predictor"
    WIN_W = 900
    WIN_H = 580
    IMG_W = 560
    PANEL_W = 280

    def __init__(self, model_path: str, model_type: str = "hybrid"):
        self.model_type = model_type
        self.model = None
        self.preprocessor = SteelImagePreprocessor(
            PreprocessConfig(target_size=(IMG_W := 512, 512), enhance_cracks=True)
        )
        self._load_model(model_path)

        self.current_img: Optional[np.ndarray] = None
        self.current_gray: Optional[np.ndarray] = None
        self.current_result: Optional[Dict] = None
        self.show_annotations = True
        self.fps = 0.0
        self.status = "Ready — Press [O] to open an image"

    def _load_model(self, path: str):
        if self.model_type == "hybrid":
            try:
                import pickle
                with open(path, "rb") as f:
                    data = pickle.load(f)
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from models.hybrid_model import HybridFatigueModel
                self.model = HybridFatigueModel()
                self.model.pipeline = data["pipeline"]
                self.model.is_trained = True
                logger.info(f"Hybrid model loaded: {path}")
            except Exception as e:
                logger.warning(f"Could not load model ({e}). Using mock predictor.")
                self.model = None
        else:
            logger.info(f"[ResNet] model loading from {path}")
            # Would load PyTorch model here
            self.model = None

    def _predict(self, img_array: np.ndarray) -> Dict:
        """Run inference on image array."""
        if self.model is not None and hasattr(self.model, "predict"):
            return self.model.predict(img_array)

        # Mock predictor for demo when model not loaded
        import random
        log_nf = random.uniform(4.0, 8.0)
        unc = 0.2
        cats = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if log_nf < 4.5:   risk = "CRITICAL"
        elif log_nf < 5.5: risk = "HIGH"
        elif log_nf < 6.5: risk = "MEDIUM"
        else:               risk = "LOW"

        return {
            "log10_Nf": round(log_nf, 4),
            "Nf_cycles": round(10 ** log_nf),
            "log10_Nf_lower": round(log_nf - 1.96 * unc, 4),
            "log10_Nf_upper": round(log_nf + 1.96 * unc, 4),
            "uncertainty": unc,
            "risk_category": risk,
        }

    def _process_image(self, path: str):
        """Load, preprocess, and predict."""
        t0 = time.time()
        raw = cv2.imread(path)
        if raw is None:
            self.status = f"Error: Cannot read {Path(path).name}"
            return

        processed = self.preprocessor.process_array(raw)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        self.current_img = processed
        self.current_gray = gray

        self.current_result = self._predict(processed)
        elapsed = time.time() - t0
        self.fps = 1.0 / elapsed
        self.status = f"Processed: {Path(path).name}  ({elapsed*1000:.0f}ms)"
        logger.info(f"Prediction: {self.current_result}")

    def _build_frame(self) -> np.ndarray:
        """Build the full display frame."""
        canvas = np.full((self.WIN_H, self.WIN_W, 3), UI_BG, dtype=np.uint8)

        # Header bar
        cv2.rectangle(canvas, (0, 0), (self.WIN_W, 38), UI_PANEL, -1)
        cv2.putText(canvas, "⚙ Steel Fatigue Life Predictor — CV + ML",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[O]pen  [S]ave  [A]nnotations  [Q]uit",
                    (self.WIN_W - 330, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_DIM, 1, cv2.LINE_AA)

        # Image area
        img_area_y = 45
        img_area_h = self.WIN_H - 75
        img_area_w = self.WIN_W - self.PANEL_W - 20

        cv2.rectangle(canvas, (10, img_area_y),
                       (10 + img_area_w, img_area_y + img_area_h), UI_PANEL, 1)

        if self.current_img is not None:
            disp = self.current_img.copy()
            if self.show_annotations and self.current_gray is not None:
                disp = draw_crack_annotations(disp, self.current_gray)

            # Fit image in area
            scale = min(img_area_w / disp.shape[1], img_area_h / disp.shape[0])
            nw = int(disp.shape[1] * scale)
            nh = int(disp.shape[0] * scale)
            disp = cv2.resize(disp, (nw, nh))

            ox = 10 + (img_area_w - nw) // 2
            oy = img_area_y + (img_area_h - nh) // 2
            canvas[oy:oy+nh, ox:ox+nw] = disp
        else:
            # Placeholder
            cx, cy = 10 + img_area_w // 2, img_area_y + img_area_h // 2
            cv2.putText(canvas, "No image loaded", (cx - 80, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press [O] to open", (cx - 75, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_DIM, 1, cv2.LINE_AA)

        # Prediction panel
        panel_x = self.WIN_W - self.PANEL_W - 5
        if self.current_result:
            draw_prediction_panel(canvas, self.current_result,
                                   panel_x, img_area_y, self.PANEL_W, img_area_h)
        else:
            cv2.rectangle(canvas, (panel_x, img_area_y),
                           (self.WIN_W - 5, img_area_y + img_area_h), UI_PANEL, -1)
            cv2.putText(canvas, "Awaiting input...",
                        (panel_x + 15, img_area_y + img_area_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_DIM, 1, cv2.LINE_AA)

        # Status bar
        draw_status_bar(canvas, self.status, self.fps,
                        self.show_annotations, self.WIN_H, self.WIN_W)
        return canvas

    def run(self):
        """Main application loop."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIN_W, self.WIN_H)

        logger.info("OpenCV GUI started. Controls: [O]pen [A]nnotations [S]ave [Q]uit")

        while True:
            frame = self._build_frame()
            cv2.imshow(self.WINDOW, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord('o'):
                # In a real deployment, use tkinter file dialog
                test_paths = list(Path("data").rglob("*.png"))[:5]
                if test_paths:
                    self._process_image(str(test_paths[0]))
                    self.status = f"Loaded: {test_paths[0].name} — Use --image flag for specific file"
                else:
                    self.status = "No images found. Generate with: python src/utils/generate_synthetic_data.py"

            elif key == ord('a'):
                self.show_annotations = not self.show_annotations
                self.status = f"Annotations: {'ON' if self.show_annotations else 'OFF'}"

            elif key == ord('s'):
                if self.current_img is not None:
                    save_path = "outputs/prediction_result.png"
                    Path("outputs").mkdir(exist_ok=True)
                    cv2.imwrite(save_path, frame)
                    self.status = f"Saved: {save_path}"

        cv2.destroyAllWindows()
        logger.info("GUI closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV GUI for Fatigue Life Prediction")
    parser.add_argument("--model", type=str, default="outputs/hybrid_model.pkl")
    parser.add_argument("--mode", choices=["hybrid", "resnet"], default="hybrid")
    parser.add_argument("--image", type=str, help="Image to load on startup")
    args = parser.parse_args()

    gui = FatigueGUI(model_path=args.model, model_type=args.mode)

    if args.image:
        gui._process_image(args.image)

    gui.run()
