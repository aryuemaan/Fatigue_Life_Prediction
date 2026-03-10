"""
predict_cli.py
--------------
Command-line interface for fatigue life prediction.
The simplest way to get a prediction from any image.

Usage:
    python predict_cli.py --image path/to/steel_image.png
    python predict_cli.py --image path/to/steel_image.png --visualize
    python predict_cli.py --batch data/raw/ --output results.csv
"""

import cv2
import numpy as np
import argparse
import json
import csv
import sys
import time
from pathlib import Path

# ASCII Art Header
BANNER = r"""
"""

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
RISK_BARS = {
    "LOW":      ("▓▓▓▓▓▓▓▓░░", "Excellent"),
    "MEDIUM":   ("▓▓▓▓▓▓░░░░", "Monitor"),
    "HIGH":     ("▓▓▓▓░░░░░░", "Inspect"),
    "CRITICAL": ("▓▓░░░░░░░░", "IMMEDIATE ACTION"),
}


def print_result(result: dict, filename: str = ""):
    """Pretty-print prediction result to terminal."""
    risk = result.get("risk_category", "UNKNOWN")
    log_nf = result.get("log10_Nf", 0)
    nf = result.get("Nf_cycles", 0)
    ci_lo = result.get("log10_Nf_lower", log_nf - 0.2)
    ci_hi = result.get("log10_Nf_upper", log_nf + 0.2)
    unc = result.get("uncertainty", 0.15)

    emoji = RISK_EMOJI.get(risk, "")
    bar, status = RISK_BARS.get(risk, ("░░░░░░░░░░", "Unknown"))

    print("\n" + "─" * 55)
    if filename:
        print(f"  File: {filename}")
    print(f"  {emoji} Risk Category:  {risk}  [{status}]")
    print(f"  Life Bar:      {bar}")
    print(f"  log₁₀(N_f):   {log_nf:.4f}")
    nf_str = f"{nf:.3e}" if nf > 1e6 else f"{int(nf):,}"
    print(f"  N_f (cycles):  {nf_str}")
    print(f"  95% CI:        [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Uncertainty:   ±{unc:.3f} (log scale)")
    print("─" * 55)

    # Interpretation
    interps = {
        "LOW":      "Material is in good condition. Expected service life is long.",
        "MEDIUM":   "Moderate degradation detected. Schedule routine inspection.",
        "HIGH":     "Significant damage features found. Increase inspection frequency.",
        "CRITICAL": "CRITICAL: High risk of imminent fatigue failure. Remove from service.",
    }
    print(f"\n  {interps.get(risk, 'No interpretation available.')}\n")


def visualize_prediction(img_path: str, result: dict):
    """Show annotated image with OpenCV."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [!] Cannot load image for visualization: {img_path}")
        return

    # Resize for display
    h, w = img.shape[:2]
    scale = min(800 / w, 600 / h)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    risk = result.get("risk_category", "UNKNOWN")
    log_nf = result.get("log10_Nf", 0)
    nf = result.get("Nf_cycles", 0)

    colors = {"LOW": (50, 220, 50), "MEDIUM": (50, 200, 255),
              "HIGH": (30, 100, 255), "CRITICAL": (50, 50, 255)}
    color = colors.get(risk, (180, 180, 180))

    # Header overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 65), (12, 14, 22), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.putText(img, f"RISK: {risk}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    nf_str = f"{nf:.2e}" if nf > 1e5 else f"{int(nf):,}"
    cv2.putText(img, f"log(Nf)={log_nf:.3f}  |  Nf={nf_str} cycles",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, img.shape[0]-4), (img.shape[1], img.shape[0]), color, -1)
    cv2.imshow("Fatigue Life Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_model(model_path: str):
    """Load the prediction model."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    if not Path(model_path).exists():
        print(f"  [!] Model not found: {model_path}")
        print(f"      Run: python src/models/hybrid_model.py --labels data/synthetic/labels.json --train")
        return None

    try:
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        from models.hybrid_model import HybridFatigueModel
        model = HybridFatigueModel()
        model.pipeline = data["pipeline"]
        model.is_trained = True
        return model
    except Exception as e:
        print(f"  [!] Model load error: {e}. Using demo predictor.")
        return None


def mock_predict(img_array) -> dict:
    """Demo predictor when model not available."""
    seed = int(np.mean(img_array) * 100) % 10000
    np.random.seed(seed)
    log_nf = float(np.random.uniform(4.2, 8.0))
    unc = 0.18
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


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="Fatigue Life Predictor CLI")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--batch", type=str, help="Folder of images for batch prediction")
    parser.add_argument("--model", type=str, default="outputs/hybrid_model.pkl")
    parser.add_argument("--output", type=str, help="Save batch results to CSV")
    parser.add_argument("--visualize", action="store_true", help="Show OpenCV visualization")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from preprocessing.preprocessor import SteelImagePreprocessor, PreprocessConfig
        preprocessor = SteelImagePreprocessor(
            PreprocessConfig(target_size=(224, 224), enhance_cracks=True)
        )
    except:
        preprocessor = None

    def predict_one(img_path: str) -> dict:
        img = cv2.imread(img_path)
        if img is None:
            return {"error": f"Cannot load {img_path}"}
        if preprocessor:
            processed = preprocessor.process_array(img)
        else:
            processed = cv2.resize(img, (224, 224))
        if model:
            return model.predict(processed)
        return mock_predict(processed)

    if args.image:
        t0 = time.time()
        result = predict_one(args.image)
        elapsed = (time.time() - t0) * 1000

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(result, args.image)
            print(f"   Inference: {elapsed:.1f}ms")

        if args.visualize:
            visualize_prediction(args.image, result)

    elif args.batch:
        batch_dir = Path(args.batch)
        image_files = sorted(list(batch_dir.glob("*.png")) +
                              list(batch_dir.glob("*.jpg")) +
                              list(batch_dir.glob("*.tif")))

        if not image_files:
            print(f"  [!] No images found in {batch_dir}")
            return

        print(f"\n  Processing {len(image_files)} images...\n")
        all_results = []
        t0 = time.time()

        for i, img_path in enumerate(image_files, 1):
            result = predict_one(str(img_path))
            result["filename"] = img_path.name
            all_results.append(result)
            risk = result.get("risk_category", "?")
            log_nf = result.get("log10_Nf", 0)
            emoji = RISK_EMOJI.get(risk, "")
            print(f"  [{i:3d}/{len(image_files)}] {img_path.name:30s} "
                  f"{emoji} {risk:8s} log(Nf)={log_nf:.3f}")

        elapsed = time.time() - t0
        print(f"\n  Processed {len(all_results)} samples in {elapsed:.1f}s")

        # Summary
        risks = [r.get("risk_category", "?") for r in all_results]
        print("\n  Risk Distribution:")
        for cat in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            n = risks.count(cat)
            pct = 100 * n / len(risks)
            print(f"     {RISK_EMOJI[cat]} {cat:8s}: {n:4d} ({pct:.1f}%)")

        # Save to CSV
        if args.output:
            with open(args.output, "w", newline="") as f:
                fields = ["filename", "log10_Nf", "Nf_cycles", "log10_Nf_lower",
                          "log10_Nf_upper", "uncertainty", "risk_category"]
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\n  Results saved to: {args.output}")

    else:
        parser.print_help()
        print("\n  Quick start:")
        print("    python predict_cli.py --image data/synthetic/images/SYN_0001.png --visualize")
        print("    python predict_cli.py --batch data/synthetic/images/ --output results.csv\n")


if __name__ == "__main__":
    main()
