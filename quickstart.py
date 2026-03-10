"""
quickstart.py
=============
One-command end-to-end demo:
  1. Generate synthetic microscopy dataset
  2. Train ResNet50 CNN (or skip if checkpoint exists)
  3. Run inference on a test sample
  4. Save annotated output image

Usage
-----
python quickstart.py                     # full pipeline
python quickstart.py --skip-train        # inference only (needs checkpoint)
python quickstart.py --arch custom_cnn   # use custom CNN instead
"""

import subprocess, sys, os, json, time, argparse
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────
G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
B  = "\033[94m"; D = "\033[2m";  E = "\033[0m"
M  = "\033[95m"

BANNER = f"""
"""

def run(cmd, desc=""):
    print(f"\n{B} {desc}{E}")
    print(f"{D}  {cmd}{E}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"{R}  ✗ Failed (exit {result.returncode}){E}")
        return False
    return True


def main():
    print(BANNER)
    p = argparse.ArgumentParser()
    p.add_argument("--arch",        default="resnet50", choices=["custom_cnn","resnet50","vgg16"])
    p.add_argument("--samples",     type=int, default=200)
    p.add_argument("--epochs",      type=int, default=5, help="Quick demo: use 5, production: use 60")
    p.add_argument("--skip-train",  action="store_true")
    p.add_argument("--batch-size",  type=int, default=8)
    args = p.parse_args()

    steps = []

    # ── Step 1: Generate data ──────────────────────────────────
    labels_path = Path("data/synthetic/labels.json")
    if not labels_path.exists():
        steps.append(("Generate synthetic dataset",
            f"python src/utils/generate_synthetic_data.py --samples {args.samples}"))
    else:
        print(f"{G}✓{E} Synthetic dataset already exists ({labels_path})")

    ckpt = Path(f"outputs/best_{args.arch}.pth")
    if not args.skip_train:
        steps.append(("Train CNN model",
            f"python src/models/trainer.py "
            f"--arch {args.arch} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--labels data/synthetic/labels.json "
            f"--save_dir outputs"))
    else:
        print(f"{Y}  Skipping training — using existing checkpoint: {ckpt}{E}")

    # Run queued steps
    for desc, cmd in steps:
        if not run(cmd, desc):
            sys.exit(1)

    # ── Step 3: Inference demo ─────────────────────────────────
    print(f"\n{B} Running inference demo{E}")
    sys.path.insert(0, "src")

    try:
        imgs = sorted(Path("data/synthetic/images").glob("*.png"))[:3]
        if not imgs:
            print(f"{Y}  No images found — generate data first{E}")
            return

        from inference.inference_engine import InferenceEngine
        ckpt_str = str(ckpt)
        engine = InferenceEngine(ckpt_str)

        Path("outputs/demo").mkdir(parents=True, exist_ok=True)
        print(f"\n  {'Sample':<35} {'Risk':8} {'log(Nf)':9} {'Nf Cycles':>14}  {'ms':>5}")
        print("  " + "─"*75)

        for img_path in imgs:
            r = engine.predict(str(img_path), return_gradcam=True, annotate=True)
            risk_colors = {'LOW':G,'MEDIUM':Y,'HIGH':Y,'CRITICAL':R}
            col = risk_colors.get(r.risk_category, D)
            print(f"  {img_path.name:<35} {col}{r.risk_category:8}{E} "
                  f"{r.log10_Nf:9.4f} {r.Nf_cycles:>14.3e}  {r.inference_ms:>5.0f}")
            out_path = f"outputs/demo/{img_path.stem}_annotated.png"
            engine.save_annotated(r, out_path)

        print(f"\n{G} Demo complete!{E}")
        print(f"   Annotated images saved in: {M}outputs/demo/{E}")
        print(f"\n{D}Next steps:{E}")
        print(f"   {G}1.{E} Open {B}dashboard.html{E} in your browser for the interactive UI")
        print(f"   {G}2.{E} Start the API: {B}uvicorn api.app:app --port 8000{E}")
        print(f"   {G}3.{E} Replace synthetic data with real microscopy images in {B}data/raw/{E}")

    except Exception as ex:
        print(f"{R}  Inference demo failed: {ex}{E}")
        print(f"{D}  This is normal if running without a trained model.{E}")
        print(f"  Open {B}dashboard.html{E} in your browser for a demo with mock predictions.")


if __name__ == "__main__":
    main()
