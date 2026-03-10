"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic microscopy images of steel alloys with
labeled fatigue life values for training and testing.

Usage:
    python src/utils/generate_synthetic_data.py --samples 500 --output data/synthetic
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import math


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def generate_grain_structure(size=512, n_grains=50, irregularity=0.3):
    """Simulate polycrystalline grain structure using Voronoi tessellation."""
    img = np.zeros((size, size), dtype=np.uint8)

    # Random seed points for Voronoi
    seeds = np.random.randint(10, size - 10, size=(n_grains, 2))

    # Color each grain slightly differently (texture variation)
    for y in range(size):
        for x in range(0, size, 4):  # Sparse for speed
            dists = np.sqrt(((seeds - [x, y]) ** 2).sum(axis=1))
            nearest = np.argmin(dists)
            intensity = 100 + (nearest * 3) % 80
            img[y, min(x, size-1)] = intensity

    # Interpolate gaps
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # Add noise for realism
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Grain boundaries (darker lines)
    blurred = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(blurred, 20, 60)
    img[edges > 0] = np.clip(img[edges > 0].astype(int) - 40, 0, 255)

    return img


def add_cracks(img, n_cracks=3, max_length=150, severity=0.5):
    """Add fatigue cracks to the image."""
    result = img.copy()
    crack_data = []

    for _ in range(n_cracks):
        # Crack start point
        x = np.random.randint(50, img.shape[1] - 50)
        y = np.random.randint(50, img.shape[0] - 50)

        # Crack direction and properties
        angle = np.random.uniform(0, 2 * math.pi)
        length = int(max_length * severity * np.random.uniform(0.5, 1.5))
        width = max(1, int(severity * 3))

        # Draw branching crack
        pts = [(x, y)]
        for step in range(length // 5):
            angle += np.random.uniform(-0.3, 0.3)  # Wander
            nx = int(pts[-1][0] + 5 * math.cos(angle))
            ny = int(pts[-1][1] + 5 * math.sin(angle))
            nx = np.clip(nx, 0, img.shape[1] - 1)
            ny = np.clip(ny, 0, img.shape[0] - 1)
            pts.append((nx, ny))

        # Draw main crack
        for i in range(1, len(pts)):
            cv2.line(result, pts[i-1], pts[i], 15, width)

        # Optional branch
        if np.random.random() > 0.5 and len(pts) > 10:
            branch_start = pts[len(pts)//2]
            b_angle = angle + np.random.uniform(0.4, 1.2)
            b_len = length // 3
            bx, by = branch_start
            for _ in range(b_len // 5):
                b_angle += np.random.uniform(-0.2, 0.2)
                nbx = int(bx + 5 * math.cos(b_angle))
                nby = int(by + 5 * math.sin(b_angle))
                nbx = np.clip(nbx, 0, img.shape[1] - 1)
                nby = np.clip(nby, 0, img.shape[0] - 1)
                cv2.line(result, (bx, by), (nbx, nby), 10, max(1, width-1))
                bx, by = nbx, nby

        crack_data.append({
            "start": (x, y),
            "length": length,
            "severity": severity
        })

    return result, crack_data


def add_pores(img, n_pores=20, max_radius=8):
    """Add micro-pores/voids typical in cast alloys."""
    result = img.copy()
    for _ in range(n_pores):
        x = np.random.randint(5, img.shape[1] - 5)
        y = np.random.randint(5, img.shape[0] - 5)
        r = np.random.randint(1, max_radius)
        cv2.circle(result, (x, y), r, 20, -1)
        # Bright rim (oxidation)
        cv2.circle(result, (x, y), r + 1, 180, 1)
    return result


def add_inclusions(img, n_inclusions=10):
    """Add second-phase inclusions (brighter regions)."""
    result = img.copy()
    for _ in range(n_inclusions):
        x = np.random.randint(10, img.shape[1] - 10)
        y = np.random.randint(10, img.shape[0] - 10)
        rx = np.random.randint(2, 12)
        ry = np.random.randint(2, 8)
        angle = np.random.randint(0, 180)
        cv2.ellipse(result, (x, y), (rx, ry), angle, 0, 360, 220, -1)
    return result


def fatigue_life_model(grain_size, n_cracks, crack_severity, porosity, n_inclusions):
    """
    Physics-inspired fatigue life formula (simplified).
    Returns log10(N_f) where N_f is cycles to failure.
    
    Based on:
    - Smaller grains → longer life (Hall-Petch like)
    - More/larger cracks → shorter life
    - More pores → shorter life
    - Inclusions → crack initiation sites
    """
    # Base life for perfect material
    log_Nf_base = 7.0  # ~10^7 cycles

    # Grain size effect (fine grain = longer life)
    grain_effect = -0.5 * np.log10(grain_size / 50.0 + 1)

    # Crack effect
    crack_effect = -1.2 * n_cracks * crack_severity

    # Porosity effect
    pore_effect = -0.8 * (porosity / 20.0)

    # Inclusion effect
    inclusion_effect = -0.3 * (n_inclusions / 10.0)

    # Random scatter (material variability)
    scatter = np.random.normal(0, 0.15)

    log_Nf = log_Nf_base + grain_effect + crack_effect + pore_effect + inclusion_effect + scatter
    log_Nf = np.clip(log_Nf, 3.5, 8.5)  # Physical bounds

    return float(log_Nf)


def generate_sample(sample_id, size=512):
    """Generate a single labeled sample."""
    # Randomize material condition
    n_grains = np.random.randint(30, 120)
    grain_size = 500 // n_grains  # Proxy: fewer grains = larger grains

    # Damage state
    damage_level = np.random.choice(['pristine', 'mild', 'moderate', 'severe'],
                                     p=[0.2, 0.3, 0.3, 0.2])

    damage_map = {
        'pristine': (0, 0.1, 5, 3),
        'mild':     (1, 0.3, 10, 6),
        'moderate': (2, 0.6, 15, 10),
        'severe':   (4, 0.9, 25, 15),
    }
    n_cracks, crack_sev, n_pores, n_inclusions = damage_map[damage_level]

    # Generate base grain image
    img = generate_grain_structure(size=size, n_grains=n_grains)

    # Add damage features
    img, crack_data = add_cracks(img, n_cracks=n_cracks, severity=crack_sev)
    img = add_pores(img, n_pores=n_pores)
    img = add_inclusions(img, n_inclusions=n_inclusions)

    # Compute fatigue life label
    log_Nf = fatigue_life_model(grain_size, n_cracks, crack_sev, n_pores, n_inclusions)
    Nf = 10 ** log_Nf

    # Convert to BGR for saving
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    metadata = {
        "sample_id": sample_id,
        "damage_level": damage_level,
        "n_grains": n_grains,
        "grain_size_um": grain_size,
        "n_cracks": n_cracks,
        "crack_severity": crack_sev,
        "n_pores": n_pores,
        "n_inclusions": n_inclusions,
        "log10_Nf": round(log_Nf, 4),
        "Nf_cycles": round(Nf, 0),
        "risk_category": (
            "CRITICAL" if log_Nf < 4.5 else
            "HIGH"     if log_Nf < 5.5 else
            "MEDIUM"   if log_Nf < 6.5 else
            "LOW"
        )
    }

    return img_bgr, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fatigue microstructure dataset")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--size", type=int, default=512, help="Image size (NxN)")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    print(f"Generating {args.samples} synthetic steel microstructure samples...")

    for i in tqdm(range(args.samples)):
        img, meta = generate_sample(f"SYN_{i:04d}", size=args.size)
        img_path = img_dir / f"SYN_{i:04d}.png"
        cv2.imwrite(str(img_path), img)
        meta["image_path"] = str(img_path)
        all_metadata.append(meta)

    # Save labels
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Save CSV for easy loading
    import csv
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"\n✅ Dataset generated!")
    print(f"   Images: {img_dir}")
    print(f"   Labels: {labels_path}")
    print(f"   CSV:    {csv_path}")
    print(f"\nLabel distribution:")
    cats = [m["risk_category"] for m in all_metadata]
    for cat in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = cats.count(cat)
        print(f"  {cat:8s}: {count:4d} samples ({100*count/len(all_metadata):.1f}%)")


if __name__ == "__main__":
    main()
