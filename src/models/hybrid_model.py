"""
hybrid_model.py
---------------
Hybrid model: OpenCV feature extraction + ML regression.
Works WITHOUT GPU. Interpretable. Fast inference.

Pipeline:
  Image → OpenCV Features → PCA → Gradient Boosting / Random Forest → log_Nf

Usage:
    python src/models/hybrid_model.py --labels data/synthetic/labels.json --train
"""

import numpy as np
import cv2
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from loguru import logger

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.feature_extractor import SteelFeatureExtractor, FeatureVector
from preprocessing.preprocessor import SteelImagePreprocessor, PreprocessConfig


class HOGFeatureExtractor:
    """HOG (Histogram of Oriented Gradients) for additional shape features."""

    def __init__(self, win_size=(64, 64), block_size=(16, 16),
                 block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )
        self.win_size = win_size

    def compute(self, gray: np.ndarray) -> np.ndarray:
        resized = cv2.resize(gray, self.win_size)
        return self.hog.compute(resized).flatten()


class HybridFatigueModel:
    """
    Full hybrid pipeline: preprocessing + feature extraction + ML model.
    """

    def __init__(self, use_hog: bool = True, use_ml_type: str = "gbm"):
        self.preprocessor = SteelImagePreprocessor(
            PreprocessConfig(target_size=(224, 224), enhance_cracks=True)
        )
        self.feature_extractor = SteelFeatureExtractor()
        self.hog_extractor = HOGFeatureExtractor() if use_hog else None
        self.use_hog = use_hog

        # ML pipeline
        if use_ml_type == "gbm":
            regressor = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=5, subsample=0.8,
                random_state=42
            )
        else:
            regressor = RandomForestRegressor(
                n_estimators=300, max_depth=12,
                min_samples_split=4, random_state=42, n_jobs=-1
            )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("model", regressor)
        ])
        self.is_trained = False
        self.feature_names = FeatureVector.feature_names()

    def _extract_features(self, img_path: str) -> np.ndarray:
        """Extract all features from an image file."""
        result = self.preprocessor.process(img_path)
        gray = result["gray"]
        bgr = result["processed"]

        # Structured features
        fv = self.feature_extractor.extract(gray)
        structured = fv.to_array()

        if self.use_hog and self.hog_extractor:
            hog_feat = self.hog_extractor.compute(gray)
            return np.concatenate([structured, hog_feat])
        return structured

    def _extract_features_from_array(self, img_array: np.ndarray) -> np.ndarray:
        """Extract features directly from numpy array."""
        processed = self.preprocessor.process_array(img_array)
        if processed.ndim == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        fv = self.feature_extractor.extract(gray)
        structured = fv.to_array()

        if self.use_hog and self.hog_extractor:
            hog_feat = self.hog_extractor.compute(gray)
            return np.concatenate([structured, hog_feat])
        return structured

    def build_features(self, labels_json: str) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix from dataset."""
        with open(labels_json) as f:
            data = json.load(f)

        X, y = [], []
        failed = 0
        logger.info(f"Extracting features from {len(data)} images...")

        for item in data:
            try:
                feat = self._extract_features(item["image_path"])
                X.append(feat)
                y.append(item["log10_Nf"])
            except Exception as e:
                logger.warning(f"Failed {item['image_path']}: {e}")
                failed += 1

        logger.info(f"Extracted {len(X)} feature vectors ({failed} failed)")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, labels_json: str, cv_folds: int = 5) -> Dict:
        """Train the hybrid model."""
        X, y = self.build_features(labels_json)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        logger.info(f"Feature matrix: {X.shape}")
        logger.info("Cross-validating...")

        cv_scores = cross_val_score(
            self.pipeline, X, y, cv=cv_folds,
            scoring="r2", n_jobs=-1
        )
        logger.info(f"CV R² scores: {cv_scores}")
        logger.info(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit on all data
        self.pipeline.fit(X, y)
        train_preds = self.pipeline.predict(X)
        train_r2 = r2_score(y, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y, train_preds))

        self.is_trained = True
        logger.info(f"Final train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")

        return {
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

    def predict(self, img_input) -> Dict:
        """
        Predict fatigue life.
        img_input: str (path) or np.ndarray
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load().")

        if isinstance(img_input, str):
            feat = self._extract_features(img_input)
        else:
            feat = self._extract_features_from_array(img_input)

        feat = np.nan_to_num(feat.reshape(1, -1), nan=0.0)
        log_Nf = float(self.pipeline.predict(feat)[0])
        Nf = 10 ** log_Nf

        # Approximate uncertainty from GBM staged predictions
        if hasattr(self.pipeline["model"], "staged_predict"):
            staged = list(self.pipeline["model"].staged_predict(
                self.pipeline[:-1].transform(feat)
            ))
            uncertainty = float(np.std(staged[-20:])) if len(staged) >= 20 else 0.1
        else:
            uncertainty = 0.15  # Default for RF

        return {
            "log10_Nf": round(log_Nf, 4),
            "Nf_cycles": round(Nf),
            "log10_Nf_lower": round(log_Nf - 1.96 * uncertainty, 4),
            "log10_Nf_upper": round(log_Nf + 1.96 * uncertainty, 4),
            "uncertainty": round(uncertainty, 4),
            "risk_category": self._risk_category(log_Nf),
        }

    @staticmethod
    def _risk_category(log_Nf: float) -> str:
        if log_Nf < 4.5:   return "CRITICAL"
        if log_Nf < 5.5:   return "HIGH"
        if log_Nf < 6.5:   return "MEDIUM"
        return "LOW"

    def get_feature_importance(self) -> Dict[str, float]:
        """Return top feature importances (GBM/RF only)."""
        if not self.is_trained:
            return {}
        model = self.pipeline["model"]
        if hasattr(model, "feature_importances_"):
            pca = self.pipeline["pca"]
            # Map PCA components back - simplified: just return top original features
            importances = model.feature_importances_
            names = [f"PCA_{i}" for i in range(len(importances))]
            sorted_idx = np.argsort(importances)[::-1][:10]
            return {names[i]: float(importances[i]) for i in sorted_idx}
        return {}

    def save(self, path: str):
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "is_trained": self.is_trained,
                "use_hog": self.use_hog,
            }, f)
        logger.info(f"Hybrid model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.is_trained = data["is_trained"]
        self.use_hog = data["use_hog"]
        logger.info(f"Hybrid model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/synthetic/labels.json")
    parser.add_argument("--save", default="outputs/hybrid_model.pkl")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str, help="Image path to predict")
    args = parser.parse_args()

    model = HybridFatigueModel(use_hog=True, use_ml_type="gbm")

    if args.train:
        metrics = model.train(args.labels)
        print(f"\n📊 Training Metrics:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")
        model.save(args.save)
        print(f"\n✅ Model saved: {args.save}")

    if args.predict:
        if not args.train:
            model.load(args.save)
        result = model.predict(args.predict)
        print(f"\n🔬 Prediction for {args.predict}:")
        for k, v in result.items():
            print(f"   {k}: {v}")
