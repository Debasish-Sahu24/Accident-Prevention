import numpy as np
import pandas as pd
import os, pickle
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(BASE_DIR / "src"))
from pipeline import VisualFeatureExtractor, NumericalProcessor, FusionLayer, AccidentPredictor

# ── Load dataset ─────────────────────────────────────────────────────────────
print("[Data] Loading india_accident_main.csv …")
df = pd.read_csv(DATA_DIR / "india_accident_main.csv")
print(f"       {len(df):,} rows loaded | Columns: {list(df.columns)}")

# ── Feature extraction ────────────────────────────────────────────────────────
print("\n[Features] Extracting visual features …")
visual_feats = VisualFeatureExtractor.extract_from_dataframe(df)
print(f"           Visual shape: {visual_feats.shape}")

print("[Features] Processing numerical features …")
num_proc = NumericalProcessor()
num_feats = num_proc.fit_transform(df)
print(f"           Numerical shape: {num_feats.shape}")

# ── Late Fusion ───────────────────────────────────────────────────────────────
print("\n[Fusion] Concatenating feature vectors …")
X = FusionLayer.fuse_batch(visual_feats, num_feats)
y = df["accident_occurred"].values
print(f"         Fused matrix: {X.shape}  |  Label distribution: "
      f"{(y==0).sum()} neg / {(y==1).sum()} pos")

# ── Compute scale_pos_weight for XGBoost ─────────────────────────────────────
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
spw = round(neg_count / max(pos_count, 1), 2)
print(f"\n[XGBoost] scale_pos_weight = {spw}")

# ── Train ─────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

predictor = AccidentPredictor(scale_pos_weight=spw)
predictor.train(X_train, y_train, apply_smote=True)

# ── Evaluate ──────────────────────────────────────────────────────────────────
predictor.evaluate(X_test, y_test)

# ── Save artefacts ────────────────────────────────────────────────────────────
predictor.save(str(MODEL_DIR / "xgb_accident_model.json"))

with open(MODEL_DIR / "numerical_scaler.pkl", "wb") as f:
    pickle.dump(num_proc.scaler, f)
print(f"[Save] Scaler saved → {MODEL_DIR / 'numerical_scaler.pkl'}")

# Save feature column names for the API
import json
feature_meta = {
    "visual_cols":   ["object_count_yolo", "bbox_velocity_variance", "road_density_seg"],
    "numerical_cols":["weather_index", "road_friction", "hour_of_day",
                      "traffic_volume_lag_5min", "is_peak_hour", "is_night",
                      "month_sin", "road_type_enc"],
    "fused_dim":     FusionLayer.FUSED_DIM,
    "scale_pos_weight": spw,
    "accident_rate": round(float(pos_count / len(y)), 4),
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
print(f"[Save] Metadata → {MODEL_DIR / 'model_meta.json'}")
print("\n✅ Training pipeline complete!")
