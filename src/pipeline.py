"""
pipeline.py — Late Fusion Accident Prediction Pipeline
Indian Roads Edition
=====================================================
Visual Features (YOLO) ──┐
                          ├──► Fusion Layer ──► XGBoost ──► Risk Score 0-100%
Numerical Features ───────┘

Author: AccidentAI System
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  VISUAL FEATURE EXTRACTOR  (YOLO proxy for offline training)
# ─────────────────────────────────────────────────────────────────────────────

class VisualFeatureExtractor:
    """
    In production: wraps YOLOv11 to process live video frames.
    During training: reads pre-computed columns from the CSV dataset.
    """

    VISUAL_DIM = 3   # [object_count, bbox_velocity_variance, road_density]

    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.35):
        self.model_path = model_path
        self.conf = conf
        self._model = None
        self._track_history: dict = {}

    def _load_model(self):
        """Lazy-load YOLO only when needed (inference mode)."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                print(f"[YOLO] Loaded {self.model_path}")
            except Exception as e:
                print(f"[YOLO] Could not load model: {e}")
                self._model = None

    # ── LIVE INFERENCE ────────────────────────────────────────────────────────
    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single BGR video frame and return visual feature vector.

        Args:
            frame: np.ndarray  shape (H, W, 3)  BGR

        Returns:
            np.ndarray shape (3,)
            [object_count, bbox_velocity_variance, road_density_0_to_1]
        """
        import cv2
        from collections import deque

        self._load_model()
        if self._model is None:
            return np.zeros(self.VISUAL_DIM, dtype=np.float32)

        h, w = frame.shape[:2]
        results = self._model.track(frame, conf=self.conf, persist=True, verbose=False)
        boxes   = results[0].boxes

        # Feature 1 — Object count
        obj_count = float(len(boxes))

        # Feature 2 — Bounding-box velocity variance (via tracked centroids)
        ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None \
               else np.arange(len(boxes))
        xyxy = boxes.xyxy.cpu().numpy()
        velocities = []
        for tid, box in zip(ids, xyxy):
            cx, cy = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
            if tid not in self._track_history:
                self._track_history[tid] = deque(maxlen=5)
            hist = self._track_history[tid]
            if len(hist) > 0:
                velocities.append(np.sqrt((cx-hist[-1][0])**2 + (cy-hist[-1][1])**2))
            hist.append((cx, cy))
        vel_var = float(np.var(velocities)) if len(velocities) > 1 else 0.0

        # Feature 3 — Road density via segmentation masks
        if results[0].masks is not None:
            import cv2 as _cv2
            masks = results[0].masks.data.cpu().numpy()
            union = np.zeros((h, w), dtype=bool)
            for m in masks:
                r = _cv2.resize(m.astype(np.uint8), (w, h))
                union |= r.astype(bool)
            road_density = float(union.sum()) / (h * w)
        else:
            area = sum((b[2]-b[0])*(b[3]-b[1]) for b in xyxy)
            road_density = min(float(area) / (h * w), 1.0)

        return np.array([obj_count, vel_var, road_density], dtype=np.float32)

    # ── TRAINING MODE (use CSV pre-computed columns) ──────────────────────────
    @staticmethod
    def extract_from_dataframe(df: pd.DataFrame) -> np.ndarray:
        """
        Pull pre-computed YOLO-proxy columns from the training CSV.

        Expected columns:
            object_count_yolo, bbox_velocity_variance, road_density_seg
        """
        cols = ["object_count_yolo", "bbox_velocity_variance", "road_density_seg"]
        return df[cols].values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NUMERICAL FEATURE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class NumericalProcessor:
    """
    Encodes + scales tabular features.
    Tabular input: [weather_index, road_friction, hour_of_day, traffic_volume_lag_5min]
    Extended input: adds is_peak_hour, is_night, month, road_type_encoded
    """

    NUMERICAL_DIM = 8

    ROAD_TYPE_MAP = {
        "Urban Arterial": 0, "State Highway": 1, "National Highway": 2,
        "City Ring Road": 3, "Expressway": 4, "Village Road": 5,
        "Industrial Zone": 6, "Hill Road": 7, "Coastal Road": 8,
    }

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        feats = pd.DataFrame()
        feats["weather_index"]          = df["weather_index"]
        feats["road_friction"]          = df["road_friction"]
        feats["hour_of_day"]            = df["hour_of_day"]
        feats["traffic_volume_lag5"]    = df["traffic_volume_lag_5min"]
        feats["is_peak_hour"]           = df["is_peak_hour"]
        feats["is_night"]               = df["is_night"]
        feats["month_sin"]              = np.sin(2*np.pi*df["month"]/12)
        feats["road_type_enc"]          = df["road_type"].map(self.ROAD_TYPE_MAP).fillna(0)
        return feats.values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(df)
        self.scaler.fit(X)
        self._fitted = True
        return self.scaler.transform(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(df)
        return self.scaler.transform(X) if self._fitted else X

    def transform_single(self, row: dict) -> np.ndarray:
        """Transform a single dict row at inference time."""
        df_row = pd.DataFrame([row])
        # fill missing fields with defaults
        df_row.setdefault("is_peak_hour", [int(8 <= row.get("hour_of_day",12) <= 20)])
        df_row.setdefault("is_night",     [int(row.get("hour_of_day",12) < 5 or
                                              row.get("hour_of_day",12) >= 22)])
        df_row.setdefault("month",        [6])
        df_row.setdefault("road_type",    ["National Highway"])
        return self.transform(df_row)

    def save(self, path: str):
        with open(path, "wb") as f: pickle.dump(self.scaler, f)

    def load(self, path: str):
        with open(path, "rb") as f: self.scaler = pickle.load(f)
        self._fitted = True


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FUSION LAYER
# ─────────────────────────────────────────────────────────────────────────────

class FusionLayer:
    """
    Late (feature-level) fusion.
    Concatenates: visual (3-dim) + numerical (8-dim) → fused (11-dim)

    Layout:
    ┌────────────────────────┬──────────────────────────────────────────────────┐
    │ VISUAL (indices 0-2)   │ NUMERICAL (indices 3-10)                         │
    │ obj_count              │ weather_index  road_friction  hour_of_day         │
    │ vel_variance           │ lag5_traffic   is_peak        is_night            │
    │ road_density           │ month_sin      road_type_enc                      │
    └────────────────────────┴──────────────────────────────────────────────────┘
    """
    VISUAL_DIM    = 3
    NUMERICAL_DIM = 8
    FUSED_DIM     = 11

    @staticmethod
    def fuse(visual: np.ndarray, numerical: np.ndarray) -> np.ndarray:
        v = visual.flatten()
        n = numerical.flatten()
        return np.concatenate([v, n]).astype(np.float32)

    @staticmethod
    def fuse_batch(visual_batch: np.ndarray, numerical_batch: np.ndarray) -> np.ndarray:
        return np.hstack([visual_batch, numerical_batch]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  XGBOOST MODEL WITH SMOTE BALANCING
# ─────────────────────────────────────────────────────────────────────────────

class AccidentPredictor:
    """
    XGBoost classifier (dart booster) trained on fused features.
    Handles class imbalance via SMOTE + scale_pos_weight.
    """

    def __init__(self, scale_pos_weight: float = 10.0):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            booster         = "dart",
            n_estimators    = 300,
            max_depth       = 6,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            scale_pos_weight= scale_pos_weight,
            use_label_encoder=False,
            eval_metric     = "auc",
            random_state    = 42,
            n_jobs          = -1,
        )
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray,
              apply_smote: bool = True,
              eval_fraction: float = 0.15):
        from sklearn.model_selection import train_test_split

        pos = y.sum(); neg = len(y) - pos
        print(f"[Train] Samples={len(y)} | Positives={pos} ({100*pos/len(y):.1f}%) | "
              f"Negatives={neg}")

        # SMOTE oversampling
        if apply_smote and pos < neg:
            from imblearn.over_sampling import SMOTE
            k = min(5, int(pos) - 1)
            if k >= 1:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X, y = sm.fit_resample(X, y)
                print(f"[SMOTE] After resampling: {len(y)} samples, "
                      f"{int(y.sum())} positives")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=eval_fraction, stratify=y, random_state=42)

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self._trained = True
        print("[Train] ✅ Training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of accident class (index 1)."""
        return self.model.predict_proba(X)[:, 1]

    def risk_score(self, fused_vector: np.ndarray) -> float:
        """Single-sample inference → 0–100% risk score."""
        X = fused_vector.reshape(1, -1)
        return float(self.predict_proba(X)[0]) * 100.0

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        from sklearn.metrics import (classification_report, roc_auc_score,
                                     confusion_matrix)
        proba = self.predict_proba(X)
        pred  = (proba >= 0.5).astype(int)
        print("\n── Evaluation Report ──")
        print(classification_report(y, pred, digits=4))
        print(f"ROC-AUC : {roc_auc_score(y, proba):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y, pred)}")

    def save(self, path: str):
        self.model.save_model(path)
        print(f"[Model] Saved → {path}")

    def load(self, path: str):
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        self._trained = True
        print(f"[Model] Loaded ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REAL-TIME INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeInferenceEngine:
    """
    Wires everything together for frame-by-frame inference.
    Usage:
        engine = RealTimeInferenceEngine(predictor, num_processor)
        engine.run_on_video("highway_feed.mp4", tabular_df)
    """

    RISK_LEVELS = [
        (80, "🔴 CRITICAL",   "#FF0000"),
        (60, "🟠 HIGH",       "#FF6600"),
        (40, "🟡 MODERATE",   "#FFCC00"),
        (20, "🟢 LOW",        "#33CC33"),
        ( 0, "⚪ MINIMAL",    "#AAAAAA"),
    ]

    def __init__(self,
                 predictor:  AccidentPredictor,
                 num_proc:   NumericalProcessor,
                 yolo_extractor: VisualFeatureExtractor | None = None):
        self.predictor   = predictor
        self.num_proc    = num_proc
        self.yolo        = yolo_extractor or VisualFeatureExtractor()
        self._scores_buf = []          # rolling buffer for smoothing

    def classify_risk(self, score: float) -> tuple[str, str]:
        for threshold, label, color in self.RISK_LEVELS:
            if score >= threshold:
                return label, color
        return "⚪ MINIMAL", "#AAAAAA"

    def infer_single(self, frame: np.ndarray, tabular_row: dict) -> dict:
        """
        Args:
            frame:        BGR numpy array from cv2 capture
            tabular_row:  dict with keys: weather_index, road_friction,
                          hour_of_day, traffic_volume_lag_5min, etc.
        Returns:
            dict with risk_score, risk_label, visual_features, fused_vector
        """
        # Extract features
        visual_feats  = self.yolo.extract_from_frame(frame)
        numerical_feat = self.num_proc.transform_single(tabular_row).flatten()

        # Fuse
        fused = FusionLayer.fuse(visual_feats, numerical_feat)

        # Predict
        raw_score = self.predictor.risk_score(fused)

        # Smooth over last 5 frames
        self._scores_buf.append(raw_score)
        if len(self._scores_buf) > 5:
            self._scores_buf.pop(0)
        smoothed = float(np.mean(self._scores_buf))

        label, color = self.classify_risk(smoothed)
        return {
            "risk_score":      round(smoothed, 2),
            "risk_label":      label,
            "risk_color":      color,
            "visual_features": visual_feats.tolist(),
            "fused_vector":    fused.tolist(),
        }

    def run_on_video(self, video_source, tabular_df: pd.DataFrame,
                     display: bool = True):
        """
        Main inference loop.

        Args:
            video_source:  path to video file OR 0 for webcam
            tabular_df:    DataFrame aligned with video frames
                           (rows consumed sequentially)
        """
        import cv2
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")

        frame_idx = 0
        print(f"[Inference] Starting loop on {video_source}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Fetch corresponding tabular row (cycle through dataset)
            row_dict = tabular_df.iloc[frame_idx % len(tabular_df)].to_dict()
            result   = self.infer_single(frame, row_dict)

            score = result["risk_score"]
            label = result["risk_label"]

            # Overlay on frame
            if display:
                overlay = frame.copy()
                bar_width = int(score / 100 * frame.shape[1])
                cv2.rectangle(overlay, (0, 0), (bar_width, 30),
                              (0, 0, 200) if score > 60 else (0, 180, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, f"Risk: {score:.1f}%  {label}",
                            (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                            (255, 255, 255), 2)
                cv2.imshow("Accident Risk Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx:>5} | {label} | Score={score:.1f}%")
            frame_idx += 1

        cap.release()
        if display:
            cv2.destroyAllWindows()
        print("[Inference] Loop ended.")
