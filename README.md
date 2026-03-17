# 🛣️ Indian Road Accident Prediction System
### Late Fusion AI — YOLO Visual + XGBoost Numerical

---

## 📁 Project Structure

```
accident_prediction/
├── data/
│   ├── india_accident_main.csv          # 50,000 rows — main training set
│   ├── india_accident_citywise.csv      # 10,000 rows — city-level stats
│   ├── india_realtime_sensor.csv        # 20,000 rows — junction sensors
│   ├── india_weather_road_log.csv       # 15,000 rows — weather/road history
│   └── india_vehicle_driver_profile.csv #  8,000 rows — driver profiles
├── models/                              # saved after training
│   ├── xgb_accident_model.json
│   ├── numerical_scaler.pkl
│   └── model_meta.json
├── src/
│   └── pipeline.py                      # core ML classes
├── app.py                               # Flask web server + dashboard
├── train.py                             # numerical training
|── train_image.py                       # image training script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🗺️ Dataset Descriptions

| File | Rows | Key Features |
|------|------|-------------|
| `india_accident_main.csv` | 50,000 | timestamp, city, highway, weather, road_surface, YOLO features, risk_score, accident_occurred |
| `india_accident_citywise.csv` | 10,000 | city, cause, severity, speed_kmh, visibility, ambulance response |
| `india_realtime_sensor.csv` | 20,000 | junction_id, avg_speed, pedestrian_count, signal_phase, pothole_detected |
| `india_weather_road_log.csv` | 15,000 | state, rainfall_mm, fog_hours, road_waterlogging, pothole_index |
| `india_vehicle_driver_profile.csv` | 8,000 | vehicle_type, driver_age, alcohol_detected, fatigue_score, past_accidents |

---

| Image File |
|------|------|-------------|
| `data` | Training | Accident | Non-Accident |
| `data` | Testing | Accident | Non-Accident |
| `data` | Implementing | Accident | Non-Accident |

---

## ⚙️ Setup (VS Code)

### 1. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```
Expected output:
```
[Data] Loading india_accident_main.csv … 50,000 rows
[SMOTE] After resampling: ~8,000 balanced samples
[Train] ✅ Training complete.
ROC-AUC: ~0.87
[Save] Model → models/xgb_accident_model.json
```

### 4. Start the web server
```bash
python app.py
```
Open → **http://localhost:5000**

---

## 🔌 API Usage

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weather_index": 0.85,
    "road_friction": 0.35,
    "hour_of_day": 22,
    "traffic_volume_lag_5min": 120,
    "object_count": 45,
    "vel_variance": 180.5,
    "road_density": 0.72,
    "road_type": "National Highway",
    "month": 8
  }'
```
Response:
```json
{
  "risk_score": 78.4,
  "risk_label": "HIGH",
  "risk_pct": "78.4%"
}
```

### Python client
```python
import requests

response = requests.post("http://localhost:5000/predict", json={
    "weather_index": 0.85,
    "road_friction": 0.35,
    "hour_of_day": 22,
    "traffic_volume_lag_5min": 120,
    "object_count": 45,
    "vel_variance": 180.5,
    "road_density": 0.72,
    "road_type": "National Highway",
    "month": 8
})
print(response.json())
```

---

## 🎥 Live Image or Video Inference

```python
from src.pipeline import (
    VisualFeatureExtractor, NumericalProcessor,
    AccidentPredictor, RealTimeInferenceEngine
)
import pandas as pd

# Load components
predictor = AccidentPredictor()
predictor.load("models/xgb_accident_model.json")

num_proc = NumericalProcessor()
num_proc.load("models/numerical_scaler.pkl")

yolo = VisualFeatureExtractor("yolo11n.pt")

# Start inference on video
engine = RealTimeInferenceEngine(predictor, num_proc, yolo)
tabular_df = pd.read_csv("data/india_realtime_sensor.csv")
engine.run_on_video("highway_feed.mp4", tabular_df)
```

---

## 🚀 Deployment

### Option A — Docker (Recommended)
```bash
docker build -t accident-predictor .
docker run -p 5000:5000 accident-predictor
```

### Option B — Render.com (Free tier)
1. Push project to GitHub
2. New Web Service on https://render.com
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`

### Option C — Railway.app
```bash
railway login
railway init
railway up
```

### Option D — AWS EC2
```bash
# On EC2 instance (Ubuntu 22.04)
git clone <your-repo>
cd accident_prediction
pip install -r requirements.txt
python train.py
gunicorn -w 4 -b 0.0.0.0:80 app:app --daemon
```

---

## 🏗️ Architecture

```
[Video Frame]
      │
      ▼
┌─────────────────┐
│  YOLOv11 Track  │  → object_count
│  + Segmentation │  → bbox_velocity_variance
│                 │  → road_density
└────────┬────────┘
         │  Visual Features (3-dim)
         │
         ▼
    ┌────────────┐
    │  FUSION    │ ◄──── Numerical Features (8-dim)
    │  LAYER     │       [weather_index, road_friction,
    │  concat    │        hour_of_day, traffic_lag5,
    └────┬───────┘        is_peak, is_night, month, road_type]
         │  Fused Vector (11-dim)
         ▼
┌────────────────────┐
│  XGBoost (DART)    │
│  + SMOTE balanced  │
│  + scale_pos_weight│
└────────┬───────────┘
         │
         ▼
  Risk Score (0–100%)
  MINIMAL / LOW / MODERATE / HIGH / CRITICAL
```

---

## 📊 Risk Level Thresholds

| Score | Label | Action |
|-------|-------|--------|
| 80–100% | 🔴 CRITICAL | Alert traffic police + reroute |
| 60–79% | 🟠 HIGH | Issue speed advisory |
| 40–59% | 🟡 MODERATE | Increase monitoring |
| 20–39% | 🟢 LOW | Normal surveillance |
| 0–19% | ⚪ MINIMAL | No action needed |

---

## 🤝 Extending the System

- **More data**: integrate NCRB accident records, MoRTH annual reports
- **Better vision**: add lane detection, traffic signal state detection
- **Alerts**: connect to Twilio / AWS SNS for real-time SMS alerts
- **Maps**: overlay risk scores on Google Maps / OpenStreetMap
