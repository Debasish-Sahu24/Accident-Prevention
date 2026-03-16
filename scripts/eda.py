import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

def banner(text):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print('─'*60)

# ── 1. Main dataset ───────────────────────────────────────────────────────────
banner("DATASET 1 — india_accident_main.csv")
df = pd.read_csv(DATA_DIR / "india_accident_main.csv")
print(df.describe().round(3).to_string())
print(f"\nAccident rate: {df.accident_occurred.mean()*100:.2f}%")
print(f"\nTop 5 risky cities:\n",
      df.groupby("city")["accident_occurred"].mean().nlargest(5).round(4))
print(f"\nAccident by weather:\n",
      df.groupby("weather_condition")["accident_occurred"].mean().round(4))
print(f"\nAccident by road type:\n",
      df.groupby("road_type")["accident_occurred"].mean().round(4))
print(f"\nPeak vs off-peak accident rate:")
print(df.groupby("is_peak_hour")["accident_occurred"].mean().round(4))
print(f"\nNight vs day accident rate:")
print(df.groupby("is_night")["accident_occurred"].mean().round(4))

# ── 2. Citywise ───────────────────────────────────────────────────────────────
banner("DATASET 2 — india_accident_citywise.csv")
df2 = pd.read_csv(DATA_DIR / "india_accident_citywise.csv")
print(df2["severity"].value_counts())
print(f"\nTop causes:\n", df2["primary_cause"].value_counts().head(8))
print(f"\nMean speed by severity:\n",
      df2.groupby("severity")["speed_kmh"].mean().round(1))

# ── 3. Sensor stream ──────────────────────────────────────────────────────────
banner("DATASET 3 — india_realtime_sensor.csv")
df3 = pd.read_csv(DATA_DIR / "india_realtime_sensor.csv")
print(f"Pothole detected: {df3.pothole_detected.mean()*100:.1f}% of frames")
print(f"Wrong-way driving: {df3.wrong_way_driving.mean()*100:.1f}% of frames")
print(f"Near-miss events: {df3.near_miss_event.mean()*100:.1f}% of frames")
print(f"\nSignal phase distribution:\n", df3["signal_phase"].value_counts())

# ── 4. Weather log ────────────────────────────────────────────────────────────
banner("DATASET 4 — india_weather_road_log.csv")
df4 = pd.read_csv(DATA_DIR / "india_weather_road_log.csv")
print(f"Correlation with accidents:\n",
      df4[["rainfall_mm","temperature_c","humidity_pct",
           "wind_speed_kmh","fog_hours","pothole_index",
           "accidents_that_day"]].corr()["accidents_that_day"].round(3))

# ── 5. Driver profile ─────────────────────────────────────────────────────────
banner("DATASET 5 — india_vehicle_driver_profile.csv")
df5 = pd.read_csv(DATA_DIR / "india_vehicle_driver_profile.csv")
print(f"Alcohol detected: {df5.alcohol_detected.mean()*100:.1f}%")
print(f"Mobile phone use: {df5.mobile_phone_use.mean()*100:.1f}%")
print(f"Invalid licence: {(1-df5.license_valid).mean()*100:.1f}%")
print(f"\nPast accidents by vehicle type:\n",
      df5.groupby("vehicle_type")["past_accident_count"].mean().round(3))

banner("✅ EDA Complete")
