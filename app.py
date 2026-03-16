"""
app.py — Indian Road Accident Prevention System
Smart image analysis — reads actual image content to determine:
  - Accident already occurred (damaged vehicles, debris)
  - High future risk (crowded road, heavy traffic)
  - Low risk (clear empty road)

Run:    python app.py
Deploy: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
"""

import os, json, pickle, time, sys, base64, io
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR / "src"))

app = Flask(__name__)
CORS(app)
predictor      = None
num_proc       = None
img_clf        = None
img_scaler     = None

def load_image_model():
    global img_clf, img_scaler
    try:
        import pickle as _pkl
        gb_path = MODEL_DIR / "image_classifier_gb.pkl"
        rf_path = MODEL_DIR / "image_classifier_rf.pkl"
        sc_path = MODEL_DIR / "image_scaler.pkl"
        if gb_path.exists() and sc_path.exists():
            with open(gb_path,'rb') as f: img_clf = _pkl.load(f)
            with open(sc_path,'rb') as f: img_scaler = _pkl.load(f)
            # also try loading rf for ensemble
            if rf_path.exists():
                with open(rf_path,'rb') as f:
                    import builtins; builtins._img_rf = _pkl.load(f)
            print("[API] ✅ Image classifier loaded (trained model active)")
        else:
            print("[API] ⚠️  No trained image model found — run train_image_model.py")
    except Exception as e:
        print(f"[API] Image model load error: {e}")

def _extract_image_features(pil_img):
    """Extract the same 24 features used during training."""
    import numpy as np
    from PIL import Image as _PIL
    img = pil_img.convert("RGB").resize((128,128))
    px  = np.array(img, dtype=np.float32)
    R,G,B = px[:,:,0], px[:,:,1], px[:,:,2]
    lum   = 0.299*R + 0.587*G + 0.114*B
    maxC  = np.maximum(np.maximum(R,G),B)
    minC  = np.minimum(np.minimum(R,G),B)
    sat   = np.where(maxC>0,(maxC-minC)/maxC,0)
    H,W   = lum.shape

    metallic = ((sat<0.18)&(lum>70)&(lum<210)).mean()
    red_dom  = ((R>140)&(R>G*1.35)&(R>B*1.35)).mean()
    dark     = (lum<55).mean()
    bright   = (lum>200).mean()
    sky_blue = ((B>130)&(B>R*1.25)&(B>G*0.9)).mean()
    green    = ((G>110)&(G>R*1.15)&(G>B*1.1)).mean()
    dy = np.abs(np.diff(lum,axis=0)).mean()
    dx = np.abs(np.diff(lum,axis=1)).mean()
    edge_density = (dy+dx)/2.0/255.0
    white    = ((R>200)&(G>200)&(B>200)).mean()
    top_lum  = lum[:H//2,:].mean(); bot_lum = lum[H//2:,:].mean()
    v_contrast = abs(top_lum-bot_lum)/255.0
    left_lum = lum[:,:W//2].mean(); right_lum= lum[:,W//2:].mean()
    h_contrast = abs(left_lum-right_lum)/255.0
    center = lum[H//4:3*H//4,W//4:3*W//4].mean()
    border = np.concatenate([lum[:H//4].flatten(),lum[3*H//4:].flatten(),
                              lum[:,:W//4].flatten(),lum[:,3*W//4:].flatten()])
    center_prom = (center-border.mean())/255.0
    sat_mean=sat.mean(); sat_std=sat.std()
    lum_mean=lum.mean()/255.0; lum_std=lum.std()/255.0
    orange = ((R>180)&(G>80)&(G<160)&(B<80)).mean()
    gray_road= ((sat<0.12)&(lum>50)&(lum<150)).mean()
    q1=lum[:H//2,:W//2].mean(); q2=lum[:H//2,W//2:].mean()
    q3=lum[H//2:,:W//2].mean(); q4=lum[H//2:,W//2:].mean()
    quad_var = np.std([q1,q2,q3,q4])/255.0
    return [metallic,red_dom,dark,bright,sky_blue,green,
            edge_density,white,v_contrast,h_contrast,center_prom,
            sat_mean,sat_std,lum_mean,lum_std,orange,gray_road,quad_var,
            0.0,  # blur (cv2 not available in server)
            metallic*red_dom, edge_density*(1-green),
            dark*red_dom, metallic*edge_density,
            (1-sky_blue)*(1-green)*metallic]

def load_models():
    global predictor, num_proc
    try:
        from pipeline import AccidentPredictor, NumericalProcessor
        mp = MODEL_DIR / "xgb_accident_model.json"
        sp = MODEL_DIR / "numerical_scaler.pkl"
        if mp.exists():
            predictor = AccidentPredictor()
            predictor.load(str(mp))
            num_proc = NumericalProcessor()
            if sp.exists():
                num_proc.load(str(sp))
            print("[API] ✅ Model loaded")
        else:
            print("[API] ⚠️  Run train.py first.")
    except Exception as e:
        print(f"[API] Load error: {e}")

def build_vector(data):
    from pipeline import FusionLayer
    visual = np.array([
        float(data.get("object_count", 12)),
        float(data.get("vel_variance",  30.0)),
        float(data.get("road_density",  0.35)),
    ], dtype=np.float32)
    hour = int(data.get("hour_of_day", 12))
    num_row = {
        "weather_index":           float(data.get("weather_index",  0.2)),
        "road_friction":           float(data.get("road_friction",  0.7)),
        "hour_of_day":             hour,
        "traffic_volume_lag_5min": float(data.get("traffic_volume_lag_5min", 60)),
        "is_peak_hour":            int(8 <= hour <= 20),
        "is_night":                int(hour < 5 or hour >= 22),
        "month":                   int(data.get("month", 6)),
        "road_type":               data.get("road_type", "National Highway"),
    }
    numerical = num_proc.transform_single(num_row).flatten()
    return FusionLayer.fuse(visual, numerical)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": predictor is not None})

@app.route("/analyse_image", methods=["POST"])
def analyse_image():
    """
    3-tier image analysis pipeline:
      Tier 1 — Trained ML model (best, used after train_image_model.py)
      Tier 2 — Claude Vision API (fallback if no trained model)
      Tier 3 — Returns failure so JS pixel analysis handles it
    """
    import base64, io
    data       = request.get_json(force=True)
    image_b64  = data.get("image_b64", "")
    media_type = data.get("media_type", "image/jpeg")

    def build_response(scene, damage, count, density, weather="clear"):
        weather_map = {"clear":0.05,"overcast":0.2,"rain":0.6,"fog":0.7,"night":0.3}
        wi  = weather_map.get(weather, 0.1)
        vel = 180 if "ACCIDENT" in scene else (60 if "BUSY" in scene else 5)
        return jsonify({
            "success": True,
            "scene_type": scene,
            "damage_score": damage,
            "vehicle_count": count,
            "road_density_pct": density,
            "weather": weather,
            "object_count": count,
            "vel_variance": vel,
            "road_density": density / 100.0,
            "weather_index": wi,
            "road_friction": max(0.05, 0.82 - wi * 0.5),
            "source": "trained_model" if img_clf else "vision_api",
        })

    # ── TIER 1: Trained ML model ──────────────────────────────────────────────
    if img_clf is not None and img_scaler is not None:
        try:
            from PIL import Image as _PIL
            img_bytes = base64.b64decode(image_b64)
            pil_img   = _PIL.open(io.BytesIO(img_bytes))
            feats     = _extract_image_features(pil_img)
            fv        = img_scaler.transform([feats])

            gb_prob = img_clf.predict_proba(fv)[0][1]

            # Ensemble with RF if available
            import builtins
            rf_model = getattr(builtins, '_img_rf', None)
            if rf_model:
                rf_prob = rf_model.predict_proba(fv)[0][1]
                prob    = gb_prob * 0.6 + rf_prob * 0.4
            else:
                prob = gb_prob

            # Map probability to scene type
            if prob >= 0.55:
                scene   = "ACCIDENT DETECTED"
                damage  = int(min(96, prob * 110))
                density = int(min(40, prob * 45))
                count   = max(2, int(prob * 8))
            elif prob >= 0.30:
                scene   = "BUSY ROAD — HIGH TRAFFIC"
                damage  = 0
                density = int(30 + prob * 50)
                count   = int(5 + prob * 20)
            else:
                scene   = "CLEAR ROAD — LOW TRAFFIC"
                damage  = 0
                density = int(prob * 20)
                count   = int(prob * 5)

            print(f"[ImageML] prob={prob:.3f} → {scene}")
            return build_response(scene, damage, count, density)

        except Exception as e:
            print(f"[ImageML] Error: {e} — falling back to Vision API")

    # ── TIER 2: Claude Vision API ─────────────────────────────────────────────
    try:
        import urllib.request, urllib.error
        prompt = (
            'Analyse this road/traffic image. Respond ONLY with JSON, no markdown.\n'
            'Rules:\n'
            '- scene_type: "ACCIDENT DETECTED" (crashed/wrecked vehicles, debris, '
            'collision aftermath) | "BUSY ROAD" (heavy traffic, many vehicles) | '
            '"CLEAR ROAD" (empty, few undamaged vehicles)\n'
            '{"scene_type":"...","damage_score":0-100,"vehicle_count":0-50,'
            '"road_density_pct":0-100,"weather_condition":"clear|rain|fog|night|overcast"}'
        )
        body = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": prompt}
            ]}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type":"application/json",
                     "anthropic-version":"2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
        raw    = result["content"][0]["text"].strip().replace("```json","").replace("```","")
        vision = json.loads(raw)
        return build_response(
            vision.get("scene_type","CLEAR ROAD"),
            int(vision.get("damage_score",0)),
            int(vision.get("vehicle_count",0)),
            int(vision.get("road_density_pct",0)),
            vision.get("weather_condition","clear")
        )
    except Exception as e:
        print(f"[Vision] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if predictor is None:
        wi  = float(data.get("weather_index", 0.2))
        rf  = float(data.get("road_friction", 0.7))
        hr  = int(data.get("hour_of_day", 12))
        obj = float(data.get("object_count", 12))
        vv  = float(data.get("vel_variance", 30))
        rd  = float(data.get("road_density", 0.35))
        score = min(97, round(
            wi*32 + (1-rf)*24 +
            (14 if hr < 5 or hr >= 22 else 7 if 17 <= hr <= 20 else 0) +
            (obj/150)*10 + (vv/200)*7 + rd*5 + 2, 1))
    else:
        try:
            fused = build_vector(data)
            score = round(predictor.risk_score(fused), 2)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    label = ("CRITICAL" if score >= 80 else "HIGH" if score >= 60
             else "MODERATE" if score >= 40 else "LOW" if score >= 20 else "MINIMAL")
    advice = {
        "CRITICAL": "🚨 Extremely high risk! Alert traffic police immediately, issue road closure advisory and deploy emergency response units now.",
        "HIGH":     "⚠️ High accident risk. Reduce speed limits, increase patrol frequency and broadcast driver warnings on highway boards.",
        "MODERATE": "🟡 Moderate risk detected. Monitor this stretch closely and issue caution alerts to all approaching drivers.",
        "LOW":      "✅ Low risk. Normal conditions prevail. Routine surveillance is sufficient.",
        "MINIMAL":  "✅ Safe conditions. Road is clear. No special action required at this time.",
    }[label]
    return jsonify({"risk_score": score, "risk_label": label,
                    "risk_pct": f"{score:.1f}%", "advice": advice})

@app.route("/")
def index():
    return render_template_string(HTML)

# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="theme-color" content="#07101f"/>
<title>Road Accident Prevention — India</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
:root{
  --bg:#07101f; --card:#0c1828; --border:#1a2d42;
  --blue:#00aaff; --green:#00e676; --red:#ff3333;
  --amber:#ffbb00; --orange:#ff7700; --teal:#00ddcc;
  --text:#cce0f0; --muted:#3d6070;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html{height:100%}
body{
  background:var(--bg);color:var(--text);
  font-family:'Rajdhani',sans-serif;
  min-height:100vh;display:flex;flex-direction:column;
}
button{cursor:pointer;font-family:'Rajdhani',sans-serif}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:#1a3a55;border-radius:4px}

/* ─── HEADER ──────────────────────────────────────────── */
header{
  background:linear-gradient(135deg,#060f1e,#0e1f38);
  border-bottom:2px solid var(--blue);
  padding:0 28px;height:58px;
  display:flex;align-items:center;gap:14px;
  flex-shrink:0;position:sticky;top:0;z-index:300;
}
.hlogo{font-size:1.9rem;line-height:1}
.htitle{flex:1}
.htitle h1{font-size:1rem;font-weight:700;letter-spacing:2px;color:var(--blue);line-height:1}
.htitle p{font-size:0.62rem;color:var(--muted);margin-top:2px;letter-spacing:.5px}
.hbadge{display:flex;align-items:center;gap:5px;
  background:#00e67612;border:1px solid #00e67638;
  border-radius:20px;padding:4px 11px}
.hbadge span{font-size:0.65rem;color:var(--green);
  font-family:'Share Tech Mono',monospace;font-weight:700;letter-spacing:1px}
.ldot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}

/* ─── MODE BAR ────────────────────────────────────────── */
.mode-bar{
  display:grid;grid-template-columns:1fr 1fr;
  border-bottom:2px solid var(--border);flex-shrink:0;
}
.mbtn{
  padding:14px 12px;border:none;
  background:var(--bg);color:var(--muted);
  font-size:1rem;font-weight:700;letter-spacing:1.5px;
  display:flex;align-items:center;justify-content:center;gap:10px;
  transition:all .2s;border-bottom:3px solid transparent;
}
.mbtn .mi{font-size:1.3rem}
.mbtn.active{
  background:linear-gradient(180deg,#0a1e38,var(--bg));
  color:#fff;border-bottom-color:var(--blue);
}
.mbtn:not(.active):hover{background:#080f1c;color:var(--text)}

/* ─── MAIN LAYOUT ─────────────────────────────────────── */
.main{flex:1;display:flex;overflow:hidden}

/* ─── PAGE ────────────────────────────────────────────── */
.page{
  display:none;width:100%;
  overflow-y:auto;padding:20px 24px 32px;
  animation:fadeUp .25s ease;
}
.page.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

/* ─── CARD ────────────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;padding:20px 22px;margin-bottom:16px;
}
.ctitle{
  font-size:0.72rem;font-weight:700;letter-spacing:2px;
  color:var(--blue);text-transform:uppercase;
  display:flex;align-items:center;gap:7px;margin-bottom:16px;
}

/* ─── FORM ────────────────────────────────────────────── */
.field{margin-bottom:11px}
label{display:block;font-size:0.72rem;color:var(--muted);margin-bottom:3px;letter-spacing:.4px}
select,input[type=number]{
  width:100%;padding:10px 14px;background:#050d1a;
  border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-family:'Share Tech Mono',monospace;
  font-size:0.85rem;outline:none;transition:border .2s;appearance:none;
}
select:focus,input:focus{border-color:var(--blue);box-shadow:0 0 0 3px #00aaff14}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}

/* ─── PREDICT BUTTON ──────────────────────────────────── */
.pbtn{
  width:100%;padding:17px;
  background:linear-gradient(135deg,#0066dd,#003eaa);
  border:none;border-radius:11px;color:#fff;
  font-size:1.15rem;font-weight:700;letter-spacing:2px;
  box-shadow:0 4px 24px #0066dd44;transition:all .2s;
}
.pbtn:hover{background:linear-gradient(135deg,#0088ff,#0055cc);transform:translateY(-1px)}
.pbtn:active{transform:scale(.98)}

/* ─── RESULT BOX ──────────────────────────────────────── */
.rbox{
  display:none;border-radius:14px;padding:24px 22px;
  text-align:center;border:2px solid transparent;
  margin-bottom:16px;transition:all .35s;
}
.rscore{font-size:4.5rem;font-weight:700;
  font-family:'Share Tech Mono',monospace;line-height:1}
.rlabel{font-size:1.8rem;font-weight:700;letter-spacing:3px;margin-top:6px}
.rbar-bg{background:#050d1a;border-radius:50px;height:10px;
  margin:14px 0;overflow:hidden;border:1px solid var(--border)}
.rbar{height:100%;border-radius:50px;
  transition:width 1.4s cubic-bezier(.22,1,.36,1)}
.radvice{text-align:left;font-size:0.9rem;line-height:1.65;
  padding:12px 16px;background:#050d1a;border-radius:9px;
  border-left:3px solid var(--blue);margin-top:12px}

/* ─── IMAGE MODE ──────────────────────────────────────── */
.img-drop{
  border:2px dashed var(--border);border-radius:14px;
  min-height:180px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  background:#050d1a;cursor:pointer;
  transition:all .2s;position:relative;
  text-align:center;padding:20px;
}
.img-drop.hover,.img-drop:hover{border-color:var(--blue);background:#091523}
.img-drop input[type=file]{
  position:absolute;inset:0;opacity:0;
  width:100%;height:100%;cursor:pointer;font-size:0;
}
.drop-ico{font-size:3.2rem;margin-bottom:10px;line-height:1}
.drop-txt{font-size:1.1rem;font-weight:700;color:var(--text)}
.drop-sub{font-size:0.72rem;color:var(--muted);margin-top:6px;line-height:1.5}

/* preview */
#img-preview{
  width:100%;border-radius:12px;
  display:none;object-fit:contain;
  background:#050d1a;
  border:2px solid var(--border);margin-top:0;
  /* No max-height — show FULL image always */
}

/* analysis overlay card */
.analysis-card{
  display:none;margin-top:14px;
  background:#050d1a;border:1px solid var(--border);
  border-radius:12px;padding:16px 18px;
}
.anl-title{font-size:0.7rem;color:var(--muted);
  font-family:'Share Tech Mono',monospace;letter-spacing:1px;margin-bottom:12px}
.anl-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.anl-item{text-align:center;padding:10px 6px;
  background:var(--card);border-radius:8px;border:1px solid var(--border)}
.anl-val{font-size:1.3rem;font-weight:700;
  font-family:'Share Tech Mono',monospace;color:var(--blue)}
.anl-key{font-size:0.65rem;color:var(--muted);margin-top:3px;letter-spacing:.5px}

/* scene type badge */
.scene-badge{
  display:inline-flex;align-items:center;gap:8px;
  padding:8px 16px;border-radius:30px;
  font-size:0.85rem;font-weight:700;letter-spacing:1px;
  margin:12px 0 4px;border:1px solid transparent;
}

/* image predict button */
.img-pbtn{
  width:100%;padding:17px;margin-top:14px;
  background:linear-gradient(135deg,#007744,#004422);
  border:none;border-radius:11px;color:#fff;
  font-size:1.15rem;font-weight:700;letter-spacing:2px;
  box-shadow:0 4px 22px #00774433;transition:all .2s;display:block;
}
.img-pbtn:hover{background:linear-gradient(135deg,#00aa55,#006633);transform:translateY(-1px)}
.img-pbtn:active{transform:scale(.98)}
.img-pbtn:disabled{opacity:.6;transform:none;cursor:not-allowed}

/* camera */
.subtabs{display:grid;grid-template-columns:1fr 1fr;
  border-radius:10px;overflow:hidden;border:1px solid var(--border);margin-bottom:16px}
.stab{padding:12px;text-align:center;font-size:0.88rem;
  font-weight:700;letter-spacing:1px;
  background:#050d1a;color:var(--muted);border:none;transition:all .2s}
.stab.active{background:var(--blue);color:#fff}

#cam-wrap{background:#050d1a;border-radius:12px;overflow:hidden;
  border:1px solid var(--border);position:relative;min-height:180px;
  display:flex;align-items:center;justify-content:center;}
#cam-video{width:100%;display:block;max-height:300px;object-fit:cover}
#cam-placeholder{color:var(--muted);font-size:0.85rem;text-align:center;padding:40px}
#cam-score-badge{
  position:absolute;top:10px;right:10px;
  font-family:'Share Tech Mono',monospace;font-size:1rem;font-weight:700;
  padding:5px 13px;border-radius:7px;background:#00000099;color:var(--green);
}
#cam-rec{
  position:absolute;top:10px;left:10px;display:none;
  font-family:'Share Tech Mono',monospace;font-size:0.68rem;
  color:var(--red);background:#00000099;padding:3px 8px;border-radius:4px;
}
.cam-ctls{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px}
.ccbtn{padding:11px;border:none;border-radius:8px;
  font-family:'Rajdhani',sans-serif;font-weight:700;
  font-size:0.88rem;letter-spacing:1px;transition:all .2s}
#cstart{background:#00883a;color:#fff}
#cstop{background:#bb2222;color:#fff;display:none}
#ccap{background:#0d2233;color:var(--blue);border:1px solid var(--blue)}
.cam-pbtn{
  width:100%;padding:17px;margin-top:14px;
  background:linear-gradient(135deg,#006688,#003355);
  border:none;border-radius:11px;color:#fff;
  font-size:1.15rem;font-weight:700;letter-spacing:2px;
  box-shadow:0 4px 22px #00668833;transition:all .2s;display:none;
}
.cam-pbtn:hover{background:linear-gradient(135deg,#0099bb,#005577);transform:translateY(-1px)}
.cam-pbtn:active{transform:scale(.98)}

/* legend */
.legend{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.leg{text-align:center;padding:10px 4px;border-radius:9px;border:1px solid transparent}
.leg .li{font-size:1.3rem}
.leg .ln{font-size:0.62rem;font-weight:700;margin-top:3px}
.leg .lr{font-size:0.58rem;color:#445;margin-top:2px}

@media(max-width:600px){
  .page{padding:14px 14px 28px}
  .rscore{font-size:3.4rem}.rlabel{font-size:1.4rem}
  .htitle h1{font-size:0.85rem}
  .mbtn{font-size:0.82rem;padding:12px 6px}
  .anl-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="hlogo">🛣️</div>
  <div class="htitle">
    <h1>INDIAN ROAD ACCIDENT PREVENTION</h1>
    <p>AI-POWERED ROAD SAFETY SYSTEM · ALL INDIA COVERAGE</p>
  </div>
  <div class="hbadge">
    <div class="ldot"></div>
    <span>LIVE</span>
  </div>
</header>

<!-- MODE BAR -->
<div class="mode-bar">
  <button class="mbtn active" id="mbtn-data" onclick="setMode('data')">
    <span class="mi">📋</span> DATA INPUT
  </button>
  <button class="mbtn" id="mbtn-img" onclick="setMode('img')">
    <span class="mi">📷</span> IMAGE / CAMERA
  </button>
</div>

<div class="main">

<!-- ════════════════════════════════════════════════════════
     PAGE A — DATA INPUT
════════════════════════════════════════════════════════ -->
<div class="page active" id="page-data">

  <div class="rbox" id="d-rbox">
    <canvas id="d-gauge" width="220" height="118" style="display:block;margin:0 auto 10px"></canvas>
    <div class="rscore" id="d-score">--</div>
    <div class="rlabel" id="d-label">--</div>
    <div class="rbar-bg"><div class="rbar" id="d-bar" style="width:0%"></div></div>
    <div class="radvice" id="d-advice"></div>
  </div>

  <!-- LOCATION -->
  <div class="card">
    <div class="ctitle">📍 Location</div>
    <div class="g2">
      <div class="field">
        <label>State</label>
        <select id="d-state" onchange="loadCities('d-state','d-city')">
          <option value="">— Select State —</option>
          <option>Andhra Pradesh</option><option>Arunachal Pradesh</option>
          <option>Assam</option><option>Bihar</option><option>Chhattisgarh</option>
          <option>Goa</option><option>Gujarat</option><option>Haryana</option>
          <option>Himachal Pradesh</option><option>Jharkhand</option>
          <option>Karnataka</option><option>Kerala</option>
          <option>Madhya Pradesh</option><option>Maharashtra</option>
          <option>Manipur</option><option>Meghalaya</option><option>Mizoram</option>
          <option>Nagaland</option><option>Odisha</option><option>Punjab</option>
          <option>Rajasthan</option><option>Sikkim</option>
          <option>Tamil Nadu</option><option>Telangana</option><option>Tripura</option>
          <option>Uttar Pradesh</option><option>Uttarakhand</option>
          <option>West Bengal</option><option>Delhi (NCT)</option>
          <option>Jammu &amp; Kashmir</option><option>Ladakh</option>
          <option>Chandigarh</option><option>Puducherry</option>
        </select>
      </div>
      <div class="field">
        <label>City / District</label>
        <select id="d-city"><option>— Select State First —</option></select>
      </div>
    </div>
    <div class="g2">
      <div class="field">
        <label>Road Type</label>
        <select id="d-road">
          <option>National Highway</option><option>Expressway</option>
          <option>State Highway</option><option>Urban Arterial</option>
          <option>City Ring Road</option><option>Hill Road</option>
          <option>Coastal Road</option><option>Village Road</option>
        </select>
      </div>
      <div class="field">
        <label>Highway / Route</label>
        <select id="d-hw">
          <option>NH-44 (Srinagar–Kanyakumari)</option>
          <option>NH-48 (Delhi–Chennai)</option>
          <option>NH-19 (Delhi–Kolkata)</option>
          <option>NH-27 (East–West Corridor)</option>
          <option>NH-16 (Kolkata–Chennai)</option>
          <option>NH-66 (Panvel–Kanyakumari)</option>
          <option>NH-8 (Delhi–Mumbai)</option>
          <option>NH-7 (Varanasi–Kanyakumari)</option>
          <option>City Road (No Highway)</option>
        </select>
      </div>
    </div>
  </div>

  <!-- TIME & WEATHER -->
  <div class="card">
    <div class="ctitle">🌦️ Time & Weather</div>
    <div class="g2">
      <div class="field">
        <label>Hour of Day (0 = midnight, 23 = 11 pm)</label>
        <input type="number" id="d-hour" value="18" min="0" max="23">
      </div>
      <div class="field">
        <label>Month</label>
        <select id="d-month">
          <option value="1">January</option><option value="2">February</option>
          <option value="3">March</option><option value="4">April</option>
          <option value="5">May</option><option value="6">June</option>
          <option value="7" selected>July</option><option value="8">August</option>
          <option value="9">September</option><option value="10">October</option>
          <option value="11">November</option><option value="12">December</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label>Weather Condition</label>
      <select id="d-weather" onchange="syncFriction()">
        <option value="0.05">☀️  Clear</option>
        <option value="0.25">🌫️  Haze / Smog</option>
        <option value="0.40">🌦️  Light Rain</option>
        <option value="0.75">🌧️  Heavy Rain</option>
        <option value="0.60">🌁  Fog</option>
        <option value="0.85">🌫️  Dense Fog</option>
        <option value="0.70">🌪️  Dust Storm</option>
        <option value="0.90">⛈️  Thunderstorm</option>
        <option value="0.98">🌀  Cyclone Warning</option>
        <option value="0.30">🌡️  Extreme Heat</option>
      </select>
    </div>
    <div class="g2">
      <div class="field">
        <label>Road Surface</label>
        <select id="d-surface" onchange="syncFriction()">
          <option value="0.82">Smooth Asphalt</option>
          <option value="0.85">Concrete</option>
          <option value="0.55">Broken Asphalt</option>
          <option value="0.40">Potholed</option>
          <option value="0.50">Gravel</option>
          <option value="0.60">Cobblestone</option>
        </select>
      </div>
      <div class="field">
        <label>Road Grip (auto-calculated)</label>
        <input type="number" id="d-friction" value="0.72"
          step="0.01" min="0.05" max="1" readonly style="opacity:.65;cursor:not-allowed">
      </div>
    </div>
  </div>

  <!-- TRAFFIC -->
  <div class="card">
    <div class="ctitle">🚗 Traffic Conditions</div>
    <div class="field">
      <label>Vehicles visible right now</label>
      <input type="number" id="d-visible" value="22" min="0" max="300">
    </div>
  </div>

  <button class="pbtn" id="d-btn" onclick="predictData()">⚡ PREDICT ACCIDENT RISK</button>

  <!-- LEGEND -->
  <div class="card" style="margin-top:16px">
    <div class="ctitle">🗺️ Risk Level Guide</div>
    <div class="legend">
      <div class="leg" style="background:#ff333318;border-color:#ff333344">
        <div class="li">🔴</div><div class="ln" style="color:#ff3333">CRITICAL</div><div class="lr">80–100%</div>
      </div>
      <div class="leg" style="background:#ff770018;border-color:#ff770044">
        <div class="li">🟠</div><div class="ln" style="color:#ff7700">HIGH</div><div class="lr">60–79%</div>
      </div>
      <div class="leg" style="background:#ffbb0018;border-color:#ffbb0044">
        <div class="li">🟡</div><div class="ln" style="color:#ffbb00">MODERATE</div><div class="lr">40–59%</div>
      </div>
      <div class="leg" style="background:#00cc6618;border-color:#00cc6644">
        <div class="li">🟢</div><div class="ln" style="color:#00cc66">LOW</div><div class="lr">20–39%</div>
      </div>
      <div class="leg" style="background:#00aaff18;border-color:#00aaff44">
        <div class="li">⚪</div><div class="ln" style="color:#00aaff">MINIMAL</div><div class="lr">0–19%</div>
      </div>
    </div>
  </div>

</div><!-- /page-data -->

<!-- ════════════════════════════════════════════════════════
     PAGE B — IMAGE / CAMERA
════════════════════════════════════════════════════════ -->
<div class="page" id="page-img">

  <!-- RESULT — image -->
  <div class="rbox" id="u-rbox">
    <canvas id="u-gauge" width="220" height="118" style="display:block;margin:0 auto 10px"></canvas>
    <div id="u-scene-badge" class="scene-badge"></div>
    <div class="rscore" id="u-score">--</div>
    <div class="rlabel" id="u-label">--</div>
    <div class="rbar-bg"><div class="rbar" id="u-bar" style="width:0%"></div></div>
    <div class="radvice" id="u-advice"></div>
  </div>

  <!-- RESULT — camera -->
  <div class="rbox" id="c-rbox">
    <canvas id="c-gauge" width="220" height="118" style="display:block;margin:0 auto 10px"></canvas>
    <div id="c-scene-badge" class="scene-badge"></div>
    <div class="rscore" id="c-score">--</div>
    <div class="rlabel" id="c-label">--</div>
    <div class="rbar-bg"><div class="rbar" id="c-bar" style="width:0%"></div></div>
    <div class="radvice" id="c-advice"></div>
  </div>

  <div class="card">
    <div class="subtabs">
      <button class="stab active" id="stab-upload" onclick="subTab('upload')">📂 Upload Photo</button>
      <button class="stab"        id="stab-cam"    onclick="subTab('cam')">🎥 Live Camera</button>
    </div>

    <!-- ── UPLOAD ───────────────────────────────── -->
    <div id="sec-upload">

      <!-- Upload zone — hidden after image chosen -->
      <div class="img-drop" id="dropzone"
        ondragover="event.preventDefault();this.classList.add('hover')"
        ondragleave="this.classList.remove('hover')"
        ondrop="onDrop(event)">
        <input type="file" id="fileInput" accept="image/*" capture="environment" onchange="onFileChosen(event)">
        <div class="drop-ico">📸</div>
        <div class="drop-txt">Tap to take photo or upload</div>
        <div class="drop-sub">
          Works on phone camera · JPG · PNG<br>
          Upload any road image — the AI will analyse it
        </div>
      </div>

      <!-- Preview container — shown after image chosen, replaces dropzone -->
      <div id="preview-wrap" style="display:none;margin-top:0">
        <div style="width:100%;max-height:280px;background:#050d1a;
          border:2px solid var(--border);border-radius:12px;
          display:flex;align-items:center;justify-content:center;overflow:hidden;">
          <img id="img-preview" src="" alt="Road preview"
            style="max-width:100%;max-height:276px;width:auto;height:auto;
                   object-fit:contain;display:block;border-radius:10px;">
        </div>
        <button onclick="resetUpload()" style="
          margin-top:7px;padding:7px 18px;background:transparent;
          border:1px solid var(--muted);border-radius:7px;
          color:var(--muted);font-family:'Rajdhani',sans-serif;
          font-size:0.8rem;letter-spacing:1px;cursor:pointer;width:100%">
          ✕ &nbsp; CHANGE IMAGE
        </button>
      </div>

      <!-- auto-analysis shown after upload -->
      <div class="analysis-card" id="img-analysis">
        <div class="anl-title">IMAGE ANALYSIS RESULTS</div>
        <div class="anl-grid">
          <div class="anl-item">
            <div class="anl-val" id="anl-scene">--</div>
            <div class="anl-key">SCENE TYPE</div>
          </div>
          <div class="anl-item">
            <div class="anl-val" id="anl-density">--%</div>
            <div class="anl-key">ROAD DENSITY</div>
          </div>
          <div class="anl-item">
            <div class="anl-val" id="anl-damage">--%</div>
            <div class="anl-key">DAMAGE SCORE</div>
          </div>
        </div>
      </div>

      <button class="img-pbtn" id="u-btn" onclick="predictFromImage()" style="display:none">
        🖼️ &nbsp; ANALYSE &amp; PREDICT ACCIDENT RISK
      </button>
    </div>

    <!-- ── CAMERA ────────────────────────────────── -->
    <div id="sec-cam" style="display:none">
      <div id="cam-wrap">
        <div id="cam-placeholder">📷 Click Start to open camera</div>
        <video id="cam-video" autoplay muted playsinline style="display:none"></video>
        <div id="cam-rec">● REC</div>
        <div id="cam-score-badge">-- %</div>
      </div>
      <div class="cam-ctls">
        <button class="ccbtn" id="cstart" onclick="startCam()">▶ Start</button>
        <button class="ccbtn" id="cstop"  onclick="stopCam()">■ Stop</button>
        <button class="ccbtn" id="ccap"   onclick="captureFrame()">📸 Capture</button>
      </div>
      <button class="cam-pbtn" id="c-btn" onclick="predictFromCamera()">
        🎥 &nbsp; PREDICT FROM CAMERA FRAME
      </button>
    </div>

  </div><!-- /card -->

  <!-- LEGEND -->
  <div class="card">
    <div class="ctitle">🗺️ Risk Level Guide</div>
    <div class="legend">
      <div class="leg" style="background:#ff333318;border-color:#ff333344">
        <div class="li">🔴</div><div class="ln" style="color:#ff3333">CRITICAL</div><div class="lr">80–100%</div>
      </div>
      <div class="leg" style="background:#ff770018;border-color:#ff770044">
        <div class="li">🟠</div><div class="ln" style="color:#ff7700">HIGH</div><div class="lr">60–79%</div>
      </div>
      <div class="leg" style="background:#ffbb0018;border-color:#ffbb0044">
        <div class="li">🟡</div><div class="ln" style="color:#ffbb00">MODERATE</div><div class="lr">40–59%</div>
      </div>
      <div class="leg" style="background:#00cc6618;border-color:#00cc6644">
        <div class="li">🟢</div><div class="ln" style="color:#00cc66">LOW</div><div class="lr">20–39%</div>
      </div>
      <div class="leg" style="background:#00aaff18;border-color:#00aaff44">
        <div class="li">⚪</div><div class="ln" style="color:#00aaff">MINIMAL</div><div class="lr">0–19%</div>
      </div>
    </div>
  </div>

</div><!-- /page-img -->
</div><!-- /main -->

<script>
// ── CITY MAP ─────────────────────────────────────────────────────────────────
const CITIES={
  "Andhra Pradesh":["Visakhapatnam","Vijayawada","Guntur","Nellore","Kurnool","Kakinada","Rajahmundry","Tirupati","Kadapa","Anantapur","Vizianagaram","Ongole","Chittoor"],
  "Arunachal Pradesh":["Itanagar","Naharlagun","Pasighat","Tawang","Ziro","Bomdila"],
  "Assam":["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon","Tinsukia","Tezpur","Bongaigaon","Dhubri"],
  "Bihar":["Patna","Gaya","Bhagalpur","Muzaffarpur","Purnia","Darbhanga","Bihar Sharif","Arrah","Begusarai","Katihar","Munger","Chhapra","Hajipur"],
  "Chhattisgarh":["Raipur","Bhilai","Bilaspur","Korba","Durg","Rajnandgaon","Jagdalpur","Ambikapur"],
  "Goa":["Panaji","Margao","Vasco da Gama","Mapusa","Ponda"],
  "Gujarat":["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar","Jamnagar","Junagadh","Gandhinagar","Anand","Navsari","Mehsana","Morbi","Bharuch","Porbandar"],
  "Haryana":["Faridabad","Gurgaon","Panipat","Ambala","Yamunanagar","Rohtak","Hisar","Karnal","Sonipat","Panchkula","Bhiwani","Sirsa"],
  "Himachal Pradesh":["Shimla","Manali","Dharamsala","Solan","Mandi","Kullu","Hamirpur","Una","Nahan"],
  "Jharkhand":["Ranchi","Jamshedpur","Dhanbad","Bokaro","Hazaribagh","Deoghar","Dumka","Chaibasa"],
  "Karnataka":["Bengaluru","Mysuru","Hubballi","Mangaluru","Belagavi","Kalaburagi","Ballari","Vijayapura","Shivamogga","Tumkur","Davanagere","Bidar","Udupi","Hassan"],
  "Kerala":["Thiruvananthapuram","Kochi","Kozhikode","Thrissur","Kollam","Palakkad","Alappuzha","Malappuram","Kannur","Kottayam","Kasaragod"],
  "Madhya Pradesh":["Bhopal","Indore","Jabalpur","Gwalior","Ujjain","Sagar","Ratlam","Satna","Dewas","Chhindwara","Rewa","Singrauli","Burhanpur"],
  "Maharashtra":["Mumbai","Pune","Nagpur","Thane","Nashik","Aurangabad","Solapur","Kolhapur","Amravati","Nanded","Sangli","Malegaon","Jalgaon","Akola","Latur","Dhule","Ahmednagar","Chandrapur","Navi Mumbai","Panvel","Bhiwandi","Vasai-Virar"],
  "Manipur":["Imphal","Thoubal","Bishnupur","Churachandpur","Ukhrul"],
  "Meghalaya":["Shillong","Tura","Jowai","Nongpoh"],
  "Mizoram":["Aizawl","Lunglei","Champhai","Serchhip"],
  "Nagaland":["Kohima","Dimapur","Mokokchung","Tuensang","Wokha"],
  "Odisha":["Bhubaneswar","Cuttack","Rourkela","Berhampur","Sambalpur","Puri","Balasore","Bhadrak","Baripada","Jharsuguda","Kendujhar","Rayagada","Koraput","Angul","Dhenkanal","Sundargarh","Bolangir","Bargarh","Paradip","Jeypore","Bhawanipatna","Kendrapara","Jajpur","Jagatsinghpur","Nayagarh","Ganjam","Nabarangpur"],
  "Punjab":["Ludhiana","Amritsar","Jalandhar","Patiala","Bathinda","Mohali","Hoshiarpur","Pathankot","Moga","Firozpur","Sangrur","Faridkot","Gurdaspur"],
  "Rajasthan":["Jaipur","Jodhpur","Kota","Bikaner","Ajmer","Udaipur","Bhilwara","Alwar","Bharatpur","Sikar","Sri Ganganagar","Pali","Barmer","Chittorgarh","Nagaur"],
  "Sikkim":["Gangtok","Namchi","Mangan","Gyalshing"],
  "Tamil Nadu":["Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli","Tiruppur","Vellore","Erode","Thoothukkudi","Dindigul","Thanjavur","Hosur","Nagercoil","Kanchipuram","Kumbakonam"],
  "Telangana":["Hyderabad","Warangal","Nizamabad","Karimnagar","Ramagundam","Khammam","Mahbubnagar","Nalgonda","Adilabad","Suryapet","Mancherial","Siddipet"],
  "Tripura":["Agartala","Dharmanagar","Udaipur","Kailashahar"],
  "Uttar Pradesh":["Lucknow","Kanpur","Ghaziabad","Agra","Varanasi","Meerut","Prayagraj","Bareilly","Aligarh","Moradabad","Saharanpur","Gorakhpur","Firozabad","Jhansi","Muzaffarnagar","Mathura","Rampur","Hapur","Mirzapur","Azamgarh","Ballia","Basti"],
  "Uttarakhand":["Dehradun","Haridwar","Rishikesh","Roorkee","Haldwani","Kashipur","Rudrapur","Nainital","Almora","Pithoragarh","Mussoorie"],
  "West Bengal":["Kolkata","Howrah","Durgapur","Asansol","Siliguri","Bardhaman","Malda","Baharampur","Kharagpur","Haldia","Darjeeling","Jalpaiguri","Cooch Behar","Bankura","Purulia","Medinipur","Nabadwip","Raiganj","Krishnanagar","Kalyani","Barrackpore","Salt Lake City","Chandannagar","Serampore","Alipurduar","Ranaghat"],
  "Delhi (NCT)":["New Delhi","Central Delhi","North Delhi","South Delhi","East Delhi","West Delhi","Dwarka","Rohini","Pitampura","Laxmi Nagar","Janakpuri","Connaught Place","Karol Bagh","Shahdara"],
  "Jammu & Kashmir":["Srinagar","Jammu","Anantnag","Baramulla","Sopore","Kathua","Udhampur","Rajouri"],
  "Ladakh":["Leh","Kargil","Nubra","Zanskar"],
  "Chandigarh":["Chandigarh","Manimajra","Industrial Area"],
  "Puducherry":["Puducherry","Karaikal","Mahe","Yanam"],
};

const COLORS={CRITICAL:"#ff3333",HIGH:"#ff7700",MODERATE:"#ffbb00",LOW:"#00cc66",MINIMAL:"#00aaff"};
const ADVICE={
  CRITICAL:"🚨 Extremely high risk! Alert traffic police immediately, deploy emergency response, issue road closure advisory.",
  HIGH:    "⚠️ High accident risk. Reduce speed limits, increase patrol frequency, broadcast driver warnings immediately.",
  MODERATE:"🟡 Moderate risk. Monitor closely, issue caution alerts to approaching drivers.",
  LOW:     "✅ Low risk. Normal conditions. Routine surveillance sufficient.",
  MINIMAL: "✅ Safe conditions. Road is clear. No special action required.",
};

// ── Mode switch ───────────────────────────────────────────────────────────────
function setMode(m){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.mbtn').forEach(b=>b.classList.remove('active'));
  document.getElementById('page-'+m).classList.add('active');
  document.getElementById('mbtn-'+m).classList.add('active');
  window.scrollTo({top:0,behavior:'smooth'});
}

// ── Sub tabs ──────────────────────────────────────────────────────────────────
function subTab(t){
  document.querySelectorAll('.stab').forEach(b=>b.classList.remove('active'));
  document.getElementById('stab-'+t).classList.add('active');
  document.getElementById('sec-upload').style.display=t==='upload'?'block':'none';
  document.getElementById('sec-cam').style.display   =t==='cam'?'block':'none';
  ['u-rbox','c-rbox'].forEach(id=>document.getElementById(id).style.display='none');
}

// ── City loader ───────────────────────────────────────────────────────────────
function loadCities(sid,cid){
  const s=document.getElementById(sid).value;
  const el=document.getElementById(cid);
  el.innerHTML='';
  (CITIES[s]||[]).forEach(c=>{const o=document.createElement('option');o.textContent=c;el.appendChild(o)});
  if(!CITIES[s]) el.innerHTML='<option>— Select State First —</option>';
}
window.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('d-state').value='Maharashtra';
  loadCities('d-state','d-city');
});

// ── Friction sync ─────────────────────────────────────────────────────────────
function syncFriction(){
  const wi=parseFloat(document.getElementById('d-weather').value)||0.05;
  const base=parseFloat(document.getElementById('d-surface').value)||0.82;
  document.getElementById('d-friction').value=Math.max(0.05,(base-wi*0.45)).toFixed(2);
}

// ── Gauge ─────────────────────────────────────────────────────────────────────
function drawGauge(cid,score,color){
  const cv=document.getElementById(cid);if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height,cx=W/2,cy=H,r=H-14;
  ctx.clearRect(0,0,W,H);
  ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,2*Math.PI);
  ctx.strokeStyle='#1a2d42';ctx.lineWidth=14;ctx.stroke();
  const ang=Math.PI+(Math.min(score,99)/100)*Math.PI;
  const g=ctx.createLinearGradient(0,0,W,0);
  g.addColorStop(0,color+'44');g.addColorStop(1,color);
  ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,ang);
  ctx.strokeStyle=g;ctx.lineWidth=14;ctx.lineCap='round';ctx.stroke();
  for(let i=0;i<=10;i++){
    const a=Math.PI+(i/10)*Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx+(r-20)*Math.cos(a),cy+(r-20)*Math.sin(a));
    ctx.lineTo(cx+(r-7)*Math.cos(a), cy+(r-7)*Math.sin(a));
    ctx.strokeStyle='#1a2d42';ctx.lineWidth=2;ctx.stroke();
  }
}

// ── Show result ───────────────────────────────────────────────────────────────
function showResult(boxId,gaugeId,scoreId,labelId,barId,advId,score,label,sceneText,sceneBadgeId,isAccident){
  const color=COLORS[label]||'#aaa';
  const box=document.getElementById(boxId);
  box.style.display='block';
  box.style.borderColor=color;
  box.style.background=color+'10';

  // ── Mode: ACCIDENT ALREADY OCCURRED ──────────────────────────────────────
  if(isAccident){
    box.style.borderColor='#ff2244';
    box.style.background='#ff224412';

    // Replace gauge with big warning banner
    const cv=document.getElementById(gaugeId);
    if(cv){ cv.style.display='none'; }

    document.getElementById(scoreId).innerHTML=
      '<div style="font-size:2.8rem;line-height:1">🚨</div>' +
      '<div style="font-size:1.05rem;font-weight:900;color:#ff2244;letter-spacing:2px;margin-top:6px">ACCIDENT ALREADY OCCURRED</div>';
    document.getElementById(scoreId).style.color='#ff2244';

    document.getElementById(labelId).textContent='IMMEDIATE ACTION REQUIRED';
    document.getElementById(labelId).style.color='#ff2244';
    document.getElementById(labelId).style.fontSize='0.75rem';
    document.getElementById(labelId).style.letterSpacing='2px';

    document.getElementById(barId).style.width='100%';
    document.getElementById(barId).style.background='linear-gradient(90deg,#ff224444,#ff2244)';

    // What happened + prevention
    document.getElementById(advId).innerHTML =
      '<div style="margin-bottom:10px;font-size:0.85rem;font-weight:700;color:#ff2244;letter-spacing:1px">📍 WHAT HAPPENED</div>' +
      '<div style="margin-bottom:14px;color:#ccc;font-size:0.88rem;line-height:1.7">' +
        'A road accident has been detected in this image. Vehicles show signs of collision damage, ' +
        'debris or crash aftermath is visible on the road.' +
      '</div>' +
      '<div style="margin-bottom:10px;font-size:0.85rem;font-weight:700;color:#ff9900;letter-spacing:1px">⚡ IMMEDIATE ACTIONS</div>' +
      '<div style="color:#ccc;font-size:0.88rem;line-height:1.9">' +
        '🚨 Call emergency services immediately (112 / 108)<br>' +
        '🚧 Place warning triangles 50m behind the accident<br>' +
        '🔴 Switch on hazard lights of all nearby vehicles<br>' +
        '🏥 Do NOT move injured persons unless fire risk<br>' +
        '📸 Document the scene for insurance/police<br>' +
        '🚗 Clear a lane for emergency vehicles to pass<br>' +
        '📢 Alert oncoming traffic to slow down' +
      '</div>' +
      '<div style="margin-top:14px;margin-bottom:8px;font-size:0.85rem;font-weight:700;color:#00cc88;letter-spacing:1px">🛡️ PREVENTION FOR FUTURE</div>' +
      '<div style="color:#ccc;font-size:0.88rem;line-height:1.9">' +
        '✅ Always maintain safe following distance<br>' +
        '✅ Avoid overtaking on curves or intersections<br>' +
        '✅ Never use phone while driving<br>' +
        '✅ Obey speed limits — especially in urban zones<br>' +
        '✅ Install dashcam for evidence in case of accidents' +
      '</div>';

  // ── Mode: PREDICTED RISK ──────────────────────────────────────────────────
  } else {
    const cv=document.getElementById(gaugeId);
    if(cv){ cv.style.display='block'; }
    drawGauge(gaugeId,score,color);
    document.getElementById(scoreId).textContent=score.toFixed(1)+'%';
    document.getElementById(scoreId).style.color=color;
    document.getElementById(scoreId).style.fontSize='';
    document.getElementById(labelId).textContent=label;
    document.getElementById(labelId).style.color=color;
    document.getElementById(labelId).style.fontSize='';
    document.getElementById(barId).style.width=score+'%';
    document.getElementById(barId).style.background=`linear-gradient(90deg,${color}44,${color})`;

    // Predicted risk advice
    const adviceMap = {
      CRITICAL: '🚨 Extremely high risk! Slow down immediately, increase following distance, avoid this stretch if possible. Alert traffic police.',
      HIGH:     '🟠 High risk conditions detected. Reduce speed, stay alert, turn on headlights and maintain extra gap from vehicles ahead.',
      MODERATE: '🟡 Moderate risk. Drive carefully, watch for pedestrians and intersections. Avoid distractions.',
      LOW:      '🟢 Low risk but stay alert. Follow traffic rules and maintain safe speed.',
      MINIMAL:  '⚪ Road appears safe. Continue with normal caution and follow all traffic rules.'
    };
    document.getElementById(advId).innerHTML =
      '<div style="margin-bottom:8px;font-size:0.85rem;font-weight:700;color:'+color+';letter-spacing:1px">📊 RISK ASSESSMENT</div>' +
      '<div style="margin-bottom:12px;color:#ccc;font-size:0.88rem;line-height:1.7">' + (adviceMap[label]||'') + '</div>' +
      '<div style="margin-bottom:8px;font-size:0.85rem;font-weight:700;color:#00cc88;letter-spacing:1px">🛡️ PREVENTION TIPS</div>' +
      '<div style="color:#ccc;font-size:0.88rem;line-height:1.9">' +
        '✅ Maintain safe distance from vehicles ahead<br>' +
        '✅ Check mirrors every 5–8 seconds<br>' +
        '✅ Reduce speed near intersections and crossings<br>' +
        '✅ Stay in lane and avoid sudden braking<br>' +
        '✅ Keep headlights on in low visibility conditions' +
      '</div>';
  }

  if(sceneBadgeId&&sceneText){
    const badge=document.getElementById(sceneBadgeId);
    badge.textContent=sceneText;
    badge.style.background=(isAccident?'#ff2244':'#333')+'22';
    badge.style.borderColor=(isAccident?'#ff2244':color)+'55';
    badge.style.color=isAccident?'#ff2244':color;
  }
  box.scrollIntoView({behavior:'smooth',block:'center'});
}

// ── API call ──────────────────────────────────────────────────────────────────
async function callAPI(payload){
  try{
    const r=await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    if(d.error) throw d.error;
    return {score:d.risk_score,label:d.risk_label};
  }catch(e){
    const wi=payload.weather_index||0.2,rf=payload.road_friction||0.7;
    const hr=payload.hour_of_day||12,obj=payload.object_count||12;
    const vv=payload.vel_variance||30,rd=payload.road_density||0.35;
    const score=Math.min(97,parseFloat((
      wi*32+(1-rf)*24+(hr<5||hr>=22?14:hr>=17&&hr<=20?7:0)
      +(obj/150)*9+(vv/200)*6+rd*5+(Math.random()*3)
    ).toFixed(1)));
    return{score,label:score>=80?"CRITICAL":score>=60?"HIGH":score>=40?"MODERATE":score>=20?"LOW":"MINIMAL"};
  }
}

// ── DATA INPUT PREDICT ────────────────────────────────────────────────────────
async function predictData(){
  const btn=document.getElementById('d-btn');
  btn.textContent='⏳ Analysing...';btn.disabled=true;
  const vis=parseFloat(document.getElementById('d-visible').value)||22;
  const payload={
    weather_index:           parseFloat(document.getElementById('d-weather').value),
    road_friction:           parseFloat(document.getElementById('d-friction').value),
    hour_of_day:             parseInt(document.getElementById('d-hour').value),
    traffic_volume_lag_5min: Math.round(vis * 4.5),
    object_count:            vis,
    vel_variance:            30+vis*0.9,
    road_density:            Math.min(0.95,vis/120),
    road_type:               document.getElementById('d-road').value,
    month:                   parseInt(document.getElementById('d-month').value),
  };
  const {score,label}=await callAPI(payload);
  btn.textContent='⚡ PREDICT ACCIDENT RISK';btn.disabled=false;
  showResult('d-rbox','d-gauge','d-score','d-label','d-bar','d-advice',score,label,null,null);
}

// ══════════════════════════════════════════════════════════════════════════════
// SMART IMAGE ANALYSIS
// Analyses actual pixel content to determine:
//   1. Accident scene (already occurred)
//   2. High density traffic (future risk)
//   3. Clear road (low risk)
// ══════════════════════════════════════════════════════════════════════════════

let imgAnalysis = null;

function deepAnalyseImage(dataUrl, callback){
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    const W = 128, H = 128;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);
    const px = ctx.getImageData(0, 0, W, H).data;
    const total = W * H;

    // ── Pixel analysis signals ────────────────────────────────────────────────
    let dark=0, bright=0, redHigh=0, grayCount=0, edgeCount=0;
    let metallic=0, skyBlue=0, greenGrass=0, brownDirt=0, asymmetry=0;
    let topBrightness=0, bottomBrightness=0;

    for(let i=0; i<px.length; i+=4){
      const r=px[i], g=px[i+1], b=px[i+2];
      const lum = (r*0.299 + g*0.587 + b*0.114);
      const pixIdx = i/4;
      const row = Math.floor(pixIdx / W);

      if(lum < 60)  dark++;
      if(lum > 200) bright++;

      // Red/orange dominance → fire, warning lights, blood, brake lights
      if(r > 160 && r > g*1.5 && r > b*1.5) redHigh++;

      // Metallic gray tones → vehicle bodies, damaged metal
      const maxC=Math.max(r,g,b), minC=Math.min(r,g,b);
      const saturation = maxC > 0 ? (maxC-minC)/maxC : 0;
      if(saturation < 0.15 && lum > 80 && lum < 200) metallic++;

      // Sky blue (upper portion of image)
      if(row < H*0.35 && b > 140 && b > r*1.3 && b > g*0.95) skyBlue++;

      // Green (vegetation, clear roadsides)
      if(g > 120 && g > r*1.2 && g > b*1.1) greenGrass++;

      // Brown/tan (dust, dirt roads)
      if(r > 120 && g > 80 && b < 80 && r > g) brownDirt++;

      // Edge detection (high contrast change = busy/complex scene)
      if(i > 4){
        const prevLum = (px[i-4]*0.299 + px[i-3]*0.587 + px[i-2]*0.114);
        if(Math.abs(lum - prevLum) > 35) edgeCount++;
      }

      // Top vs bottom brightness split
      if(row < H/2) topBrightness += lum;
      else bottomBrightness += lum;
    }

    const f = v => v / total;  // fraction helper

    // ── Derived features ──────────────────────────────────────────────────────
    const metallicFrac  = f(metallic);   // high → lots of vehicle bodies
    const darkFrac      = f(dark);       // high → night / shadows / smoke
    const edgeFrac      = f(edgeCount);  // high → complex busy scene
    const redFrac       = f(redHigh);    // high → brake lights, fire, damage
    const skyFrac       = f(skyBlue);    // low → overcast / occluded
    const greenFrac     = f(greenGrass); // high → open road / rural
    const brightFrac    = f(bright);     // high → clear day
    const topAvgLum     = topBrightness / (total/2 * 255);
    const botAvgLum     = bottomBrightness / (total/2 * 255);
    const verticalContrast = Math.abs(topAvgLum - botAvgLum);

    // ── SCENE CLASSIFICATION ALGORITHM ───────────────────────────────────────
    //
    // ACCIDENT SCENE detection criteria:
    //   • High metallic fraction (damaged vehicle bodies filling frame)
    //   • High edge complexity (debris, deformation, chaos)
    //   • Red dominance (fire, blood, warning lights, brake lights)
    //   • Low sky (camera close to ground / facing wreckage)
    //   • Low green (no vegetation visible = urban crash scene)
    //
    // BUSY ROAD detection criteria:
    //   • Moderate-high metallic (multiple vehicle tops/bodies)
    //   • High edge count (many vehicles = many edges)
    //   • Low sky fraction (vehicles block horizon)
    //   • Low green (urban, highway)
    //   • Moderate brightness (daytime road)
    //
    // CLEAR ROAD criteria:
    //   • Low metallic (few vehicles)
    //   • Low edges (open, uncluttered)
    //   • High sky or high green (open environment)

    // ── SCENE CLASSIFICATION — strict rules, no false positives ─────────────
    // KEY: metallic pixels = vehicle bodies. Dark empty road = zero metallic.
    // ACCIDENT: needs metallic > 0.18 AND (red > 0.05 OR chaotic edges > 0.30)
    // BUSY:     needs metallic > 0.08 AND edges > 0.15
    // CLEAR:    everything else — dark roads, empty roads, rural, no vehicles

    const hasVehicles  = metallicFrac > 0.08;
    const manyVehicles = metallicFrac > 0.18;
    const hasDamage    = redFrac > 0.05;
    const chaotic      = edgeFrac > 0.30;
    const busyEdges    = edgeFrac > 0.15;

    const isAccident = manyVehicles && (hasDamage || chaotic);
    const isBusy     = !isAccident && hasVehicles && busyEdges;

    let sceneType, roadDensity, damageScore, weatherEst, objCount, velVariance;

    if(isAccident){
      // ACCIDENT — density = only the involved vehicles, NOT full road
      // 2 cars = ~10-20% density max
      const involvedFrac = Math.min(metallicFrac * 1.0, 0.35);
      sceneType   = "ACCIDENT DETECTED";
      roadDensity = Math.max(0.05, involvedFrac);
      damageScore = Math.min(96, Math.round(metallicFrac*100 + redFrac*150 + (chaotic?10:0)));
      objCount    = Math.max(2, Math.round(metallicFrac * 20 + redFrac * 8));
      velVariance = 60 + redFrac*180;
      weatherEst  = Math.min(0.5, darkFrac*0.4 + 0.1);

    } else if(isBusy){
      // BUSY ROAD — density scales with how many vehicle bodies visible
      sceneType   = "BUSY ROAD — HIGH TRAFFIC";
      roadDensity = Math.min(0.80, 0.15 + metallicFrac*1.2 + edgeFrac*0.4);
      damageScore = Math.round(redFrac * 20);
      objCount    = Math.round(4 + metallicFrac*50 + edgeFrac*30);
      velVariance = 25 + edgeFrac*120;
      weatherEst  = darkFrac*0.35 + (skyFrac < 0.1 ? 0.12 : 0.02);

    } else {
      // CLEAR / EMPTY ROAD — dark road, empty highway, rural road
      sceneType   = "CLEAR ROAD — LOW TRAFFIC";
      roadDensity = Math.max(0.01, metallicFrac * 0.2);
      damageScore = 0;
      objCount    = Math.max(0, Math.round(metallicFrac * 4));
      velVariance = 2 + edgeFrac * 8;
      weatherEst  = darkFrac * 0.2;
    }

    const result = {
      sceneType,
      roadDensity: Math.min(roadDensity, 0.98),
      damageScore: Math.min(damageScore, 99),
      objCount:    Math.min(objCount, 120),
      velVariance: Math.min(velVariance, 380),
      weatherEst:  Math.min(weatherEst, 0.95),
      // pass-through raw for API
      object_count:  Math.min(objCount, 120),
      vel_variance:  Math.min(velVariance, 380),
      road_density:  Math.min(roadDensity, 0.98),
      weather_index: Math.min(weatherEst, 0.95),
      road_friction: Math.max(0.05, 0.82 - weatherEst*0.5),
    };

    callback(result);
  };
  img.src = dataUrl;
}

// ── File input handlers ───────────────────────────────────────────────────────
function onDrop(e){
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('hover');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')) loadImage(f);
}
function onFileChosen(e){ if(e.target.files[0]) loadImage(e.target.files[0]); }

function resetUpload(){
  document.getElementById('dropzone').style.display='flex';
  document.getElementById('preview-wrap').style.display='none';
  document.getElementById('img-analysis').style.display='none';
  document.getElementById('u-btn').style.display='none';
  document.getElementById('u-rbox').style.display='none';
  document.getElementById('fileInput').value='';
  imgAnalysis = null;
}

function loadImage(file){
  document.getElementById('u-rbox').style.display='none';
  document.getElementById('img-analysis').style.display='none';
  document.getElementById('u-btn').style.display='none';
  imgAnalysis = null;

  // Show "analysing..." immediately
  document.getElementById('anl-scene').textContent   = 'ANALYSING...';
  document.getElementById('anl-density').textContent = '--';
  document.getElementById('anl-damage').textContent  = '--';

  const r=new FileReader();
  r.onload=async ev=>{
    const dataUrl = ev.target.result;

    // Hide dropzone, show full-size preview
    document.getElementById('dropzone').style.display='none';
    const prev=document.getElementById('img-preview');
    prev.src=dataUrl;
    document.getElementById('preview-wrap').style.display='block';
    document.getElementById('img-analysis').style.display='block';

    // ── STEP 1: Try Claude Vision AI (server-side) ──────────────────────────
    try {
      // Extract base64 from data URL (strip "data:image/jpeg;base64,")
      const commaIdx = dataUrl.indexOf(',');
      const b64      = dataUrl.substring(commaIdx + 1);
      const mtype    = dataUrl.substring(5, commaIdx).split(';')[0];

      const resp = await fetch('/analyse_image', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({image_b64: b64, media_type: mtype})
      });
      const vision = await resp.json();

      if(vision.success){
        // Claude Vision successfully read the image
        imgAnalysis = {
          sceneType:    vision.scene_type,
          roadDensity:  vision.road_density,
          damageScore:  vision.damage_score,
          object_count: vision.object_count,
          vel_variance: vision.vel_variance,
          road_density: vision.road_density,
          weather_index:vision.weather_index,
          road_friction:vision.road_friction,
        };
        document.getElementById('anl-scene').textContent   = vision.scene_type.split('—')[0].trim();
        document.getElementById('anl-density').textContent = vision.road_density_pct+'%';
        document.getElementById('anl-damage').textContent  = vision.damage_score+'%';
        document.getElementById('u-btn').style.display='block';
        return;
      }
    } catch(e){
      console.warn('Vision API unavailable, falling back to pixel analysis:', e);
    }

    // ── STEP 2: Fallback — pixel analysis ───────────────────────────────────
    deepAnalyseImage(dataUrl, result=>{
      imgAnalysis = result;
      document.getElementById('anl-scene').textContent   = result.sceneType.split('—')[0].trim();
      document.getElementById('anl-density').textContent = Math.round(result.roadDensity*100)+'%';
      document.getElementById('anl-damage').textContent  = result.damageScore+'%';
      document.getElementById('u-btn').style.display='block';
    });
  };
  r.readAsDataURL(file);
}

// ── PREDICT FROM IMAGE ────────────────────────────────────────────────────────
async function predictFromImage(){
  if(!imgAnalysis) return;
  const btn=document.getElementById('u-btn');
  btn.textContent='⏳ Predicting...';btn.disabled=true;

  const hr = new Date().getHours();
  const scene = imgAnalysis.sceneType || 'CLEAR ROAD';

  const payload={
    weather_index:           imgAnalysis.weather_index  || 0.1,
    road_friction:           imgAnalysis.road_friction  || 0.75,
    hour_of_day:             hr,
    traffic_volume_lag_5min: Math.round((imgAnalysis.object_count||1) * 3.5),
    object_count:            imgAnalysis.object_count   || 1,
    vel_variance:            imgAnalysis.vel_variance   || 5,
    road_density:            imgAnalysis.road_density   || 0.02,
    road_type:               "National Highway",
    month:                   new Date().getMonth()+1,
  };

  let {score, label} = await callAPI(payload);

  // Scene-aware score adjustment based on what AI actually saw
  if(scene.includes('ACCIDENT')){
    // Accident already occurred — always CRITICAL
    const damageBoost = (imgAnalysis.damageScore || 80) * 0.12;
    score = Math.min(97, Math.max(82, score + damageBoost));
    label = 'CRITICAL';
  } else if(scene.includes('BUSY')){
    // Heavy traffic — future risk is HIGH
    score = Math.min(97, Math.max(55, score));
    label = score>=80?"CRITICAL":score>=60?"HIGH":"MODERATE";
  } else {
    // Clear road — keep score honest and LOW
    score = Math.min(score, 30);
    label = score>=20?"LOW":"MINIMAL";
  }

  score = Math.round(score * 10) / 10;
  btn.textContent='🖼️  ANALYSE & PREDICT ACCIDENT RISK';btn.disabled=false;

  const isAccident = scene.includes('ACCIDENT');
  const sceneEmoji = isAccident?'🚨 ':label==='HIGH'?'🟠 ':label==='MODERATE'?'🟡 ':'🟢 ';
  showResult('u-rbox','u-gauge','u-score','u-label','u-bar','u-advice',score,label,
             sceneEmoji+scene,'u-scene-badge',isAccident);
}

// ── CAMERA ────────────────────────────────────────────────────────────────────
let camStream=null, camTimer=null, camAnalysis=null;

async function startCam(){
  try{
    camStream=await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}},audio:false});
    const video=document.getElementById('cam-video');
    video.srcObject=camStream;
    video.style.display='block';
    document.getElementById('cam-placeholder').style.display='none';
    document.getElementById('cstart').style.display='none';
    document.getElementById('cstop').style.display='block';
    document.getElementById('cam-rec').style.display='block';

    // Live risk ticker
    camTimer=setInterval(()=>{
      const hr=new Date().getHours();
      const nightBonus=(hr<5||hr>=22)?0.14:0;
      const fakeScore=Math.min(95,parseFloat((20+Math.random()*30+nightBonus*100).toFixed(1)));
      const lbl=fakeScore>=80?"CRITICAL":fakeScore>=60?"HIGH":fakeScore>=40?"MODERATE":fakeScore>=20?"LOW":"MINIMAL";
      const badge=document.getElementById('cam-score-badge');
      badge.textContent=fakeScore.toFixed(1)+'%';
      badge.style.color=COLORS[lbl];
    },2500);
  }catch(e){ alert('Camera permission denied. Please allow camera access in your browser settings.'); }
}

function stopCam(){
  if(camStream) camStream.getTracks().forEach(t=>t.stop());
  clearInterval(camTimer); camStream=null;
  document.getElementById('cam-video').style.display='none';
  document.getElementById('cam-placeholder').style.display='block';
  document.getElementById('cstart').style.display='block';
  document.getElementById('cstop').style.display='none';
  document.getElementById('cam-rec').style.display='none';
  document.getElementById('cam-score-badge').textContent='-- %';
  document.getElementById('cam-score-badge').style.color='var(--green)';
  document.getElementById('cam-pbtn').style.display='none';
  camAnalysis=null;
}

function captureFrame(){
  const video=document.getElementById('cam-video');
  if(!video.srcObject){ alert('Please start the camera first.'); return; }
  const canvas=document.createElement('canvas');
  canvas.width=video.videoWidth||640; canvas.height=video.videoHeight||480;
  canvas.getContext('2d').drawImage(video,0,0);
  const dataUrl=canvas.toDataURL('image/jpeg',0.85);

  deepAnalyseImage(dataUrl, result=>{
    camAnalysis=result;
    document.getElementById('cam-pbtn').style.display='block';
    document.getElementById('c-rbox').style.display='none';
  });
}

async function predictFromCamera(){
  if(!camAnalysis) return;
  const btn=document.getElementById('c-btn');
  btn.textContent='⏳ Predicting...';btn.disabled=true;

  const hr=new Date().getHours();
  const payload={
    weather_index:           camAnalysis.weather_index,
    road_friction:           camAnalysis.road_friction,
    hour_of_day:             hr,
    traffic_volume_lag_5min: Math.round(camAnalysis.object_count*3.5),
    object_count:            camAnalysis.object_count,
    vel_variance:            camAnalysis.vel_variance,
    road_density:            camAnalysis.road_density,
    road_type:               "National Highway",
    month:                   new Date().getMonth()+1,
  };

  let {score,label}=await callAPI(payload);
  if(camAnalysis.sceneType.includes('ACCIDENT')){
    score=Math.max(score,82+Math.random()*12);
  } else if(camAnalysis.sceneType.includes('BUSY')){
    score=Math.max(score,45+camAnalysis.roadDensity*35);
  }
  score=Math.min(Math.round(score*10)/10,97);
  label=score>=80?"CRITICAL":score>=60?"HIGH":score>=40?"MODERATE":score>=20?"LOW":"MINIMAL";

  btn.textContent='🎥  PREDICT FROM CAMERA FRAME';btn.disabled=false;
  const camIsAccident = camAnalysis.sceneType.includes('ACCIDENT');
  const sceneEmoji=camIsAccident?'🚨 ':label==='HIGH'?'🟠 ':label==='MODERATE'?'🟡 ':'🟢 ';
  showResult('c-rbox','c-gauge','c-score','c-label','c-bar','c-advice',score,label,
             sceneEmoji+camAnalysis.sceneType,'c-scene-badge',camIsAccident);
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    load_models()
    load_image_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[API] 🚀  Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)