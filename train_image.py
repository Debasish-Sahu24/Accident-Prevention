# ══════════════════════════════════════════════════════════════════════════════
# ★★★  CHANGE THESE 3 LINES TO MATCH YOUR DATASET  ★★★
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH    = r"D:\Acc Pred\data"
                                      # folder name inside your project folder
                                      # e.g. "dataset" or "road_accident_data"

ACCIDENT_FOLDER = "Accident"          # name of the accident images folder
                                      # common names: "accident", "crash",
                                      #               "Accident", "positive"

NORMAL_FOLDER   = "Non Accident"      # name of the normal/safe images folder
                                      # common names: "non_accident", "normal",
                                      #               "no_crash", "negative",
                                      #               "Non Accident", "safe"

# ══════════════════════════════════════════════════════════════════════════════
# ★  OPTIONAL SETTINGS (you don't need to change these)  ★
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_SIZE      = (128, 128)          # resize all images to this size
MAX_IMAGES      = 5000                # max images per class (0 = use all)
TEST_SPLIT      = 0.20                # 20% images used for testing accuracy
RANDOM_SEED     = 42

# ══════════════════════════════════════════════════════════════════════════════
# DO NOT CHANGE ANYTHING BELOW THIS LINE
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, json, pickle, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("\n" + "═"*60)
print("  ACCIDENT IMAGE CLASSIFIER — TRAINING")
print("═"*60)

# ── Step 1: Check packages ────────────────────────────────────────────────────
print("\n[1/7] Checking required packages...")
try:
    from PIL import Image
    print("  ✅ Pillow (image loading)")
except ImportError:
    print("  ❌ Pillow not found. Run:  pip install pillow")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, roc_auc_score)
    print("  ✅ Scikit-learn (ML)")
except ImportError:
    print("  ❌ scikit-learn not found. Run:  pip install scikit-learn")
    sys.exit(1)

try:
    import cv2
    USE_CV2 = True
    print("  ✅ OpenCV (advanced features)")
except ImportError:
    USE_CV2 = False
    print("  ⚠️  OpenCV not found — using basic features only")
    print("     (Optional) Install with:  pip install opencv-python")

# ── Step 2: Find dataset ──────────────────────────────────────────────────────
print(f"\n[2/7] Looking for dataset at: {BASE_DIR / DATASET_PATH}")

dataset_root = BASE_DIR / DATASET_PATH

# Auto-detect structure
acc_path = None
non_path = None

# Try direct structure: dataset/accident/ and dataset/non_accident/
candidate_acc = dataset_root / ACCIDENT_FOLDER
candidate_non = dataset_root / NORMAL_FOLDER
if candidate_acc.exists() and candidate_non.exists():
    acc_path = candidate_acc
    non_path = candidate_non
    print(f"  ✅ Found Structure A (direct folders)")

# Try train subfolder: dataset/train/accident/
if acc_path is None:
    candidate_acc = dataset_root / "train" / ACCIDENT_FOLDER
    candidate_non = dataset_root / "train" / NORMAL_FOLDER
    if candidate_acc.exists() and candidate_non.exists():
        acc_path = candidate_acc
        non_path = candidate_non
        print(f"  ✅ Found Structure B (train subfolder)")

# Try one level up
if acc_path is None:
    candidate_acc = BASE_DIR / ACCIDENT_FOLDER
    candidate_non = BASE_DIR / NORMAL_FOLDER
    if candidate_acc.exists() and candidate_non.exists():
        acc_path = candidate_acc
        non_path = candidate_non
        print(f"  ✅ Found folders directly in project root")

if acc_path is None:
    print(f"\n  ❌ DATASET NOT FOUND!")
    print(f"\n  Expected to find:")
    print(f"    {dataset_root / ACCIDENT_FOLDER}")
    print(f"    {dataset_root / NORMAL_FOLDER}")
    print(f"\n  FIX: Open train_image_model.py and change:")
    print(f"    DATASET_PATH    = '{DATASET_PATH}'  ← your dataset folder name")
    print(f"    ACCIDENT_FOLDER = '{ACCIDENT_FOLDER}'  ← your accident folder name")
    print(f"    NORMAL_FOLDER   = '{NORMAL_FOLDER}'  ← your normal folder name")
    print(f"\n  Then run again.")
    sys.exit(1)

print(f"  📁 Accident images: {acc_path}")
print(f"  📁 Normal images:   {non_path}")

# ── Step 3: Load images ───────────────────────────────────────────────────────
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_image_paths(folder):
    paths = []
    for ext in VALID_EXTS:
        paths += list(Path(folder).rglob(f"*{ext}"))
        paths += list(Path(folder).rglob(f"*{ext.upper()}"))
    return sorted(set(paths))

acc_imgs = get_image_paths(acc_path)
non_imgs = get_image_paths(non_path)

print(f"\n[3/7] Loading images...")
print(f"  Found {len(acc_imgs):,} accident images")
print(f"  Found {len(non_imgs):,} normal images")

if len(acc_imgs) == 0:
    print(f"\n  ❌ No images found in: {acc_path}")
    print(f"  Make sure the folder contains .jpg or .png files")
    sys.exit(1)

if len(non_imgs) == 0:
    print(f"\n  ❌ No images found in: {non_path}")
    sys.exit(1)

# Limit if needed
if MAX_IMAGES > 0:
    np.random.seed(RANDOM_SEED)
    if len(acc_imgs) > MAX_IMAGES:
        acc_imgs = list(np.random.choice(acc_imgs, MAX_IMAGES, replace=False))
        print(f"  ⚠️  Limiting accident images to {MAX_IMAGES:,}")
    if len(non_imgs) > MAX_IMAGES:
        non_imgs = list(np.random.choice(non_imgs, MAX_IMAGES, replace=False))
        print(f"  ⚠️  Limiting normal images to {MAX_IMAGES:,}")

# ── Step 4: Feature extraction ────────────────────────────────────────────────
def extract_features(img_path):
    """
    Extract 24 visual features from a single image.
    These features are designed to distinguish accident scenes from normal roads.
    """
    try:
        img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        px  = np.array(img, dtype=np.float32)
    except Exception:
        return None

    R, G, B = px[:,:,0], px[:,:,1], px[:,:,2]
    lum = 0.299*R + 0.587*G + 0.114*B

    maxC = np.maximum(np.maximum(R,G),B)
    minC = np.minimum(np.minimum(R,G),B)
    sat  = np.where(maxC>0, (maxC-minC)/maxC, 0)
    H, W = lum.shape
    total = H * W

    # ── 1. Metallic gray fraction (vehicle bodies, damaged metal)
    metallic = ((sat < 0.18) & (lum > 70) & (lum < 210)).mean()

    # ── 2. Red dominance (fire, blood, brake lights, warning signs)
    red_dom  = ((R > 140) & (R > G*1.35) & (R > B*1.35)).mean()

    # ── 3. Dark pixel fraction (night, smoke, shadows)
    dark     = (lum < 55).mean()

    # ── 4. Bright pixel fraction (clear day, headlights)
    bright   = (lum > 200).mean()

    # ── 5. Sky blue fraction (open environment indicator)
    sky_blue = ((B > 130) & (B > R*1.25) & (B > G*0.9)).mean()

    # ── 6. Green fraction (vegetation = open road)
    green    = ((G > 110) & (G > R*1.15) & (G > B*1.1)).mean()

    # ── 7. Edge density (complexity = debris/wreckage/chaos)
    dy = np.abs(np.diff(lum, axis=0)).mean()
    dx = np.abs(np.diff(lum, axis=1)).mean()
    edge_density = (dy + dx) / 2.0 / 255.0

    # ── 8. White fraction (dust, smoke, airbags, fog)
    white    = ((R > 200) & (G > 200) & (B > 200)).mean()

    # ── 9. Vertical contrast (road vs vehicles)
    top_lum  = lum[:H//2, :].mean()
    bot_lum  = lum[H//2:, :].mean()
    v_contrast = abs(top_lum - bot_lum) / 255.0

    # ── 10. Horizontal contrast (side to side imbalance = chaos)
    left_lum = lum[:, :W//2].mean()
    right_lum= lum[:, W//2:].mean()
    h_contrast = abs(left_lum - right_lum) / 255.0

    # ── 11. Center vs border brightness (vehicle in center of frame)
    center   = lum[H//4:3*H//4, W//4:3*W//4].mean()
    border   = np.concatenate([lum[:H//4].flatten(), lum[3*H//4:].flatten(),
                                lum[:,  :W//4].flatten(), lum[:, 3*W//4:].flatten()])
    center_prominence = (center - border.mean()) / 255.0

    # ── 12. Saturation stats (colorful = more vehicles/signals)
    sat_mean = sat.mean()
    sat_std  = sat.std()

    # ── 13. Luminance stats
    lum_mean = lum.mean() / 255.0
    lum_std  = lum.std()  / 255.0

    # ── 14. Orange fraction (fire, warning lights)
    orange   = ((R > 180) & (G > 80) & (G < 160) & (B < 80)).mean()

    # ── 15. Gray road fraction (asphalt)
    gray_road= ((sat < 0.12) & (lum > 50) & (lum < 150)).mean()

    # ── 16. Quadrant variance (uneven scene = accident)
    q1 = lum[:H//2, :W//2].mean()
    q2 = lum[:H//2, W//2:].mean()
    q3 = lum[H//2:, :W//2].mean()
    q4 = lum[H//2:, W//2:].mean()
    quad_var = np.std([q1,q2,q3,q4]) / 255.0

    # ── 17. CV2 features (if available)
    blur_score = 0.0
    if USE_CV2:
        gray_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray_cv, cv2.CV_64F).var() / 10000.0

    features = [
        metallic,       # 1
        red_dom,        # 2
        dark,           # 3
        bright,         # 4
        sky_blue,       # 5
        green,          # 6
        edge_density,   # 7
        white,          # 8
        v_contrast,     # 9
        h_contrast,     # 10
        center_prominence, # 11
        sat_mean,       # 12
        sat_std,        # 13
        lum_mean,       # 14
        lum_std,        # 15
        orange,         # 16
        gray_road,      # 17
        quad_var,       # 18
        blur_score,     # 19
        metallic * red_dom,          # 20 interaction: metallic + damage
        edge_density * (1 - green),  # 21 interaction: chaos on non-green
        dark * red_dom,              # 22 interaction: fire at night
        metallic * edge_density,     # 23 interaction: wrecked metal
        (1 - sky_blue) * (1 - green) * metallic,  # 24 urban vehicle density
    ]
    return features

print(f"\n[4/7] Extracting features from {len(acc_imgs)+len(non_imgs):,} images...")
print("  This may take a few minutes depending on dataset size...")

def load_all(paths, label, desc):
    X, y, skipped = [], [], 0
    t0 = time.time()
    for i, p in enumerate(paths):
        f = extract_features(p)
        if f is not None:
            X.append(f)
            y.append(label)
        else:
            skipped += 1
        if (i+1) % 200 == 0 or (i+1) == len(paths):
            elapsed = time.time() - t0
            speed = (i+1) / elapsed
            remaining = (len(paths) - i - 1) / speed if speed > 0 else 0
            print(f"  {desc}: {i+1:,}/{len(paths):,} done  "
                  f"({speed:.0f} img/s, ~{remaining:.0f}s remaining)", end='\r')
    print()
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} unreadable images")
    return X, y

X_acc, y_acc = load_all(acc_imgs, label=1, desc="Accident  ")
X_non, y_non = load_all(non_imgs, label=0, desc="Normal    ")

X = np.array(X_acc + X_non, dtype=np.float32)
y = np.array(y_acc + y_non, dtype=np.int32)

print(f"\n  ✅ Features extracted:")
print(f"     Accident images : {len(X_acc):,}")
print(f"     Normal images   : {len(X_non):,}")
print(f"     Feature vector  : {X.shape[1]} features per image")

# ── Step 5: Train ─────────────────────────────────────────────────────────────
print(f"\n[5/7] Training model...")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)

print(f"  Training set : {len(X_train):,} images")
print(f"  Test set     : {len(X_test):,} images")

# Train Gradient Boosting (best for structured features)
print(f"\n  Training Gradient Boosting classifier...")
t0 = time.time()
clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    verbose=0
)
clf.fit(X_train, y_train)
print(f"  ✅ Training done in {time.time()-t0:.1f}s")

# Also train Random Forest as backup ensemble
print(f"  Training Random Forest classifier...")
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"  ✅ Done")

# ── Step 6: Evaluate ──────────────────────────────────────────────────────────
print(f"\n[6/7] Evaluating accuracy...")

# Ensemble: average both model probabilities
gb_proba = clf.predict_proba(X_test)[:,1]
rf_proba = rf.predict_proba(X_test)[:,1]
ens_proba = (gb_proba * 0.6 + rf_proba * 0.4)
ens_pred  = (ens_proba >= 0.5).astype(int)

acc   = accuracy_score(y_test, ens_pred)
try:
    auc = roc_auc_score(y_test, ens_proba)
except:
    auc = 0.0

report = classification_report(y_test, ens_pred,
                                target_names=['Normal Road','Accident'])
cm     = confusion_matrix(y_test, ens_pred)

tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

print(f"\n  {'═'*45}")
print(f"  ACCURACY      : {acc*100:.1f}%")
print(f"  ROC-AUC       : {auc:.3f}")
print(f"  {'═'*45}")
print(f"\n  Confusion Matrix:")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │              Predicted                  │")
print(f"  │         Normal    Accident               │")
print(f"  │  Normal   {tn:5d}     {fp:5d}   (correctly safe) │")
print(f"  │  Accident {fn:5d}     {tp:5d}   (correctly crash)│")
print(f"  └─────────────────────────────────────────┘")
print(f"\n  Missed accidents (false negatives): {fn}")
print(f"  False alarms   (false positives)  : {fp}")
print(f"\n{report}")

# Feature importance
feat_names = [
    'metallic','red_dominant','dark','bright','sky_blue','green',
    'edge_density','white','v_contrast','h_contrast','center_prom',
    'sat_mean','sat_std','lum_mean','lum_std','orange','gray_road',
    'quad_var','blur','metallic*red','chaos*non-green','dark*red',
    'metallic*edge','urban_density'
]
importances = clf.feature_importances_
ranked = sorted(zip(feat_names, importances), key=lambda x:-x[1])
print(f"\n  Top 10 most important features:")
for name, imp in ranked[:10]:
    bar = '█' * int(imp * 200)
    print(f"  {name:22s} {imp:.4f}  {bar}")

# ── Step 7: Save models ───────────────────────────────────────────────────────
print(f"\n[7/7] Saving models to models/...")

# Save both classifiers + scaler
with open(MODEL_DIR / "image_classifier_gb.pkl",  'wb') as f:
    pickle.dump(clf, f)
with open(MODEL_DIR / "image_classifier_rf.pkl",  'wb') as f:
    pickle.dump(rf, f)
with open(MODEL_DIR / "image_scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# Save metadata
meta = {
    "accuracy":       round(acc, 4),
    "roc_auc":        round(auc, 4),
    "n_features":     X.shape[1],
    "image_size":     list(IMAGE_SIZE),
    "n_accident":     len(X_acc),
    "n_normal":       len(X_non),
    "feature_names":  feat_names,
    "use_cv2":        USE_CV2,
    "trained_on":     str(acc_path),
}
with open(MODEL_DIR / "image_model_meta.json", 'w') as f:
    json.dump(meta, f, indent=2)

print(f"  ✅ models/image_classifier_gb.pkl")
print(f"  ✅ models/image_classifier_rf.pkl")
print(f"  ✅ models/image_scaler.pkl")
print(f"  ✅ models/image_model_meta.json")