import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
SRC_DIR    = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import AccidentPredictor, NumericalProcessor, FusionLayer, VisualFeatureExtractor

# ── Colours for terminal ──────────────────────────────────────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"

def banner(txt):
    print(f"\n{BLD}{CYN}{'═'*60}")
    print(f"  {txt}")
    print(f"{'═'*60}{RST}")

def ok(txt):   print(f"  {GRN}✅ {txt}{RST}")
def warn(txt): print(f"  {YLW}⚠️  {txt}{RST}")
def bad(txt):  print(f"  {RED}❌ {txt}{RST}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load everything
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1 — Loading Model & Data")

model_path  = MODEL_DIR / "xgb_accident_model.json"
scaler_path = MODEL_DIR / "numerical_scaler.pkl"
meta_path   = MODEL_DIR / "model_meta.json"

if not model_path.exists():
    bad("Model not found!  Please run  python train.py  first.")
    sys.exit(1)

predictor = AccidentPredictor()
predictor.load(str(model_path))

num_proc = NumericalProcessor()
num_proc.load(str(scaler_path))

with open(meta_path) as f:
    meta = json.load(f)

print(f"  Model:   {model_path}")
print(f"  Scaler:  {scaler_path}")
print(f"  Meta:    accident_rate={meta['accident_rate']*100:.1f}%  "
      f"| fused_dim={meta['fused_dim']}  "
      f"| scale_pos_weight={meta['scale_pos_weight']}")

df = pd.read_csv(DATA_DIR / "india_accident_main.csv")
ok(f"Dataset loaded — {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Rebuild features  (same as train.py)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2 — Rebuilding Feature Matrix")

visual_feats = VisualFeatureExtractor.extract_from_dataframe(df)
num_feats    = num_proc.transform(df)          # already fitted scaler
X            = FusionLayer.fuse_batch(visual_feats, num_feats)
y            = df["accident_occurred"].values

print(f"  X shape : {X.shape}")
print(f"  Labels  : {(y==0).sum():,} non-accident  |  {(y==1).sum():,} accident")

# Stratified 80/20 split  (same seed as train.py → same test set)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"  Test set: {len(y_test):,} rows  "
      f"({(y_test==0).sum():,} neg / {(y_test==1).sum():,} pos)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Get predictions
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3 — Running Predictions on Test Set")

y_proba = predictor.predict_proba(X_test)           # probabilities 0-1
y_pred  = (y_proba >= 0.5).astype(int)              # binary at 0.5 threshold
y_score = y_proba * 100                             # 0-100% risk scores

ok(f"Predictions complete for {len(y_test):,} test samples")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Compute ALL metrics
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 4 — Computing Accuracy Metrics")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    log_loss, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score,
)

accuracy          = accuracy_score(y_test, y_pred)
balanced_acc      = balanced_accuracy_score(y_test, y_pred)
precision         = precision_score(y_test, y_pred, zero_division=0)
recall            = recall_score(y_test, y_pred, zero_division=0)
f1                = f1_score(y_test, y_pred, zero_division=0)
roc_auc           = roc_auc_score(y_test, y_proba)
avg_precision     = average_precision_score(y_test, y_proba)
logloss           = log_loss(y_test, y_proba)
mcc               = matthews_corrcoef(y_test, y_pred)
kappa             = cohen_kappa_score(y_test, y_pred)
tn, fp, fn, tp    = confusion_matrix(y_test, y_pred).ravel()
specificity       = tn / (tn + fp) if (tn+fp) > 0 else 0
miss_rate         = fn / (fn + tp) if (fn+tp) > 0 else 0   # false negative rate

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Print full report
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 5 — FULL ACCURACY REPORT")

print(f"""
{BLD}┌─────────────────────────────────────────────────────────┐
│           INDIAN ROAD ACCIDENT PREDICTION AI              │
│                   MODEL ACCURACY REPORT                   │
└─────────────────────────────────────────────────────────┘{RST}

{BLD}━━━ MAIN SCORES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RST}

  Overall Accuracy     : {BLD}{accuracy*100:.2f}%{RST}
  ↳ Out of every 100 test cases, the model got {accuracy*100:.0f} correct.
  ↳ {YLW}Note: Accuracy alone is misleading for rare-event data (only
    {meta["accident_rate"]*100:.1f}% of roads have accidents). Use ROC-AUC.{RST}

  Balanced Accuracy    : {BLD}{balanced_acc*100:.2f}%{RST}
  ↳ Average accuracy across both classes (accidents AND safe roads).
  ↳ This is the fair accuracy number for imbalanced datasets.

  ROC-AUC Score        : {BLD}{roc_auc:.4f}{RST}   {GRN if roc_auc>0.85 else YLW}{'🟢 Excellent' if roc_auc>0.9 else '🟡 Good' if roc_auc>0.8 else '🔴 Needs improvement'}{RST}
  ↳ Ranges 0.5 (random guessing) to 1.0 (perfect).
  ↳ {roc_auc:.4f} means the model correctly ranks a risky road above a 
    safe one {roc_auc*100:.1f}% of the time.

  Average Precision    : {BLD}{avg_precision:.4f}{RST}
  ↳ Area under the Precision-Recall curve. More meaningful than 
    ROC-AUC when accidents are rare.

{BLD}━━━ CONFUSION MATRIX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RST}

                    Predicted: SAFE   Predicted: ACCIDENT
  Actual: SAFE          {tn:>6,}              {fp:>6,}
  Actual: ACCIDENT      {fn:>6,}              {tp:>6,}

  True Negatives  (TN) = {tn:,}   ← Correctly said "safe" when safe
  False Positives (FP) = {fp:,}   ← Said "accident" but was actually safe   (false alarm)
  False Negatives (FN) = {fn:,}   ← Said "safe" but accident actually happened (DANGEROUS!)
  True Positives  (TP) = {tp:,}   ← Correctly said "accident" when accident

{BLD}━━━ DETECTION PERFORMANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RST}

  Precision            : {BLD}{precision*100:.2f}%{RST}
  ↳ When the model says "ACCIDENT RISK", it is correct {precision*100:.1f}% of the time.
  ↳ {fp:,} false alarms out of {fp+tp:,} total accident alerts.

  Recall (Sensitivity) : {BLD}{recall*100:.2f}%{RST}
  ↳ Out of all REAL accidents, the model caught {recall*100:.1f}% of them.
  ↳ Missed {fn:,} actual accidents out of {fn+tp:,} total.

  Specificity          : {BLD}{specificity*100:.2f}%{RST}
  ↳ When road is actually SAFE, model correctly says safe {specificity*100:.1f}% of the time.

  Miss Rate            : {BLD}{miss_rate*100:.2f}%{RST}  {RED if miss_rate>0.2 else GRN}{'⚠️ High — consider lowering threshold' if miss_rate>0.2 else '✅ Good'}{RST}
  ↳ Percentage of real accidents the model FAILED to detect.
  ↳ In road safety, you want this as LOW as possible.

  F1 Score             : {BLD}{f1:.4f}{RST}
  ↳ Harmonic mean of Precision + Recall. Best single number for
    imbalanced datasets. Ranges 0 to 1 (higher = better).

{BLD}━━━ ADVANCED METRICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RST}

  Matthews Correlation : {BLD}{mcc:.4f}{RST}
  ↳ Ranges -1 to +1. Considers all 4 cells of confusion matrix.
  ↳ {mcc:.2f} = {'Strong positive correlation' if mcc>0.5 else 'Moderate correlation' if mcc>0.3 else 'Weak correlation'}

  Cohen's Kappa        : {BLD}{kappa:.4f}{RST}
  ↳ Agreement beyond chance. {kappa:.2f} = {'Almost perfect' if kappa>0.8 else 'Substantial' if kappa>0.6 else 'Moderate' if kappa>0.4 else 'Fair'} agreement.

  Log Loss             : {BLD}{logloss:.4f}{RST}
  ↳ Penalises confident wrong predictions. Lower = better.
    A perfect model = 0. Random model ≈ 0.693.
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Threshold analysis  (what if we lower the alarm threshold?)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 6 — THRESHOLD SENSITIVITY ANALYSIS")

print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>8}  {'False Alarms':>13}  {'Missed':>8}")
print(f"  {'─'*70}")

best_f1, best_thresh = 0, 0.5
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    yp = (y_proba >= thresh).astype(int)
    p  = precision_score(y_test, yp, zero_division=0)
    r  = recall_score(y_test, yp, zero_division=0)
    f  = f1_score(y_test, yp, zero_division=0)
    cm = confusion_matrix(y_test, yp).ravel()
    tn_,fp_,fn_,tp_ = (cm[0],cm[1],cm[2],cm[3]) if len(cm)==4 else (cm[0],0,0,0)
    flag = " ← current" if thresh == 0.5 else (" ← best F1" if f > best_f1 and thresh != 0.5 else "")
    if f > best_f1:
        best_f1, best_thresh = f, thresh
    print(f"  {thresh:>10.1f}  {p*100:>9.1f}%  {r*100:>9.1f}%  {f:>8.4f}  {fp_:>13,}  {fn_:>8,}  {flag}")

print(f"\n  {YLW}💡 For road safety: use a LOWER threshold (0.2–0.3) to catch")
print(f"     more accidents even at cost of more false alarms.{RST}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Feature importance
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 7 — FEATURE IMPORTANCE (What drives the prediction?)")

feature_names = [
    "Object Count",         # visual[0]
    "Speed Variance",       # visual[1]
    "Road Occupancy",       # visual[2]
    "Weather Index",        # numerical[0]
    "Road Friction",        # numerical[1]
    "Hour of Day",          # numerical[2]
    "Traffic Volume",       # numerical[3]
    "Is Peak Hour",         # numerical[4]
    "Is Night",             # numerical[5]
    "Month (Seasonal)",     # numerical[6]
    "Road Type",            # numerical[7]
]

try:
    importances = predictor.model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    print(f"\n  {'Rank':>4}  {'Feature':<22}  {'Importance':>10}  Bar")
    print(f"  {'─'*60}")
    for rank, idx in enumerate(sorted_idx):
        bar = "█" * int(importances[idx] * 200)
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  {rank+1:>4}  {name:<22}  {importances[idx]:>10.4f}  {GRN}{bar}{RST}")
except Exception as e:
    warn(f"Could not read feature importances: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Risk score distribution analysis
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 8 — RISK SCORE DISTRIBUTION")

safe_scores     = y_score[y_test == 0]
accident_scores = y_score[y_test == 1]

print(f"""
  Safe roads (actual):
    Mean risk score    : {safe_scores.mean():.1f}%
    Median risk score  : {np.median(safe_scores):.1f}%
    Std deviation      : {safe_scores.std():.1f}%

  Accident roads (actual):
    Mean risk score    : {accident_scores.mean():.1f}%
    Median risk score  : {np.median(accident_scores):.1f}%
    Std deviation      : {accident_scores.std():.1f}%

  Separation gap      : {accident_scores.mean() - safe_scores.mean():.1f}%
  ↳ Higher = better model separation between safe and dangerous roads.
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save charts
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 9 — SAVING VISUALISATION CHARTS")

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12), facecolor="#07101f")
fig.suptitle("Indian Road Accident Prediction — Model Evaluation Report",
             fontsize=16, fontweight="bold", color="#00aaff", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

CARD  = "#0d1a2e"
BLUE  = "#00aaff"
GREEN = "#00e676"
RED2  = "#ff3333"
AMBE  = "#ffbb00"

# ── Chart 1: Confusion Matrix ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
cm_data = np.array([[tn, fp], [fn, tp]])
cax = ax1.imshow(cm_data, cmap="Blues", vmin=0)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{cm_data[i,j]:,}", ha="center", va="center",
                 fontsize=13, fontweight="bold",
                 color="white" if cm_data[i,j] > cm_data.max()/2 else "black")
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(["Predicted\nSAFE", "Predicted\nACCIDENT"], color="white", fontsize=9)
ax1.set_yticklabels(["Actual\nSAFE", "Actual\nACCIDENT"], color="white", fontsize=9)
ax1.set_title("Confusion Matrix", color=BLUE, fontweight="bold", pad=10)
ax1.tick_params(colors="white")
for spine in ax1.spines.values(): spine.set_edgecolor("#1a2d42")

# ── Chart 2: ROC Curve ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color=BLUE, lw=2, label=f"Model (AUC = {roc_auc:.3f})")
ax2.plot([0,1],[0,1], color="#334455", lw=1, linestyle="--", label="Random guess (0.5)")
ax2.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax2.set_xlabel("False Positive Rate", color="white", fontsize=9)
ax2.set_ylabel("True Positive Rate", color="white", fontsize=9)
ax2.set_title("ROC Curve", color=BLUE, fontweight="bold", pad=10)
ax2.legend(fontsize=8, facecolor=CARD, edgecolor="#1a2d42", labelcolor="white")
ax2.tick_params(colors="white"); ax2.grid(color="#1a2d42", linestyle="--", alpha=0.5)
for spine in ax2.spines.values(): spine.set_edgecolor("#1a2d42")

# ── Chart 3: Precision-Recall Curve ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(CARD)
prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
ax3.plot(rec_c, prec_c, color=GREEN, lw=2, label=f"AP = {avg_precision:.3f}")
ax3.axhline(meta["accident_rate"], color="#334455", lw=1, linestyle="--",
            label=f"Baseline ({meta['accident_rate']*100:.1f}%)")
ax3.fill_between(rec_c, prec_c, alpha=0.08, color=GREEN)
ax3.set_xlabel("Recall", color="white", fontsize=9)
ax3.set_ylabel("Precision", color="white", fontsize=9)
ax3.set_title("Precision-Recall Curve", color=BLUE, fontweight="bold", pad=10)
ax3.legend(fontsize=8, facecolor=CARD, edgecolor="#1a2d42", labelcolor="white")
ax3.tick_params(colors="white"); ax3.grid(color="#1a2d42", linestyle="--", alpha=0.5)
for spine in ax3.spines.values(): spine.set_edgecolor("#1a2d42")

# ── Chart 4: Risk Score Distribution ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(CARD)
bins = np.linspace(0, 100, 40)
ax4.hist(safe_scores,     bins=bins, alpha=0.7, color=GREEN,  label="Safe roads",    density=True)
ax4.hist(accident_scores, bins=bins, alpha=0.8, color=RED2,   label="Accidents",     density=True)
ax4.axvline(50, color=AMBE, lw=1.5, linestyle="--", label="50% threshold")
ax4.set_xlabel("Risk Score (%)", color="white", fontsize=9)
ax4.set_ylabel("Density", color="white", fontsize=9)
ax4.set_title("Risk Score Distribution", color=BLUE, fontweight="bold", pad=10)
ax4.legend(fontsize=8, facecolor=CARD, edgecolor="#1a2d42", labelcolor="white")
ax4.tick_params(colors="white"); ax4.grid(color="#1a2d42", linestyle="--", alpha=0.3)
for spine in ax4.spines.values(): spine.set_edgecolor("#1a2d42")

# ── Chart 5: Feature Importance ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(CARD)
try:
    importances = predictor.model.feature_importances_
    names_short = ["Obj Count","Spd Var","Road Occ","Weather","Friction",
                   "Hour","Traffic","Peak Hr","Night","Season","Road Type"]
    sorted_idx = np.argsort(importances)
    colors_bar = [RED2 if importances[i]>importances.mean()*1.5 else BLUE for i in sorted_idx]
    bars = ax5.barh(range(len(importances)), importances[sorted_idx],
                    color=colors_bar, edgecolor="#1a2d42", height=0.7)
    ax5.set_yticks(range(len(importances)))
    labels = [names_short[i] if i < len(names_short) else f"F{i}" for i in sorted_idx]
    ax5.set_yticklabels(labels, color="white", fontsize=8)
    ax5.set_xlabel("Importance Score", color="white", fontsize=9)
    ax5.set_title("Feature Importance", color=BLUE, fontweight="bold", pad=10)
    ax5.tick_params(colors="white"); ax5.grid(color="#1a2d42", linestyle="--", alpha=0.3, axis="x")
    for spine in ax5.spines.values(): spine.set_edgecolor("#1a2d42")
except Exception:
    ax5.text(0.5, 0.5, "Feature importance\nnot available",
             ha="center", va="center", color="white", transform=ax5.transAxes)

# ── Chart 6: Metrics Summary Card ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(CARD)
ax6.axis("off")

metrics_data = [
    ("Accuracy",          f"{accuracy*100:.1f}%",    accuracy),
    ("Balanced Accuracy", f"{balanced_acc*100:.1f}%", balanced_acc),
    ("ROC-AUC",           f"{roc_auc:.3f}",           roc_auc),
    ("Avg Precision",     f"{avg_precision:.3f}",     avg_precision),
    ("Precision",         f"{precision*100:.1f}%",    precision),
    ("Recall",            f"{recall*100:.1f}%",       recall),
    ("F1 Score",          f"{f1:.3f}",                f1),
    ("Specificity",       f"{specificity*100:.1f}%",  specificity),
    ("Miss Rate",         f"{miss_rate*100:.1f}%",    1 - miss_rate),
    ("MCC",               f"{mcc:.3f}",               (mcc+1)/2),
]

ax6.set_title("Metrics Summary", color=BLUE, fontweight="bold", pad=10)
y_pos = 0.95
for name, val, score in metrics_data:
    color = GREEN if score >= 0.8 else AMBE if score >= 0.6 else RED2
    ax6.text(0.03, y_pos, name,  transform=ax6.transAxes,
             fontsize=9, color="white", va="top")
    ax6.text(0.72, y_pos, val,   transform=ax6.transAxes,
             fontsize=9, color=color, va="top", fontweight="bold")
    # mini bar
    bar_w = score * 0.22
    ax6.add_patch(plt.Rectangle((0.72, y_pos-0.05), bar_w, 0.03,
                                 transform=ax6.transAxes,
                                 color=color, alpha=0.35, clip_on=False))
    ax6.add_patch(plt.Rectangle((0.72, y_pos-0.05), 0.22,  0.03,
                                 transform=ax6.transAxes,
                                 fill=False, edgecolor="#1a2d42", clip_on=False))
    y_pos -= 0.092

report_path = MODEL_DIR / "evaluation_report.png"
fig.savefig(report_path, dpi=150, bbox_inches="tight", facecolor="#07101f")
plt.close()
ok(f"Chart saved → {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Final verdict
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 10 — FINAL VERDICT")

print()
grade_map = [(0.95,"🏆 EXCELLENT — Production ready"),
             (0.90,"🟢 VERY GOOD — Minor tuning possible"),
             (0.80,"🟡 GOOD — Suitable with human review"),
             (0.70,"🟠 FAIR — Needs more training data"),
             (0.0, "🔴 POOR — Requires significant work")]
grade = next(g for t,g in grade_map if roc_auc >= t)
print(f"  Overall Grade  (ROC-AUC = {roc_auc:.3f})  →  {BLD}{grade}{RST}")

print(f"""
  {BLD}Interpreting for road safety deployment:{RST}
  • The model correctly ranks risky roads {roc_auc*100:.1f}% of the time
  • It catches {recall*100:.0f}% of actual high-risk situations
  • {miss_rate*100:.0f}% of real accident scenarios are missed
  • For deployment: recommend threshold = 0.2-0.3 (lower = safer)

  {BLD}How to improve accuracy further:{RST}
  1. Collect real-world Indian accident data (MORTH datasets)
  2. Add more features: speed cameras, pothole sensors, signal timing
  3. Retrain with actual YOLO video frame features (not CSV proxies)
  4. Try ensemble: XGBoost + LightGBM + CatBoost voting
  5. Add location-specific models per highway / city

  Files saved to: {MODEL_DIR}/
    ├── evaluation_report.png   ← 6 visualisation charts
""")

print(f"  {'═'*58}")
print(f"  {GRN}Evaluation complete!{RST}")
print(f"  {'═'*58}\n")