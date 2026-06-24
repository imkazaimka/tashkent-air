"""
Operational evaluation of the forecaster in four scenarios the user asked about:

           |  next-day (h=1)        |  week-ahead (h=7)
  ---------+------------------------+--------------------------
  WITH a   |  (1) how good tomorrow |  (2) can it warn a week
  live     |      with a sensor?    |      ahead with a sensor?
  sensor   |                        |
  ---------+------------------------+--------------------------
  WITHOUT  |  (3) tomorrow, model   |  (4) week ahead, model
  a sensor |      inputs only       |      inputs only

"WITH a sensor"  = the model also sees the most recent REAL reading (as of the decision
                   day, i.e. h days before the target) + its recent trend.
"WITHOUT"        = CAMS + weather forecast for the target day + calendar only.

Target = real US-Embassy PM2.5. Threshold for "bad air" = 35 µg/m³ (unhealthy).
Temporal split: train < 2024-06-01, test >= 2024-06-01 (held-out, incl. winter 2024-25).

Honest caveat: the target-day weather/CAMS fields are reanalysis used as a stand-in for a
perfect forecast. At h=7 a real forecast is less skilful, so the true week-ahead numbers are
an OPTIMISTIC ceiling; the with-vs-without-sensor gap and the h=1-vs-h=7 drop are the robust
findings.

Output: models/scenario_metrics.json, figures/scenario_confusion.png
Run:    python src/scenario_evaluation.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

THR = 35.0
SPLIT = "2024-06-01"
WC = ["temperature_2m", "wind_speed_10m", "boundary_layer_height", "relative_humidity_2m",
      "surface_pressure", "shortwave_radiation", "precipitation", "pm2_5", "pm10",
      "nitrogen_dioxide", "ozone", "carbon_monoxide", "dust"]


def build():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    gt = gt[(gt.date >= dm.date.min()) & (gt.date <= dm.date.max())]
    d = dm.merge(gt, on="date", how="left").sort_values("date").reset_index(drop=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365); d["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return d


def fit_eval(d, h, sensor):
    df = d.copy()
    # target = real PM2.5 on the prediction day (row i); the decision is taken h days earlier.
    df["target"] = df["pm25_ground"]
    # forecast available for the TARGET day: weather + CAMS air-quality forecast (WC, calendar).
    feats = WC + ["doy_sin", "doy_cos"]
    # CAMS's own recent history (free, no ground sensor) — known at the decision day (h days back):
    df["cams_lag"] = df["pm2_5"].shift(h)
    df["cams_roll7"] = df["pm2_5"].shift(h).rolling(7, min_periods=3).mean()
    feats = feats + ["cams_lag", "cams_roll7"]
    if sensor:                                           # what a LIVE sensor adds at the decision day
        df["real_lag"] = df["pm25_ground"].shift(h)      # last real obs, h days before target
        df["real_roll7"] = df["pm25_ground"].shift(h).rolling(7, min_periods=3).mean()
        df["real_trend"] = df["pm25_ground"].shift(h) - df["pm25_ground"].shift(h + 3)
        feats = feats + ["real_lag", "real_roll7", "real_trend"]
    df = df.dropna(subset=["target"])
    tr, te = df[df.date < SPLIT], df[df.date >= SPLIT]
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                          min_child_samples=20, random_state=42, verbose=-1)
    m.fit(tr[feats], np.log1p(tr["target"]))
    pred = np.expm1(m.predict(te[feats]))
    y = te["target"].values
    yb, pb = (y > THR).astype(int), (pred > THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    return {
        "h": h, "sensor": sensor, "n_test": int(len(te)), "n_bad": int(yb.sum()),
        "R2": round(float(r2_score(y, pred)), 2), "MAE": round(float(mean_absolute_error(y, pred)), 1),
        "precision": round(float(precision_score(yb, pb, zero_division=0)), 2),
        "recall": round(float(recall_score(yb, pb, zero_division=0)), 2),
        "f1": round(float(f1_score(yb, pb, zero_division=0)), 2),
        "accuracy": round(float((tp + tn) / max(len(yb), 1)), 2),
        "roc_auc": round(float(roc_auc_score(yb, pred)) if yb.sum() else float("nan"), 2),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def main():
    d = build()
    scen = [(1, True), (7, True), (1, False), (7, False)]
    res = [fit_eval(d, h, s) for h, s in scen]
    (C.ROOT / "models" / "scenario_metrics.json").write_text(json.dumps(res, indent=2))

    print(f"Target = real embassy PM2.5; bad-air threshold = {THR:.0f} µg/m³; "
          f"test = held-out days from {SPLIT}\n")
    hdr = f"{'scenario':<34}{'R2':>6}{'MAE':>6}{'prec':>6}{'recall':>7}{'F1':>6}{'AUC':>6}"
    print(hdr); print("-" * len(hdr))
    for r in res:
        tag = f"{'WITH' if r['sensor'] else 'NO'} sensor · {'next-day' if r['h']==1 else 'week-ahead'}"
        print(f"{tag:<34}{r['R2']:>6}{r['MAE']:>6}{r['precision']:>6}{r['recall']:>7}"
              f"{r['f1']:>6}{r['roc_auc']:>6}")
    print("\nConfusion tables (bad-air day = PM2.5 > 35):")
    for r in res:
        tag = f"{'WITH' if r['sensor'] else 'NO'} sensor, {'next-day' if r['h']==1 else 'week-ahead'}"
        print(f"  {tag:<26} TP={r['TP']:>3} FP={r['FP']:>3} FN={r['FN']:>3} TN={r['TN']:>3}  "
              f"(of {r['n_test']} days, {r['n_bad']} truly bad)")

    # ---- figure: 2x2 confusion matrices ----
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    order = [(0, 0, res[0]), (0, 1, res[1]), (1, 0, res[2]), (1, 1, res[3])]
    for rr, cc, r in order:
        ax = axes[rr, cc]
        M = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(M, cmap="Blues")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                    color="white" if v > M.max() * 0.5 else "#1b2a4a", fontweight="bold")
        tag = f"{'WITH' if r['sensor'] else 'NO'} sensor · {'next-day' if r['h']==1 else 'week-ahead'}"
        ax.set_title(f"{tag}\nrecall {r['recall']:.0%} · precision {r['precision']:.0%} · "
                     f"AUC {r['roc_auc']:.2f}", fontsize=10.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred good", "pred bad"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["truly good", "truly bad"])
        ax.set_xticks(np.arange(-.5, 2), minor=True); ax.set_yticks(np.arange(-.5, 2), minor=True)
    fig.suptitle("Bad-air-day detection (PM2.5 > 35) — held-out test, four scenarios",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.ROOT / "figures" / "scenario_confusion.png", dpi=140); plt.close(fig)
    print("\nSaved figures/scenario_confusion.png and models/scenario_metrics.json")


if __name__ == "__main__":
    main()
