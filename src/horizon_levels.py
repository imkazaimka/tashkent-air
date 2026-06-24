"""
Three things a real warning tool needs, beyond a single "bad day" flag:

  (1) SEVERITY LEVELS — predict the AQI/PM2.5 category, not just bad/good, and report
      how well each level is recovered (per-level recall + a 6x6 truth table).
  (2) CONFIDENCE — quantile prediction intervals (10th–90th percentile) with their
      MEASURED coverage on held-out data ("our 80% band contains the truth X% of the time").
  (3) HORIZON DECAY — how skill falls off as we forecast 1, 2, ... 14 days ahead, at each
      threat level, under two honest bounds:
        - "today only"   : use only what is known now (no future weather) -> the realistic
                           decay as memory of today fades toward climatology.
        - "+ weather fc" : also give the model the target-day weather (a PERFECT-forecast
                           proxy) -> an optimistic ceiling. A real 14-day system lies between,
                           near the ceiling for the first few days and sliding to the floor.

PM2.5 categories (µg/m³):  Good <12 | Moderate <35 | USG <55.4 | Unhealthy <150.4 |
                           Very Unhealthy <250.4 | Hazardous >=250.4
Target = real US-Embassy PM2.5. Temporal split: train <2024-06-01, test >=2024-06-01.

Output: figures/forecast_horizon.png, figures/aqi_levels.png, models/horizon_levels.json
Run:    python src/horizon_levels.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

SPLIT = "2024-06-01"
BREAKS = [12, 35, 55.4, 150.4, 250.4]
LABELS = ["Good", "Moderate", "USG", "Unhealthy", "Very\nUnhealthy", "Hazardous"]
WEATHER = ["temperature_2m", "wind_speed_10m", "boundary_layer_height",
           "relative_humidity_2m", "surface_pressure"]
FCAST = WEATHER + ["pm2_5", "nitrogen_dioxide", "dust", "shortwave_radiation"]


def load():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    gt = gt[(gt.date >= dm.date.min()) & (gt.date <= dm.date.max())]
    d = dm.merge(gt, on="date", how="left").sort_values("date").reset_index(drop=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    return d


def features(d, h, setup):
    df = d.copy()
    df["target"] = df["pm25_ground"]
    feats = ["doy_sin", "doy_cos"]                         # target-day calendar (always known)
    # known at the decision day t = target - h
    df["cams_lag"] = df["pm2_5"].shift(h)
    df["cams_roll7"] = df["pm2_5"].shift(h).rolling(7, min_periods=3).mean()
    df["real_lag"] = df["pm25_ground"].shift(h)
    df["real_roll7"] = df["pm25_ground"].shift(h).rolling(7, min_periods=3).mean()
    df["real_trend"] = df["pm25_ground"].shift(h) - df["pm25_ground"].shift(h + 3)
    feats += ["cams_lag", "cams_roll7", "real_lag", "real_roll7", "real_trend"]
    for c in WEATHER:                                       # today's weather
        df[c + "_dec"] = df[c].shift(h); feats.append(c + "_dec")
    if setup == "forecast":                                # + perfect target-day forecast proxy
        feats += FCAST
    return df.dropna(subset=["target"]), feats


def split(df):
    return df[df.date < SPLIT], df[df.date >= SPLIT]


def fit_predict(df, feats, obj="regression", alpha=None):
    tr, te = split(df)
    kw = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              random_state=42, verbose=-1)
    if obj == "quantile":
        kw.update(objective="quantile", alpha=alpha)
    m = lgb.LGBMRegressor(**kw)
    m.fit(tr[feats], np.log1p(tr["target"]))
    return np.expm1(m.predict(te[feats])), te["target"].values


def main():
    d = load()
    out = {}

    # ---------- (3) HORIZON DECAY 1..14 ----------
    horizons = list(range(1, 15))
    rec35, rec55, auc_today, auc_fc = [], [], [], []
    for h in horizons:
        dt, ft = features(d, h, "today")
        p, y = fit_predict(dt, ft)
        rec35.append(recall_score(y > 35, p > 35, zero_division=0))
        rec55.append(recall_score(y > 55.4, p > 55.4, zero_division=0))
        auc_today.append(roc_auc_score(y > 35, p) if (y > 35).any() else np.nan)
        dff, ff = features(d, h, "forecast")
        pf, yf = fit_predict(dff, ff)
        auc_fc.append(roc_auc_score(yf > 35, pf) if (yf > 35).any() else np.nan)
    out["horizon"] = {"days": horizons, "recall35_today": rec35, "recall55_today": rec55,
                      "auc35_today": auc_today, "auc35_forecast": auc_fc}
    print("HORIZON DECAY (today-only, with sensor):")
    print("  day :  " + " ".join(f"{h:>4}" for h in horizons))
    print("  rec>35: " + " ".join(f"{v:>4.2f}" for v in rec35))
    print("  rec>55: " + " ".join(f"{v:>4.2f}" for v in rec55))
    print("  AUC today : " + " ".join(f"{v:>4.2f}" for v in auc_today))
    print("  AUC +w.fc : " + " ".join(f"{v:>4.2f}" for v in auc_fc))

    # ---------- (1) SEVERITY LEVELS + (2) CONFIDENCE (next-day, forecast+sensor) ----------
    df1, f1 = features(d, 1, "forecast")
    p50, y = fit_predict(df1, f1)

    # CONFIDENCE via conformalized quantile regression (CQR): fit q10/q90 on an inner
    # train split, then widen by the calibration-set residual quantile to hit true 80%.
    tr, te = split(df1)
    cut = tr["date"].quantile(0.80)
    trf, cal = tr[tr.date <= cut], tr[tr.date > cut]

    def qmodel(a):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=500,
                              learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              random_state=42, verbose=-1)
        m.fit(trf[f1], np.log1p(trf["target"])); return m

    mlo, mhi = qmodel(0.10), qmodel(0.90)
    qlo_c, qhi_c = np.expm1(mlo.predict(cal[f1])), np.expm1(mhi.predict(cal[f1]))
    yc = cal["target"].values
    E = np.maximum(qlo_c - yc, yc - qhi_c)                 # conformity scores
    Q = float(np.quantile(E, 0.80, method="higher"))       # widen to reach 80% coverage
    p10 = np.clip(np.expm1(mlo.predict(te[f1])) - Q, 0, None)
    p90 = np.expm1(mhi.predict(te[f1])) + Q
    cat_true = np.digitize(y, BREAKS)
    cat_pred = np.digitize(p50, BREAKS)
    cm = confusion_matrix(cat_true, cat_pred, labels=range(6))
    per_recall = recall_score(cat_true, cat_pred, labels=range(6), average=None, zero_division=0)
    support = [int((cat_true == k).sum()) for k in range(6)]
    within1 = np.mean(np.abs(cat_true - cat_pred) <= 1)
    cover80 = float(np.mean((y >= p10) & (y <= p90)))      # calibrated coverage
    width = float(np.median(p90 - p10))
    out["levels"] = {"per_recall": [round(float(r), 2) for r in per_recall], "support": support,
                     "exact_acc": round(float((cat_true == cat_pred).mean()), 2),
                     "within1_acc": round(float(within1), 2),
                     "pi80_coverage": round(cover80, 2), "pi80_median_width": round(width, 1)}
    print("\nSEVERITY LEVELS (next-day, forecast+sensor):")
    for k in range(6):
        print(f"  {LABELS[k].replace(chr(10),' '):<15} support={support[k]:>3}  recall={per_recall[k]:.2f}")
    print(f"  exact-category accuracy {(cat_true==cat_pred).mean():.2f}; within-one-level {within1:.2f}")
    print(f"\nCONFIDENCE: 80% interval covers truth {cover80:.0%} of the time "
          f"(target 80%); median width ±{width/2:.0f} µg/m³")

    (C.ROOT / "models" / "horizon_levels.json").write_text(json.dumps(out, indent=2))

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})

    # forecast horizon
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(horizons, rec35, "o-", color="#e67e22", lw=2, label="catch ‘unhealthy-for-sensitive’ (>35)")
    ax[0].plot(horizons, rec55, "s-", color="#c0392b", lw=2, label="catch ‘unhealthy’ (>55)")
    ax[0].axhline(0.5, color="#999", ls=":", lw=1)
    ax[0].set(xlabel="forecast lead time (days ahead)", ylabel="catch rate (recall)",
              ylim=(0, 1), title="Catch rate vs forecast lead time (today's readings only)")
    ax[0].set_xticks(horizons); ax[0].legend(fontsize=9, loc="lower left")
    ax[1].plot(horizons, auc_fc, "^-", color="#2980b9", lw=2, label="with a weather forecast (optimistic)")
    ax[1].plot(horizons, auc_today, "o-", color="#7f8c8d", lw=2, label="today’s readings only (realistic floor)")
    ax[1].axhline(0.5, color="#999", ls=":", lw=1, label="coin-flip")
    ax[1].fill_between(horizons, auc_today, auc_fc, color="#2980b9", alpha=0.08)
    ax[1].set(xlabel="forecast lead time (days ahead)", ylabel="bad-air detection (ROC-AUC)",
              ylim=(0.5, 1), title="Value of a weather forecast vs lead time")
    ax[1].set_xticks(horizons); ax[1].legend(fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(C.ROOT / "figures" / "forecast_horizon.png", dpi=140); plt.close(fig)

    # aqi levels + confidence
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    im = ax[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        if v:
            ax[0].text(j, i, str(v), ha="center", va="center", fontsize=10,
                       color="white" if v > cm.max() * 0.5 else "#1b2a4a", fontweight="bold")
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels(LABELS, fontsize=8, rotation=35, ha="right")
    ax[0].set_yticks(range(6)); ax[0].set_yticklabels(LABELS, fontsize=8)
    ax[0].set(xlabel="predicted level", ylabel="true level",
              title=f"Severity-level truth table (next-day)\nexact {(cat_true==cat_pred).mean():.0%}, "
                    f"within-one-level {within1:.0%}")
    # confidence: sorted true value with 80% band
    o = np.argsort(p50)
    xx = np.arange(len(o))
    ax[1].fill_between(xx, np.minimum(p10, p90)[o], np.maximum(p10, p90)[o], color="#2980b9",
                       alpha=0.2, label="80% confidence band")
    ax[1].plot(xx, p50[o], color="#2980b9", lw=1.4, label="prediction (median)")
    ax[1].scatter(xx, y[o], s=8, color="#c0392b", alpha=0.5, label="actual")
    for b in BREAKS:
        ax[1].axhline(b, color="#bbb", ls=":", lw=0.8)
    ax[1].set(xlabel="test days (sorted by prediction)", ylabel="PM2.5 (µg/m³)", ylim=(0, 200),
              title=f"Confidence band (next-day)\n80% band holds {cover80:.0%} of the time")
    ax[1].legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(C.ROOT / "figures" / "aqi_levels.png", dpi=140); plt.close(fig)
    print("\nSaved figures/forecast_horizon.png, figures/aqi_levels.png, models/horizon_levels.json")


if __name__ == "__main__":
    main()
