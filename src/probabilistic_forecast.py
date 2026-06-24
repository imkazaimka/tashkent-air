"""
Probabilistic warning forecaster — three upgrades the user asked for:

(3) PREDICT THE ODDS, NOT THE NUMBER. For each day the model outputs the probability that
    PM2.5 will EXCEED each threshold:  P(>10), P(>20), P(>35), P(>55), P(>150).
    e.g. "tomorrow: >10 with 98%, >20 with 60%, >35 with 25% ...".

(1) THE CONFIDENCE IS COMPUTED, NOT ASSUMED. Each probability is isotonically CALIBRATED on
    held-out data, so "70%" really means it happens ~70% of the time (reliability diagram + Brier).
    Any confidence level (80%, 90%, ...) is then just read off the calibrated curve, not hard-coded.

(2) THE DECISION KNOB ("third variator"). A warning fires when P(>35) >= tau. Sweeping tau trades
    FALSE ALARMS against MISSED BAD DAYS: low tau -> catch almost everything but cry wolf; high tau
    -> few false alarms but miss real episodes. We report the curve and three operating points.

Setup: next-day, forecast+sensor; train <2024-06 (inner split for calibration), test >=2024-06.

Output: figures/probabilistic_forecast.png, figures/warning_threshold.png, models/probabilistic.json
Run:    python src/probabilistic_forecast.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

SPLIT = "2024-06-01"
THRESHOLDS = [10, 20, 35, 55, 150]
FCAST = ["temperature_2m", "wind_speed_10m", "boundary_layer_height", "relative_humidity_2m",
         "surface_pressure", "pm2_5", "nitrogen_dioxide", "dust", "shortwave_radiation"]
WARN = 35  # the actionable "bad air" line for the decision-threshold analysis


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    gt = gt[(gt.date >= dm.date.min()) & (gt.date <= dm.date.max())]
    d = dm.merge(gt, on="date", how="left").sort_values("date").reset_index(drop=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    d["target"] = d["pm25_ground"].shift(-1)
    d["cams_lag"] = d["pm2_5"]; d["cams_roll7"] = d["pm2_5"].rolling(7, min_periods=3).mean()
    d["real_lag"] = d["pm25_ground"]; d["real_roll7"] = d["pm25_ground"].rolling(7, min_periods=3).mean()
    feats = FCAST + ["doy_sin", "doy_cos", "cams_lag", "cams_roll7", "real_lag", "real_roll7"]
    d = d.dropna(subset=["target"]).reset_index(drop=True)

    tr = d[d.date < SPLIT]; te = d[d.date >= SPLIT].copy()
    cut = tr.date.quantile(0.80); trf, cal = tr[tr.date <= cut], tr[tr.date > cut]

    # calibrated exceedance probability per threshold
    probs_te, probs_cal, brier = {}, {}, {}
    for t in THRESHOLDS:
        ytr = (trf["target"] > t).astype(int)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                 min_child_samples=20, random_state=42, verbose=-1)
        clf.fit(trf[feats], ytr)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(cal[feats])[:, 1], (cal["target"] > t).astype(int))
        probs_te[t] = iso.transform(clf.predict_proba(te[feats])[:, 1])
        if (te["target"] > t).sum() > 3:
            brier[t] = float(brier_score_loss((te["target"] > t).astype(int), probs_te[t]))

    # enforce monotonic non-increasing across thresholds (P>10 >= P>20 >= ...)
    P = np.vstack([probs_te[t] for t in THRESHOLDS])           # (n_thr, n_days)
    P = np.minimum.accumulate(P, axis=0)
    for i, t in enumerate(THRESHOLDS):
        probs_te[t] = P[i]

    print("PROBABILISTIC FORECAST — example test days (P that tomorrow's PM2.5 exceeds ...):")
    print(f"  {'date':<12}{'actual':>7} | " + " ".join(f">{t:<4}" for t in THRESHOLDS))
    te = te.reset_index(drop=True)
    pick = [te["target"].idxmin(), (te["target"] - 30).abs().idxmin(), te["target"].idxmax()]
    for i in pick:
        row = " ".join(f"{probs_te[t][i]*100:>4.0f}%" for t in THRESHOLDS)
        print(f"  {str(te['date'][i].date()):<12}{te['target'][i]:>7.0f} | {row}")

    print("\nCALIBRATION (are the probabilities honest?):")
    for t in THRESHOLDS:
        if t in brier:
            print(f"  P(>{t:<3}) Brier = {brier[t]:.3f}  (lower is better; base rate "
                  f"{(te['target']>t).mean():.2f})")

    # ---------- decision knob: warn if P(>35) >= tau ----------
    p35 = probs_te[WARN]; y35 = (te["target"] > WARN).values
    taus = np.linspace(0.05, 0.95, 19)
    rec, far, prec = [], [], []
    for tau in taus:
        w = p35 >= tau
        tp = (w & y35).sum(); fp = (w & ~y35).sum(); fn = (~w & y35).sum(); tn = (~w & ~y35).sum()
        rec.append(tp / max(tp + fn, 1)); far.append(fp / max(fp + tn, 1))
        prec.append(tp / max(tp + fp, 1))
    rec, far, prec = map(np.array, (rec, far, prec))
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)

    def op(mask, key):
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else int(np.argmax(key))
    i_caut = op(rec >= 0.90, rec)                              # catch >=90% of bad days
    i_bal = int(np.argmax(f1))                                 # best balance
    i_prec = op(prec >= 0.85, prec)                            # few false alarms
    ops = {"cautious (catch >=90%)": i_caut, "balanced (max F1)": i_bal,
           "precise (few false alarms)": i_prec}
    print("\nDECISION KNOB — warn if P(>35) >= tau:")
    print(f"  {'setting':<28}{'tau':>5}{'catch':>7}{'false-alarm':>13}{'precision':>11}")
    op_rows = {}
    for name, i in ops.items():
        print(f"  {name:<28}{taus[i]:>5.2f}{rec[i]:>7.0%}{far[i]:>12.0%}{prec[i]:>11.0%}")
        op_rows[name] = {"tau": round(float(taus[i]), 2), "recall": round(float(rec[i]), 2),
                         "far": round(float(far[i]), 2), "precision": round(float(prec[i]), 2)}

    json.dump({"brier": {str(k): round(v, 3) for k, v in brier.items()},
               "operating_points": op_rows,
               "sweep": {"tau": [round(float(x), 2) for x in taus],
                         "recall": [round(float(x), 2) for x in rec],
                         "far": [round(float(x), 2) for x in far],
                         "precision": [round(float(x), 2) for x in prec]}},
              open(C.ROOT / "models" / "probabilistic.json", "w"), indent=2)

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})

    # A: example exceedance ladders + reliability
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    labs = ["a clean day", "a moderate day", "a bad day"]
    cols = ["#27ae60", "#e67e22", "#c0392b"]
    x = np.arange(len(THRESHOLDS)); w = 0.26
    for k, i in enumerate(pick):
        ax[0].bar(x + (k - 1) * w, [probs_te[t][i] * 100 for t in THRESHOLDS], w,
                  color=cols[k], label=f"{labs[k]} (actual {te['target'][i]:.0f})")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f">{t}" for t in THRESHOLDS])
    ax[0].set(xlabel="PM2.5 threshold (µg/m³)", ylabel="probability of exceeding (%)", ylim=(0, 105),
              title="Tomorrow’s odds, not a single number")
    ax[0].legend(fontsize=9)
    # reliability for >35
    bins = np.linspace(0, 1, 11)
    bi = np.digitize(p35, bins) - 1
    mp, of = [], []
    for b in range(10):
        m = bi == b
        if m.sum() >= 5:
            mp.append(p35[m].mean()); of.append(y35[m].mean())
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax[1].plot(mp, of, "o-", color="#2980b9", lw=2, label="model (>35 warning)")
    ax[1].set(xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1),
              title="The probabilities are honest (calibrated)")
    ax[1].legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(C.ROOT / "figures" / "probabilistic_forecast.png", dpi=140); plt.close(fig)

    # B: the decision knob
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(taus, rec * 100, "o-", color="#c0392b", lw=2, label="bad days CAUGHT (recall)")
    ax.plot(taus, far * 100, "s-", color="#7f8c8d", lw=2, label="FALSE-ALARM rate")
    for name, i in ops.items():
        ax.axvline(taus[i], color="#2980b9", ls=":", lw=1)
        ax.text(taus[i], 102, name.split(" (")[0], rotation=90, va="bottom", ha="center",
                fontsize=8.5, color="#2980b9")
    ax.set(xlabel="warning threshold τ  (warn if P(>35) ≥ τ)", ylabel="% of days", ylim=(0, 110),
           title="Warning threshold: catch rate vs false-alarm rate")
    ax.legend(fontsize=9.5, loc="center right")
    fig.tight_layout(); fig.savefig(C.ROOT / "figures" / "warning_threshold.png", dpi=140); plt.close(fig)
    print("\nSaved figures/probabilistic_forecast.png, figures/warning_threshold.png, models/probabilistic.json")


if __name__ == "__main__":
    main()
