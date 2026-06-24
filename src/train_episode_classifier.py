"""
A2 — episode-onset classifier: will tomorrow's REAL PM2.5 exceed a health threshold?

Target = ground-truth exceedance (US-Embassy sensor). Two thresholds:
  35 ug/m3  (Unhealthy-for-Sensitive boundary)   and   55 ug/m3 (episode).
Features = the deployable A1 set (CAMS forecast + weather + dust + regional +
season). Compared against raw-CAMS, rescaled-CAMS, and persistence rules.
Special focus: recall on ONSET days (first day of an exceedance run) — the
hardest and most health-relevant case.

Run:  python src/train_episode_classifier.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"; MODELS = C.ROOT / "models"
TEST_FROM = "2024-06-01"
DROP = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_roll3_mean",
        "pm25_roll7_mean", "pm25_roll7_std", "pm25_diff1", "episode_streak"]


def assemble():
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"]).sort_values("date")
    cams = dm[["date", "pm2_5", "pm10", "dust", "nitrogen_dioxide", "ozone",
               "carbon_monoxide"]].rename(columns=lambda c: "cams_" + c if c != "date" else c)
    gt["ground_lag1"] = gt["pm25_ground"].where(gt["date"].diff().dt.days.eq(1)).shift(1)
    df = (feat.drop(columns=[c for c in DROP if c in feat.columns])
          .merge(cams, on="date").merge(gt, on="date"))
    return df.dropna(subset=["pm25_ground", "cams_pm2_5"]).sort_values("date").reset_index(drop=True)


def onset_mask(exceed: pd.Series) -> pd.Series:
    """True on the first day of each exceedance run."""
    return exceed & ~exceed.shift(1, fill_value=False)


def pr_at(y, p, thr=0.5):
    pb = p >= thr
    tp = int((y & pb).sum()); fp = int((~y & pb).sum()); fn = int((y & ~pb).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return prec, rec


def main():
    df = assemble()
    base_cols = [c for c in df.columns
                 if c not in (["date", "y", "split", "pm25_ground", "ground_lag1"])]
    tr = df[df["date"] < TEST_FROM]; te = df[df["date"] >= TEST_FROM]
    cut = tr["date"].quantile(0.85)
    trA, va = tr[tr["date"] <= cut], tr[tr["date"] > cut]
    print(f"train {len(trA)}  val {len(va)}  test {len(te)}")

    OUT = {}
    for thr in (35.0, 55.0):
        ytr = (trA["pm25_ground"] > thr).astype(int)
        yva = (va["pm25_ground"] > thr).astype(int)
        yte = (te["pm25_ground"] > thr).astype(int)
        pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

        clf = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.02, num_leaves=31,
                                 subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
                                 min_child_samples=15, reg_lambda=1.0,
                                 scale_pos_weight=pos_w, random_state=42,
                                 n_jobs=-1, verbose=-1)
        clf.fit(trA[base_cols], ytr, eval_set=[(va[base_cols], yva)],
                callbacks=[lgb.early_stopping(60, verbose=False)])
        proba = clf.predict_proba(te[base_cols])[:, 1]

        auc = roc_auc_score(yte, proba) if yte.nunique() > 1 else float("nan")
        ap = average_precision_score(yte, proba)
        # choose operating threshold maximising F1 on validation
        pva = clf.predict_proba(va[base_cols])[:, 1]
        ts = np.linspace(0.1, 0.9, 33)
        best_t = ts[np.argmax([f1_score(yva, pva >= t, zero_division=0) for t in ts])]
        prec, rec = pr_at(yte.astype(bool), proba, best_t)

        # onset recall
        onset = onset_mask(te["pm25_ground"] > thr).values
        onset_rec = float(((proba >= best_t) & onset).sum() / max(onset.sum(), 1))

        # baselines
        cams_b = pr_at(yte.astype(bool), (te["cams_pm2_5"] > thr).values.astype(float), 0.5)
        a, b = np.polyfit(trA["cams_pm2_5"], trA["pm25_ground"], 1)
        resc_b = pr_at(yte.astype(bool), ((a*te["cams_pm2_5"]+b) > thr).values.astype(float), 0.5)
        pe = te.dropna(subset=["ground_lag1"])
        pers_b = pr_at((pe["pm25_ground"] > thr).values,
                       (pe["ground_lag1"] > thr).values.astype(float), 0.5)

        OUT[f"thr_{int(thr)}"] = {
            "n_test_pos": int(yte.sum()), "roc_auc": float(auc), "pr_auc": float(ap),
            "op_threshold": float(best_t),
            "classifier": {"precision": prec, "recall": rec, "onset_recall": onset_rec},
            "raw_cams": {"precision": cams_b[0], "recall": cams_b[1]},
            "rescaled_cams": {"precision": resc_b[0], "recall": resc_b[1]},
            "persistence": {"precision": pers_b[0], "recall": pers_b[1]},
        }
        print(f"\n>{int(thr)} ug/m3  ({int(yte.sum())}/{len(te)} test days positive)")
        print(f"  classifier   ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")
        print(f"  classifier   precision={prec:.2f} recall={rec:.2f} onset_recall={onset_rec:.2f}")
        print(f"  raw CAMS     precision={cams_b[0]:.2f} recall={cams_b[1]:.2f}")
        print(f"  rescaled CAMS precision={resc_b[0]:.2f} recall={resc_b[1]:.2f}")
        print(f"  persistence  precision={pers_b[0]:.2f} recall={pers_b[1]:.2f}")
        if int(thr) == 55:
            clf.booster_.save_model(str(MODELS / "lgbm_episode_clf_55.txt"))

    (MODELS / "episode_classifier_metrics.json").write_text(json.dumps(OUT, indent=2))
    print("\nSaved models/episode_classifier_metrics.json")


if __name__ == "__main__":
    main()
