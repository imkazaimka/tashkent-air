"""
Phase 3 — train the LightGBM PM2.5 forecaster and evaluate vs persistence.

Uses the FULL feature set (incl. PM2.5 autocorrelation) — this is the predictive
model. Root-cause attribution is done separately in research.py with an
exogenous model.

Run:  python src/train.py        (after features.py)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

MODELS = C.ROOT / "models"
THR = C.PM25_THRESHOLD


def load():
    f = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    feat_cols = [c for c in f.columns if c not in ("date", "y", "split")]
    return f, feat_cols


def split_xy(f, feat_cols, which):
    s = f[f["split"] == which]
    return s[feat_cols], s["y"], s


def metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def threshold_acc(y, p, thr=THR):
    yb, pb = (y > thr), (p > thr)
    tp = int(((yb) & (pb)).sum()); fp = int((~yb & pb).sum())
    fn = int((yb & ~pb).sum()); tn = int((~yb & ~pb).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_exceed": int(yb.sum())}


def main():
    MODELS.mkdir(exist_ok=True)
    f, feat_cols = load()
    Xtr, ytr, _ = split_xy(f, feat_cols, "train")
    Xva, yva, _ = split_xy(f, feat_cols, "val")
    Xte, yte, ste = split_xy(f, feat_cols, "test")
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  | {len(feat_cols)} features")

    params = dict(objective="regression", metric=["mae", "rmse"],
                  num_leaves=63, learning_rate=0.03, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, min_child_samples=10,
                  reg_alpha=0.1, reg_lambda=1.0, verbose=-1, seed=42)
    model = lgb.train(
        params,
        lgb.Dataset(Xtr, ytr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xva, yva)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.save_model(str(MODELS / "lgbm.txt"))
    print(f"best_iteration: {model.best_iteration}")

    # --- evaluate on held-out TEST ---
    pte = model.predict(Xte)
    m = metrics(yte, pte)
    ta = threshold_acc(yte, pte)

    # baselines on the SAME test rows
    persist = ste["pm25_lag1"]
    mp = metrics(yte, persist)
    tap = threshold_acc(yte, persist)

    print("\n================  TEST-SET RESULTS  ================")
    print(f"{'metric':<14}{'LightGBM':>12}{'persistence':>14}")
    for k in ("MAE", "RMSE", "R2"):
        print(f"{k:<14}{m[k]:>12.3f}{mp[k]:>14.3f}")
    print(f"{'skill vs persist (MAE)':<28}"
          f"{(1 - m['MAE'] / mp['MAE']) * 100:>+6.1f}%")
    print(f"\nThreshold >{THR:.0f} ug/m3  (n_exceed={ta['n_exceed']} of {len(yte)} days)")
    print(f"  LightGBM   : acc {ta['accuracy']:.3f}  precision {ta['precision']:.3f}  recall {ta['recall']:.3f}")
    print(f"  persistence: acc {tap['accuracy']:.3f}  precision {tap['precision']:.3f}  recall {tap['recall']:.3f}")

    report = {"test": m, "test_persistence": mp,
              "threshold": ta, "threshold_persistence": tap,
              "best_iteration": model.best_iteration,
              "n": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)}}
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    # save test predictions for downstream plots
    pd.DataFrame({"date": ste["date"].values, "y": yte.values,
                  "pred": pte, "persistence": persist.values}
                 ).to_csv(MODELS / "test_predictions.csv", index=False)
    print(f"\nSaved model + metrics.json + test_predictions.csv to {MODELS}/")


if __name__ == "__main__":
    main()
