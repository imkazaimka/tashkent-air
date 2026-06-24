"""
Reviewer-driven improvements — compute the new evidence the paper needs:

  1. PRECISION alongside recall for every model (Table 7 gap).
  2. Leave-one-winter-out cross-validation of the calibrated model (the ~8-month
     test window is the biggest credibility issue).
  3. Precision-recall curves for the models + the >35/>55 classifiers.
  4. Bootstrap confidence intervals on the driver-importance shares (Table 4).
  5. Two cheap ML wins: ensemble (calibrated + persistence) and quantile regression.

Run:  python src/improvements.py
Outputs: models/improvements.json, figures/pr_curves.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_recall_curve, average_precision_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

THR = C.PM25_THRESHOLD
FIG = C.ROOT / "figures"; MODELS = C.ROOT / "models"
DROP = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_roll3_mean",
        "pm25_roll7_mean", "pm25_roll7_std", "pm25_diff1", "episode_streak"]
AUTOCORR = DROP
OUT = {}


def pr(y_true, score, thr_pred=None):
    yb = y_true > THR
    if thr_pred is None:
        pb = score > THR            # regression model: threshold the prediction
    else:
        pb = score >= thr_pred
    tp = int((yb & pb).sum()); fp = int((~yb & pb).sum()); fn = int((yb & ~pb).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return round(prec, 3), round(rec, 3)


def assemble():
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"]).sort_values("date")
    gt["ground_lag1"] = gt["pm25_ground"].where(gt["date"].diff().dt.days.eq(1)).shift(1)
    cams = dm[["date", "pm2_5", "pm10", "dust", "nitrogen_dioxide", "ozone",
               "carbon_monoxide"]].rename(columns=lambda c: "cams_" + c if c != "date" else c)
    df = (feat.drop(columns=[c for c in DROP if c in feat.columns])
          .merge(cams, on="date").merge(gt, on="date")
          .dropna(subset=["pm25_ground", "cams_pm2_5"]).sort_values("date").reset_index(drop=True))
    cols = [c for c in df.columns if c not in ("date", "y", "split", "pm25_ground", "ground_lag1")]
    return df, cols


def fit(Xtr, ytr, **kw):
    m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=31,
                          subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
                          min_child_samples=15, reg_lambda=1.0, random_state=42,
                          n_jobs=-1, verbose=-1, **kw)
    m.fit(Xtr, ytr); return m


def main():
    df, cols = assemble()
    tr = df[df["date"] < "2024-06-01"]; te = df[df["date"] >= "2024-06-01"]
    ylog = lambda s: np.log1p(s.clip(lower=0))

    # ---------- 1. precision + recall for all Table-7 models ----------
    print("=== 1. Table 7 with PRECISION added (test = 253 days) ===")
    m = fit(tr[cols], ylog(tr["pm25_ground"]))
    pred = np.expm1(m.predict(te[cols]))
    o = te["pm25_ground"].values
    a, b = np.polyfit(tr["cams_pm2_5"], tr["pm25_ground"], 1)
    resc = a * te["cams_pm2_5"].values + b
    pe = te.dropna(subset=["ground_lag1"])
    models = {
        "raw_cams": te["cams_pm2_5"].values, "rescaled": resc,
        "persistence": None, "calibrated": pred}
    tab = {}
    for name, p in models.items():
        if name == "persistence":
            yy = pe["pm25_ground"].values; pp = pe["ground_lag1"].values
            mae = mean_absolute_error(yy, pp); rmse = np.sqrt(mean_squared_error(yy, pp))
            r2 = r2_score(yy, pp); prec, rec = pr(yy, pp)
        else:
            mae = mean_absolute_error(o, p); rmse = np.sqrt(mean_squared_error(o, p))
            r2 = r2_score(o, p); prec, rec = pr(o, p)
        tab[name] = {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "R2": round(r2, 2),
                     "precision35": prec, "recall35": rec}
        print(f"  {name:<12} MAE={mae:4.1f} RMSE={rmse:4.1f} R2={r2:+.2f} "
              f"precision={prec:.2f} recall={rec:.2f}")
    OUT["table7_with_precision"] = tab

    # ---------- 2. leave-one-winter-out CV (calibrated) ----------
    print("\n=== 2. Leave-one-winter-out cross-validation (calibrated model) ===")
    mo = df["date"].dt.month
    wy = np.where(mo >= 11, df["date"].dt.year, df["date"].dt.year - 1)
    df2 = df.assign(winter=wy)
    winters = [w for w in sorted(set(wy)) if ((df2["winter"] == w) &
               (df2["date"].dt.month.isin([11, 12, 1, 2, 3]))).sum() >= 40]
    cv = []
    for w in winters:
        test_mask = (df2["winter"] == w) & (df2["date"].dt.month.isin([11, 12, 1, 2, 3]))
        trn = df2[~test_mask]; tst = df2[test_mask]
        if len(tst) < 40:
            continue
        mm = fit(trn[cols], ylog(trn["pm25_ground"]))
        pp = np.expm1(mm.predict(tst[cols])); oo = tst["pm25_ground"].values
        prec, rec = pr(oo, pp)
        cv.append({"winter": f"{w}-{w+1}", "n": int(len(tst)), "R2": round(r2_score(oo, pp), 2),
                   "precision35": prec, "recall35": rec})
        print(f"  winter {w}-{w+1} (n={len(tst):3d}): R2={r2_score(oo,pp):+.2f} "
              f"precision={prec:.2f} recall={rec:.2f}")
    r2s = [c["R2"] for c in cv]; recs = [c["recall35"] for c in cv]; precs = [c["precision35"] for c in cv]
    print(f"  MEAN across winters: R2={np.mean(r2s):.2f}±{np.std(r2s):.2f}  "
          f"recall={np.mean(recs):.2f}±{np.std(recs):.2f}  precision={np.mean(precs):.2f}±{np.std(precs):.2f}")
    OUT["winter_cv"] = {"folds": cv, "R2_mean": round(float(np.mean(r2s)), 2),
                        "R2_sd": round(float(np.std(r2s)), 2),
                        "recall_mean": round(float(np.mean(recs)), 2),
                        "recall_sd": round(float(np.std(recs)), 2),
                        "precision_mean": round(float(np.mean(precs)), 2),
                        "precision_sd": round(float(np.std(precs)), 2)}

    # ---------- 3. precision-recall curves ----------
    print("\n=== 3. PR curves (figure) ===")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    yb = (o > THR).astype(int)
    for name, score, col in [("calibrated regression", pred, "#16a085"),
                             ("CAMS×rescale", resc, "#e67e22"),
                             ("raw CAMS", te["cams_pm2_5"].values, "#2980b9")]:
        p_, r_, _ = precision_recall_curve(yb, score)
        ap = average_precision_score(yb, score)
        ax.plot(r_, p_, color=col, lw=2, label=f"{name} (AP={ap:.2f})")
    # dedicated >35 classifier
    clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=31,
                             scale_pos_weight=(tr["pm25_ground"] <= THR).sum() /
                             max((tr["pm25_ground"] > THR).sum(), 1),
                             random_state=42, verbose=-1)
    clf.fit(tr[cols], (tr["pm25_ground"] > THR).astype(int))
    proba = clf.predict_proba(te[cols])[:, 1]
    p_, r_, _ = precision_recall_curve(yb, proba); ap = average_precision_score(yb, proba)
    ax.plot(r_, p_, color="#8e44ad", lw=2.5, ls="--", label=f">35 classifier (AP={ap:.2f})")
    ax.axhline(yb.mean(), color="#999", ls=":", lw=1, label=f"baseline ({yb.mean():.2f})")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision–recall for the >35 µg/m³ warning")
    ax.legend(fontsize=9); plt.tight_layout(); fig.savefig(FIG / "pr_curves.png", dpi=130); plt.close(fig)
    OUT["classifier35_ap"] = round(float(ap), 3)
    print(f"  saved figures/pr_curves.png  (>35 classifier AP={ap:.2f})")

    # ---------- 4. bootstrap CIs on driver-importance shares ----------
    print("\n=== 4. Bootstrap CIs on driver-importance shares (Table 4) ===")
    f = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    exo = [c for c in f.columns if c not in (["date", "y", "split"] + AUTOCORR)]
    GROUPS = {"regional": [c for c in exo if "_pm25_lag" in c or "_transport" in c],
              "dispersion": ["boundary_layer_height", "ventilation_coef", "trapping_index",
                             "inv_wind", "stagnation_proxy", "wind_speed_10m", "surface_pressure"],
              "thermal_season": ["temperature_2m", "shortwave_radiation", "relative_humidity_2m",
                                 "doy_sin", "doy_cos", "is_heating_season"],
              "precip": ["precip", "precip_lag1", "precip_sum_3d"],
              "wind_vec": ["wind_sin", "wind_cos"]}
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(0)
    tr2 = f[f["split"].isin(["train", "val"])]; te2 = f[f["split"] == "test"].reset_index(drop=True)
    base = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             random_state=42, verbose=-1).fit(tr2[exo], tr2["y"])
    shares = {g: [] for g in GROUPS}
    for _ in range(200):
        idx = rng.integers(0, len(te2), len(te2))
        pi = permutation_importance(base, te2.iloc[idx][exo], te2.iloc[idx]["y"],
                                    n_repeats=3, random_state=1, scoring="r2")
        imp = pd.Series(pi.importances_mean, index=exo).clip(lower=0)
        tot = imp.sum() or 1
        for g, cs in GROUPS.items():
            shares[g].append(imp[[c for c in cs if c in exo]].sum() / tot * 100)
    print("  group importance share, 95% CI:")
    ci = {}
    for g in GROUPS:
        lo, hi = np.percentile(shares[g], [2.5, 97.5]); me = np.mean(shares[g])
        ci[g] = {"mean": round(float(me), 0), "lo": round(float(lo), 0), "hi": round(float(hi), 0)}
        print(f"    {g:<16} {me:4.0f}%  [{lo:3.0f}, {hi:3.0f}]")
    OUT["driver_importance_ci"] = ci

    # ---------- 5. cheap ML wins ----------
    print("\n=== 5. ML wins: ensemble & quantile ===")
    # ensemble calibrated + persistence (on rows with ground_lag1)
    pe2 = te.dropna(subset=["ground_lag1"]).copy()
    pe2["cal"] = np.expm1(m.predict(pe2[cols]))
    ens = 0.5 * pe2["cal"].values + 0.5 * pe2["ground_lag1"].values
    oo = pe2["pm25_ground"].values
    pe_p, pe_r = pr(oo, ens)
    print(f"  ensemble (0.5 cal + 0.5 persistence): R2={r2_score(oo,ens):+.2f} "
          f"precision={pe_p:.2f} recall={pe_r:.2f}")
    # quantile regression (predict 80th pct) for catching high days
    mq = fit(tr[cols], ylog(tr["pm25_ground"]), objective="quantile", alpha=0.8)
    pq = np.expm1(mq.predict(te[cols])); qp, qr = pr(o, pq)
    print(f"  quantile-0.8 model: precision={qp:.2f} recall={qr:.2f} (vs point recall "
          f"{tab['calibrated']['recall35']})")
    OUT["ml_wins"] = {"ensemble": {"r2": round(r2_score(oo, ens), 2), "precision": pe_p, "recall": pe_r},
                      "quantile80": {"precision": qp, "recall": qr}}

    (MODELS / "improvements.json").write_text(json.dumps(OUT, indent=2))
    print("\nSaved models/improvements.json and figures/pr_curves.png")


if __name__ == "__main__":
    main()
