"""
A1 — train a GROUND-TRUTH model: predict the real US-Embassy PM2.5, not CAMS.

Reframe: CAMS is mis-calibrated (~46% of real, misses 81% of episodes), but it
still carries day-to-day signal (r=0.57). So we don't discard it — we treat the
CAMS forecast as an INPUT and learn to correct it to reality, using weather,
dust, regional transport and season as context.

Deployable model (no live ground sensor needed at inference):
    real_PM2.5(D)  ~  CAMS_forecast(D) + weather(D) + regional(D) + calendar(D)
We also fit a "+live sensor" variant (adds yesterday's real PM2.5) as an upper
bound, and report it separately.

Target is modelled in log space (PM2.5 is heavy-tailed: real max 686).

Baselines on the test period:
    raw CAMS · linear rescale of CAMS (fit on train) · real-data persistence.

Run:  python src/train_ground_truth.py   (after fetch_openaq.py + features.py)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"; MODELS = C.ROOT / "models"
TEST_FROM = "2024-06-01"            # test = last winter+ in the paired window
THR, EPI = C.PM25_THRESHOLD, C.EPISODE_THRESHOLD

# CAMS autocorrelation features are excluded — the CAMS signal enters via the
# same-day CAMS pollutant forecasts instead.
DROP = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_roll3_mean",
        "pm25_roll7_mean", "pm25_roll7_std", "pm25_diff1", "episode_streak"]


def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)))


def exceed(y, p, thr):
    yb, pb = y > thr, p > thr
    tp = int((yb & pb).sum()); fp = int((~yb & pb).sum()); fn = int((yb & ~pb).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(precision=prec, recall=rec, n_exceed=int(yb.sum()))


def fit_lgbm(Xtr, ytr, Xva, yva):
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=31,
                          subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
                          min_child_samples=15, reg_lambda=1.0, random_state=42,
                          n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m


def main():
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])

    # same-day CAMS pollutant forecasts as inputs (forecastable; dust matters here)
    cams = dm[["date", "pm2_5", "pm10", "dust", "nitrogen_dioxide", "ozone",
               "carbon_monoxide"]].rename(columns=lambda c: "cams_" + c if c != "date" else c)

    # ground-truth lag (real persistence) from the full ground series
    gt = gt.sort_values("date")
    gt["ground_lag1"] = gt["pm25_ground"].where(
        gt["date"].diff().dt.days.eq(1)).shift(1)

    df = (feat.drop(columns=[c for c in DROP if c in feat.columns])
          .merge(cams, on="date").merge(gt, on="date"))
    df = df.dropna(subset=["pm25_ground", "cams_pm2_5"]).sort_values("date").reset_index(drop=True)

    base_cols = [c for c in df.columns
                 if c not in (["date", "y", "split", "pm25_ground", "ground_lag1"])]
    print(f"Paired days: {len(df)}  | features (deployable): {len(base_cols)}")

    # chronological split
    tr = df[df["date"] < TEST_FROM]; te = df[df["date"] >= TEST_FROM]
    # carve a small val tail from train for early stopping
    cut = tr["date"].quantile(0.85)
    trA, va = tr[tr["date"] <= cut], tr[tr["date"] > cut]
    print(f"train {len(trA)}  val {len(va)}  test {len(te)}  "
          f"(test {te['date'].min().date()} -> {te['date'].max().date()})")

    ylog = lambda s: np.log1p(s.clip(lower=0))
    ytr_real, yte_real = trA["pm25_ground"].values, te["pm25_ground"].values

    results = {}

    # ---- deployable model (no live sensor) ----
    mdep = fit_lgbm(trA[base_cols], ylog(trA["pm25_ground"]),
                    va[base_cols], ylog(va["pm25_ground"]))
    pdep = np.expm1(mdep.predict(te[base_cols]))
    results["ground_model_deployable"] = {**metrics(yte_real, pdep),
                                           "exceed35": exceed(yte_real, pdep, THR),
                                           "exceed55": exceed(yte_real, pdep, EPI)}

    # ---- +live-sensor variant (adds yesterday's real PM2.5) ----
    cols2 = base_cols + ["ground_lag1"]
    mlive = fit_lgbm(trA[cols2], ylog(trA["pm25_ground"]),
                     va[cols2], ylog(va["pm25_ground"]))
    plive = np.expm1(mlive.predict(te[cols2]))
    results["ground_model_plus_live_sensor"] = {**metrics(yte_real, plive),
                                                 "exceed35": exceed(yte_real, plive, THR)}

    # ---- baselines ----
    raw = te["cams_pm2_5"].values
    results["baseline_raw_cams"] = {**metrics(yte_real, raw),
                                    "exceed35": exceed(yte_real, raw, THR)}
    # linear rescale fit on TRAIN only
    a, b = np.polyfit(trA["cams_pm2_5"], trA["pm25_ground"], 1)
    resc = a * raw + b
    results["baseline_cams_rescaled"] = {**metrics(yte_real, resc),
                                         "rescale": [float(a), float(b)],
                                         "exceed35": exceed(yte_real, resc, THR)}
    # real-data persistence (where yesterday's real value exists)
    pe = te.dropna(subset=["ground_lag1"])
    results["baseline_persistence"] = {**metrics(pe["pm25_ground"].values,
                                                 pe["ground_lag1"].values),
                                       "exceed35": exceed(pe["pm25_ground"].values,
                                                          pe["ground_lag1"].values, THR),
                                       "n": int(len(pe))}

    # ---- report ----
    print("\n================ GROUND-TRUTH TEST RESULTS ================")
    print(f"{'model':<32}{'MAE':>7}{'RMSE':>7}{'R2':>7}{'>35 rec':>9}{'>35 prec':>9}")
    order = ["baseline_raw_cams", "baseline_cams_rescaled", "baseline_persistence",
             "ground_model_deployable", "ground_model_plus_live_sensor"]
    for k in order:
        r = results[k]; e = r["exceed35"]
        print(f"{k:<32}{r['MAE']:>7.1f}{r['RMSE']:>7.1f}{r['R2']:>7.3f}"
              f"{e['recall']:>9.2f}{e['precision']:>9.2f}")
    print(f"\nreal exceedance (>35) days in test: "
          f"{results['ground_model_deployable']['exceed35']['n_exceed']} of {len(te)}")
    print(f"deployable model episode (>55) recall: "
          f"{results['ground_model_deployable']['exceed55']['recall']:.2f}")

    mdep.booster_.save_model(str(MODELS / "lgbm_ground_truth.txt"))
    (MODELS / "ground_truth_model_metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"date": te["date"].values, "actual": yte_real,
                  "model": pdep, "raw_cams": raw, "rescaled_cams": resc}
                 ).to_csv(MODELS / "ground_truth_test_predictions.csv", index=False)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(te["date"], yte_real, color="#111", lw=1.4, label="real sensor")
    ax.plot(te["date"], pdep, color="#16a085", lw=1.2, label="ground-truth model")
    ax.plot(te["date"], raw, color="#2980b9", lw=1.0, alpha=0.7, label="raw CAMS")
    ax.axhline(THR, color="#c0392b", ls=":", lw=1, label="35 ug/m3")
    ax.set(title="A1: calibrated ground-truth model vs real sensor vs raw CAMS (test period)",
           ylabel="PM2.5 (ug/m3)")
    ax.legend(ncol=4, fontsize=9); plt.tight_layout()
    fig.savefig(FIG / "ground_truth_model.png", dpi=130); plt.close(fig)
    print(f"\nSaved model + metrics + predictions + figure.")


if __name__ == "__main__":
    main()
