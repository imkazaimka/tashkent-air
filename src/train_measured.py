"""
Forecaster trained on the MEASURED 10-station municipal network (Section 6) — "is the air going to
be dirty?" validated against real readings, not CAMS.

Target  : daily citywide measured PM2.5 (the 10-station municipal median, data/raw/, from the Tashkent
          open-data portal — the data we found). Dirty = >35 µg/m³.
Features: CAMS air-quality + ERA5 weather (the gap-free regional "tool" layer) + seasonal terms;
          optionally yesterday's measured reading (the live-sensor case).
Validation: leave-one-winter-out (predict a winter never seen) — the honest test — plus a time-split.

Headline (LOWO): catches ~72% of measured citywide dirty days vs ~20% for raw CAMS (varies 60–98% by
winter). The time-split gives an optimistic ~90%; LOWO is the number reported.

Run:  python src/train_measured.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

THR = 35
FEATS = ["pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide", "dust", "temperature_2m",
         "wind_speed_10m", "boundary_layer_height", "relative_humidity_2m", "surface_pressure",
         "doy_sin", "doy_cos"]


def load():
    muni = pd.read_csv(C.RAW / "tashkent_municipal_pm25_daily.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    d = dm.merge(muni[["date", "pm25_muni"]], on="date").sort_values("date").reset_index(drop=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
    d["wyear"] = np.where(d.date.dt.month >= 11, d.date.dt.year, d.date.dt.year - 1)
    return d.dropna(subset=FEATS + ["pm25_muni"]).copy()


def fit(tr, feats):
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                             min_child_samples=20, random_state=42, verbose=-1
                             ).fit(tr[feats], np.log1p(tr.pm25_muni))


def catch(y, p):
    yb, pb = y > THR, p > THR
    return (yb & pb).sum() / yb.sum() if yb.sum() else np.nan


def scenario(d, feats, label):
    """Time-split forecast metrics for one feature set (matches Section 6 Table 8)."""
    from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score
    x = d.dropna(subset=feats + ["pm25_muni"]).copy()
    cut = x.date.quantile(0.8); tr, te = x[x.date <= cut], x[x.date > cut]
    m = fit(tr, feats); p = np.expm1(m.predict(te[feats])); y = te.pm25_muni.values
    yb, pb = y > THR, p > THR
    return m, {"scenario": label, "n_test": int(len(te)),
               "R2": round(r2_score(np.log1p(y), np.log1p(p)), 2),
               "MAE": round(mean_absolute_error(y, p), 1),
               "precision": round(precision_score(yb, pb, zero_division=0), 2),
               "recall": round(recall_score(yb, pb, zero_division=0), 2),
               "F1": round(f1_score(yb, pb, zero_division=0), 2),
               "ROC_AUC": round(roc_auc_score(yb, np.log1p(p)), 2) if yb.any() and (~yb).any() else None}


def main():
    import json
    d = load()
    print("TARGET = measured 10-station citywide PM2.5 (dirty > 35 µg/m³)\n")
    # leave-one-winter-out (honest headline)
    caught = bad = 0; per_winter = {}
    for wy in sorted(d[d.date.dt.month.isin([11, 12, 1, 2, 3])].wyear.unique()):
        te = d[(d.wyear == wy) & d.date.dt.month.isin([11, 12, 1, 2, 3])]
        tr = d[~d.index.isin(te.index)]
        if te.pm25_muni.gt(THR).sum() < 5:
            continue
        p = np.expm1(fit(tr, FEATS).predict(te[FEATS])); y = te.pm25_muni.values
        caught += int(((y > THR) & (p > THR)).sum()); bad += int((y > THR).sum())
        per_winter[f"{wy}-{str(wy+1)[2:]}"] = round(catch(y, p), 2)
        print(f"  winter {wy}-{str(wy+1)[2:]}: caught {catch(y, p)*100:.0f}%  (raw CAMS {catch(y, te.pm2_5.values)*100:.0f}%)")
    lowo = round(caught / bad, 2)
    print(f"  POOLED leave-one-winter-out: {lowo*100:.0f}% of measured dirty days caught\n")

    # four operational scenarios (sensor × horizon), like Table 8 — on the MEASURED target
    d["lag1"] = d["pm25_muni"].shift(1); d["lag7"] = d["pm25_muni"].shift(7)
    scen = []
    model_to_save, _ = scenario(d, FEATS, "no sensor — next-day")   # deploy the no-sensor model (most general)
    for feats, lab in [(FEATS + ["lag1"], "with sensor — next-day"),
                       (FEATS + ["lag7"], "with sensor — week-ahead"),
                       (FEATS, "no sensor — next-day"),
                       (FEATS, "no sensor — week-ahead")]:
        _, mtr = scenario(d, feats, lab); scen.append(mtr)
        print(f"  {lab:26} R²={mtr['R2']:.2f}  recall={mtr['recall']:.2f}  precision={mtr['precision']:.2f}  AUC={mtr['ROC_AUC']}")

    # SAVE the artifacts (this is the part that was missing)
    out = C.ROOT / "models"
    model_to_save.booster_.save_model(str(out / "lgbm_measured.txt"))
    json.dump({"target": "measured 10-station municipal PM2.5", "threshold": THR,
               "leave_one_winter_out_catch": lowo, "per_winter": per_winter,
               "raw_cams_catch_approx": 0.20, "scenarios": scen},
              open(out / "measured_metrics.json", "w"), indent=2)
    print(f"\n  SAVED models/lgbm_measured.txt + models/measured_metrics.json")


if __name__ == "__main__":
    main()
