"""
Multi-sensor validation — use ALL available Tashkent ground sensors, not just one.

Addresses three reviewer critiques:
  * single sensor / thin sample  -> add the Sputnik-4 station (different location)
  * no ground truth after 2025-03 -> Sputnik-4 covers 2025-06 .. 2026-01
  * no spatial validation         -> check the CAMS bias at two locations/periods

Three analyses:
  1. CAMS vs each station (and combined): is the ~2x under-bias consistent across
     a second location and a later period?
  2. Generalisation: does the calibration model — trained ONLY on the embassy
     (2022-2025) — still predict the Sputnik-4 station (a different sensor type,
     location, and time window)? This is an out-of-distribution test.
  3. Figure: full record with both stations.

Run:  python src/validate_multi_sensor.py   (after fetch_openaq.py + train_ground_truth.py)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"; MODELS = C.ROOT / "models"
THR = C.PM25_THRESHOLD
DROP = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_roll3_mean",
        "pm25_roll7_mean", "pm25_roll7_std", "pm25_diff1", "episode_streak"]
OUT = {}


def agree(cams, obs):
    c, g = np.asarray(cams), np.asarray(obs)
    return dict(n=int(len(c)), mean_cams=float(c.mean()), mean_obs=float(g.mean()),
                ratio=float(g.mean() / c.mean()), pearson=float(stats.pearsonr(c, g)[0]),
                rmse=float(np.sqrt(np.mean((c - g) ** 2))),
                recall35=float(((g > THR) & (c > THR)).sum() / max((g > THR).sum(), 1)))


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    st = pd.read_csv(C.RAW / "openaq_all_stations_daily.csv", parse_dates=["date"])
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])

    # ---------- 1. CAMS vs each station ----------
    print("=== 1. CAMS vs each ground sensor ===")
    cams = dm[["date", "pm2_5"]].rename(columns={"pm2_5": "cams"})
    per = {}
    pooled = []
    for name, g in st.groupby("station"):
        m = g.merge(cams, on="date").dropna(subset=["pm25", "cams"])
        a = agree(m["cams"], m["pm25"]); per[name] = a
        pooled.append(m.assign(station=name))
        print(f"  {name:<12} n={a['n']:>4}  CAMS {a['mean_cams']:.1f} vs obs "
              f"{a['mean_obs']:.1f}  (real={a['ratio']:.2f}xCAMS)  r={a['pearson']:.2f}  "
              f"recall>35={a['recall35']:.2f}")
    allm = pd.concat(pooled, ignore_index=True)
    a_all = agree(allm["cams"], allm["pm25"])
    print(f"  {'COMBINED':<12} n={a_all['n']:>4}  real={a_all['ratio']:.2f}xCAMS  "
          f"r={a_all['pearson']:.2f}  recall>35={a_all['recall35']:.2f}")
    print(f"  => under-bias consistent across both locations/periods: "
          f"{'YES' if abs(per['US Embassy']['ratio']-per['Sputnik-4']['ratio'])<0.7 else 'differs'}")
    OUT["cams_vs_stations"] = {"per_station": per, "combined": a_all}

    # ---------- 2. generalisation of the embassy-trained model to Sputnik-4 ----------
    print("\n=== 2. Embassy-trained calibration model -> Sputnik-4 (out-of-distribution) ===")
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_ground_truth.txt"))
    cols = booster.feature_name()
    cams_cols = dm[["date", "pm2_5", "pm10", "dust", "nitrogen_dioxide", "ozone",
                    "carbon_monoxide"]].rename(columns=lambda c: "cams_" + c if c != "date" else c)
    base = (feat.drop(columns=[c for c in DROP if c in feat.columns]).merge(cams_cols, on="date"))
    sput = st[st["station"] == "Sputnik-4"][["date", "pm25"]]
    test = base.merge(sput, on="date").dropna(subset=["pm25"])
    have = [c for c in cols if c in test.columns]
    if len(have) == len(cols) and len(test) > 20:
        pred = np.expm1(booster.predict(test[cols]))
        obs = test["pm25"].values
        g = {"n": int(len(test)), "MAE": float(mean_absolute_error(obs, pred)),
             "R2": float(r2_score(obs, pred)), "pearson": float(stats.pearsonr(obs, pred)[0]),
             "model_mean": float(pred.mean()), "obs_mean": float(obs.mean()),
             "recall35": float(((obs > THR) & (pred > THR)).sum() / max((obs > THR).sum(), 1)),
             "rawcams_recall35": float(((obs > THR) & (test["cams_pm2_5"].values > THR)).sum()
                                       / max((obs > THR).sum(), 1))}
        OUT["generalisation_sputnik"] = g
        print(f"  n={g['n']}  model R2={g['R2']:.2f}  r={g['pearson']:.2f}  "
              f"MAE={g['MAE']:.1f}  model_mean={g['model_mean']:.1f} vs obs {g['obs_mean']:.1f}")
        print(f"  exceedance recall>35: model {g['recall35']:.2f} vs raw CAMS "
              f"{g['rawcams_recall35']:.2f}")
        print("  (model trained ONLY on the embassy 2022-2025; this is a different "
              "sensor, location and period)")
    else:
        print("  insufficient overlapping features/days for the generalisation test")

    (MODELS / "multi_sensor_results.json").write_text(json.dumps(OUT, indent=2))

    # ---------- 3. figure ----------
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(dm["date"], dm["pm2_5"].rolling(15, min_periods=5).mean(),
            color="#2980b9", lw=1.1, label="CAMS (15-day avg)")
    colors = {"US Embassy": "#111", "Sputnik-4": "#c0392b"}
    for name, g in st.groupby("station"):
        ax.scatter(g["date"], g["pm25"], s=7, alpha=0.5, color=colors.get(name, "#888"),
                   label=f"{name} sensor")
    ax.axhline(THR, color="#e67e22", ls=":", lw=1)
    ax.set(title="All Tashkent ground sensors vs CAMS — embassy (to 2025-03) + "
                 "Sputnik-4 (2025-06 on)", ylabel="PM2.5 (µg/m³)", xlabel="date")
    ax.legend(ncol=3, fontsize=9); plt.tight_layout()
    fig.savefig(FIG / "multi_sensor.png", dpi=130); plt.close(fig)
    print("\nSaved figures/multi_sensor.png and models/multi_sensor_results.json")


if __name__ == "__main__":
    main()
