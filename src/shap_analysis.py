"""
A3 — SHAP attribution for the GROUND-TRUTH model (what drives REAL PM2.5).

Explains the deployable A1 model (lgbm_ground_truth.txt) with SHAP TreeExplainer.
Outputs a beeswarm summary, grouped importance, and dependence plots for the key
drivers. SHAP values are in log-PM2.5 space (the model's target), so read them as
relative contributions.

Run:  python src/shap_analysis.py   (after train_ground_truth.py)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"; MODELS = C.ROOT / "models"
TEST_FROM = "2024-06-01"
DROP = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_roll3_mean",
        "pm25_roll7_mean", "pm25_roll7_std", "pm25_diff1", "episode_streak"]

GROUPS = {
    "CAMS forecast": ["cams_pm2_5", "cams_pm10", "cams_dust", "cams_nitrogen_dioxide",
                      "cams_ozone", "cams_carbon_monoxide"],
    "dispersion": ["boundary_layer_height", "ventilation_coef", "trapping_index",
                   "inv_wind", "stagnation_proxy", "wind_speed_10m", "surface_pressure"],
    "thermal_season": ["temperature_2m", "shortwave_radiation", "relative_humidity_2m",
                       "doy_sin", "doy_cos", "is_heating_season"],
    "wind_vector": ["wind_sin", "wind_cos"],
    "regional_transport": [f"{c['name'].lower()}_pm25_lag1" for c in C.REGIONAL_CITIES]
        + [f"{c['name'].lower()}_pm25_lag2" for c in C.REGIONAL_CITIES]
        + [f"{c['name'].lower()}_transport" for c in C.REGIONAL_CITIES],
    "precipitation": ["precip", "precip_lag1", "precip_sum_3d"],
}


def assemble():
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    cams = dm[["date", "pm2_5", "pm10", "dust", "nitrogen_dioxide", "ozone",
               "carbon_monoxide"]].rename(columns=lambda c: "cams_" + c if c != "date" else c)
    df = (feat.drop(columns=[c for c in DROP if c in feat.columns])
          .merge(cams, on="date").merge(gt, on="date"))
    return df.dropna(subset=["pm25_ground", "cams_pm2_5"]).sort_values("date").reset_index(drop=True)


def main():
    df = assemble()
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_ground_truth.txt"))
    cols = booster.feature_name()
    X = df[df["date"] >= TEST_FROM][cols].reset_index(drop=True)

    expl = shap.TreeExplainer(booster)
    sv = expl.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.Series(mean_abs, index=cols).sort_values(ascending=False)

    print("Top 12 features by mean|SHAP| (log-PM2.5 units):")
    for k in imp.head(12).index:
        print(f"    {k:<26}{imp[k]:.4f}")

    grp = {g: float(imp[[c for c in cols2 if c in cols]].sum())
           for g, cols2 in GROUPS.items()}
    tot = sum(grp.values())
    print("\nGrouped SHAP importance (% of total):")
    for g, v in sorted(grp.items(), key=lambda kv: -kv[1]):
        print(f"    {g:<20}{v/tot*100:>6.1f}%")

    # beeswarm summary
    plt.figure()
    shap.summary_plot(sv, X, max_display=14, show=False)
    plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=130); plt.close()

    # grouped bar
    fig, ax = plt.subplots(figsize=(7, 4))
    gs = sorted(grp, key=lambda g: grp[g])
    ax.barh(gs, [grp[g]/tot*100 for g in gs], color="#8e44ad")
    ax.set(title="SHAP importance by driver group (ground-truth model)", xlabel="% of total")
    plt.tight_layout(); fig.savefig(FIG / "shap_groups.png", dpi=130); plt.close(fig)

    # dependence for the two biggest non-CAMS drivers
    nonc = [c for c in imp.index if c not in GROUPS["CAMS forecast"]][:2]
    for c in nonc:
        plt.figure()
        shap.dependence_plot(c, sv, X, show=False)
        plt.tight_layout(); plt.savefig(FIG / f"shap_dep_{c}.png", dpi=120); plt.close()

    (MODELS / "shap_importance.json").write_text(json.dumps(
        {"feature_mean_abs_shap": imp.round(5).to_dict(),
         "group_share": {g: grp[g]/tot for g in grp}}, indent=2))
    print(f"\nSaved figures/shap_*.png and models/shap_importance.json")
    print("  top non-CAMS dependence plots:", nonc)


if __name__ == "__main__":
    main()
