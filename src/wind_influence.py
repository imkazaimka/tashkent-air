"""
How much does wind really influence PM2.5? Wind acts through two channels:
  (A) wind SPEED  -> dispersion (windy = pollution blown away / diluted)
  (B) wind DIRECTION -> transport (certain directions carry dirty air in)
We quantify both, raw and confounder-controlled (calm days are also cold/winter).

Computed on CAMS over the full record and cross-checked on the real embassy sensor.
Run:  python src/wind_influence.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / a.std()


def analyse(df, pm_col, label):
    d = df.dropna(subset=[pm_col, "wind_speed_10m", "temperature_2m",
                          "boundary_layer_height", "wind_direction_10m"]).copy()
    pm, ws = d[pm_col], d["wind_speed_10m"]
    print(f"\n================  {label}  (n={len(d)})  ================")

    # ---- A. WIND SPEED (dispersion) ----
    r = stats.pearsonr(ws, pm)[0]
    sl, ic = np.polyfit(ws, pm, 1)
    print(f"[A] Wind SPEED → dispersion")
    print(f"    correlation with PM2.5: r = {r:+.2f}")
    print(f"    raw slope: {sl:+.1f} µg/m³ per +1 m/s of wind")

    # calm vs windy terciles
    d["wt"] = pd.qcut(ws, 3, labels=["calm", "moderate", "windy"])
    g = d.groupby("wt", observed=True)[pm_col].mean()
    drop = (g["calm"] - g["windy"]) / g["calm"] * 100
    print(f"    calm days {g['calm']:.1f}  →  windy days {g['windy']:.1f} µg/m³  "
          f"({drop:.0f}% lower on windy days, raw)")

    # controlled: same temperature, low vs high wind (isolates wind from cold/season)
    d["tbin"] = (d["temperature_2m"] / 3).round() * 3
    diffs, n = [], 0
    for tb, gg in d.groupby("tbin"):
        if len(gg) < 25:
            continue
        med = gg["wind_speed_10m"].median()
        lo = gg[gg["wind_speed_10m"] <= med][pm_col].mean()
        hi = gg[gg["wind_speed_10m"] > med][pm_col].mean()
        diffs.append((lo - hi) / lo * 100 if lo else np.nan); n += len(gg)
    ctrl = np.nanmean(diffs)
    print(f"    at the SAME temperature, calmer-than-median days are {ctrl:.0f}% dirtier "
          f"than windier days (confounder-controlled)")

    # multiple regression: standardized betas (compare wind vs temp vs mixing height)
    X = sm.add_constant(np.column_stack([z(ws), z(d["temperature_2m"]),
                                         z(d["boundary_layer_height"])]))
    m = sm.OLS(pm.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"    standardized effect (controlling temp & mixing height): "
          f"wind β = {m.params[1]:+.1f} µg/m³ per SD  (p={m.pvalues[1]:.1g})")

    # ---- ventilation = wind × mixing height ----
    vent = d["boundary_layer_height"] * ws
    print(f"    ventilation (wind×mixing-height) vs PM2.5: r = "
          f"{stats.pearsonr(vent, pm)[0]:+.2f}")

    # ---- B. WIND DIRECTION (transport) ----
    wd = d["wind_direction_10m"]
    ene = (wd >= 30) & (wd <= 110)          # easterly / E-NE sector
    nw = (wd >= 290) | (wd <= 20)           # north-westerly (clean) sector
    print(f"[B] Wind DIRECTION → transport")
    print(f"    PM2.5 under E/NE flow: {pm[ene].mean():.1f}  vs NW flow: "
          f"{pm[nw].mean():.1f} µg/m³  "
          f"(E/NE is {(pm[ene].mean()/pm[nw].mean()-1)*100:+.0f}% dirtier)")

    # variance share: wind-only vs full simple model
    r2_wind = sm.OLS(pm.values, sm.add_constant(np.column_stack(
        [z(ws), np.sin(np.radians(wd)), np.cos(np.radians(wd))]))).fit().rsquared
    print(f"    wind alone (speed+direction) explains R² = {r2_wind:.2f} of daily PM2.5")
    return {"r_windspeed": float(r), "slope": float(sl),
            "calm_vs_windy_pct": float(drop), "controlled_pct": float(ctrl),
            "ene_vs_nw_pct": float((pm[ene].mean()/pm[nw].mean()-1)*100),
            "r2_wind_only": float(r2_wind)}


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    analyse(dm, "pm2_5", "CAMS model (full record)")

    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    real = dm.merge(gt, on="date")
    analyse(real, "pm25_ground", "Real US-Embassy sensor")


if __name__ == "__main__":
    main()
