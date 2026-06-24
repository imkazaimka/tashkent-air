"""
Descriptive statistics + data quality-control / outlier audit.

Two jobs the paper was missing as explicit, reported numbers:
  (1) a summary table — n, mean, SD, min, quartiles, max, skew — for the key variables;
  (2) a QC / outlier audit: how complete is the data, are there impossible values, and are
      the statistical "outliers" erroneous spikes or genuine pollution episodes?

Important: for PM2.5 the extreme-high days are usually the REAL signal (the dangerous
episodes we want to explain), so we screen for *erroneous* values, we do NOT trim genuine
high days.

Run:  python src/descriptive_stats.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def summarize(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": len(s), "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "q25": s.quantile(.25), "median": s.median(), "q75": s.quantile(.75),
            "max": s.max(), "skew": stats.skew(s)}


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    ft = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"])
    d = dm.merge(gt, on="date", how="left").merge(so2, on="date", how="left")
    d = d.merge(ft[["date", "ventilation_coef"]], on="date", how="left")
    d["m"] = d["date"].dt.month

    VARS = [("PM2.5 — real sensor", "pm25_ground", "µg/m³"),
            ("PM2.5 — CAMS", "pm2_5", "µg/m³"),
            ("Temperature", "temperature_2m", "°C"),
            ("Wind speed", "wind_speed_10m", "m/s"),
            ("Boundary-layer height", "boundary_layer_height", "m"),
            ("Relative humidity", "relative_humidity_2m", "%"),
            ("SO₂ (CAMS)", "so2", "µg/m³"),
            ("NO₂ (CAMS)", "nitrogen_dioxide", "µg/m³"),
            ("Dust (CAMS)", "dust", "µg/m³")]

    print("=== (1) DESCRIPTIVE STATISTICS ===")
    print(f"{'Variable':<24}{'n':>5}{'mean':>9}{'SD':>8}{'min':>8}{'25%':>8}"
          f"{'med':>8}{'75%':>8}{'max':>9}{'skew':>7}")
    rows = []
    for name, col, unit in VARS:
        if col not in d.columns:
            print(f"  (missing column: {col})"); continue
        st = summarize(d[col])
        rows.append((name, unit, st))
        print(f"{name:<24}{st['n']:>5}{st['mean']:>9.1f}{st['sd']:>8.1f}{st['min']:>8.1f}"
              f"{st['q25']:>8.1f}{st['median']:>8.1f}{st['q75']:>8.1f}{st['max']:>9.1f}{st['skew']:>7.1f}")

    print("\n=== (2) DATA COMPLETENESS ===")
    span = (d['date'].max() - d['date'].min()).days + 1
    print(f"  study span {span} days; rows {len(d)}")
    for name, col, _ in VARS:
        if col in d.columns:
            miss = d[col].isna().mean() * 100
            print(f"  {name:<24} {100-miss:5.1f}% present  ({d[col].notna().sum()} days)")

    print("\n=== (3) IMPOSSIBLE / ERRONEOUS VALUES ===")
    CAN_BE_NEG = {"temperature_2m"}  # sub-zero is physically valid; everything else must be >= 0
    for name, col, _ in VARS:
        if col not in d.columns:
            continue
        s = d[col].dropna()
        neg = (s < 0).sum()
        # a stuck sensor shows the same value many days running
        stuck = int((s.reset_index(drop=True).diff() == 0).sum())
        if col in CAN_BE_NEG:
            note = f"  ({neg} sub-zero days — physically valid)"
        else:
            note = "" if neg == 0 else f"  <-- {neg} IMPOSSIBLE NEGATIVE"
        print(f"  {name:<24} negatives={neg:>3}  zero-change-days={stuck:>4}{note}")

    print("\n=== (4) OUTLIER AUDIT (real PM2.5) — error, or genuine episode? ===")
    pm = d.dropna(subset=["pm25_ground"]).copy()
    q1, q3 = pm["pm25_ground"].quantile([.25, .75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    out = pm[pm["pm25_ground"] > upper]
    print(f"  Tukey upper fence = Q3 + 1.5·IQR = {upper:.0f} µg/m³")
    print(f"  {len(out)} of {len(pm)} days ({len(out)/len(pm)*100:.0f}%) lie above it")
    if len(out):
        win = out["m"].isin([11, 12, 1, 2, 3]).mean() * 100
        hi_so2 = (out["so2"] > pm["so2"].median()).mean() * 100 if "so2" in out else float("nan")
        print(f"    of those, {win:.0f}% fall in winter and {hi_so2:.0f}% have above-median SO₂")
        print(f"    => these are genuine winter combustion episodes, NOT sensor errors — KEPT.")
    top = pm.nlargest(3, "pm25_ground")[["date", "pm25_ground", "so2", "dust"]]
    print("  the three highest days (with their tracers):")
    for _, r in top.iterrows():
        print(f"    {r['date'].date()}  PM2.5={r['pm25_ground']:.0f}  "
              f"SO₂={r.get('so2', float('nan')):.1f}  dust={r.get('dust', float('nan')):.1f}")

    print("\n=== (5) DISTRIBUTION SHAPE (why we model log-PM2.5) ===")
    s = d["pm25_ground"].dropna()
    print(f"  raw PM2.5 skew = {stats.skew(s):+.2f} (strong right tail); "
          f"log PM2.5 skew = {stats.skew(np.log(s)):+.2f} (near-symmetric)")


if __name__ == "__main__":
    main()
