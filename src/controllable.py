"""
Can Tashkent's pollution be lowered WITHOUT fighting nature? Split the pollution into:
  UNCONTROLLABLE — dust storms (natural), transboundary transport (other regions),
                   and the winter weather that traps air.
  CONTROLLABLE   — local emissions, chiefly mazut/coal heating combustion.
and estimate how much is which, using the real embassy sensor.

Key idea: weather sets the TIMING (the "trap"); local combustion supplies the MASS.
If the mass is mostly controllable, cleaner fuel lowers pollution even on stagnant days.

Run:  python src/controllable.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

THR = C.PM25_THRESHOLD


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"])
    d = dm.merge(gt, on="date").merge(so2, on="date").dropna(subset=["pm25_ground", "dust", "so2"])
    d["m"] = d["date"].dt.month
    pm = d["pm25_ground"]
    print(f"Real embassy PM2.5, n={len(d)} days. WHO 24h guideline = 15, exceedance line = 35.\n")

    # ---- the floor (well-ventilated days) vs the loaded days ----
    summer = d[d.m.isin([6, 7, 8])]["pm25_ground"].mean()
    winter = d[d.m.isin([11, 12, 1, 2, 3])]["pm25_ground"].mean()
    floor = pm[pm < pm.quantile(.25)].mean()
    print("[A] How high is the pollution, and when?")
    print(f"    overall mean {pm.mean():.0f}   summer {summer:.0f}   winter {winter:.0f} µg/m³")
    print(f"    cleanest-quarter floor {floor:.0f} µg/m³")
    print(f"    winter EXCESS over summer = {winter - summer:.0f} µg/m³ "
          f"({(winter-summer)/winter*100:.0f}% of the winter level)")

    # ---- where do the DANGEROUS days come from? ----
    exc = d[pm > THR]
    print(f"\n[B] The {len(exc)} dangerous days (>{THR:.0f}):")
    print(f"    {(exc['m'].isin([11,12,1,2,3])).mean()*100:.0f}% fall in winter (Nov-Mar)")
    # is it a dust day or a combustion day? high dust vs high so2
    exc_dusty = (exc["dust"] > d["dust"].quantile(.75)).mean() * 100
    exc_comb = (exc["so2"] > d["so2"].quantile(.75)).mean() * 100
    print(f"    {exc_comb:.0f}% have high sulfur (combustion); {exc_dusty:.0f}% have high dust")

    # ---- dust = uncontrollable natural: is it even a problem? ----
    dusty = d[d["dust"] > d["dust"].quantile(.90)]
    print(f"\n[C] UNCONTROLLABLE — dust storms (natural):")
    print(f"    on the top-10% dust days, PM2.5 averages {dusty['pm25_ground'].mean():.0f} "
          f"vs {pm.mean():.0f} overall — dust days are {'CLEANER' if dusty['pm25_ground'].mean()<pm.mean() else 'dirtier'}.")
    print(f"    dust vs PM2.5 correlation r={stats.pearsonr(d['dust'],pm)[0]:+.2f} (dust is a summer phenomenon)")

    # ---- controllable share estimate ----
    # combustion-attributable = winter excess (shown elsewhere to be a combustion jump)
    contr_excess = winter - summer
    # a coarse counterfactual: halving local combustion roughly halves the excess
    cut50 = summer + 0.5 * contr_excess
    print(f"\n[D] CONTROLLABLE — local combustion (mazut/coal heating):")
    print(f"    the winter excess (~{contr_excess:.0f} µg/m³) carries the combustion fingerprint")
    print(f"    => it is the LARGEST and most controllable slice of the winter problem")
    print(f"    illustrative counterfactual: halving local combustion → winter mean ~"
          f"{cut50:.0f} (from {winter:.0f}), i.e. {(1-cut50/winter)*100:.0f}% lower, "
          f"WITHOUT changing the weather")
    # how many fewer exceedance days if winter levels scaled down by that factor
    scale = cut50 / winter
    wexc = d[d.m.isin([11,12,1,2,3])]
    fewer = (wexc["pm25_ground"] > THR).mean()*100 - (wexc["pm25_ground"]*scale > THR).mean()*100
    print(f"    winter exceedance days would fall from "
          f"{(wexc['pm25_ground']>THR).mean()*100:.0f}% to {(wexc['pm25_ground']*scale>THR).mean()*100:.0f}% of days")

    print("\nCAVEAT: a coarse, evidence-based split (CAMS tracers, not formal source "
          "apportionment); the counterfactual assumes pollution scales ~linearly with "
          "local emissions, which is only approximate.")


if __name__ == "__main__":
    main()
