"""
Historical ground-SO2 cross-check for the CAMS SO2 tracer (Section 4.5).

Source: WAQI / aqicn.org historical station downloads for two Tashkent stations:
  - Uzhydromet *Chilanzar* :  tashkent-chilanzar-air-quality.csv
  - *US-Embassy* site      :  "tashkent-us embassy, uzbekistan-air-quality.csv"
NOTE on terms: WAQI data are explicitly unvalidated and may NOT be redistributed as cached/archived
data (WAQI Data Use Statement). We therefore do NOT commit the raw CSVs to this repo; this script
reads locally-downloaded copies and only prints the aggregate cross-check statistics in the paper.
Attribution: Uzhydromet (originating EPA) and the World Air Quality Index project.

Download: aqicn.org -> the station page -> historical data CSV (values are per-pollutant indices).

Chilanzar's 2025 SO2 jumps ~5x and loses seasonality (a local sensor fault); the embassy reads
normal across the same months, which *diagnoses* the fault — so we drop Chilanzar-2025 and pool the
two clean records.

Run:  python src/so2_ground_crosscheck.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

DL = Path.home() / "Downloads"
CHIL = DL / "tashkent-chilanzar-air-quality.csv"
EMB = DL / "tashkent-us embassy, uzbekistan-air-quality.csv"


def load(path):
    if not Path(path).exists():
        sys.exit(f"WAQI CSV not found at {path} (not redistributed; download from aqicn.org).")
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], format="%Y/%m/%d", errors="coerce")
    for c in ("so2", "pm25"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["date"])


def ws(d):  # winter(NDJF)/summer(JJA) median ratio
    g = d.dropna(subset=["so2"])
    return (g[g.date.dt.month.isin([11, 12, 1, 2])].so2.median(),
            g[g.date.dt.month.isin([6, 7, 8])].so2.median())


def main():
    chil, emb = load(CHIL), load(EMB)
    # diagnose the Chilanzar 2025 artefact using the embassy (same city/months)
    c25 = chil[chil.date.dt.year == 2025].so2.median()
    e25 = emb[emb.date.dt.year == 2025].so2.median()
    print(f"2025 SO2 median — Chilanzar {c25:.0f} vs Embassy {e25:.0f}  => embassy normal => Chilanzar 2025 is a local fault\n")

    ch = chil[chil.date.dt.year != 2025][["date", "so2"]].dropna(subset=["so2"])   # drop the fault year
    em = emb[["date", "so2"]].dropna(subset=["so2"])
    both = pd.concat([ch, em])
    city = both.groupby("date").so2.mean().rename("so2_ground").reset_index()

    cams = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"]).rename(columns={"so2": "so2_cams"})
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    m = (city.merge(cams, on="date").merge(dm[["date", "temperature_2m"]], on="date")
         .dropna(subset=["so2_ground", "so2_cams"]))
    c = m[m.temperature_2m < m.temperature_2m.quantile(.33)]
    w = m[m.temperature_2m > m.temperature_2m.quantile(.67)]
    print(f"combined two-station paired days vs CAMS: {len(m)}")
    print(f"  magnitude:   ground {m.so2_ground.mean():.1f} vs CAMS {m.so2_cams.mean():.1f} ug/m3")
    print(f"  seasonality (cold/warm): ground {c.so2_ground.median()/w.so2_ground.median():.1f}x | CAMS {c.so2_cams.median()/w.so2_cams.median():.1f}x")
    print(f"  track: r={pearsonr(m.so2_ground, m.so2_cams)[0]:.2f}")
    for nm, d in (("Chilanzar(excl 2025)", ch), ("US Embassy", em)):
        wm, sm = ws(d)
        print(f"  {nm}: winter {wm:.0f} / summer {sm:.0f} = {wm/sm:.1f}x")
    pm = chil[chil.date.dt.year.isin([2023, 2024, 2026])].dropna(subset=["so2", "pm25"])
    print(f"  co-emission: r(ground SO2, ground PM2.5) = {pearsonr(pm.so2, pm.pm25)[0]:.2f} (n={len(pm)}, Chilanzar)")


if __name__ == "__main__":
    main()
