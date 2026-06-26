"""
Pull gridded daily ERA5-Land precipitation over the Central Asia domain via Earth Engine,
co-registered to the aerosol grid. Wet removal: rain washes aerosol out, so precip is a real driver
of how fast the air clears. Saved as per-day .npy (57x112), value in metres. Gap-free. Resumable.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_precip_grid.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "precip_grid"; OUT.mkdir(parents=True, exist_ok=True)
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
BAND = "total_precipitation_sum"


def daterange():
    d = datetime.date(2019, 1, 1); end = datetime.date(2025, 12, 31)
    while d <= end:
        yield d.isoformat(); d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle([55, 37, 75, 47]); proj = ee.Projection('EPSG:4326').atScale(20000)
    dates = list(daterange()); got = 0
    print(f"pulling ERA5 precip for {len(dates)} days -> {OUT}")
    for i, d in enumerate(dates):
        fp = OUT / f"{d}.npy"
        if fp.exists():
            got += 1; continue
        d2 = (datetime.date.fromisoformat(d) + datetime.timedelta(days=1)).isoformat()
        for _ in range(3):
            try:
                img = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(BAND).filterDate(d, d2).first().reproject(proj)
                arr = np.array(img.sampleRectangle(dom, defaultValue=0).get(BAND).getInfo(), np.float32)
                if arr.ndim == 2:
                    np.save(fp, arr); got += 1
                break
            except Exception:
                pass
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(dates)} ({d})  got {got}", flush=True)
    print(f"done: {got}/{len(dates)} precip grids -> {OUT}")


if __name__ == "__main__":
    main()
