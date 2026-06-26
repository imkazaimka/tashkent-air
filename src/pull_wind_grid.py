"""
Pull gridded daily ERA5-Land 10 m wind (u, v) over the Central Asia domain via Earth Engine,
co-registered to the aerosol grid. Used by the PHYSICS advection engine (push today's aerosol field
forward along the wind) to compete head-to-head with the ConvLSTM.

Saved as per-day .npy of shape (2, 57, 112) = [u, v] in m/s. Resumable. Lightweight (no GPU).

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_wind_grid.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "wind_grid"; OUT.mkdir(parents=True, exist_ok=True)
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
BANDS = ["u_component_of_wind_10m", "v_component_of_wind_10m"]


def daterange():
    d = datetime.date(2019, 1, 1); end = datetime.date(2025, 12, 31)
    while d <= end:
        yield d.isoformat(); d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle([55, 37, 75, 47]); proj = ee.Projection('EPSG:4326').atScale(20000)
    dates = list(daterange()); got = 0
    print(f"pulling gridded ERA5 wind for {len(dates)} days -> {OUT}")
    for i, d in enumerate(dates):
        fp = OUT / f"{d}.npy"
        if fp.exists():
            got += 1; continue
        d2 = (datetime.date.fromisoformat(d) + datetime.timedelta(days=1)).isoformat()
        for _ in range(3):
            try:
                img = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(BANDS).filterDate(d, d2).first().reproject(proj)
                u = np.array(img.select(BANDS[0]).sampleRectangle(dom, defaultValue=0).get(BANDS[0]).getInfo(), np.float32)
                v = np.array(img.select(BANDS[1]).sampleRectangle(dom, defaultValue=0).get(BANDS[1]).getInfo(), np.float32)
                if u.ndim == 2 and v.ndim == 2:
                    np.save(fp, np.stack([u, v])); got += 1
                break
            except Exception:
                pass
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(dates)} ({d})  got {got}", flush=True)
    print(f"done: {got}/{len(dates)} wind grids -> {OUT}")


if __name__ == "__main__":
    main()
