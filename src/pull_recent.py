"""
Pull the most RECENT available AOD + wind + precip over the Central Asia (Tashkent) domain, to check for
live dust storms and test the model's direction/speed read on present data. Same 20-km grid as training.
Resumable. Satellite/reanalysis have a few days' latency, so "now" really means the last available days.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_recent.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "recent"
for s in ("aod", "wind", "precip"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
DOM = [55, 37, 75, 47]                         # Central Asia / Tashkent domain (same as training)
END = datetime.date.today(); START = END - datetime.timedelta(days=24)   # live: always up to today (resumable, fetches only new days)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle(DOM); proj = ee.Projection("EPSG:4326").atScale(20000)
    d = START; got = 0; dates = []
    while d <= END:
        dates.append(d.isoformat()); d += datetime.timedelta(days=1)
    print(f"pulling recent stack {START}..{END} ({len(dates)} days) -> {OUT}", flush=True)
    for ds in dates:
        d2 = (datetime.date.fromisoformat(ds) + datetime.timedelta(days=1)).isoformat()
        fa, fw, fp = OUT/"aod"/f"{ds}.npy", OUT/"wind"/f"{ds}.npy", OUT/"precip"/f"{ds}.npy"
        if not fa.exists():
            for _ in range(3):
                try:
                    img = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").select("Optical_Depth_055")
                           .filterDate(ds, d2).filterBounds(dom).mean().multiply(0.001).toFloat().reproject(proj))
                    arr = np.array(img.sampleRectangle(dom, defaultValue=-999).get("Optical_Depth_055").getInfo(), np.float32)
                    if arr.ndim == 2: np.save(fa, arr)
                    break
                except Exception: pass
        if not fw.exists():
            for _ in range(3):
                try:
                    img = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                           .select(["u_component_of_wind_10m", "v_component_of_wind_10m"]).filterDate(ds, d2).first().reproject(proj))
                    u = np.array(img.select("u_component_of_wind_10m").sampleRectangle(dom, defaultValue=0).get("u_component_of_wind_10m").getInfo(), np.float32)
                    v = np.array(img.select("v_component_of_wind_10m").sampleRectangle(dom, defaultValue=0).get("v_component_of_wind_10m").getInfo(), np.float32)
                    if u.ndim == 2 and v.ndim == 2 and u.shape == v.shape: np.save(fw, np.stack([u, v]))
                    break
                except Exception: pass
        if not fp.exists():
            for _ in range(3):
                try:
                    img = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("total_precipitation_sum").filterDate(ds, d2).first().reproject(proj)
                    arr = np.array(img.sampleRectangle(dom, defaultValue=0).get("total_precipitation_sum").getInfo(), np.float32)
                    if arr.ndim == 2: np.save(fp, arr)
                    break
                except Exception: pass
        tags = ("A" if fa.exists() else "-") + ("W" if fw.exists() else "-") + ("P" if fp.exists() else "-")
        if fa.exists(): got += 1
        print(f"  {ds} [{tags}]", flush=True)
    print(f"done: {got}/{len(dates)} days with AOD -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
