"""
Pull an IRAN-domain stack (MAIAC AOD + ERA5 wind u/v + precip) to TEST the Central-Asia-trained dust
ConvLSTM out of domain — a true generalization check on a different environment (Tigris-Euphrates dust
in the west, the central deserts Dasht-e Kavir/Lut, and the Sistan basin in the east). Summer = peak
dust. Same 20-km grid / channel recipe as training. Resumable.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_iran.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "iran"
for s in ("aod", "wind", "precip"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
DOM = [46, 27, 66, 37]                       # lon 46-66, lat 27-37  (20x10 deg, 2:1 like training)


def daterange(s, e):
    d = s
    while d <= e:
        yield d.isoformat(); d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle(DOM); proj = ee.Projection("EPSG:4326").atScale(20000)
    dates = list(daterange(datetime.date(2022, 6, 1), datetime.date(2022, 8, 31)))
    print(f"pulling Iran stack for {len(dates)} days -> {OUT}", flush=True)
    got = 0
    for i, d in enumerate(dates):
        d2 = (datetime.date.fromisoformat(d) + datetime.timedelta(days=1)).isoformat()
        fa, fw, fp = OUT/"aod"/f"{d}.npy", OUT/"wind"/f"{d}.npy", OUT/"precip"/f"{d}.npy"
        if not fa.exists():
            for _ in range(3):
                try:
                    img = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").select("Optical_Depth_055")
                           .filterDate(d, d2).filterBounds(dom).mean().multiply(0.001).toFloat().reproject(proj))
                    arr = np.array(img.sampleRectangle(dom, defaultValue=-999).get("Optical_Depth_055").getInfo(), np.float32)
                    if arr.ndim == 2: np.save(fa, arr)
                    break
                except Exception: pass
        if not fw.exists():
            for _ in range(3):
                try:
                    img = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                           .select(["u_component_of_wind_10m", "v_component_of_wind_10m"]).filterDate(d, d2).first().reproject(proj))
                    u = np.array(img.select("u_component_of_wind_10m").sampleRectangle(dom, defaultValue=0).get("u_component_of_wind_10m").getInfo(), np.float32)
                    v = np.array(img.select("v_component_of_wind_10m").sampleRectangle(dom, defaultValue=0).get("v_component_of_wind_10m").getInfo(), np.float32)
                    if u.ndim == 2 and v.ndim == 2 and u.shape == v.shape: np.save(fw, np.stack([u, v]))
                    break
                except Exception: pass
        if not fp.exists():
            for _ in range(3):
                try:
                    img = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("total_precipitation_sum").filterDate(d, d2).first().reproject(proj)
                    arr = np.array(img.sampleRectangle(dom, defaultValue=0).get("total_precipitation_sum").getInfo(), np.float32)
                    if arr.ndim == 2: np.save(fp, arr)
                    break
                except Exception: pass
        if fa.exists() and fw.exists() and fp.exists(): got += 1
        if (i + 1) % 15 == 0: print(f"  {i+1}/{len(dates)} ({d})  complete days {got}", flush=True)
    print(f"done: {got}/{len(dates)} complete Iran days -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
