"""
Pull MAIAC AOD + ERA5 wind/precip for several DUST regions, to test the Central-Asia-trained ConvLSTM
out of domain (the generalisation result for Paper 2). Same 20-km grid / channel recipe as training.
Each region is pulled over its own dust season. Resumable.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_dustregions.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
REGIONS = {                                                   # dom=[lon0,lat0,lon1,lat1], peak dust season
    "sahara":     {"dom": [0, 12, 30, 27],   "s": (2022, 6, 15), "e": (2022, 7, 31)},   # Sahara/Sahel (Bodélé), summer
    "middleeast": {"dom": [38, 22, 58, 35],  "s": (2022, 6, 15), "e": (2022, 7, 31)},   # Arabian/Iraqi Shamal, summer
    "gobi":       {"dom": [98, 40, 118, 48], "s": (2022, 3, 20), "e": (2022, 5, 5)},    # Gobi / East Asia, spring
}


def daterange(s, e):
    d = datetime.date(*s); end = datetime.date(*e)
    while d <= end:
        yield d.isoformat(); d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    proj = ee.Projection("EPSG:4326").atScale(20000)
    for name, cfg in REGIONS.items():
        dom = ee.Geometry.Rectangle(cfg["dom"])
        out = ROOT / "data" / "satellite" / "regions" / name
        for s in ("aod", "wind", "precip"): (out / s).mkdir(parents=True, exist_ok=True)
        dates = list(daterange(cfg["s"], cfg["e"])); got = 0
        print(f"[{name}] {cfg['dom']}  {len(dates)} days -> {out}", flush=True)
        for i, d in enumerate(dates):
            d2 = (datetime.date.fromisoformat(d) + datetime.timedelta(days=1)).isoformat()
            fa, fw, fp = out/"aod"/f"{d}.npy", out/"wind"/f"{d}.npy", out/"precip"/f"{d}.npy"
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
            if (i+1) % 15 == 0: print(f"  [{name}] {i+1}/{len(dates)} got {got}", flush=True)
        print(f"[{name}] done: {got}/{len(dates)}", flush=True)
    print("ALL REGIONS DONE", flush=True)


if __name__ == "__main__":
    main()
