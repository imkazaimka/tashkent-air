"""
Pull REAL gridded daily MAIAC AOD (550 nm) over the Central Asia domain via Earth Engine — physical
AOD values, replacing the rendered-tile (255-G)/255 proxy the first multimodal model trained on.
Co-registered to the TROPOMI/wind grids (57x112 @ ~20 km). -999 = no retrieval. Resumable.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_aod_real.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "aod_real_grid"; OUT.mkdir(parents=True, exist_ok=True)
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")


def daterange():
    d = datetime.date(2019, 1, 1); end = datetime.date(2025, 12, 31)
    while d <= end:
        yield d.isoformat(); d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle([55, 37, 75, 47]); proj = ee.Projection('EPSG:4326').atScale(20000)
    dates = list(daterange()); got = 0
    print(f"pulling real MAIAC AOD (550nm) for {len(dates)} days -> {OUT}")
    for i, d in enumerate(dates):
        fp = OUT / f"{d}.npy"
        if fp.exists():
            got += 1; continue
        d2 = (datetime.date.fromisoformat(d) + datetime.timedelta(days=1)).isoformat()
        for _ in range(3):
            try:
                img = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").select("Optical_Depth_055")
                       .filterDate(d, d2).filterBounds(dom).mean().multiply(0.001).toFloat().reproject(proj))
                arr = np.array(img.sampleRectangle(dom, defaultValue=-999).get("Optical_Depth_055").getInfo(), np.float32)
                if arr.ndim == 2:
                    np.save(fp, arr); got += 1
                break
            except Exception:
                pass
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(dates)} ({d})  got {got}", flush=True)
    print(f"done: {got}/{len(dates)} real-AOD grids -> {OUT}")


if __name__ == "__main__":
    main()
