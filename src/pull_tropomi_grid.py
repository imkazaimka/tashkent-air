"""
Pull gridded daily TROPOMI fields (NO2, UVAI) over the Central Asia domain via Google Earth Engine,
co-registered to the regional AOD grid, as extra channels for the ConvLSTM movement model.

NO2  : anthropogenic pollution field (tracks BAD AIR that AOD can't isolate; ~95% coverage)
UVAI : absorbing-aerosol field (dust/smoke; ~100% coverage, fills AOD's cloud gaps)

Saved as per-day .npy arrays (57x112); -999 = no retrieval. Resumable.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_tropomi_grid.py
"""
from __future__ import annotations
import os, sys, datetime
from pathlib import Path
import numpy as np
import ee

ROOT = Path(__file__).resolve().parent.parent
OUTN = ROOT / "data" / "satellite" / "tropomi_grid" / "no2"; OUTN.mkdir(parents=True, exist_ok=True)
OUTU = ROOT / "data" / "satellite" / "tropomi_grid" / "uvai"; OUTU.mkdir(parents=True, exist_ok=True)
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
PRODS = [("no2", "COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", OUTN),
         ("uvai", "COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index", OUTU)]


def daterange():
    d = datetime.date(2019, 1, 1); end = datetime.date(2025, 12, 31)
    while d <= end:
        yield d.isoformat(); d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle([55, 37, 75, 47])
    proj = ee.Projection('EPSG:4326').atScale(20000)
    dates = list(daterange())
    print(f"pulling gridded TROPOMI NO2+UVAI for {len(dates)} days -> {OUTN.parent}")
    got = {p[0]: 0 for p in PRODS}
    for i, d in enumerate(dates):
        d2 = (datetime.date.fromisoformat(d) + datetime.timedelta(days=1)).isoformat()
        for name, cid, band, out in PRODS:
            fp = out / f"{d}.npy"
            if fp.exists():
                got[name] += 1; continue
            for _ in range(3):
                try:
                    img = ee.ImageCollection(cid).select(band).filterDate(d, d2).filterBounds(dom).mean().reproject(proj)
                    arr = img.sampleRectangle(region=dom, defaultValue=-999).get(band).getInfo()
                    np.save(fp, np.array(arr, np.float32)); got[name] += 1; break
                except Exception:
                    pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dates)} ({d})  no2 {got['no2']} uvai {got['uvai']}", flush=True)
    print(f"done: no2 {got['no2']}, uvai {got['uvai']} of {len(dates)}")


if __name__ == "__main__":
    main()
