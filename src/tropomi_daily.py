"""
Daily TROPOMI (Sentinel-5P) source tracers over Tashkent via Google Earth Engine.

Pulls daily regional-mean UV Aerosol Index (dust/smoke, works over cloud), tropospheric NO2
(anthropogenic winter pollution), and CO (combustion, background-dominated) for 2018-2025.
~100% coverage on smog days (vs ~10% for optical AOD). Server-side per-year extraction.

SO2 is omitted: too sparse/noisy at the daily scale for a moderate urban source (monthly only).

Run:  EE_PROJECT=civil-sentry-379101 python src/tropomi_daily.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "tropomi_daily_tracers.csv"
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
REG = ee.Geometry.Rectangle([68.8, 40.9, 69.7, 41.7]) if False else None  # set after init
PRODS = {"uvai": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index"),
         "no2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
         "co": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density")}


def daily_year(cid, band, year, reg):
    """Server-side: one regional-mean value per day in the year -> getInfo once."""
    start = ee.Date.fromYMD(year, 1, 1)
    ndays = ee.Number(ee.Date.fromYMD(year, 12, 31).difference(start, "day")).add(1)

    def perday(d):
        s = start.advance(ee.Number(d), "day"); e = s.advance(1, "day")
        img = ee.ImageCollection(cid).select(band).filterDate(s, e).filterBounds(reg).mean()
        v = img.reduceRegion(ee.Reducer.mean(), reg, 7000).get(band)
        return ee.Feature(None, {"date": s.format("YYYY-MM-dd"), "v": v})

    fc = ee.FeatureCollection(ee.List.sequence(0, ndays.subtract(1)).map(perday))
    return {f["properties"]["date"]: f["properties"].get("v") for f in fc.getInfo()["features"]}


def main():
    ee.Initialize(project=PROJECT)
    reg = ee.Geometry.Rectangle([68.8, 40.9, 69.7, 41.7])
    out = None
    for name, (cid, band) in PRODS.items():
        ser = {}
        for y in range(2018, 2026):
            try:
                ser.update(daily_year(cid, band, y, reg)); print(f"  {name} {y} ok", flush=True)
            except Exception as e:
                print(f"  {name} {y} ERR {str(e)[:80]}", flush=True)
        s = pd.Series(ser, name=name)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    out.index.name = "date"; out = out.sort_index()
    out.to_csv(OUT)
    cov = out.notna().mean()
    print(f"saved {OUT}  | days {len(out)} | coverage uvai {cov['uvai']:.0%} no2 {cov['no2']:.0%} co {cov['co']:.0%}")


if __name__ == "__main__":
    main()
