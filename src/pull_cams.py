"""
Pull CAMS (ECMWF Atmosphere) operational AOD *forecasts* over the Central-Asia model domain, so we can
benchmark our ConvLSTM against a real operational physics+assimilation system on the SAME days.

For each target day T and lead L (1..3) we fetch the CAMS forecast initialised at (T-L) 00:00 UTC and
valid at T, taking forecast_hour = 24*L + 6 (≈ 11:00 local in Uzbekistan, near the MODIS overpass, so it
lines up with the daily MAIAC composite we score against). We save both total AOD (directly comparable to
MAIAC total-column AOD) and CAMS's dust-only AOD. Same 20-km grid as training. Resumable.

Run:  EE_PROJECT=civil-sentry-379101 python src/pull_cams.py
"""
from __future__ import annotations
import os, datetime
from pathlib import Path
import numpy as np, ee

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "cams"
PROJECT = os.environ.get("EE_PROJECT", "civil-sentry-379101")
DOM = [55, 37, 75, 47]                                   # Central-Asia model domain (lon/lat)
START, END = datetime.date(2024, 3, 15), datetime.date(2024, 6, 15)   # 2024 spring dust — out-of-sample
LEADS = [1, 2, 3]
BAND_TOTAL = "total_aerosol_optical_depth_at_550nm_surface"
BAND_DUST = "dust_aerosol_optical_depth_at_550nm_surface"


def daterange(s, e):
    d = s
    while d <= e:
        yield d; d += datetime.timedelta(days=1)


def main():
    ee.Initialize(project=PROJECT)
    dom = ee.Geometry.Rectangle(DOM); proj = ee.Projection("EPSG:4326").atScale(20000)
    col = ee.ImageCollection("ECMWF/CAMS/NRT")
    for L in LEADS:
        (OUT / f"lead{L}").mkdir(parents=True, exist_ok=True)
    dates = list(daterange(START, END))
    print(f"CAMS forecast pull  {DOM}  {len(dates)} days x leads {LEADS} -> {OUT}", flush=True)
    for L in LEADS:
        got = 0; fh = 24 * L + 6
        for i, T in enumerate(dates):
            fout = OUT / f"lead{L}" / f"{T.isoformat()}.npy"
            if fout.exists():
                got += 1; continue
            init = (T - datetime.timedelta(days=L)).isoformat()
            for _ in range(3):
                try:
                    img = (col.filter(ee.Filter.eq("model_initialization_datetime", f"{init}T00:00:00"))
                              .filter(ee.Filter.eq("model_forecast_hour", fh)).first())
                    img = img.select([BAND_TOTAL, BAND_DUST]).reproject(proj)
                    tot = np.array(img.select(BAND_TOTAL).sampleRectangle(dom, defaultValue=-999).get(BAND_TOTAL).getInfo(), np.float32)
                    dst = np.array(img.select(BAND_DUST).sampleRectangle(dom, defaultValue=-999).get(BAND_DUST).getInfo(), np.float32)
                    if tot.ndim == 2 and dst.ndim == 2 and tot.shape == dst.shape:
                        np.save(fout, np.stack([tot, dst])); got += 1
                    break
                except Exception:
                    pass
            if (i + 1) % 20 == 0: print(f"  lead{L} {i+1}/{len(dates)} got {got}", flush=True)
        print(f"[lead{L}] done: {got}/{len(dates)}", flush=True)
    print("CAMS PULL DONE", flush=True)


if __name__ == "__main__":
    main()
