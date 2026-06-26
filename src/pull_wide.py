"""
Pull WIDE-regional daily MODIS true-color + MAIAC AOD over the Tashkent region from NASA GIBS.
The wide field of view (deserts to the west, Fergana to the east, steppe to the north) is what reveals
the SOURCE / direction of incoming aerosol — local build-up vs a regional plume. Resumable.

Run:  python src/pull_wide.py
"""
from __future__ import annotations
import sys, time, io, csv, datetime
from pathlib import Path
import requests, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DRGB = ROOT / "data" / "satellite" / "modis_wide"; DRGB.mkdir(parents=True, exist_ok=True)
DAOD = ROOT / "data" / "satellite" / "aod_wide"; DAOD.mkdir(parents=True, exist_ok=True)
MAN = ROOT / "data" / "satellite" / "wide_manifest.csv"
S, W, N, E = 38.5, 64.0, 43.0, 73.0          # wide regional bbox
WD, HT = 600, 400
# Tashkent location in pixel space (for downstream sector/direction features)
TX = int((69.24 - W) / (E - W) * WD); TY = int((N - 41.31) / (N - S) * HT)
URL = ("https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
       "&TIME={d}T00:00:00Z&BBOX={s},{w},{n},{e}&CRS=EPSG:4326"
       "&LAYERS={lyr}&WRAP=DAY&FORMAT={fmt}&WIDTH={wd}&HEIGHT={ht}")


def daterange():
    for y in range(2018, 2025):
        d = datetime.date(y, 10, 1); end = datetime.date(y + 1, 3, 31)
        while d <= end:
            yield d.isoformat(); d += datetime.timedelta(days=1)


def pull(date, lyr, fmt, conv):
    u = URL.format(d=date, s=S, w=W, n=N, e=E, lyr=lyr, fmt=fmt, wd=WD, ht=HT)
    for _ in range(3):
        try:
            r = requests.get(u, timeout=40)
            if r.status_code == 200 and len(r.content) > 500:
                return Image.open(io.BytesIO(r.content)).convert(conv)
            return None
        except Exception:
            time.sleep(2)
    return None


def main():
    print(f"WIDE pull -> {DRGB} & {DAOD}  | bbox {S},{W},{N},{E}  | Tashkent px ({TX},{TY})")
    dates = list(daterange()); rows = []; gr = ga = 0
    for i, d in enumerate(dates):
        rfp, afp = DRGB / f"{d}.jpg", DAOD / f"{d}.png"
        if not rfp.exists():
            im = pull(d, "MODIS_Terra_CorrectedReflectance_TrueColor", "image/jpeg", "RGB"); time.sleep(0.2)
            if im is not None: im.save(rfp, quality=88)
        if not afp.exists():
            im = pull(d, "MODIS_Combined_MAIAC_L2G_AerosolOpticalDepth", "image/png", "RGBA"); time.sleep(0.2)
            if im is not None: im.save(afp)
        gr += rfp.exists(); ga += afp.exists()
        rows.append([d, rfp.name if rfp.exists() else "", afp.name if afp.exists() else ""])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dates)} ({d})  rgb {gr} aod {ga}", flush=True)
    with open(MAN, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["date", "rgb", "aod"]); wr.writerows(rows)
    print(f"done: rgb {gr}, aod {ga} of {len(dates)} -> manifest {MAN}")


if __name__ == "__main__":
    main()
