"""
Pull daily MODIS MAIAC Aerosol Optical Depth (AOD) frames over Tashkent from NASA GIBS, co-registered
to the true-color frames (same bbox/size). PNG with alpha — the alpha channel is the retrieval mask
(opaque = AOD retrieved; transparent = no retrieval, i.e. cloud/gap), which doubles as a cloud flag.

Run:  python src/pull_aod.py
"""
from __future__ import annotations
import sys, time, io, datetime
from pathlib import Path
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "aod"; OUT.mkdir(parents=True, exist_ok=True)
S, W, N, E = 40.8, 68.6, 41.8, 69.9
PX = 128
LAYER = "MODIS_Combined_MAIAC_L2G_AerosolOpticalDepth"
URL = ("https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
       "&TIME={d}T00:00:00Z&BBOX={s},{w},{n},{e}&CRS=EPSG:4326"
       "&LAYERS={lyr}&WRAP=DAY&FORMAT=image/png&WIDTH={px}&HEIGHT={px}")


def daterange():
    for y in range(2018, 2025):
        d = datetime.date(y, 10, 1); end = datetime.date(y + 1, 3, 31)
        while d <= end:
            yield d.isoformat(); d += datetime.timedelta(days=1)


def pull(date):
    u = URL.format(d=date, s=S, w=W, n=N, e=E, lyr=LAYER, px=PX)
    for _ in range(3):
        try:
            r = requests.get(u, timeout=30)
            if r.status_code == 200 and len(r.content) > 300:
                return Image.open(io.BytesIO(r.content)).convert("RGBA")
            return None
        except Exception:
            time.sleep(2)
    return None


def main():
    dates = list(daterange()); got = 0
    print(f"pulling {len(dates)} daily MAIAC AOD frames -> {OUT}")
    for i, d in enumerate(dates):
        fp = OUT / f"{d}.png"
        if fp.exists():
            got += 1; continue
        im = pull(d); time.sleep(0.25)
        if im is not None:
            im.save(fp); got += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dates)}  ({d})  got {got}", flush=True)
    print(f"done: {got}/{len(dates)} AOD frames -> {OUT}")


if __name__ == "__main__":
    main()
