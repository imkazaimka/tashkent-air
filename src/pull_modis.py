"""
Pull daily MODIS Terra true-color (CorrectedReflectance) frames over Tashkent from NASA GIBS.
No auth needed. Resumable. Saves frames + a manifest with per-frame brightness stats (a cloud proxy).

Winters Oct 1 – Mar 31, 2018-19 .. 2024-25 (the period with US-Embassy PM2.5 labels).

Run:  python src/pull_modis.py
"""
from __future__ import annotations
import sys, time, io, csv
from pathlib import Path
import requests, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "satellite" / "modis"; OUT.mkdir(parents=True, exist_ok=True)
MAN = ROOT / "data" / "satellite" / "modis_manifest.csv"
S, W, N, E = 40.8, 68.6, 41.8, 69.9          # Tashkent bbox (south, west, north, east)
PX = 128
LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
URL = ("https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
       "&TIME={d}T00:00:00Z&BBOX={s},{w},{n},{e}&CRS=EPSG:4326"
       "&LAYERS={lyr}&WRAP=DAY&FORMAT=image/jpeg&WIDTH={px}&HEIGHT={px}")


def daterange():
    import datetime
    for y in range(2018, 2025):
        d = datetime.date(y, 10, 1); end = datetime.date(y + 1, 3, 31)
        while d <= end:
            yield d.isoformat(); d += datetime.timedelta(days=1)


def pull(date):
    u = URL.format(d=date, s=S, w=W, n=N, e=E, lyr=LAYER, px=PX)
    for _ in range(3):
        try:
            r = requests.get(u, timeout=30)
            if r.status_code == 200 and len(r.content) > 800:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            return None
        except Exception:
            time.sleep(2)
    return None


def main():
    rows = []
    dates = list(daterange())
    print(f"pulling {len(dates)} daily MODIS frames -> {OUT}")
    for i, d in enumerate(dates):
        fp = OUT / f"{d}.jpg"
        if fp.exists():
            im = Image.open(fp).convert("RGB")
        else:
            im = pull(d); time.sleep(0.25)
            if im is not None:
                im.save(fp, quality=85)
        if im is None:
            rows.append([d, "", 0, 0, 0]); continue
        a = np.asarray(im).astype(float) / 255.0
        bright = float(a.mean())                      # whiteness ~ cloud proxy
        # "haze/grey" proxy: low saturation + mid brightness
        mx, mn = a.max(2), a.min(2); sat = float((mx - mn).mean())
        rows.append([d, fp.name, round(bright, 4), round(sat, 4), round(float(a.std()), 4)])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dates)}  ({d})", flush=True)
    with open(MAN, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["date", "file", "brightness", "saturation", "contrast"]); wr.writerows(rows)
    got = sum(1 for r in rows if r[1])
    print(f"done: {got}/{len(dates)} frames retrieved; manifest -> {MAN}")


if __name__ == "__main__":
    main()
