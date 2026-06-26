"""
Pull daily REGIONAL aerosol fields over Central Asia from NASA GIBS, for the ConvLSTM movement model.
Domain captures the western dust sources (Aralkum, Kyzylkum, Karakum deserts) and the populated east
(Tashkent, Fergana). These gridded fields are the frames the ConvLSTM learns aerosol motion from.

  AOD  : MODIS Combined MAIAC AOD (PNG, alpha = retrieval mask)  -> the aerosol field
  RGB  : MODIS Terra true-color (JPG)                            -> visual / dust context

Run:  python src/pull_region.py
"""
from __future__ import annotations
import sys, time, io, csv, datetime
from pathlib import Path
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DAOD = ROOT / "data" / "satellite" / "region_aod"; DAOD.mkdir(parents=True, exist_ok=True)
DRGB = ROOT / "data" / "satellite" / "region_rgb"; DRGB.mkdir(parents=True, exist_ok=True)
MAN = ROOT / "data" / "satellite" / "region_manifest.csv"
# Central Asia domain (~1700 x 1100 km)
S, W, N, E = 37.0, 55.0, 47.0, 75.0
WD, HT = 320, 160
TX = int((69.24 - W) / (E - W) * WD); TY = int((N - 41.31) / (N - S) * HT)   # Tashkent pixel
URL = ("https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
       "&TIME={d}T00:00:00Z&BBOX={s},{w},{n},{e}&CRS=EPSG:4326"
       "&LAYERS={lyr}&WRAP=DAY&FORMAT={fmt}&WIDTH={wd}&HEIGHT={ht}")


def daterange():
    d = datetime.date(2019, 1, 1); end = datetime.date(2025, 12, 31)
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
    print(f"REGION pull  bbox {S},{W},{N},{E}  grid {WD}x{HT}  Tashkent px ({TX},{TY})")
    dates = list(daterange()); rows = []; ga = gr = 0
    for i, d in enumerate(dates):
        afp, rfp = DAOD / f"{d}.png", DRGB / f"{d}.jpg"
        if not afp.exists():
            im = pull(d, "MODIS_Combined_MAIAC_L2G_AerosolOpticalDepth", "image/png", "RGBA"); time.sleep(0.18)
            if im is not None: im.save(afp)
        if not rfp.exists():
            im = pull(d, "MODIS_Terra_CorrectedReflectance_TrueColor", "image/jpeg", "RGB"); time.sleep(0.18)
            if im is not None: im.save(rfp, quality=85)
        ga += afp.exists(); gr += rfp.exists()
        rows.append([d, afp.name if afp.exists() else "", rfp.name if rfp.exists() else ""])
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(dates)} ({d})  aod {ga} rgb {gr}", flush=True)
    with open(MAN, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["date", "aod", "rgb"]); wr.writerows(rows)
    print(f"done: aod {ga}, rgb {gr} of {len(dates)} -> {MAN}")


if __name__ == "__main__":
    main()
