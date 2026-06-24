"""
SO2 cross-check logger — addresses the central weakness of Section 4.5 (all tracers are CAMS).

Each run appends one row: today's CAMS SO2 (model) and the independent ground SO2 measured by
the Uzhydromet stations (via WAQI), for Tashkent. Run daily (e.g. cron) and, over a season, the
accumulated series lets us finally test whether CAMS SO2 tracks REAL SO2 — the validation the
paper flags as missing. Ground SO2 is reported by WAQI as a US-EPA AQI sub-index; we convert to
µg/m³ with the EPA 1-hour SO2 breakpoints.

Run:  python src/log_so2_crosscheck.py    ->  appends to data/so2_crosscheck.csv
"""
from __future__ import annotations
import os, sys, csv, datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
load_dotenv(C.ROOT / ".env")

OUT = C.ROOT / "data" / "so2_crosscheck.csv"
STATIONS = {"@14722": "Chilanzar", "@14723": "Yunusabad", "@11219": "Embassy"}


def aqi_to_ugm3(a):
    if a is None:
        return None
    ppb = a * 35 / 50 if a <= 50 else 35 + (a - 50) * (75 - 35) / 50   # EPA 1-h SO2 breakpoints
    return round(ppb * 2.62, 1)                                         # ppb -> µg/m³ (~25 °C)


def main():
    W = os.getenv("WAQI_TOKEN")
    ground = {}
    for uid, nm in STATIONS.items():
        try:
            j = requests.get(f"https://api.waqi.info/feed/{uid}/", params={"token": W}, timeout=20).json()
            v = j.get("data", {}).get("iaqi", {}).get("so2", {}).get("v")
            ground[nm] = aqi_to_ugm3(v)
        except Exception:
            ground[nm] = None
    vals = [v for v in ground.values() if v is not None]
    ground_mean = round(sum(vals) / len(vals), 1) if vals else None

    cams = None
    try:
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": C.TASHKENT["lat"], "longitude": C.TASHKENT["lon"],
            "hourly": "sulphur_dioxide", "past_days": 1, "forecast_days": 1, "timezone": "auto"},
            timeout=30).json()
        s = [x for x in r["hourly"]["sulphur_dioxide"] if x is not None][-24:]
        cams = round(sum(s) / len(s), 1) if s else None
    except Exception:
        pass

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    row = {"date": today, "cams_so2": cams, "ground_so2_mean": ground_mean,
           **{f"ground_{n}": ground[n] for n in STATIONS.values()}}
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{today}: CAMS SO2={cams}  ground mean={ground_mean} µg/m³  "
          f"({', '.join(f'{n}={ground[n]}' for n in STATIONS.values())})  -> {OUT.name}")
    print("Run daily; once a few weeks accumulate, correlate cams_so2 vs ground_so2_mean.")


if __name__ == "__main__":
    main()
