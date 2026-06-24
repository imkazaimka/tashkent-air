"""
Fetch ground-truth PM2.5 from ALL available OpenAQ sensors in Tashkent, for
validating the CAMS reanalysis. Two stations exist (different locations, different
periods), giving spatial coverage and extending ground truth past the embassy's
2025-03 shutdown:

  US Embassy   loc 8881   pm25 sensor 25916     reference-grade  2018-11 -> 2025-03
  Sputnik-4    loc 4902926 pm25 sensor 13465748 low-cost (AirGradient) 2025-06 -> 2026-01

Output:
  data/raw/openaq_embassy_pm25_daily.csv   (date, pm25_ground)   — back-compat
  data/raw/openaq_all_stations_daily.csv   (date, station, pm25, type)

Run:  python src/fetch_openaq.py
"""
from __future__ import annotations
import sys, os, time
from pathlib import Path
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from dotenv import load_dotenv
load_dotenv(C.ROOT / ".env")

BASE = "https://api.openaq.org/v3"
STATIONS = [
    {"name": "US Embassy", "sensor": 25916,    "type": "reference"},
    {"name": "Sputnik-4",  "sensor": 13465748, "type": "low-cost"},
]


def fetch_daily(sensor_id: int, hdr: dict) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        r = requests.get(f"{BASE}/sensors/{sensor_id}/days", headers=hdr,
                         params={"limit": 1000, "page": page}, timeout=60)
        r.raise_for_status()
        res = r.json().get("results", [])
        if not res:
            break
        for x in res:
            day = x.get("period", {}).get("datetimeFrom", {}).get("utc")
            val = x.get("value")
            if day is not None and val is not None:
                rows.append((day[:10], val))
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["date", "pm25"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def main():
    key = os.getenv("OPENAQ_TOKEN")
    if not key:
        print("OPENAQ_TOKEN missing in .env"); return
    hdr = {"X-API-Key": key}

    frames = []
    for st in STATIONS:
        df = fetch_daily(st["sensor"], hdr)
        df["station"] = st["name"]; df["type"] = st["type"]
        frames.append(df)
        print(f"  {st['name']:<12} ({st['type']:<9}): {len(df):>4} days  "
              f"{df['date'].min().date()} -> {df['date'].max().date()}  "
              f"mean {df['pm25'].mean():.1f}")

    allst = pd.concat(frames, ignore_index=True)
    allst.to_csv(C.RAW / "openaq_all_stations_daily.csv", index=False)

    # back-compat: embassy-only file used elsewhere
    emb = allst[allst["station"] == "US Embassy"][["date", "pm25"]].rename(
        columns={"pm25": "pm25_ground"})
    emb.to_csv(C.RAW / "openaq_embassy_pm25_daily.csv", index=False)

    print(f"\nSaved openaq_all_stations_daily.csv ({len(allst)} rows, "
          f"{allst['station'].nunique()} stations) and openaq_embassy_pm25_daily.csv")


if __name__ == "__main__":
    main()
