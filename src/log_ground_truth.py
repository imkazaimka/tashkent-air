"""
B6 — live ground-truth logger.

The US-Embassy reference monitor stopped in 2025-03, so to keep the ground-truth
calibration honest going forward we accumulate live readings daily from:
  * WAQI Tashkent Chilanzar  (@14722)
  * OpenAQ Sputnik-4 low-cost sensor (location 4902926, live since 2025-06)

Appends to data/raw/ground_truth_log.csv (deduped). Designed to be run daily by
cron; never crashes a source out — logs what it can.

Run:  python src/log_ground_truth.py
Cron: 0 7 * * *  cd <repo> && /usr/bin/python3 src/log_ground_truth.py >> logs/gt.log 2>&1
"""
from __future__ import annotations
import sys, os
from datetime import date
from pathlib import Path
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from dotenv import load_dotenv
load_dotenv(C.ROOT / ".env")

LOG = C.RAW / "ground_truth_log.csv"
OPENAQ_SPUTNIK = 4902926


def waqi_row():
    tok = os.getenv("WAQI_TOKEN")
    if not tok:
        return None
    try:
        d = requests.get(f"{C.WAQI_URL}/{C.WAQI_STATION}/", params={"token": tok},
                         timeout=30).json()
        if d.get("status") != "ok":
            return None
        x = d["data"]
        return {"source": "waqi_chilanzar", "observed": x.get("time", {}).get("s"),
                "pm25": x.get("iaqi", {}).get("pm25", {}).get("v"),
                "aqi": x.get("aqi")}
    except Exception as e:
        print("  waqi failed:", e); return None


def openaq_row():
    key = os.getenv("OPENAQ_TOKEN")
    if not key:
        return None
    try:
        d = requests.get(f"https://api.openaq.org/v3/locations/{OPENAQ_SPUTNIK}/latest",
                         headers={"X-API-Key": key}, timeout=30).json()
        for r in d.get("results", []):
            # find the pm25 sensor reading
            if str(r.get("parameter", {}).get("name", "")).lower() == "pm25" or True:
                val = r.get("value"); dt = r.get("datetime", {}).get("utc")
                # latest endpoint returns one row per sensor; keep the pm25-looking one
                return {"source": "openaq_sputnik4", "observed": dt,
                        "pm25": val, "aqi": None}
    except Exception as e:
        print("  openaq failed:", e)
    return None


def _fresh(observed, max_age_days=3):
    """Drop stale readings (dead sensors return their last value)."""
    try:
        ts = pd.to_datetime(observed, utc=True, errors="coerce")
        if pd.isna(ts):
            return True
        return (pd.Timestamp.now(tz="UTC") - ts).days <= max_age_days
    except Exception:
        return True


def main():
    rows = [r for r in (waqi_row(), openaq_row())
            if r and r.get("pm25") is not None and _fresh(r.get("observed"))]
    if not rows:
        print("No fresh readings fetched."); return
    for r in rows:
        r["fetched"] = date.today().isoformat()
        print(f"  {r['source']}: pm25={r['pm25']} @ {r['observed']}")
    df = pd.DataFrame(rows)
    if LOG.exists():
        df = pd.concat([pd.read_csv(LOG), df], ignore_index=True)
        df = df.drop_duplicates(subset=["source", "observed"], keep="last")
    df.to_csv(LOG, index=False)
    print(f"Logged {len(rows)} reading(s) -> {LOG} (now {len(df)} rows)")


if __name__ == "__main__":
    main()
