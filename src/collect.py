"""
Phase 1 data collection for the Tashkent air-quality forecaster.

Fetches (all from Open-Meteo unless noted):
  1. Tashkent historical air quality   -> data/raw/tashkent_air_quality_hourly.parquet
  2. Tashkent historical weather (ERA5) -> data/raw/tashkent_weather_hourly.parquet
  3. Regional PM2.5 (6 cities)          -> data/raw/regional_pm25_hourly.parquet
  4. WAQI ground-station snapshot       -> data/raw/waqi_snapshot.csv   (validation)

Then aggregates everything to one row per day -> data/processed/daily_merged.csv

Run:  python src/collect.py
"""
from __future__ import annotations
import sys, time, json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from dotenv import load_dotenv
import os
load_dotenv(C.ROOT / ".env")


# ---------------------------------------------------------------- HTTP helper
def get_json(url: str, params: dict, retries: int = 4, pause: float = 2.0) -> dict:
    """GET with exponential backoff; raises on persistent failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            # 429 = rate limited; back off harder
            wait = pause * (2 ** attempt)
            print(f"    HTTP {r.status_code}, retry in {wait:.0f}s ({r.text[:120]})")
            time.sleep(wait)
        except requests.RequestException as e:
            wait = pause * (2 ** attempt)
            print(f"    network error: {e}; retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def hourly_to_df(payload: dict) -> pd.DataFrame:
    """Open-Meteo 'hourly' block -> DataFrame indexed by tz-naive local time."""
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


# ---------------------------------------------------------------- daily agg
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly -> daily. Sum for precip, vector-mean for wind dir, mean for the rest."""
    out = {}
    for col in df.columns:
        if col == "wind_direction_10m":
            continue  # handled below
        if col in C.SUM_VARS:
            out[col] = df[col].resample("1D").sum(min_count=1)
        else:
            out[col] = df[col].resample("1D").mean()

    daily = pd.DataFrame(out)

    # Circular (vector) daily mean for wind direction, weighted by wind speed.
    if "wind_direction_10m" in df.columns and "wind_speed_10m" in df.columns:
        rad = np.radians(df["wind_direction_10m"])
        spd = df["wind_speed_10m"].fillna(0)
        # meteorological "from" convention
        u = -spd * np.sin(rad)
        v = -spd * np.cos(rad)
        u_d = u.resample("1D").mean()
        v_d = v.resample("1D").mean()
        daily["wind_direction_10m"] = (np.degrees(np.arctan2(-u_d, -v_d)) % 360)

    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------- end dates
def archive_end() -> str:
    # ERA5 archive lags ~5 days
    return (date.today() - timedelta(days=5)).isoformat()


def aq_end() -> str:
    # CAMS air-quality updates with ~1 day lag
    return (date.today() - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------- fetchers
def fetch_air_quality() -> pd.DataFrame:
    print("[1/4] Tashkent air quality ...")
    payload = get_json(C.AQ_URL, {
        "latitude": C.TASHKENT["lat"], "longitude": C.TASHKENT["lon"],
        "hourly": ",".join(C.AQ_VARS),
        "start_date": C.START_DATE, "end_date": aq_end(),
        "timezone": C.TIMEZONE,
    })
    hourly = hourly_to_df(payload)
    hourly.to_parquet(C.RAW / "tashkent_air_quality_hourly.parquet")
    daily = aggregate_daily(hourly)
    # drop leading rows before AQ data actually starts
    daily = daily[daily["pm2_5"].notna()]
    print(f"      hourly rows={len(hourly)}, daily rows={len(daily)}, "
          f"range={daily.index.min().date()} -> {daily.index.max().date()}")
    return daily


def fetch_weather() -> pd.DataFrame:
    print("[2/4] Tashkent weather (ERA5) ...")
    payload = get_json(C.ARCHIVE_URL, {
        "latitude": C.TASHKENT["lat"], "longitude": C.TASHKENT["lon"],
        "hourly": ",".join(C.WEATHER_VARS),
        "start_date": C.START_DATE, "end_date": archive_end(),
        "timezone": C.TIMEZONE,
    })
    hourly = hourly_to_df(payload)
    hourly.to_parquet(C.RAW / "tashkent_weather_hourly.parquet")
    daily = aggregate_daily(hourly)
    print(f"      hourly rows={len(hourly)}, daily rows={len(daily)}, "
          f"range={daily.index.min().date()} -> {daily.index.max().date()}")
    return daily


def fetch_regional() -> pd.DataFrame:
    print("[3/4] Regional PM2.5 (6 cities) ...")
    frames, raw_long = [], []
    for city in C.REGIONAL_CITIES:
        print(f"      - {city['name']}")
        payload = get_json(C.AQ_URL, {
            "latitude": city["lat"], "longitude": city["lon"],
            "hourly": "pm2_5",
            "start_date": C.START_DATE, "end_date": aq_end(),
            "timezone": C.TIMEZONE,
        })
        hourly = hourly_to_df(payload)
        long = hourly.reset_index()[["time", "pm2_5"]].copy()
        long["city"] = city["name"]
        raw_long.append(long)

        daily = hourly[["pm2_5"]].resample("1D").mean()
        daily = daily.rename(columns={"pm2_5": f"{city['name'].lower()}_pm25"})
        frames.append(daily)
        time.sleep(0.5)  # be polite to the free API

    pd.concat(raw_long, ignore_index=True).to_parquet(
        C.RAW / "regional_pm25_hourly.parquet")
    out = pd.concat(frames, axis=1)
    out.index.name = "date"
    out = out[out.notna().any(axis=1)]
    print(f"      daily rows={len(out)}, cols={list(out.columns)}")
    return out


def fetch_waqi_snapshot() -> None:
    """Current ground-station reading. WAQI's free feed gives the latest obs +
    short forecast, not deep history — so validation data is accumulated daily."""
    print("[4/4] WAQI ground-station snapshot ...")
    token = os.getenv("WAQI_TOKEN")
    if not token:
        print("      WAQI_TOKEN missing in .env — skipping")
        return
    try:
        payload = get_json(f"{C.WAQI_URL}/{C.WAQI_STATION}/",
                           {"token": token}, retries=2)
    except RuntimeError as e:
        print(f"      WAQI fetch failed: {e}")
        return
    if payload.get("status") != "ok":
        print(f"      WAQI error: {payload.get('data')}")
        return
    d = payload["data"]
    row = {
        "observed_time": d.get("time", {}).get("s"),
        "aqi": d.get("aqi"),
        "pm25": d.get("iaqi", {}).get("pm25", {}).get("v"),
        "pm10": d.get("iaqi", {}).get("pm10", {}).get("v"),
        "station": d.get("city", {}).get("name"),
        "fetched": date.today().isoformat(),
    }
    path = C.RAW / "waqi_snapshot.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=["observed_time"], keep="last")
    df.to_csv(path, index=False)
    print(f"      {row['station']}: AQI={row['aqi']} PM2.5={row['pm25']} "
          f"@ {row['observed_time']}")


# ---------------------------------------------------------------- main
def main() -> None:
    C.RAW.mkdir(parents=True, exist_ok=True)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)

    aq = fetch_air_quality()
    wx = fetch_weather()
    reg = fetch_regional()
    fetch_waqi_snapshot()

    print("\nMerging to one row per day ...")
    merged = aq.join(wx, how="outer").join(reg, how="outer")
    merged = merged.sort_index()
    # keep only days where we have the target (Tashkent pm2_5)
    merged = merged[merged["pm2_5"].notna()]

    out_path = C.PROCESSED / "daily_merged.csv"
    merged.to_csv(out_path)
    print(f"\nSaved {out_path}")
    print(f"  shape: {merged.shape[0]} days x {merged.shape[1]} columns")
    print(f"  range: {merged.index.min().date()} -> {merged.index.max().date()}")
    miss = (merged.isna().mean() * 100).round(1)
    worst = miss[miss > 0].sort_values(ascending=False)
    if len(worst):
        print("  columns with missing values (%):")
        for k, v in worst.items():
            print(f"    {k:28s} {v:5.1f}%")
    else:
        print("  no missing values.")


if __name__ == "__main__":
    main()
