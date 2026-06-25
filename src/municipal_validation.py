"""
Citywide validation against Tashkent's 10-station municipal low-cost network (Section 5.3).

Source: Tashkent Open Data Portal, dataset 133 "Air-quality monitoring in Tashkent"
  download API: https://opendata-back.tashkent.uz/ru/api/data/all/133/download  (~117 MB JSON,
  hourly PM2.5 + weather from 10 stations, Mar 2023 - Feb 2026, fields in Russian).

Pipeline: parse -> QC (drop sensor faults >985 ug/m3) -> daily city MEDIAN across stations
(robust to the few faulty units) -> validate against CAMS -> train the calibration on the
citywide series -> figure. Cached daily series in data/raw/ so this runs offline.

Run:  python src/municipal_validation.py
"""
from __future__ import annotations
import sys, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

RAW = C.RAW / "tashkent_municipal_pm25_daily.csv"
URL = "https://opendata-back.tashkent.uz/ru/api/data/all/133/download"
COLS = {"Дата": "date", "Станция": "station", "PM2.5": "pm25"}


def build_daily_from_portal(dst="/tmp/tk133.json"):
    """Download dataset 133 and build the QC'd daily city-median series."""
    subprocess.run(["curl", "-sL", "-m", "120", URL, "-o", dst], check=True)
    s = open(dst, encoding="utf-8", errors="ignore").read()
    s = s[: s.rfind("},") + 1] + "]"                     # drop truncated tail record
    df = pd.DataFrame(json.loads(s)).rename(columns=COLS)
    df["date"] = pd.to_datetime(df["date"]); df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["pm25"]); df = df[(df.pm25 >= 0) & (df.pm25 <= 985)]   # drop sensor faults
    daily = (df.groupby("date").agg(pm25_muni=("pm25", "median"), n_st=("station", "nunique"))
             .reset_index())
    daily = daily[daily.n_st >= 3]                       # >=3 stations for a city median
    daily.to_csv(RAW, index=False)
    return daily


def main():
    daily = pd.read_csv(RAW, parse_dates=["date"]) if RAW.exists() else build_daily_from_portal()
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    m = dm.merge(daily, on="date").dropna(subset=["pm25_muni", "pm2_5"])
    print(f"citywide days: {len(m)}  | CAMS {m.pm2_5.mean():.1f}  municipal {m.pm25_muni.mean():.1f}"
          f"  -> under-read {m.pm25_muni.mean()/m.pm2_5.mean():.2f}x")
    print(f"corr(CAMS, municipal): {pearsonr(m.pm2_5, m.pm25_muni)[0]:.2f}")

    # calibration on the citywide series
    import lightgbm as lgb
    from sklearn.metrics import r2_score
    m = m.copy(); doy = m.date.dt.dayofyear
    m["doy_sin"], m["doy_cos"] = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
    feats = [c for c in ["pm2_5","pm10","nitrogen_dioxide","carbon_monoxide","dust","temperature_2m",
             "wind_speed_10m","boundary_layer_height","relative_humidity_2m","surface_pressure",
             "doy_sin","doy_cos"] if c in m.columns]
    m = m.dropna(subset=feats); cut = m.date.quantile(0.8); tr, te = m[m.date <= cut], m[m.date > cut]
    reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                            min_child_samples=20, random_state=42, verbose=-1).fit(tr[feats], np.log1p(tr.pm25_muni))
    pred = np.expm1(reg.predict(te[feats]))
    rc = lambda p, y, t=35: ((p > t) & (y > t)).sum() / (y > t).sum()
    print(f"R2:  raw CAMS {r2_score(np.log1p(te.pm25_muni), np.log1p(te.pm2_5)):.2f}"
          f"  -> calibrated {r2_score(np.log1p(te.pm25_muni), np.log1p(pred)):.2f}")
    print(f"bad-day(>35) recall: raw CAMS {rc(te.pm2_5.values, te.pm25_muni.values):.2f}"
          f"  -> calibrated {rc(pred, te.pm25_muni.values):.2f}")


if __name__ == "__main__":
    main()
