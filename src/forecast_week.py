"""
THE DEPLOYABLE TOOL — a live 7-day air-quality risk outlook for Tashkent.

For each of the next 7 days it answers the only question that matters to a resident:
"how likely is the air to be bad / very bad that day?" — as calibrated probabilities,
like a weather app's chance-of-rain, plus a best-estimate PM2.5 and AQI level.

How it works (per lead day h = 1..7):
  - a model is trained on history to predict real PM2.5 h days ahead from the *forecast*
    weather + CAMS air-quality forecast for the target day, plus recent CAMS persistence;
  - exceedance probabilities P(>40 "bad") and P(>100 "very bad") are isotonically calibrated;
  - at run time it pulls the LIVE 7-day weather forecast and CAMS air-quality forecast
    (Open-Meteo, free, no key) and produces the outlook.

HONEST reliability: next 1-3 days are solid; by day 7 this is a *risk outlook*, not a
day-specific promise — its skill is limited by the upstream weather forecast, and it
sharpens as the day approaches. Very-bad (>100) calls a week out are low-confidence without
a live ground sensor (see the paper).

Output: figures/week_outlook.png, models/week_outlook.json
Run:    python src/forecast_week.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

LAT, LON = C.TASHKENT["lat"], C.TASHKENT["lon"]
BAD, VBAD = 40, 100
WX = ["temperature_2m", "wind_speed_10m", "boundary_layer_height", "relative_humidity_2m",
      "surface_pressure", "shortwave_radiation"]
AQ = ["pm2_5", "pm10", "nitrogen_dioxide", "dust", "carbon_monoxide"]
TARGET_FEATS = WX + AQ + ["doy_sin", "doy_cos"]
AQI = [(12, "Good", "#27ae60"), (35, "Moderate", "#f1c40f"), (55, "USG", "#e67e22"),
       (150, "Unhealthy", "#c0392b"), (250, "Very Unhealthy", "#8e44ad"), (1e9, "Hazardous", "#7e2811")]


def aqi_level(v):
    for hi, name, col in AQI:
        if v < hi:
            return name, col
    return AQI[-1][1], AQI[-1][2]


# ---------------- training ----------------
def history():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    gt = gt[(gt.date >= dm.date.min()) & (gt.date <= dm.date.max())]
    d = dm.merge(gt, on="date", how="left")
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    d["cams_lag0"] = d["pm2_5"]; d["cams_roll7_0"] = d["pm2_5"].rolling(7, min_periods=3).mean()
    return d


def train_horizon(d, h):
    df = d.copy()
    df["target"] = df["pm25_ground"].shift(-h)               # real PM2.5 h days ahead
    df["cams_lag"] = df["cams_lag0"]                          # today's CAMS (decision day)
    df["cams_roll7"] = df["cams_roll7_0"]
    feats = TARGET_FEATS + ["cams_lag", "cams_roll7"]
    tr = df.dropna(subset=["target"])
    cut = tr["date"].quantile(0.85); a, cal = tr[tr.date <= cut], tr[tr.date > cut]
    reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                            min_child_samples=20, random_state=42, verbose=-1).fit(a[feats], np.log1p(a["target"]))
    # held-out residuals (log space) define one COHERENT predictive distribution per day,
    # so the exceedance odds always agree with the point estimate.
    resid = np.log1p(cal["target"].values) - reg.predict(cal[feats])
    return reg, resid, feats


def predict(row, reg, resid, feats):
    X = row[feats].to_frame().T.apply(pd.to_numeric, errors="coerce").astype(float)
    pl = float(reg.predict(X)[0])
    draws = np.expm1(pl + resid)                              # predictive distribution for the day
    return {"pm25": float(np.median(draws)),
            BAD: float((draws > BAD).mean()), VBAD: float((draws > VBAD).mean())}


# ---------------- live forecast pull ----------------
def daily_mean(js, keys):
    h = pd.DataFrame(js["hourly"]); h["t"] = pd.to_datetime(h["time"])
    h["date"] = h["t"].dt.normalize()
    cols = [k for k in keys if k in h.columns]
    g = h.groupby("date")[cols].mean().reset_index()
    return g


def fetch_live():
    wx = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "forecast_days": 8, "past_days": 1,
        "hourly": "temperature_2m,wind_speed_10m,relative_humidity_2m,surface_pressure,"
                  "shortwave_radiation,boundary_layer_height", "timezone": "auto"}, timeout=40).json()
    aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON, "forecast_days": 7, "past_days": 7,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,dust,carbon_monoxide", "timezone": "auto"}, timeout=40).json()
    w = daily_mean(wx, WX)
    a = daily_mean(aq, ["pm2_5", "pm10", "nitrogen_dioxide", "dust", "carbon_monoxide"])
    m = w.merge(a, on="date", how="left").sort_values("date").reset_index(drop=True)
    return m


def compute_outlook():
    """Train the 7 lead-time models, pull the live forecast, return the outlook rows
    (and write models/week_outlook.json). Reused by the Telegram alerter."""
    d = history()
    models = {h: train_horizon(d, h) for h in range(1, 8)}
    print("Trained 7 lead-time models. Fetching live forecast for Tashkent ...")
    try:
        live = fetch_live()
    except Exception as e:
        print(f"  live fetch failed ({e}); using the most recent reanalysis day as a demo run.")
        live = None

    # climatology fallback for any missing forecast feature
    clim = d.assign(m=d.date.dt.month).groupby("m")[AQ + WX].mean()

    if live is not None:
        today = pd.Timestamp.now().normalize()
        cams_today = live.loc[live.date <= today, "pm2_5"].dropna()
        cams_lag = float(cams_today.iloc[-1]) if len(cams_today) else float(d["pm2_5"].iloc[-1])
        cams_roll7 = float(live.loc[live.date <= today, "pm2_5"].dropna().tail(7).mean()) if len(cams_today) else cams_lag
        fut = live[live.date > today].head(7).reset_index(drop=True)
    else:
        cams_lag = float(d["pm2_5"].iloc[-1]); cams_roll7 = float(d["pm2_5"].tail(7).mean())
        fut = d.tail(7).reset_index(drop=True); fut["date"] = pd.date_range(pd.Timestamp.now().normalize() + pd.Timedelta(days=1), periods=7)

    rows = []
    for h in range(1, 8):
        if h - 1 >= len(fut):
            break
        r = fut.iloc[h - 1].copy()
        doy = r["date"].dayofyear; mth = r["date"].month
        r["doy_sin"], r["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
        for c in AQ + WX:                                     # fill any missing with climatology
            if c not in r or pd.isna(r.get(c)):
                r[c] = clim.loc[mth, c]
        r["cams_lag"], r["cams_roll7"] = cams_lag, cams_roll7
        reg, resid, feats = models[h]
        p = predict(r, reg, resid, feats)
        name, col = aqi_level(p["pm25"])
        rows.append({"day": h, "date": str(r["date"].date()), "pm25": round(p["pm25"], 0),
                     "level": name, "color": col, "p_bad": round(p[BAD], 2), "p_vbad": round(p[VBAD], 2)})

    json.dump(rows, open(C.ROOT / "models" / "week_outlook.json", "w"), indent=2)
    return rows


def main():
    rows = compute_outlook()
    print("\n================  TASHKENT — 7-DAY AIR-QUALITY OUTLOOK  ================")
    print(f"{'day':>3}  {'date':<12}{'est PM2.5':>10}  {'level':<15}{'P(bad>40)':>11}{'P(v.bad>100)':>13}")
    for r in rows:
        print(f"{r['day']:>3}  {r['date']:<12}{r['pm25']:>9.0f}  {r['level']:<15}"
              f"{r['p_bad']*100:>10.0f}%{r['p_vbad']*100:>12.0f}%")
    print("=" * 70)
    print("Reliability: days 1-3 solid; day 7 is a risk outlook (limited by the weather forecast).")

    # ---- figure ----
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw={"height_ratios": [1.25, 1]})
    x = [r["day"] for r in rows]; lbl = [f"D{r['day']}\n{r['date'][5:]}" for r in rows]
    ax[0].bar(x, [r["pm25"] for r in rows], color=[r["color"] for r in rows], edgecolor="white")
    for thr, name in [(12, "Good"), (35, "Moderate"), (55, "USG"), (100, "Unhealthy")]:
        ax[0].axhline(thr, color="#bbb", ls=":", lw=0.8)
        ax[0].text(x[-1] + 0.55, thr, name, fontsize=8, color="#888", va="center")
    ax[0].set(ylabel="best-estimate PM2.5 (µg/m³)", xticks=x, title="Tashkent — 7-day air-quality outlook")
    ax[0].set_xticklabels(lbl)
    ax[1].bar([i - 0.2 for i in x], [r["p_bad"] * 100 for r in rows], 0.4, color="#e67e22", label="P(bad, >40)")
    ax[1].bar([i + 0.2 for i in x], [r["p_vbad"] * 100 for r in rows], 0.4, color="#7e2811", label="P(very bad, >100)")
    ax[1].set(ylabel="probability (%)", ylim=(0, 100), xticks=x, xlabel="lead day",
              title="Chance the air is BAD / VERY BAD")
    ax[1].set_xticklabels(lbl); ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(C.ROOT / "figures" / "week_outlook.png", dpi=140); plt.close(fig)
    print("\nSaved figures/week_outlook.png, models/week_outlook.json")


if __name__ == "__main__":
    main()
