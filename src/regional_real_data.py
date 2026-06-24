"""
Use neighbouring cities' REAL ground sensors (not CAMS) for the Tashkent analysis.

Motivation: Section 5 showed CAMS is unreliable at the neighbouring cities (r as low
as 0.19). So we re-test the transport hypothesis (H1) using the cities' real reference
monitors as the upwind signal, and check whether adding them helps the Tashkent
forecasting model. The TARGET stays Tashkent throughout — we only swap the upwind
inputs from model to measurement.

Cities with usable real sensors: Almaty, Bishkek, Dushanbe, Ashgabat.
(Fergana and Samarkand have no public sensor.)

Run:  python src/regional_real_data.py
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from dotenv import load_dotenv
load_dotenv(C.ROOT / ".env")

BASE = "https://api.openaq.org/v3"
THR = C.PM25_THRESHOLD
# city -> (reference sensor id, bearing from Tashkent, CAMS column)
SENS = {
    "Almaty":   (25903, 45,  "almaty_pm25"),
    "Bishkek":  (23972, 50,  "bishkek_pm25"),
    "Dushanbe": (25215, 180, "dushanbe_pm25"),
    "Ashgabat": (23772, 240, "ashgabat_pm25"),
}


def fetch_daily(sid, hdr):
    rows, page = [], 1
    while True:
        r = requests.get(f"{BASE}/sensors/{sid}/days", headers=hdr,
                         params={"limit": 1000, "page": page}, timeout=60)
        if r.status_code != 200:
            break
        res = r.json().get("results", [])
        if not res:
            break
        for x in res:
            d = x.get("period", {}).get("datetimeFrom", {}).get("utc"); v = x.get("value")
            if d and v is not None:
                rows.append((d[:10], v))
        page += 1; time.sleep(0.25)
    df = pd.DataFrame(rows, columns=["date", "v"]); df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date")


def z(a):
    a = np.asarray(a, float); s = a.std(); return (a - a.mean()) / s if s else a - a.mean()


def interaction(y, upwind, align, vent, temp):
    """HAC OLS; return (beta, p) on the upwind x inbound-alignment interaction."""
    d = pd.DataFrame({"y": y, "u": upwind, "a": align, "ua": upwind * align,
                      "v": vent, "t": temp}).dropna()
    X = sm.add_constant(np.column_stack([z(d.u), z(d.a), z(d.ua), z(d.v), z(d.t)]))
    m = sm.OLS(d.y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(m.params[3]), float(m.pvalues[3]), int(len(d))


def main():
    hdr = {"X-API-Key": os.getenv("OPENAQ_TOKEN")}
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])

    df = feat[["date", "y", "wind_sin", "wind_cos", "ventilation_coef",
               "temperature_2m"]].copy()
    OUT = {"h1_real_vs_cams": {}}

    # fetch real sensors, attach real + CAMS lag1 per city
    print("Fetching real neighbour sensors ...")
    realcols = []
    for city, (sid, bearing, camscol) in SENS.items():
        g = fetch_daily(sid, hdr).rename(columns={"v": f"{city}_real"})
        df = df.merge(g, on="date", how="left")
        df[f"{city}_real_lag1"] = df[f"{city}_real"].shift(1)
        df[f"{city}_cams_lag1"] = dm.set_index("date")[camscol].reindex(df["date"]).shift(1).values
        df[f"{city}_align"] = (df["wind_cos"] * np.cos(np.radians(bearing))
                               + df["wind_sin"] * np.sin(np.radians(bearing)))
        realcols.append(city)

    # ---- 1. H1 transport test: real upwind vs CAMS upwind (target = Tashkent) ----
    print("\n=== H1 transport: real upwind sensor vs CAMS upwind (Tashkent target) ===")
    print(f"  {'city':<10}{'CAMS beta (p)':<22}{'REAL beta (p)':<22}{'n_real'}")
    for city in realcols:
        bc, pc, _ = interaction(df["y"], df[f"{city}_cams_lag1"], df[f"{city}_align"],
                                df["ventilation_coef"], df["temperature_2m"])
        br, pr, nr = interaction(df["y"], df[f"{city}_real_lag1"], df[f"{city}_align"],
                                 df["ventilation_coef"], df["temperature_2m"])
        OUT["h1_real_vs_cams"][city] = {"cams_beta": bc, "cams_p": pc,
                                        "real_beta": br, "real_p": pr, "n_real": nr}
        sig = "***" if pr < 0.001 else "**" if pr < 0.01 else "*" if pr < 0.05 else "ns"
        print(f"  {city:<10}{f'{bc:+.2f} (p={pc:.3g})':<22}"
              f"{f'{br:+.2f} (p={pr:.3g}) {sig}':<22}{nr}")

    # ---- 2. does adding real upwind sensors help the Tashkent forecast? ----
    print("\n=== 2. Add real upwind sensors to the Tashkent ground-truth model ===")
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    DROP = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_roll3_mean",
            "pm25_roll7_mean", "pm25_roll7_std", "pm25_diff1", "episode_streak"]
    cams = dm[["date", "pm2_5", "pm10", "dust", "nitrogen_dioxide", "ozone",
               "carbon_monoxide"]].rename(columns=lambda c: "cams_" + c if c != "date" else c)
    base = feat.drop(columns=[c for c in DROP if c in feat.columns]).merge(cams, on="date")
    real = df[["date"] + [f"{c}_real_lag1" for c in realcols]]
    full = base.merge(real, on="date").merge(gt, on="date").dropna(subset=["pm25_ground", "cams_pm2_5"])
    base_cols = [c for c in base.columns if c not in ("date", "y", "split")]
    extra_cols = [f"{c}_real_lag1" for c in realcols]

    tr = full[full["date"] < "2024-06-01"]; te = full[full["date"] >= "2024-06-01"]
    def fit_eval(cols):
        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
                              random_state=42, verbose=-1)
        # rows where the chosen cols are present
        trX = tr.dropna(subset=cols); teX = te.dropna(subset=cols)
        m.fit(trX[cols], np.log1p(trX["pm25_ground"]))
        p = np.expm1(m.predict(teX[cols])); o = teX["pm25_ground"].values
        rec = ((o > THR) & (p > THR)).sum() / max((o > THR).sum(), 1)
        return r2_score(o, p), float(rec), len(teX)
    r2a, reca, na = fit_eval(base_cols)
    r2b, recb, nb = fit_eval(base_cols + extra_cols)
    print(f"  without real neighbours: R2={r2a:.3f} recall>35={reca:.2f} (n_test={na})")
    print(f"  with    real neighbours: R2={r2b:.3f} recall>35={recb:.2f} (n_test={nb})")
    OUT["model_improvement"] = {"baseline": {"r2": r2a, "recall": reca, "n": na},
                                "with_real_neighbours": {"r2": r2b, "recall": recb, "n": nb}}
    (C.ROOT / "models" / "regional_real_results.json").write_text(json.dumps(OUT, indent=2))
    print("\nSaved models/regional_real_results.json")


if __name__ == "__main__":
    main()
