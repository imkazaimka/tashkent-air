"""
H4 — is sulfur-rich heating fuel (mazut/coal) the MAIN source of Tashkent's PM2.5?

Local rumour says mazut (heavy fuel oil) is the main problem. We have no fuel-use
data, so we test it with chemical tracers: mazut/coal burning leaves a SULFUR (SO2)
+ COMBUSTION (CO) fingerprint; dust storms leave a mineral (dust) fingerprint;
traffic leaves an NO2 / weekday fingerprint. We ask which fingerprint best explains
the REAL embassy PM2.5 (using ground truth as target avoids CAMS's internal
chemistry circularity).

Honest caveats (printed): CAMS SO2/CO are MODEL estimates, may under-represent
sporadic mazut burning; SO2 cannot separate mazut from coal or other sulfur sources.

Run:  python src/mazut_hypothesis.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from dotenv import load_dotenv
load_dotenv(C.ROOT / ".env")

SO2_CACHE = C.RAW / "cams_so2_daily.csv"


def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / a.std()


def get_so2():
    if SO2_CACHE.exists():
        return pd.read_csv(SO2_CACHE, parse_dates=["date"])
    r = requests.get(C.AQ_URL, params={"latitude": C.TASHKENT["lat"], "longitude": C.TASHKENT["lon"],
        "hourly": "sulphur_dioxide", "start_date": "2022-08-01", "end_date": "2026-06-22",
        "timezone": C.TIMEZONE}, timeout=60).json()
    h = pd.DataFrame(r["hourly"]); h["time"] = pd.to_datetime(h["time"])
    so2 = h.set_index("time")["sulphur_dioxide"].resample("1D").mean().rename("so2").reset_index()
    so2 = so2.rename(columns={"time": "date"}); so2["date"] = so2["date"].dt.normalize()
    so2.to_csv(SO2_CACHE, index=False); return so2


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    ft = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])[["date", "ventilation_coef"]]
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    so2 = get_so2()
    d = (dm.merge(so2, on="date").merge(ft, on="date").merge(gt, on="date")
         .dropna(subset=["pm25_ground", "so2", "carbon_monoxide", "nitrogen_dioxide", "dust"]))
    d["month"] = d["date"].dt.month
    d["weekend"] = d["date"].dt.dayofweek >= 5
    TRACERS = {"SO2 (mazut/coal sulfur)": "so2", "CO (combustion)": "carbon_monoxide",
               "NO2 (traffic/combustion)": "nitrogen_dioxide", "dust (storms)": "dust"}
    y = d["pm25_ground"]
    print(f"Target = REAL embassy PM2.5, n={len(d)} days\n")

    # 1. which tracer best correlates with REAL pollution
    print("[1] Correlation of each CAMS tracer with REAL PM2.5:")
    for name, col in TRACERS.items():
        print(f"    {name:<26} r = {stats.pearsonr(d[col], y)[0]:+.2f}")

    # 2. fingerprint of the worst days
    print("\n[2] Enrichment on the worst-10% real-PM2.5 days (×typical):")
    hi = d[y > y.quantile(.90)]; lo = d[y < y.quantile(.50)]
    for name, col in TRACERS.items():
        print(f"    {name:<26} {hi[col].mean()/lo[col].mean():.1f}×")

    # 3. joint regression: which tracer dominates (standardized betas)
    print("\n[3] Joint regression  PM2.5 ~ SO2 + CO + NO2 + dust (standardized, HAC):")
    X = sm.add_constant(np.column_stack([z(d[c]) for c in TRACERS.values()]))
    m = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    names = list(TRACERS)
    for i, nm in enumerate(names):
        print(f"    {nm:<26} β = {m.params[i+1]:+5.1f} µg/m³/SD  (p={m.pvalues[i+1]:.1g})")
    print(f"    total R² = {m.rsquared:.2f}")

    # 4. does the combustion signal survive controlling for weather (dispersion+season)?
    print("\n[4] Combustion signal BEYOND meteorology:")
    base = sm.add_constant(np.column_stack([np.log(d["ventilation_coef"]+1),
                                            np.sin(2*np.pi*d["month"]/12), np.cos(2*np.pi*d["month"]/12)]))
    resid = y.values - sm.OLS(y.values, base).fit().predict(base)
    comb = sm.add_constant(np.column_stack([z(d["so2"]), z(d["carbon_monoxide"])]))
    mc = sm.OLS(resid, comb).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"    after removing dispersion & season, SO2+CO still explain "
          f"R²={mc.rsquared:.2f} of what's left (SO2 β={mc.params[1]:+.1f} p={mc.pvalues[1]:.1g}, "
          f"CO β={mc.params[2]:+.1f} p={mc.pvalues[2]:.1g})")

    # 5. winter excess: is the winter PM2.5 jump a combustion jump?
    print("\n[5] Winter (Nov-Mar) vs summer (Jun-Aug) means:")
    w = d[d.month.isin([11,12,1,2,3])]; s = d[d.month.isin([6,7,8])]
    for lab, col in [("PM2.5 (real)", "pm25_ground"), ("SO2", "so2"),
                     ("CO", "carbon_monoxide"), ("dust", "dust"), ("NO2", "nitrogen_dioxide")]:
        print(f"    {lab:<13} winter {w[col].mean():6.1f}  summer {s[col].mean():6.1f}  "
              f"({w[col].mean()/s[col].mean():.1f}× higher in winter)")

    # 6. weekday vs weekend (traffic would drop on weekends; heating would not)
    print("\n[6] Weekday vs weekend (traffic test):")
    wd = d[~d.weekend]; we = d[d.weekend]
    print(f"    PM2.5: weekday {wd['pm25_ground'].mean():.1f}  weekend {we['pm25_ground'].mean():.1f}  "
          f"({(we['pm25_ground'].mean()/wd['pm25_ground'].mean()-1)*100:+.0f}%)")
    print(f"    NO2  : weekday {wd['nitrogen_dioxide'].mean():.1f}  weekend {we['nitrogen_dioxide'].mean():.1f}")

    # ---- figure: source fingerprint ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    trc = ["so2", "carbon_monoxide", "nitrogen_dioxide", "dust"]
    tl = ["SO₂\n(mazut/coal)", "CO\n(combustion)", "NO₂\n(traffic)", "dust\n(storms)"]
    col = ["#c0392b", "#e67e22", "#e8a33d", "#7f8c8d"]
    enr = [hi[c].mean()/lo[c].mean() for c in trc]
    ax[0].bar(tl, enr, color=col); ax[0].axhline(1, color="k", lw=1)
    for i, v in enumerate(enr): ax[0].text(i, v+0.06, f"{v:.1f}×", ha="center", fontsize=10.5)
    ax[0].set(ylabel="× a typical day", title="What is elevated on the WORST days?")
    pm = ["pm25_ground", "so2", "carbon_monoxide", "nitrogen_dioxide", "dust"]
    pl = ["PM2.5", "SO₂", "CO", "NO₂", "dust"]
    rat = [w[c].mean()/s[c].mean() for c in pm]
    ax[1].bar(pl, rat, color=["#111", "#c0392b", "#e67e22", "#e8a33d", "#7f8c8d"])
    ax[1].axhline(1, color="k", lw=1)
    for i, v in enumerate(rat): ax[1].text(i, v+0.06, f"{v:.1f}×", ha="center", fontsize=10.5)
    ax[1].set(ylabel="winter ÷ summer", title="Winter rise: combustion tracks PM2.5; dust does not")
    plt.tight_layout(); fig.savefig(C.ROOT/"figures"/"mazut_fingerprint.png", dpi=140); plt.close(fig)
    print("\nSaved figures/mazut_fingerprint.png")

    print("\nCAVEATS: CAMS SO2/CO are model estimates (may under-capture illegal/sporadic "
          "mazut); SO2 cannot separate mazut from coal or other sulfur sources. This tests a "
          "SULFUR-RICH COMBUSTION fingerprint, not mazut uniquely.")


if __name__ == "__main__":
    main()
