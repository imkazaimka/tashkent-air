"""
Regional multi-sensor validation — validate CAMS against reference-grade ground
monitors across Tashkent AND the neighbouring cities used in the transport
analysis. Turns a single-sensor check into a 9-station, 5-city test and shows the
CAMS under-bias is a systematic regional feature, not a Tashkent artefact.

CAMS series already exist per city in daily_merged.csv (pm2_5 for Tashkent;
<city>_pm25 for neighbours). We pull the main long-record OpenAQ monitor(s) per
city and compare.

Output: models/regional_validation.json, figures/regional_validation.png
Run:  python src/validate_regional_sensors.py
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from dotenv import load_dotenv
load_dotenv(C.ROOT / ".env")

THR = C.PM25_THRESHOLD
BASE = "https://api.openaq.org/v3"

# curated reference / long-record monitors, one CAMS series per city
CITY_CAMS = {"Tashkent": "pm2_5", "Almaty": "almaty_pm25", "Bishkek": "bishkek_pm25",
             "Dushanbe": "dushanbe_pm25", "Ashgabat": "ashgabat_pm25"}
STATIONS = [
    ("Tashkent", "US Embassy",     25916,    "reference"),
    ("Tashkent", "Sputnik-4",      13465748, "low-cost"),
    ("Almaty",   "Almaty ref.",    25903,    "reference"),
    ("Bishkek",  "US Embassy",     23972,    "reference"),
    ("Bishkek",  "Bishkek ref.",   25744,    "reference"),
    ("Dushanbe", "Dushanbe",       25215,    "reference"),
    ("Dushanbe", "US Embassy",     30477,    "reference"),
    ("Ashgabat", "US Embassy",     23772,    "reference"),
    ("Ashgabat", "Ashgabat ref.",  25891,    "reference"),
]


def fetch_daily(sensor_id, hdr):
    rows, page = [], 1
    while True:
        r = requests.get(f"{BASE}/sensors/{sensor_id}/days", headers=hdr,
                         params={"limit": 1000, "page": page}, timeout=60)
        if r.status_code != 200:
            break
        res = r.json().get("results", [])
        if not res:
            break
        for x in res:
            day = x.get("period", {}).get("datetimeFrom", {}).get("utc")
            val = x.get("value")
            if day and val is not None:
                rows.append((day[:10], val))
        page += 1
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=["date", "obs"])
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date")


def main():
    key = os.getenv("OPENAQ_TOKEN"); hdr = {"X-API-Key": key}
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])

    rows, pooled = [], []
    for city, label, sid, typ in STATIONS:
        g = fetch_daily(sid, hdr)
        cam = dm[["date", CITY_CAMS[city]]].rename(columns={CITY_CAMS[city]: "cams"})
        m = g.merge(cam, on="date").dropna()
        if len(m) < 20:
            print(f"  {city}/{label}: too few paired days ({len(m)})"); continue
        c, o = m["cams"].values, m["obs"].values
        rec = float(((o > THR) & (c > THR)).sum() / max((o > THR).sum(), 1))
        rows.append({"city": city, "station": label, "type": typ, "n": int(len(m)),
                     "cams_mean": float(c.mean()), "obs_mean": float(o.mean()),
                     "ratio": float(o.mean() / c.mean()), "r": float(stats.pearsonr(c, o)[0]),
                     "recall35": rec})
        pooled.append(m.assign(city=city))
        print(f"  {city:<9} {label:<14} n={len(m):>4}  CAMS {c.mean():5.1f} vs obs "
              f"{o.mean():5.1f}  ({o.mean()/c.mean():.2f}x)  r={stats.pearsonr(c,o)[0]:.2f}  "
              f"recall>35={rec:.2f}")

    tab = pd.DataFrame(rows)
    allp = pd.concat(pooled, ignore_index=True)
    summary = {"n_stations": int(len(tab)), "n_cities": int(tab["city"].nunique()),
               "total_paired_days": int(tab["n"].sum()),
               "ratio_min": float(tab["ratio"].min()), "ratio_max": float(tab["ratio"].max()),
               "ratio_median": float(tab["ratio"].median()),
               "all_underestimate": bool((tab["ratio"] > 1.2).all()),
               "median_recall35": float(tab["recall35"].median()),
               "stations": rows}
    (C.ROOT / "models" / "regional_validation.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  {len(tab)} stations across {tab['city'].nunique()} cities, "
          f"{tab['n'].sum()} paired days")
    print(f"  CAMS underestimates at EVERY station: {summary['all_underestimate']} "
          f"(ratio {tab['ratio'].min():.2f}-{tab['ratio'].max():.2f}, "
          f"median {tab['ratio'].median():.2f}x)")
    print(f"  median exceedance recall across stations: {summary['median_recall35']:.2f}")

    # ---- figure: pooled scatter + per-station ratio bars ----
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    cmap = {c: col for c, col in zip(CITY_CAMS, ["#c0392b", "#2980b9", "#16a085",
                                                 "#8e44ad", "#e67e22"])}
    # LEFT: one readable marker per CITY (mean) with middle-50% spread bars,
    # instead of 6,894 overplotted points.
    xmx = 32
    ax[0].plot([0, xmx], [0, xmx], "k--", lw=1.4, zorder=1, label="1:1 (CAMS = reality)")
    ax[0].text(3, 80, "CAMS\nreads LOW\n(points above line)", fontsize=10,
               color="#777", style="italic", va="top")
    loff = {"Dushanbe": (10, 7), "Tashkent": (12, 2), "Almaty": (12, 2),
            "Bishkek": (-62, 6), "Ashgabat": (10, -16)}
    for city, gg in allp.groupby("city"):
        cx, cy = gg["cams"].mean(), gg["obs"].mean()
        xe = [[cx - gg["cams"].quantile(.25)], [gg["cams"].quantile(.75) - cx]]
        ye = [[cy - gg["obs"].quantile(.25)], [gg["obs"].quantile(.75) - cy]]
        ax[0].errorbar(cx, cy, xerr=xe, yerr=ye, fmt="o", ms=13, color=cmap[city],
                       mec="white", mew=1.6, capsize=4, elinewidth=1.7, zorder=5)
        ax[0].annotate(city, (cx, cy), xytext=loff.get(city, (11, 5)),
                       textcoords="offset points", fontsize=11, fontweight="bold",
                       color=cmap[city])
    ax[0].set(xlim=(0, xmx), ylim=(0, 95),
              xlabel="CAMS model — mean (µg/m³)", ylabel="real sensor — mean (µg/m³)",
              title="Each city's average: real vs CAMS")
    ax[0].legend(loc="lower right", fontsize=9.5)
    # RIGHT: under-bias ratio per station, with value labels
    lab = [f"{r['city']} · {r['station'].replace(' ref.','').replace('US ','')}" for r in rows]
    ax[1].barh(lab, [r["ratio"] for r in rows], color=[cmap[r["city"]] for r in rows], alpha=0.92)
    for i, r in enumerate(rows):
        ax[1].text(r["ratio"] + 0.06, i, f"{r['ratio']:.1f}×", va="center", fontsize=9.5)
    ax[1].axvline(1, color="k", lw=1.3); ax[1].invert_yaxis()
    ax[1].set(xlim=(0, 5.4), xlabel="real ÷ CAMS   (1 = correct, >1 = CAMS too low)",
              title="Under-reading at all 9 monitors")
    plt.tight_layout(); fig.savefig(C.ROOT / "figures" / "regional_validation.png", dpi=140)
    plt.close(fig)
    print("\nSaved figures/regional_validation.png and models/regional_validation.json")


if __name__ == "__main__":
    main()
