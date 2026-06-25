"""
Ground-tracer cross-check for the source attribution (Section 4): SO2, CO, NO2 from the THREE
official Tashkent stations with real history — Uzhydromet Chilanzar & Yunusabad, and the US-Embassy
site — versus the CAMS tracers the attribution uses.

Source: WAQI / aqicn.org historical station downloads (the only 3 Tashkent stations with multi-year
history; the ~10 "citizen" stations are May-2026-onward and PM-only-real, so excluded). WAQI data are
unvalidated and non-redistributable -> we read local Downloads copies and print/plot aggregates only;
the raw CSVs are NOT committed. Attribution: Uzhydromet + WAQI.

Tracers are used RELATIVELY (correlation, cold/warm ratio), never as absolute mass — so the WAQI
index units are fine. Chilanzar's 2025 SO2 is a local sensor fault (the embassy reads normal over the
same months) and is dropped.

Findings the paper cites:
  SO2  -> all 3 stations winter/summer ~2.0x; rises with cold; tracks CAMS  -> mazut/coal confirmed
  CO   -> ground cold/warm ~CAMS, r~0.7                                     -> combustion confirmed
  NO2  -> ground ~flat (cold/warm ~1.0) while CAMS ~3x                      -> traffic flat, not driver

Run:  python src/ground_tracers_crosscheck.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

DL = Path.home() / "Downloads"
FILES = {"Chilanzar": "tashkent-chilanzar-air-quality.csv",
         "Embassy":   "tashkent-us embassy, uzbekistan-air-quality.csv",
         "Yunusabad": "tashkent-yunusabad-air-quality.csv"}


def load(name):
    p = DL / FILES[name]
    if not p.exists():
        sys.exit(f"WAQI CSV not found: {p} (not redistributed; download from aqicn.org).")
    d = pd.read_csv(p, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], format="%Y/%m/%d", errors="coerce")
    for c in ("pm25", "no2", "so2", "co"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["date"])


def main():
    st = {k: load(k) for k in FILES}
    st["Chilanzar"].loc[st["Chilanzar"].date.dt.year == 2025, "so2"] = np.nan   # 2025 fault
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    cams_so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"]).rename(columns={"so2": "v"})

    def pool(names, col):
        x = pd.concat([st[n][["date", col]].dropna(subset=[col]) for n in names])
        return x.groupby("date")[col].mean().rename("ground").reset_index()

    def check(ground, cams):
        m = ground.merge(cams, on="date").merge(dm[["date", "temperature_2m"]], on="date").dropna()
        c = m[m.temperature_2m < m.temperature_2m.quantile(.33)]
        w = m[m.temperature_2m > m.temperature_2m.quantile(.67)]
        return (len(m), pearsonr(m.ground, m.v)[0],
                c.ground.median() / max(w.ground.median(), .5),
                c.v.median() / max(w.v.median(), .01))

    specs = [("SO2", ["Chilanzar", "Embassy", "Yunusabad"], cams_so2),
             ("CO",  ["Chilanzar", "Embassy", "Yunusabad"], dm[["date", "carbon_monoxide"]].rename(columns={"carbon_monoxide": "v"})),
             ("NO2", ["Embassy", "Yunusabad"],              dm[["date", "nitrogen_dioxide"]].rename(columns={"nitrogen_dioxide": "v"}))]
    res = {}
    print(f"{'tracer':6}{'n':>5}{'r':>7}{'ground c/w':>12}{'CAMS c/w':>10}")
    for name, stns, cams in specs:
        n, r, gcw, ccw = check(pool(stns, name.lower()), cams)
        res[name] = (n, r, gcw, ccw)
        print(f"{name:6}{n:>5}{r:>7.2f}{gcw:>12.1f}{ccw:>10.1f}")

    # figure: ground vs CAMS cold/warm ratio per tracer
    labels = ["SO₂\n(mazut/coal)", "CO\n(combustion)", "NO₂\n(traffic)"]
    g = [res[k][2] for k in ("SO2", "CO", "NO2")]
    cm = [res[k][3] for k in ("SO2", "CO", "NO2")]
    rs = [res[k][1] for k in ("SO2", "CO", "NO2")]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=160)
    ax.bar(x - w/2, g, w, label="ground (3 stations)", color="#c0392b")
    ax.bar(x + w/2, cm, w, label="CAMS", color="#16314f", alpha=.55, hatch="//")
    ax.axhline(1, color="#888", lw=1, ls="--")
    ax.text(2.0, 1.06, "flat = no seasonal rise", fontsize=8, color="#555")
    for i, r in enumerate(rs):
        ax.text(i, max(g[i], cm[i]) + .12, f"r={r:.2f}", ha="center", fontsize=8.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 3.8)
    ax.set_ylabel("winter ÷ summer (cold ÷ warm)")
    ax.set_title("Ground stations: combustion tracers rise in winter, traffic stays flat")
    ax.legend(fontsize=8.5, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = C.ROOT / "figures" / "ground_tracers.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
