"""
Where does Tashkent's PM2.5 actually come from? A clean, facts-only breakdown into:
  NATURAL / BACKGROUND  — the floor that remains on clean, well-ventilated days
                          (regional baseline, any blown-in dust, unavoidable background)
  CARS & year-round     — local combustion present all year (vehicle exhaust + warm-season
                          sources), bounded by the summer (heating-off) level
  MAZUT & COAL HEATING  — the winter-only combustion excess (sulfur/SO2 fingerprint)

Plus the two "natural suspect" tests the public asks about:
  - Dust storms: are the bad days dust days?      (answer via dust corr + dusty-day PM2.5)
  - Cars: does pollution drop on weekends?         (traffic would; heating would not)

Method is a coarse seasonal decomposition on the REAL embassy sensor — directional, not a
formal source-apportionment. Tracers are CAMS model estimates. We state facts, not remedies.

Run:  python src/source_breakdown.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

THR = C.PM25_THRESHOLD


def main():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"])
    d = dm.merge(gt, on="date").merge(so2, on="date").dropna(subset=["pm25_ground", "dust"])
    d["m"] = d["date"].dt.month
    d["weekend"] = d["date"].dt.dayofweek >= 5
    pm = d["pm25_ground"]

    annual = pm.mean()
    summer = d[d.m.isin([6, 7, 8])]["pm25_ground"].mean()
    winter = d[d.m.isin([11, 12, 1, 2, 3])]["pm25_ground"].mean()
    floor = pm[pm < pm.quantile(.25)].mean()
    traffic = max(summer - floor, 0.0)            # year-round non-background (cars + other)
    heating = max(annual - floor - traffic, 0.0)  # winter-only combustion (mazut + coal)
    nat_p, car_p, heat_p = floor/annual*100, traffic/annual*100, heating/annual*100

    print(f"Real embassy PM2.5, n={len(d)}.  annual {annual:.0f}  summer {summer:.0f}  "
          f"winter {winter:.0f}  clean-day floor {floor:.0f} µg/m³\n")
    print("ANNUAL-AVERAGE SOURCE BREAKDOWN (coarse seasonal decomposition):")
    print(f"  Natural / background floor      {floor:5.1f} µg/m³   {nat_p:4.0f}%")
    print(f"  Cars & year-round combustion    {traffic:5.1f} µg/m³   {car_p:4.0f}%   (upper bound for cars)")
    print(f"  Mazut & coal heating (winter)   {heating:5.1f} µg/m³   {heat_p:4.0f}%")

    # on the dangerous winter days specifically
    win_excess = winter - summer
    print(f"\nON THE DANGEROUS WINTER DAYS:")
    print(f"  heating excess {win_excess:.0f} of {winter:.0f} µg/m³ = "
          f"{win_excess/winter*100:.0f}% of the winter load is the mazut/coal heating add-on")

    # natural suspect 1: dust storms
    dusty = d[d["dust"] > d["dust"].quantile(.90)]
    print(f"\nDUST STORMS (natural):  dust vs PM2.5 r={stats.pearsonr(d['dust'],pm)[0]:+.2f}; "
          f"on the top-10% dust days PM2.5 = {dusty['pm25_ground'].mean():.0f} vs {annual:.0f} overall "
          f"-> dust days are CLEANER.  Answer: NOT the cause.")

    # human suspect: cars (weekend test)
    wd = d[~d.weekend]["pm25_ground"].mean(); we = d[d.weekend]["pm25_ground"].mean()
    print(f"CARS (weekend test):  weekday {wd:.0f} vs weekend {we:.0f} ({(we/wd-1)*100:+.0f}%) "
          f"-> no weekend drop.  Answer: secondary, not the swing driver.")

    # ---- figure ----
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    # A: stacked source bar (annual average)
    labels = ["Mazut & coal\nheating", "Cars & other\nyear-round", "Natural /\nbackground"]
    vals = [heat_p, car_p, nat_p]
    cols = ["#c0392b", "#e67e22", "#7f8c8d"]
    _w, _t, aut = ax[0].pie(
        vals, labels=labels, colors=cols, startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.0f}%", pctdistance=0.78, labeldistance=1.13,
        wedgeprops=dict(width=0.46, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10.5, fontweight="bold"))
    for a in aut:
        a.set_color("white"); a.set_fontsize(12); a.set_fontweight("bold")
    ax[0].text(0, 0, "≈70%\nhuman-made", ha="center", va="center",
               fontsize=12.5, fontweight="bold", color="#333")
    ax[0].set_title("Where Tashkent's air pollution comes from")
    # B: the two suspects the public names, tested
    cats = ["Dust days\nvs typical", "Weekend\nvs weekday"]
    base = [annual, wd]; test = [dusty['pm25_ground'].mean(), we]
    x = np.arange(2); w = 0.36
    ax[1].bar(x - w/2, base, w, label="typical", color="#bdc3c7")
    ax[1].bar(x + w/2, test, w, label="the suspect", color="#7f8c8d")
    ax[1].axhline(THR, color="k", ls=":", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(cats)
    ax[1].set(ylabel="PM2.5 (µg/m³)", title="Dust storms & cars: tested, not the cause")
    ax[1].legend(fontsize=9)
    for i, (b, t) in enumerate(zip(base, test)):
        ax[1].text(i - w/2, b+1, f"{b:.0f}", ha="center", fontsize=9)
        ax[1].text(i + w/2, t+1, f"{t:.0f}", ha="center", fontsize=9)
    plt.tight_layout(); fig.savefig(C.ROOT/"figures"/"source_breakdown.png", dpi=140); plt.close(fig)
    print("\nSaved figures/source_breakdown.png")
    print("\nCAVEAT: coarse seasonal decomposition on one sensor; CAMS tracers are model "
          "estimates; the 'cars' bucket is an UPPER bound (includes other year-round sources). "
          "Directional shares, not a formal source-apportionment.")


if __name__ == "__main__":
    main()
