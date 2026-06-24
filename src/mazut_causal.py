"""
The mazut causal chain + clean controllable/natural split.

  (1) COLD (and how long it lasts) -> heating demand -> mazut burning (SO2 rises)
  (2) mazut (SO2) -> PM2.5, and how much of the pollution it is responsible for
  (3) is mazut a SUMMER thing too? (if not, winter pollution needs the FUEL, not just
      the dense air)
  (4) in winter: is it the dense air or the mazut? (2x2: emission x dispersion)
  (5) clean facts: % of pollution that is us (controllable) vs natural

Tracers are CAMS model estimates (SO2 = sulfur fuel proxy); correlational, not proof.
Run:  python src/mazut_causal.py
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
    so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"])
    ft = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])[["date", "ventilation_coef"]]
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    d = dm.merge(so2, on="date").merge(ft, on="date").dropna(subset=["so2", "temperature_2m"])
    d["m"] = d["date"].dt.month
    d["HDD"] = (18 - d["temperature_2m"]).clip(lower=0)          # heating-degree-days
    d["HDD7"] = d["HDD"].rolling(7, min_periods=3).mean()        # how cold, sustained
    dg = d.merge(gt, on="date")                                   # subset with real PM2.5

    print("=== (1) COLD (and its persistence) -> mazut (SO2) ===")
    print(f"  SO2 vs heating-degree-days (same day):  r = {stats.pearsonr(d.HDD, d.so2)[0]:+.2f}")
    print(f"  SO2 vs 7-day sustained cold (HDD7):      r = {stats.pearsonr(d.dropna(subset=['HDD7']).HDD7, d.dropna(subset=['HDD7']).so2)[0]:+.2f}")
    for lo, hi, lab in [(20, 99, ">20°C (warm)"), (5, 20, "5–20°C"), (-5, 5, "−5–5°C (cold)"),
                        (-99, -5, "<−5°C (deep cold)")]:
        sub = d[(d.temperature_2m >= lo) & (d.temperature_2m < hi)]
        if len(sub) > 10:
            print(f"    {lab:<18} mean SO2 {sub.so2.mean():5.1f}  (n={len(sub)})")

    print("\n=== (2) mazut (SO2) -> PM2.5, and how responsible ===")
    print(f"  SO2 vs real PM2.5:  r = {stats.pearsonr(dg.so2, dg.pm25_ground)[0]:+.2f}  "
          f"(so SO2 alone explains ~{stats.pearsonr(dg.so2, dg.pm25_ground)[0]**2*100:.0f}% of PM2.5 variance)")
    # partial: SO2 beyond dispersion, and dispersion beyond SO2
    import statsmodels.api as sm
    z = lambda a: (a - a.mean())/a.std()
    X = sm.add_constant(np.column_stack([z(dg.so2), z(np.log(dg.ventilation_coef+1))]))
    m = sm.OLS(dg.pm25_ground.values, X).fit()
    print(f"  joint model PM2.5 ~ SO2 + dispersion: R²={m.rsquared:.2f}  "
          f"(SO2 β={m.params[1]:+.1f}, dispersion β={m.params[2]:+.1f} µg/m³/SD)")

    print("\n=== (3) Is mazut a SUMMER thing? (test the 'fuel vs dense air' question) ===")
    w = d[d.m.isin([11,12,1,2,3])]; s = d[d.m.isin([6,7,8])]
    print(f"  SO2: winter {w.so2.mean():.1f}  vs summer {s.so2.mean():.1f}  "
          f"-> mazut is a WINTER fuel; summer SO2 {s.so2.mean():.1f} is the non-heating baseline")
    mazut_share = (w.so2.mean()-s.so2.mean())/w.so2.mean()*100
    print(f"  ~{mazut_share:.0f}% of winter SO2 is the heating-season (mazut/coal) ADD-ON")

    print("\n=== (4) In winter: dense air OR mazut? (2x2 on real PM2.5) ===")
    wg = dg[dg.m.isin([11,12,1,2,3])].copy()
    so2_hi = wg.so2 > wg.so2.median(); disp_bad = wg.ventilation_coef < wg.ventilation_coef.median()
    cells = {}
    for sl, sm_ in [(True,"high mazut"),(False,"low mazut")]:
        for dl, dm_ in [(True,"trapped air"),(False,"good dispersion")]:
            c = wg[(so2_hi==sl)&(disp_bad==dl)]
            cells[(sm_,dm_)] = c.pm25_ground.mean()
            print(f"    {sm_:<11} + {dm_:<15}: PM2.5 = {c.pm25_ground.mean():5.1f}  (n={len(c)})")
    print(f"  -> high-mazut+good-dispersion ({cells[('high mazut','good dispersion')]:.0f}) "
          f"vs low-mazut+trapped-air ({cells[('low mazut','trapped air')]:.0f}): "
          f"{'MAZUT matters even without dense air' if cells[('high mazut','good dispersion')]>cells[('low mazut','trapped air')] else 'dense air dominates'}")

    print("\n=== (5) CLEAN FACTS: us (controllable) vs natural ===")
    pm = dg.pm25_ground
    floor = pm[pm < pm.quantile(.25)].mean()    # well-ventilated background
    overall = pm.mean()
    natural = floor / overall * 100
    print(f"  background/natural floor (clean days): {floor:.0f} µg/m³ = {natural:.0f}% of the mean ({overall:.0f})")
    print(f"  controllable local-combustion excess:  {overall-floor:.0f} µg/m³ = {100-natural:.0f}% of the mean")
    print(f"  => roughly {100-natural:.0f}% of the pollution is US (local emissions), "
          f"~{natural:.0f}% background/natural")

    # ---- figure: cold->mazut->pollution chain + 2x2 ----
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    tb = pd.cut(d.temperature_2m, [-30,-5,5,15,25,45])
    g = d.groupby(tb, observed=True).agg(so2=("so2","mean"), pm=("pm2_5","mean"))
    xlab = [str(i) for i in g.index]
    axb = ax[0]; axb2 = axb.twinx()
    axb.bar(range(len(g)), g.so2, color="#c0392b", alpha=0.8, label="SO₂ (mazut/coal tracer)")
    axb2.plot(range(len(g)), g.pm, "ko-", lw=2, label="PM2.5")
    axb.set_xticks(range(len(g))); axb.set_xticklabels(xlab, fontsize=9)
    axb.set(xlabel="temperature band (°C)", ylabel="SO₂ (µg/m³)",
            title="Colder → more mazut/coal (SO₂) → more PM2.5")
    axb2.set_ylabel("PM2.5 (µg/m³)")
    cell_lab = ["low mazut/coal\n+good disp.", "low mazut/coal\n+trapped",
                "high mazut/coal\n+good disp.", "high mazut/coal\n+trapped"]
    vals = [cells[("low mazut","good dispersion")], cells[("low mazut","trapped air")],
            cells[("high mazut","good dispersion")], cells[("high mazut","trapped air")]]
    ax[1].bar(cell_lab, vals, color=["#9aa3ad","#7f8c8d","#e67e22","#c0392b"])
    for i,v in enumerate(vals): ax[1].text(i, v+1, f"{v:.0f}", ha="center", fontsize=10)
    ax[1].axhline(THR, color="k", ls=":", lw=1); ax[1].set(ylabel="winter PM2.5 (µg/m³)",
            title="Winter: mazut/coal matters even with good dispersion")
    plt.tight_layout(); fig.savefig(C.ROOT/"figures"/"mazut_causal.png", dpi=140); plt.close(fig)
    print("\nSaved figures/mazut_causal.png")
    print("\nCAVEAT: SO2 is a CAMS model proxy for sulfur fuel; correlations are not proof, "
          "and the controllable/natural split is coarse.")


if __name__ == "__main__":
    main()
