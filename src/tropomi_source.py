"""
Satellite source attribution from TROPOMI (Sentinel-5P) daily tracers over Tashkent.

Daily regional-mean UV Aerosol Index (dust/smoke, all-weather), NO2 (anthropogenic), CO are used to
classify the SOURCE of each smog day (embassy PM2.5 > 35) — dust vs winter anthropogenic — independent
of, and validated against, the ground-tracer attribution of Paper 1.

Outputs the coverage win (TROPOMI vs optical AOD), the UVAI->dust validation, the source-regime
separation, and the seasonal source mix.

Run:  python src/tropomi_source.py
"""
from __future__ import annotations
import sys, json, glob, datetime
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; OUT = ROOT / "models"; THR = 35
NAVY="#16314f"; ACC="#1f7a8c"; RED="#c0392b"; AMBER="#e8a33d"; GREY="#9aa3ad"; GREEN="#2e7d52"


def load():
    tro = pd.read_csv(ROOT/"data"/"processed"/"tropomi_daily_tracers.csv", parse_dates=["date"])
    emb = pd.read_csv(ROOT/"data"/"raw"/"openaq_embassy_pm25_daily.csv")
    dc=[c for c in emb.columns if "date" in c.lower()][0]; pc=[c for c in emb.columns if "pm" in c.lower()][0]
    emb["date"]=pd.to_datetime(emb[dc]); emb["pm"]=pd.to_numeric(emb[pc],errors="coerce")
    dm = pd.read_csv(ROOT/"data"/"processed"/"daily_merged.csv", parse_dates=["date"])[["date","dust","pm10","pm2_5","nitrogen_dioxide","carbon_monoxide"]]
    so2 = pd.read_csv(ROOT/"data"/"raw"/"cams_so2_daily.csv", parse_dates=["date"]).rename(columns={"so2":"cams_so2"})
    d = tro.merge(emb[["date","pm"]],on="date").merge(dm,on="date",how="left").merge(so2,on="date",how="left")
    d["month"]=d.date.dt.month
    return d


def aod_coverage_on_smog():
    """Optical AOD retrieval rate on smog days, from the wide AOD frames already pulled."""
    emb = pd.read_csv(ROOT/"data"/"raw"/"openaq_embassy_pm25_daily.csv")
    dc=[c for c in emb.columns if "date" in c.lower()][0]; pc=[c for c in emb.columns if "pm" in c.lower()][0]
    lab={pd.to_datetime(r[dc]).date():pd.to_numeric(r[pc],errors="coerce") for _,r in emb.iterrows()}
    cov=[]
    for fp in glob.glob(str(ROOT/"data"/"satellite"/"aod_wide"/"*.png")):
        dd=datetime.date.fromisoformat(Path(fp).stem)
        if lab.get(dd,0)>THR:
            cov.append((np.asarray(Image.open(fp).convert("RGBA"))[...,3]>10).mean())
    cov=np.array(cov)
    return (cov>0.5).mean() if len(cov) else np.nan, len(cov)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = load()
    smog = d[d.pm>THR].copy()
    print(f"days {len(d)} | smog days {len(smog)}")

    # ---- (1) UVAI validates dust ----
    v = d.dropna(subset=["uvai","dust"])
    r_dust = np.corrcoef(v.uvai, v.dust)[0,1]
    v2 = d.dropna(subset=["uvai","pm10","pm2_5"]); coarse = v2.pm10/v2.pm2_5.clip(lower=1)
    r_coarse = np.corrcoef(v2.uvai, coarse)[0,1]
    print(f"TROPOMI UVAI vs CAMS dust: r={r_dust:.2f} | vs coarse fraction: r={r_coarse:.2f}")

    # ---- (2) source classification (percentile of climatology) ----
    for c in ["uvai","no2","cams_so2"]:
        d[c+"_pct"] = d[c].rank(pct=True)
        smog[c+"_pct"] = smog[c].map(lambda x: (d[c]<x).mean() if pd.notna(x) else np.nan)
    def classify(r):
        if pd.notna(r.uvai_pct) and r.uvai_pct>0.70 and r.uvai>0:   # absorbing aerosol present
            return "Dust (regional)"
        if pd.notna(r.no2_pct) and r.no2_pct>0.55:
            return "Anthropogenic (winter)"
        return "Anthropogenic (winter)" if r.month in (11,12,1,2) else "Mixed/other"
    smog["sat_source"] = smog.apply(classify, axis=1)
    mix = smog.sat_source.value_counts()
    print("\nTROPOMI source classification of smog days:"); print(mix.to_string())

    # ---- (3) coverage win ----
    cov_aod, n_aod = aod_coverage_on_smog()
    sm = smog
    cov = {"TROPOMI UVAI": sm.uvai.notna().mean(), "TROPOMI CO": sm.co.notna().mean(),
           "TROPOMI NO₂": sm.no2.notna().mean(), "Optical AOD": cov_aod}
    print(f"\ncoverage on smog days: " + " | ".join(f"{k} {v:.0%}" for k,v in cov.items()))

    # ---- (4) seasonal source mix ----
    smog["season"] = np.where(smog.month.isin([11,12,1,2]),"Winter",np.where(smog.month.isin([6,7,8]),"Summer","Spring/Autumn"))
    seas = pd.crosstab(smog.season, smog.sat_source, normalize="index")*100

    json.dump({"uvai_dust_r":round(float(r_dust),2),"uvai_coarse_r":round(float(r_coarse),2),
               "coverage":{k:round(float(x),3) for k,x in cov.items()},
               "source_mix":mix.to_dict(),"n_smog":int(len(smog))}, open(OUT/"tropomi_source_metrics.json","w"),indent=2)

    # ================= FIGURES =================
    fig, ax = plt.subplots(1,3,figsize=(13,3.8),dpi=160)
    # A: coverage
    ks=list(cov); vals=[cov[k]*100 for k in ks]; cols=[ACC,ACC,ACC,GREY]
    ax[0].bar(range(4),vals,color=cols)
    for i,vv in enumerate(vals): ax[0].text(i,vv+2,f"{vv:.0f}%",ha="center",fontsize=9,fontweight="bold")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(ks,fontsize=8,rotation=15); ax[0].set_ylabel("% of smog days retrieved"); ax[0].set_ylim(0,108); ax[0].set_title("Coverage on smog days\n(satellite sees through cloud)",fontsize=10)
    # B: UVAI vs dust
    sc=ax[1].scatter(v.dust, v.uvai, s=6, alpha=.3, c=v.month, cmap="twilight")
    ax[1].set(xlabel="CAMS dust (µg/m³)", ylabel="TROPOMI UV Aerosol Index", title=f"UVAI tracks dust (r={r_dust:.2f})"); ax[1].axhline(0,color="#bbb",lw=.6)
    # C: source space
    cmap={"Dust (regional)":AMBER,"Anthropogenic (winter)":RED,"Mixed/other":GREY}
    for s,c in cmap.items():
        ss=smog[smog.sat_source==s]; ax[2].scatter(ss.no2*1e5, ss.uvai, s=14,alpha=.6,color=c,label=f"{s} ({len(ss)})")
    ax[2].set(xlabel="TROPOMI NO₂ (×10⁻⁵ mol/m²)", ylabel="TROPOMI UVAI", title="Smog days in TROPOMI source space"); ax[2].axhline(0,color="#bbb",lw=.6); ax[2].legend(fontsize=7.5)
    for a in ax:
        for s in ("top","right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG/"tropomi_source.png",dpi=160,bbox_inches="tight",facecolor="white"); plt.close()
    print("\nsaved figures/tropomi_source.png + models/tropomi_source_metrics.json")
    print("\nseasonal source mix (%):"); print(seas.round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
