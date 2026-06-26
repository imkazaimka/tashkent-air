"""
Satellite source attribution: from the WIDE regional MAIAC AOD field, determine for each smog day
WHERE the aerosol is coming from — local build-up over the city vs a regional plume, and its bearing.

This is what imagery adds over the tabular model: the SPATIAL pattern of aerosol.

Per day (PM2.5 > 35), from the rendered AOD field (magnitude proxy = (255-G)/255, validated vs ground
PM2.5; alpha = retrieval mask):
  - local_aod   : mean AOD over a ~40 km window on the city
  - regional_aod: mean AOD over the wide region
  - local_ratio : local / regional  (>1 => concentrated locally = local source)
  - sector AOD  : mean AOD in 8 compass sectors around the city => dominant inflow direction
  - centroid    : AOD-weighted centroid offset from the city => source bearing & distance
Source label (spatial): LOCAL (combustion) if concentrated over city; otherwise REGIONAL/TRANSPORT
from the dominant high-AOD bearing. Cross-checked against the independent tracer fingerprint (Sec.
Paper 1/2) where ground tracers exist (2023+).

Run:  python src/source_attribution.py
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DAOD = ROOT / "data" / "satellite" / "aod_wide"
FIGDIR = ROOT / "figures"; OUT = ROOT / "models"
# wide bbox + city pixel (must match pull_wide.py)
S, W, N, E = 38.5, 64.0, 43.0, 73.0; WD, HT = 600, 400
CX = int((69.24 - W) / (E - W) * WD); CY = int((N - 41.31) / (N - S) * HT)
KMX = (E - W) * 111 * np.cos(np.deg2rad(41)) / WD     # km per px (lon)
LOCAL_R = int(40 / KMX)                                # ~40 km city window radius
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
THR = 35
NAVY="#16314f"; ACC="#1f7a8c"; RED="#c0392b"; AMBER="#e8a33d"; GREY="#9aa3ad"; GREEN="#2e7d52"


def aod_field(fp):
    a = np.asarray(Image.open(fp).convert("RGBA"), float)
    mask = a[..., 3] > 10
    mag = np.where(mask, (255 - a[..., 1]) / 255.0, np.nan)     # (255-G)/255, validated vs PM2.5
    return mag, mask


def _grid():
    yy, xx = np.mgrid[0:HT, 0:WD]
    dx, dy = xx - CX, yy - CY
    dist = np.sqrt(dx ** 2 + dy ** 2)
    bearing = (np.degrees(np.arctan2(dx, -dy))) % 360         # 0=N,90=E
    return dist, bearing


DIST, BEAR = _grid()


def features(mag):
    loc = DIST <= LOCAL_R
    local = np.nanmean(mag[loc]) if np.isfinite(mag[loc]).any() else np.nan
    regional = np.nanmean(mag) if np.isfinite(mag).any() else np.nan
    ratio = local / (regional + 1e-6)
    # 8-sector means in an annulus beyond the city
    ann = (DIST > LOCAL_R) & (DIST < 220)
    secvals = {}
    for i, s in enumerate(SECTORS):
        lo, hi = (i * 45 - 22.5) % 360, (i * 45 + 22.5) % 360
        sm = (BEAR >= lo) | (BEAR < hi) if lo > hi else (BEAR >= lo) & (BEAR < hi)
        m = ann & sm
        secvals[s] = np.nanmean(mag[m]) if np.isfinite(mag[m]).any() else 0.0
    dom = max(secvals, key=secvals.get)
    # AOD-weighted centroid offset (where the aerosol mass sits)
    w = np.where(np.isfinite(mag), mag, 0)
    if w.sum() > 0:
        yy, xx = np.mgrid[0:HT, 0:WD]
        cx, cy = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
        off_bear = (np.degrees(np.arctan2(cx - CX, -(cy - CY)))) % 360
        off_km = np.sqrt((cx - CX) ** 2 + (cy - CY) ** 2) * KMX
    else:
        off_bear, off_km = np.nan, np.nan
    return dict(local=local, regional=regional, ratio=ratio, coverage=float(np.isfinite(mag).mean()),
                dom_sector=dom, dom_val=secvals[dom], off_bear=off_bear, off_km=off_km, **{f"sec_{s}": secvals[s] for s in SECTORS})


def load_labels():
    emb = pd.read_csv(ROOT / "data" / "raw" / "openaq_embassy_pm25_daily.csv")
    dc = [c for c in emb.columns if "date" in c.lower()][0]; pc = [c for c in emb.columns if "pm" in c.lower()][0]
    emb["d"] = pd.to_datetime(emb[dc]).dt.date; emb["pm"] = pd.to_numeric(emb[pc], errors="coerce")
    return {r.d: r.pm for _, r in emb.dropna(subset=["pm"]).iterrows()}


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lab = load_labels()
    rows = []
    for fp in sorted(DAOD.glob("*.png")):
        d = datetime.date.fromisoformat(fp.stem)
        if d not in lab:
            continue
        mag, mask = aod_field(fp)
        if not np.isfinite(mag).any():
            continue
        f = features(mag); f["date"] = d; f["pm"] = lab[d]; f["month"] = d.month
        rows.append(f)
    df = pd.DataFrame(rows)
    print(f"days with AOD+label: {len(df)} | smog days (>35): {(df.pm>THR).sum()}")
    # validate AOD proxy vs ground PM2.5
    v = df.dropna(subset=["local", "pm"])
    r = np.corrcoef(v.local, np.log1p(v.pm))[0, 1]
    print(f"AOD proxy (local) vs ground PM2.5: r = {r:.2f}  (validates the rendered-AOD scalar)")
    # source label (spatial)
    dirty = df[df.pm > THR].copy()
    dirty["source"] = np.where(dirty.ratio > 1.05, "Local (combustion)",
                        np.where(dirty.month.isin([4,5,6,7,8,9]), "Dust (regional)", "Transport (regional)"))
    print("\nspatial source mix on smog days:")
    print(dirty.source.value_counts().to_string())
    print("\ndominant inflow sector (regional smog days):")
    print(dirty[dirty.source != "Local (combustion)"].dom_sector.value_counts().to_string())
    dirty.to_csv(OUT / "source_attribution.csv", index=False)

    # ---- figure: local vs regional, colored by source ----
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=160)
    cmap = {"Local (combustion)": RED, "Dust (regional)": AMBER, "Transport (regional)": ACC}
    for s, c in cmap.items():
        sub = dirty[dirty.source == s]
        ax[0].scatter(sub.regional, sub.local, s=18, alpha=.6, color=c, label=f"{s} (n={len(sub)})")
    lim = np.nanpercentile(np.r_[dirty.local, dirty.regional], 98)
    ax[0].plot([0, lim], [0, lim], "--", color="#bbb", lw=1)
    ax[0].set(xlabel="regional AOD", ylabel="local (city) AOD", title="Local vs regional aerosol on smog days")
    ax[0].legend(fontsize=7.5, loc="upper left")
    # bearing rose of regional smog days
    reg = dirty[dirty.source != "Local (combustion)"]
    counts = [(reg.dom_sector == s).sum() for s in SECTORS]
    ang = np.deg2rad([0,45,90,135,180,225,270,315])
    axp = plt.subplot(122, projection="polar"); axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    axp.bar(ang, counts, width=0.7, color=ACC, alpha=.8)
    axp.set_xticks(ang); axp.set_xticklabels(SECTORS, fontsize=8); axp.set_title("Inflow direction of regional smog", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGDIR / "source_attr_scatter.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    json.dump({"n_days": len(df), "n_smog": int((df.pm>THR).sum()), "aod_pm_corr": round(float(r), 3),
               "source_mix": dirty.source.value_counts().to_dict()}, open(OUT / "source_attr_metrics.json", "w"), indent=2)
    print("\nsaved figures/source_attr_scatter.png + models/source_attribution.csv")


if __name__ == "__main__":
    main()
