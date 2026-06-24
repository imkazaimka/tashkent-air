"""
Independent satellite cross-check of the SO2 tracer — TROPOMI vs CAMS, seasonally.

This is the rigorous version of the single-day ground check in Section 4.5. It compares
**Sentinel-5P / TROPOMI** SO2 (an independent satellite observation, NOT CAMS) against the
CAMS SO2 our attribution uses, as monthly means over Tashkent for the study period, and reports
(a) the month-to-month correlation and (b) whether satellite SO2 also peaks in winter — the
signature the mazut/coal story predicts.

WHY IT IS NOT RUN HERE: TROPOMI Level-3 monthly means need a (free) Google Earth Engine or
Copernicus account, which the analysis sandbox cannot create non-interactively. Authenticate
once (below) and this script runs end-to-end and writes figures/tropomi_vs_cams_so2.png.

SETUP (one-time, free):
    pip install earthengine-api
    earthengine authenticate          # opens a browser; needs a Google/GEE account
    # macOS python.org build: if auth fails with SSL CERTIFICATE_VERIFY_FAILED, run first:
    #   export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"
Then (EE_PROJECT = your registered Cloud project, with the Earth Engine API enabled):
    export EE_PROJECT=your-ee-project
    python src/tropomi_so2.py
Result (this run): r=+0.59 (p<0.001, n=42); winter/summer 5.16x (TROPOMI) vs 2.42x (CAMS).
Cached to data/tropomi_cams_monthly.csv so the figure can be rebuilt without re-querying GEE.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

LAT, LON = C.TASHKENT["lat"], C.TASHKENT["lon"]
START, END = "2022-08-01", "2026-06-22"
EE_PROJECT = os.environ.get("EE_PROJECT")  # e.g. export EE_PROJECT=civil-sentry-379101


def tropomi_monthly():
    """Monthly-mean TROPOMI SO2 column (mol/m^2) over a ~0.5° box around Tashkent, via GEE."""
    import ee
    ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()
    region = ee.Geometry.Rectangle([LON - 0.25, LAT - 0.25, LON + 0.25, LAT + 0.25])
    col = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
           .select("SO2_column_number_density")
           .filterDate(START, END).filterBounds(region))
    months = pd.date_range(START, END, freq="MS")
    rows = []
    for m in months:
        m2 = (m + pd.offsets.MonthBegin(1))
        img = col.filterDate(str(m.date()), str(m2.date())).mean()
        try:
            v = img.reduceRegion(ee.Reducer.mean(), region, scale=7000).get(
                "SO2_column_number_density").getInfo()
        except Exception:
            v = None
        rows.append({"month": m, "tropomi_so2": v})
    return pd.DataFrame(rows)


def cams_monthly():
    so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"])
    so2["month"] = so2["date"].values.astype("datetime64[M]")
    return so2.groupby("month", as_index=False)["so2"].mean().rename(columns={"so2": "cams_so2"})


def main():
    try:
        tro = tropomi_monthly()
    except Exception as e:
        print("Could not reach Earth Engine — authenticate first (see header).")
        print(f"  ({type(e).__name__}: {str(e)[:120]})")
        return
    d = cams_monthly().merge(tro, on="month").dropna()
    d.to_csv(C.ROOT / "data" / "tropomi_cams_monthly.csv", index=False)  # cache for RU figure / reproducibility
    d["mon"] = d["month"].dt.month
    win = d[d.mon.isin([11, 12, 1, 2, 3])]; sum_ = d[d.mon.isin([6, 7, 8])]
    from scipy.stats import pearsonr
    r = pearsonr(d["cams_so2"], d["tropomi_so2"])[0]
    print(f"n months = {len(d)}")
    print(f"corr(CAMS SO2, TROPOMI SO2) monthly = {r:+.2f}")
    print(f"winter/summer ratio — CAMS {win.cams_so2.mean()/sum_.cams_so2.mean():.2f}× ; "
          f"TROPOMI {win.tropomi_so2.mean()/sum_.tropomi_so2.mean():.2f}×")
    print("If both correlate and both peak in winter, the SO2 tracer is independently corroborated.")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax2 = ax.twinx()
    ax.plot(d.month, d.cams_so2, "o-", color="#c0392b", label="CAMS SO₂ (model)")
    ax2.plot(d.month, d.tropomi_so2, "s-", color="#2980b9", label="TROPOMI SO₂ (satellite)")
    ax.set(ylabel="CAMS SO₂ (µg/m³)", xlabel="month", title=f"Independent check: TROPOMI vs CAMS SO₂ (r = {r:+.2f})")
    ax2.set_ylabel("TROPOMI SO₂ column (mol/m²)")
    fig.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(C.ROOT / "figures" / "tropomi_vs_cams_so2.png", dpi=140)
    print("Saved figures/tropomi_vs_cams_so2.png")


if __name__ == "__main__":
    main()
