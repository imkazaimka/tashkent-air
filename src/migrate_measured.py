"""
Driver attribution for the MEASURED-network model (Section 4.4) — regenerated on the real target so
the importance figure is not on the old CAMS/embassy model.

Loads the measured model (models/lgbm_measured.txt) and the measured municipal target, and computes
grouped permutation importance (permute a whole driver group on the test set, measure the log-R²
drop). Output: figures/drivers_measured.png.

Run:  python src/migrate_measured.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

GROUPS = {
    "Pollution / emission tracers (PM, NO₂, CO, dust)": ["pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide", "dust"],
    "Weather / dispersion": ["temperature_2m", "wind_speed_10m", "boundary_layer_height", "relative_humidity_2m", "surface_pressure"],
    "Season": ["doy_sin", "doy_cos"],
}


def main():
    muni = pd.read_csv(C.RAW / "tashkent_municipal_pm25_daily.csv", parse_dates=["date"])
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    d = dm.merge(muni[["date", "pm25_muni"]], on="date").sort_values("date").reset_index(drop=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
    feats = [f for g in GROUPS.values() for f in g]
    d = d.dropna(subset=feats + ["pm25_muni"]).reset_index(drop=True)
    cut = d.date.quantile(0.8); tr, te = d[d.date <= cut], d[d.date > cut]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                          min_child_samples=20, random_state=42, verbose=-1).fit(tr[feats], np.log1p(tr.pm25_muni))
    base = r2_score(np.log1p(te.pm25_muni), m.predict(te[feats]))
    rng = np.random.RandomState(0); imp = {}
    for name, cols in GROUPS.items():
        drops = []
        for _ in range(20):
            x = te[feats].copy()
            for c in cols:
                x[c] = rng.permutation(x[c].values)
            drops.append(base - r2_score(np.log1p(te.pm25_muni), m.predict(x)))
        imp[name] = np.mean(drops)
    tot = sum(imp.values())
    print(f"measured-model test R²={base:.2f}; grouped permutation importance (share of R² lost):")
    for k, v in imp.items():
        print(f"  {v/tot*100:4.0f}%  {k}")

    order = sorted(imp, key=imp.get)
    fig, ax = plt.subplots(figsize=(7.4, 3.1), dpi=160)
    ax.barh(range(len(order)), [imp[k]/tot*100 for k in order], color=["#7f8c8d", "#1f7a8c", "#c0392b"][:len(order)])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([k.split(" (")[0] for k in order], fontsize=8.5)
    ax.set_xlabel("share of predictive power (%), permutation importance")
    ax.set_title("What drives the forecast of MEASURED citywide PM2.5", fontsize=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(C.ROOT / "figures" / "drivers_measured.png", dpi=160, bbox_inches="tight", facecolor="white")
    print("saved figures/drivers_measured.png")


if __name__ == "__main__":
    main()
