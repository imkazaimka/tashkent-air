"""
TIER B (ground truth) — validate CAMS and re-test the hypotheses against the
US-Embassy reference monitor (OpenAQ sensor 25916).

Part 1: how well does CAMS reanalysis match real sensors? (level, correlation,
        episode detection)
Part 2: do H-A (transport) and H-B (temperature) survive when the TARGET is the
        real sensor instead of CAMS? Weather is ERA5 and regional pollution is
        CAMS — both independent of the embassy monitor, so this is a genuine test.

Run:  python src/validate_ground_truth.py   (after fetch_openaq.py + features.py)
Outputs: console report, models/ground_truth_results.json, figures/ground_truth_*.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"; MODELS = C.ROOT / "models"
RNG = np.random.default_rng(42)
OUT = {}
ENE = ["Fergana", "Almaty", "Bishkek"]
SW = ["Samarkand", "Ashgabat"]
BEAR = {c["name"]: c["bearing"] for c in C.REGIONAL_CITIES}


def z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 0 else a - a.mean()


def inter_beta(y, city, sin, cos, bearing, vent, temp):
    al = cos * np.cos(np.radians(bearing)) + sin * np.sin(np.radians(bearing))
    X = np.column_stack([np.ones_like(y), z(city), z(al), z(city * al), z(vent), z(temp)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b[3]


def main():
    feat = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    df = feat.merge(gt, on="date", how="inner").rename(columns={"y": "cams"})
    df = df.dropna(subset=["cams", "pm25_ground"])
    print(f"Paired CAMS vs ground: {len(df)} days "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    # ---------------- Part 1: agreement ----------------
    c, g = df["cams"].values, df["pm25_ground"].values
    pear = stats.pearsonr(c, g); spear = stats.spearmanr(c, g)
    bias = float(c.mean() - g.mean())
    rmse = float(np.sqrt(np.mean((c - g) ** 2)))
    sl, ic, r, p, se = stats.linregress(c, g)   # ground ~ CAMS
    thr = C.PM25_THRESHOLD
    gb, cb = g > thr, c > thr
    recall = float((gb & cb).sum() / gb.sum())     # of real exceedances, CAMS caught
    print("\n=== Part 1: CAMS reanalysis vs ground sensor ===")
    print(f"  mean CAMS {c.mean():.1f}  vs ground {g.mean():.1f}  "
          f"(bias {bias:+.1f} ug/m3, CAMS = {c.mean()/g.mean()*100:.0f}% of real)")
    print(f"  Pearson r = {pear[0]:.3f}   Spearman r = {spear[0]:.3f}")
    print(f"  RMSE = {rmse:.1f} ug/m3")
    print(f"  ground ~ CAMS:  slope {sl:.2f}  intercept {ic:.1f}  (R^2 {r**2:.3f})")
    print(f"  exceedance (>{thr:.0f}) days: ground {int(gb.sum())}, CAMS {int(cb.sum())}; "
          f"CAMS recall of real exceedances = {recall:.2f}")
    OUT["agreement"] = {"n": len(df), "mean_cams": float(c.mean()),
                        "mean_ground": float(g.mean()), "bias": bias,
                        "pearson_r": float(pear[0]), "spearman_r": float(spear[0]),
                        "rmse": rmse, "ground_vs_cams_slope": float(sl),
                        "ground_vs_cams_intercept": float(ic), "r2": float(r**2),
                        "ground_exceed_days": int(gb.sum()),
                        "cams_exceed_days": int(cb.sum()), "cams_recall": recall}
    _fig_agreement(df, sl, ic)

    # ---------------- Part 2: re-test hypotheses on GROUND target ----------------
    print("\n=== Part 2: hypotheses re-tested with REAL sensor as target ===")
    need = ["pm25_ground", "wind_sin", "wind_cos", "ventilation_coef",
            "temperature_2m"] + [f"{c['name'].lower()}_pm25_lag1" for c in C.REGIONAL_CITIES]
    d = df.dropna(subset=need).reset_index(drop=True)
    y = d["pm25_ground"].values
    sin, cos = d["wind_sin"].values, d["wind_cos"].values
    vent, temp = d["ventilation_coef"].values, d["temperature_2m"].values

    # H-A: transport interaction on ground truth (+ placebo wind shuffle)
    print("H-A transport (target = ground sensor):")
    ha = {}
    for name in ENE + SW:
        city = d[f"{name.lower()}_pm25_lag1"].values
        obs = inter_beta(y, city, sin, cos, BEAR[name], vent, temp)
        null = np.array([inter_beta(y, city, *(lambda p: (sin[p], cos[p]))(
            RNG.permutation(len(sin))), BEAR[name], vent, temp) for _ in range(500)])
        pval = float(np.mean(null >= obs) if obs >= 0 else np.mean(null <= obs))
        ha[name] = {"beta": float(obs), "placebo_p": pval}
        sig = "PASS" if pval < 0.05 else "ns"
        print(f"    {name:<10} inter_beta={obs:+.3f}  placebo_p={pval:.3f}  [{sig}]")
    OUT["H_A_ground"] = ha

    # H-B: temperature on ground truth
    print("H-B temperature (target = ground sensor):")
    zt = z(temp)
    q = sm.OLS(y, sm.add_constant(np.column_stack([zt, zt**2]))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})
    # ventilation-controlled
    lv = np.log(vent + 1)
    bv = sm.OLS(y, sm.add_constant(lv)).fit()
    resid = y - bv.predict(sm.add_constant(lv))
    qr = sm.OLS(resid, sm.add_constant(np.column_stack([zt, zt**2]))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})
    hot = d[temp > np.quantile(temp, 0.75)]
    hs, _, _, hp, _ = stats.linregress(hot["temperature_2m"], hot["pm25_ground"])
    print(f"    raw quad:  lin={q.params[1]:+.2f}(p={q.pvalues[1]:.2g})  "
          f"quad={q.params[2]:+.2f}(p={q.pvalues[2]:.2g})")
    print(f"    ventilation-controlled: lin={qr.params[1]:+.2f}(p={qr.pvalues[1]:.2g})")
    print(f"    hot-tail slope={hs:+.3f}(p={hp:.2g})  "
          f"({'AC signal' if hs>0 and hp<0.05 else 'no AC signal'})")
    OUT["H_B_ground"] = {
        "raw_lin": float(q.params[1]), "raw_lin_p": float(q.pvalues[1]),
        "raw_quad": float(q.params[2]), "raw_quad_p": float(q.pvalues[2]),
        "vent_ctrl_lin": float(qr.params[1]), "vent_ctrl_lin_p": float(qr.pvalues[1]),
        "hot_tail_slope": float(hs), "hot_tail_p": float(hp)}
    _fig_temp_ground(d)

    def _jd(o):
        import numpy as _np
        if isinstance(o, _np.bool_): return bool(o)
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        return str(o)
    (MODELS / "ground_truth_results.json").write_text(json.dumps(OUT, indent=2, default=_jd))
    print("\nSaved models/ground_truth_results.json and figures/ground_truth_*.png")


def _fig_agreement(df, sl, ic):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    ax[0].plot(df["date"], df["pm25_ground"], color="#c0392b", lw=0.8, label="ground (embassy)")
    ax[0].plot(df["date"], df["cams"], color="#2980b9", lw=0.8, label="CAMS reanalysis")
    ax[0].set(title="Tashkent PM2.5: real sensor vs model", ylabel="ug/m3"); ax[0].legend()
    ax[1].scatter(df["cams"], df["pm25_ground"], s=8, alpha=0.4, color="#555")
    xs = np.linspace(df["cams"].min(), df["cams"].max(), 50)
    ax[1].plot(xs, sl*xs+ic, "r-", label=f"ground={sl:.2f}*CAMS+{ic:.0f}")
    ax[1].plot(xs, xs, "k--", lw=0.8, label="1:1")
    ax[1].set(title="CAMS underestimates real PM2.5", xlabel="CAMS", ylabel="ground"); ax[1].legend()
    plt.tight_layout(); fig.savefig(FIG / "ground_truth_agreement.png", dpi=130); plt.close(fig)


def _fig_temp_ground(d):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    b = d.groupby(pd.qcut(d["temperature_2m"], 12, duplicates="drop"),
                  observed=True)["pm25_ground"].mean()
    ax.plot([iv.mid for iv in b.index], b.values, "o-", color="#c0392b")
    ax.set(xlabel="temperature (C)", ylabel="ground PM2.5 (ug/m3)",
           title="H-B on REAL sensor data: PM2.5 vs temperature")
    plt.tight_layout(); fig.savefig(FIG / "ground_truth_temperature.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
