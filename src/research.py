"""
Scientific analysis — test three hypotheses about Tashkent PM2.5.

  H-A  Transport:    pollution is advected from other regions, conditional on wind.
  H-B  Temperature:  pollution rises on COLD days (heating) AND HOT days (AC) — a U.
  H-C  Attribution:  what is the single dominant driver?

Method notes (this is an observational study on CAMS reanalysis, see caveats):
  * For attribution we use an EXOGENOUS model (PM2.5 autocorrelation removed), so
    we explain root causes, not "pollution is persistent".
  * The transport test is a wind-direction quasi-experiment: holding an upwind
    city's pollution fixed, is Tashkent dirtier when wind blows FROM it than
    TOWARD it? (common-cause weather can't easily fake that asymmetry).
  * OLS uses HAC (Newey-West) standard errors to honour serial correlation.
  * Dispersion (boundary layer) is controlled where relevant to separate
    "more emissions" from "less dispersion".

Run:  python src/research.py        (after features.py + train.py)
Outputs: figures/*.png  and  models/research_results.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"; FIG.mkdir(exist_ok=True)
MODELS = C.ROOT / "models"
R = {}  # results collector

AUTOCORR = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
            "pm25_roll3_mean", "pm25_roll7_mean", "pm25_roll7_std",
            "pm25_diff1", "episode_streak"]

GROUPS = {
    "dispersion": ["boundary_layer_height", "ventilation_coef", "trapping_index",
                   "inv_wind", "stagnation_proxy", "wind_speed_10m", "surface_pressure"],
    "thermal_season": ["temperature_2m", "shortwave_radiation",
                       "relative_humidity_2m", "doy_sin", "doy_cos",
                       "is_heating_season"],
    "wind_vector": ["wind_sin", "wind_cos"],
    "regional_transport": [f"{c['name'].lower()}_pm25_lag1" for c in C.REGIONAL_CITIES]
                          + [f"{c['name'].lower()}_pm25_lag2" for c in C.REGIONAL_CITIES]
                          + [f"{c['name'].lower()}_transport" for c in C.REGIONAL_CITIES],
    "precipitation": ["precip", "precip_lag1", "precip_sum_3d"],
}


def lgbm(X, y):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
                          min_child_samples=20, reg_lambda=1.0, random_state=42,
                          n_jobs=-1, verbose=-1)
    m.fit(X, y)
    return m


def alignment(f, bearing):
    """cos(wind_from_dir - city_bearing): +1 = wind blows FROM city toward Tashkent."""
    br = np.radians(bearing)
    return f["wind_cos"] * np.cos(br) + f["wind_sin"] * np.sin(br)


# ====================================================================== H-A
def hypothesis_A(f):
    print("\n" + "=" * 70 + "\nH-A  REGIONAL TRANSPORT\n" + "=" * 70)
    y = f["y"]
    res = {"lagged_corr": {}, "wind_interaction": {}}

    # A1 — simple lagged correlation city(D-1) -> Tashkent(D)
    print("A1. Lagged correlation  city_pm25(D-1) -> Tashkent(D):")
    for c in C.REGIONAL_CITIES:
        col = f"{c['name'].lower()}_pm25_lag1"
        r = f[[col, "y"]].corr().iloc[0, 1]
        res["lagged_corr"][c["name"]] = float(r)
        print(f"    {c['name']:<10} r = {r:+.3f}")

    # A2 — wind-direction quasi-experiment (the key test)
    # y ~ city_lag1 + align + city_lag1:align + ventilation + temperature
    # positive, significant interaction => transport (upwind pollution matters
    # MORE when wind is inbound), beyond shared weather.
    print("\nA2. Wind-direction interaction test (HAC SE). "
          "Positive city_lag1xINBOUND => transport:")
    ctrl = f[["ventilation_coef", "temperature_2m"]].copy()
    for c in C.REGIONAL_CITIES:
        col = f"{c['name'].lower()}_pm25_lag1"
        al = alignment(f, c["bearing"])
        d = pd.DataFrame({
            "y": y, "city": f[col], "align": al,
            "inter": f[col] * al,
            "vent": ctrl["ventilation_coef"], "temp": ctrl["temperature_2m"],
        }).dropna()
        # standardize predictors for comparable coefficients
        Xz = (d[["city", "align", "inter", "vent", "temp"]]
              - d[["city", "align", "inter", "vent", "temp"]].mean()) \
            / d[["city", "align", "inter", "vent", "temp"]].std()
        Xz = sm.add_constant(Xz)
        ols = sm.OLS(d["y"].values, Xz.values).fit(
            cov_type="HAC", cov_kwds={"maxlags": 7})
        names = ["const", "city", "align", "inter", "vent", "temp"]
        beta = dict(zip(names, ols.params)); p = dict(zip(names, ols.pvalues))
        res["wind_interaction"][c["name"]] = {
            "beta_interaction": float(beta["inter"]),
            "p_interaction": float(p["inter"]),
            "beta_city": float(beta["city"]), "p_city": float(p["city"]),
        }
        sig = "***" if p["inter"] < 0.001 else "**" if p["inter"] < 0.01 \
            else "*" if p["inter"] < 0.05 else "ns"
        print(f"    {c['name']:<10} inter_beta={beta['inter']:+.3f} "
              f"p={p['inter']:.3g} {sig:<4} (city_beta={beta['city']:+.2f})")

    R["H_A"] = res
    _fig_transport(f, res)


def _fig_transport(f, res):
    cities = [c["name"] for c in C.REGIONAL_CITIES]
    inter = [res["wind_interaction"][c]["beta_interaction"] for c in cities]
    psig = [res["wind_interaction"][c]["p_interaction"] for c in cities]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b" if p < 0.05 and b > 0 else "#7f8c8d"
              for b, p in zip(inter, psig)]
    ax.bar(cities, inter, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("inbound-wind x upwind-PM2.5\ninteraction (std coef)")
    ax.set_title("H-A: transport signal per neighbouring city\n"
                 "(red = positive & p<0.05 => advection)")
    plt.xticks(rotation=30); plt.tight_layout()
    fig.savefig(FIG / "H_A_transport.png", dpi=130); plt.close(fig)


# ====================================================================== H-B
def hypothesis_B(f, exo_model, exo_cols):
    print("\n" + "=" * 70 + "\nH-B  TEMPERATURE (U-shape?)\n" + "=" * 70)
    d = f[["temperature_2m", "y", "ventilation_coef"]].dropna()
    t, y = d["temperature_2m"], d["y"]
    res = {}

    # B1 — raw binned relationship
    bins = pd.qcut(t, 12, duplicates="drop")
    binned = y.groupby(bins, observed=True).agg(["mean", "sem", "count"])
    centers = [iv.mid for iv in binned.index]

    # B2 — quadratic OLS (HAC): y ~ z + z^2
    z = (t - t.mean()) / t.std()
    X = sm.add_constant(np.column_stack([z, z**2]))
    q = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    res["quad_raw"] = {"b_lin": float(q.params[1]), "p_lin": float(q.pvalues[1]),
                       "b_quad": float(q.params[2]), "p_quad": float(q.pvalues[2])}
    print(f"B2. Raw quadratic  y~temp+temp^2 :  lin={q.params[1]:+.2f}"
          f"(p={q.pvalues[1]:.2g})  quad={q.params[2]:+.2f}(p={q.pvalues[2]:.2g})")
    print("    quad>0 with both significant => U-shape; quad~0 => monotonic")

    # B3 — dispersion-controlled: remove ventilation effect, re-test temp.
    # Isolates EMISSION-driven temperature effect from dispersion.
    res_v = sm.OLS(y.values, sm.add_constant(
        np.log(d["ventilation_coef"].values + 1))).fit()
    resid = y.values - res_v.predict(sm.add_constant(
        np.log(d["ventilation_coef"].values + 1)))
    Xr = sm.add_constant(np.column_stack([z, z**2]))
    qr = sm.OLS(resid, Xr).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    res["quad_ventilation_controlled"] = {
        "b_lin": float(qr.params[1]), "p_lin": float(qr.pvalues[1]),
        "b_quad": float(qr.params[2]), "p_quad": float(qr.pvalues[2])}
    print(f"B3. Ventilation-controlled residual: lin={qr.params[1]:+.2f}"
          f"(p={qr.pvalues[1]:.2g})  quad={qr.params[2]:+.2f}(p={qr.pvalues[2]:.2g})")

    # B4 — HOT-tail test: among warm days, does PM2.5 rise with temperature?
    hot = d[t > t.quantile(0.75)]
    sl, ic, rr, pp, se = stats.linregress(hot["temperature_2m"], hot["y"])
    res["hot_tail"] = {"slope": float(sl), "p": float(pp), "r": float(rr),
                       "n": int(len(hot)), "temp_min": float(hot["temperature_2m"].min())}
    print(f"B4. Hot-tail (top quartile temp, n={len(hot)}, T>"
          f"{hot['temperature_2m'].min():.1f}C): slope={sl:+.3f} p={pp:.2g} "
          f"({'rises' if sl>0 and pp<0.05 else 'no rise'} with heat)")

    # B4b — AC test: hot-tail slope AFTER removing dispersion (summer's deep
    # boundary layer dilutes, which could mask an AC emission signal).
    dh = pd.DataFrame({"t": t.values, "r": resid})
    hot_r = dh[dh["t"] > t.quantile(0.75)]
    sl2, _, _, pp2, _ = stats.linregress(hot_r["t"], hot_r["r"])
    res["hot_tail_ventilation_controlled"] = {
        "slope": float(sl2), "p": float(pp2), "n": int(len(hot_r))}
    print(f"B4b. Hot-tail, dispersion-controlled: slope={sl2:+.3f} p={pp2:.2g} "
          f"({'AC signal' if sl2>0 and pp2<0.05 else 'still no AC signal'})")

    # B5 — partial dependence of temperature from EXOGENOUS model
    grid = np.linspace(t.quantile(0.02), t.quantile(0.98), 40)
    base = f[exo_cols].median(numeric_only=True)
    Xpd = pd.DataFrame([base] * len(grid))[exo_cols]
    Xpd["temperature_2m"] = grid
    pd_pred = exo_model.predict(Xpd)
    res["pdp_temp"] = {"temp": grid.tolist(), "pm25": pd_pred.tolist()}

    R["H_B"] = res
    _fig_temperature(centers, binned, t, resid, z, q, qr, grid, pd_pred)


def _fig_temperature(centers, binned, t, resid, z, q, qr, grid, pd_pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # raw binned
    ax[0].errorbar(centers, binned["mean"], yerr=binned["sem"], fmt="o-",
                   color="#2c3e50", capsize=3)
    ax[0].set(title="B1 raw: PM2.5 vs temperature",
              xlabel="temperature (C)", ylabel="mean PM2.5 (ug/m3)")
    # ventilation-controlled residual binned
    db = pd.DataFrame({"t": t.values, "r": resid})
    bb = db.groupby(pd.qcut(db["t"], 12, duplicates="drop"), observed=True)["r"].mean()
    ax[1].plot([iv.mid for iv in bb.index], bb.values, "s-", color="#e67e22")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set(title="B3 dispersion-controlled\n(emission-only temp effect)",
              xlabel="temperature (C)", ylabel="PM2.5 residual")
    # partial dependence
    ax[2].plot(grid, pd_pred, "-", color="#c0392b", lw=2)
    ax[2].set(title="B5 model partial dependence",
              xlabel="temperature (C)", ylabel="predicted PM2.5")
    plt.tight_layout(); fig.savefig(FIG / "H_B_temperature.png", dpi=130); plt.close(fig)


# ====================================================================== H-C
def hypothesis_C(f):
    print("\n" + "=" * 70 + "\nH-C  ATTRIBUTION — main driver\n" + "=" * 70)
    exo_cols = [c for c in f.columns
                if c not in (["date", "y", "split"] + AUTOCORR)]
    tr = f[f["split"].isin(["train", "val"])]
    te = f[f["split"] == "test"]
    Xtr, ytr = tr[exo_cols], tr["y"]
    Xte, yte = te[exo_cols], te["y"]

    full = lgbm(Xtr, ytr)
    r2_full = r2_score(yte, full.predict(Xte))
    print(f"Exogenous model R^2 (no PM2.5 autocorrelation): {r2_full:.3f}")
    print("  (vs full predictive model ~0.73 — the gap is pure persistence)")

    # leave-one-group-out  ->  unique contribution
    print("\nC. Feature-group importance:")
    print(f"  {'group':<20}{'alone R2':>10}{'dROP R2':>10}")
    rows = {}
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in exo_cols]
        # group alone
        a = lgbm(Xtr[cols], ytr)
        r2_alone = r2_score(yte, a.predict(Xte[cols]))
        # leave-one-out
        rest = [c for c in exo_cols if c not in cols]
        b = lgbm(Xtr[rest], ytr)
        r2_drop = r2_score(yte, b.predict(Xte[rest]))
        rows[g] = {"r2_alone": float(r2_alone),
                   "delta_r2_when_removed": float(r2_full - r2_drop)}
        print(f"  {g:<20}{r2_alone:>10.3f}{r2_full - r2_drop:>10.3f}")

    # permutation importance on full exogenous model, aggregated by group
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(full, Xte, yte, n_repeats=20,
                                random_state=42, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=exo_cols)
    grp_imp = {}
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in exo_cols]
        grp_imp[g] = float(imp[cols].sum())
    tot = sum(v for v in grp_imp.values() if v > 0)
    print("\n  Permutation importance share (% of positive total):")
    for g, v in sorted(grp_imp.items(), key=lambda kv: -kv[1]):
        print(f"    {g:<20}{v / tot * 100:>6.1f}%")

    R["H_C"] = {"r2_exogenous": float(r2_full), "groups": rows,
                "perm_importance": grp_imp,
                "perm_share": {g: float(v / tot) for g, v in grp_imp.items()}}
    _fig_drivers(rows, grp_imp, tot)
    return full, exo_cols


def _fig_drivers(rows, grp_imp, tot):
    gs = list(GROUPS.keys())
    alone = [rows[g]["r2_alone"] for g in gs]
    share = [grp_imp[g] / tot * 100 for g in gs]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].barh(gs, alone, color="#16a085"); ax[0].invert_yaxis()
    ax[0].set(title="Group-alone R^2 (standalone power)", xlabel="R^2")
    ax[1].barh(gs, share, color="#2980b9"); ax[1].invert_yaxis()
    ax[1].set(title="Permutation-importance share (%)", xlabel="% of total")
    plt.tight_layout(); fig.savefig(FIG / "H_C_drivers.png", dpi=130); plt.close(fig)


def seasonal_context(f):
    g = f.assign(m=f["date"].dt.month).groupby("m").agg(
        pm25=("y", "mean"), temp=("temperature_2m", "mean"),
        blh=("boundary_layer_height", "mean"))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(g.index, g["pm25"], color="#95a5a6", label="PM2.5")
    ax1.set(xlabel="month", ylabel="PM2.5 (ug/m3)")
    ax2 = ax1.twinx()
    ax2.plot(g.index, g["temp"], "r-o", label="temp")
    ax2.plot(g.index, g["blh"] / 20, "b-s", label="BLH/20")
    ax2.set_ylabel("temp (C)  /  BLH/20 (m)")
    ax1.set_title("Seasonal context: PM2.5 vs temperature & boundary layer")
    fig.legend(loc="upper center", ncol=3)
    plt.tight_layout(); fig.savefig(FIG / "seasonal_context.png", dpi=130); plt.close(fig)


def main():
    f = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    print(f"Loaded {len(f)} days for analysis.")
    full_exo, exo_cols = hypothesis_C(f)      # trains exogenous model (reused by B)
    hypothesis_A(f)
    hypothesis_B(f, full_exo, exo_cols)
    seasonal_context(f)

    (MODELS / "research_results.json").write_text(json.dumps(R, indent=2))
    print(f"\nSaved figures/ and models/research_results.json")


if __name__ == "__main__":
    main()
