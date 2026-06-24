"""
TIER A validation — falsify and stress-test the three findings WITHOUT new data.

A finding only counts as "proven (so far)" if it survives attempts to break it.
Tests:
  H-A transport
    1. Placebo wind shuffle  — destroy wind/pollution day-matching; the E/NE
       interaction MUST collapse to the null. (falsification)
    2. Bearing sweep         — vary the assumed source direction; the transport
       signal MUST peak near each city's TRUE compass bearing. (specificity)
    3. Wind-speed dose-response — transport effect should strengthen with wind.
  H-B temperature
    4. Matched pairs         — same temperature, different boundary layer; if
       PM2.5 still differs, dispersion (not heating) drives the cold effect.
  Robustness (all)
    5. Block-bootstrap 95% CIs on every headline number (respects autocorrelation).
    6. Per-winter replication — do signs hold in each winter independently?

Run:  python src/tier_a_validation.py     (after research.py)
Outputs: console PASS/FAIL, models/tier_a_results.json, figures/tier_a_*.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"
MODELS = C.ROOT / "models"
RNG = np.random.default_rng(42)
OUT = {}

ENE = ["Fergana", "Almaty", "Bishkek"]   # cities with positive transport in research.py
SW = ["Samarkand", "Ashgabat"]
BEAR = {c["name"]: c["bearing"] for c in C.REGIONAL_CITIES}


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a - a.mean()


def inter_beta(y, city, sin, cos, bearing, vent, temp):
    """OLS coefficient on (upwind_PM2.5 x inbound_alignment), controlling for
    ventilation and temperature. Standardized design, fast lstsq."""
    al = cos * np.cos(np.radians(bearing)) + sin * np.sin(np.radians(bearing))
    inter = city * al
    X = np.column_stack([np.ones_like(y), z(city), z(al), z(inter), z(vent), z(temp)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[3]


def load():
    f = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    need = ["y", "wind_sin", "wind_cos", "ventilation_coef", "temperature_2m",
            "boundary_layer_height", "wind_speed_10m"] \
        + [f"{c['name'].lower()}_pm25_lag1" for c in C.REGIONAL_CITIES]
    return f.dropna(subset=need).reset_index(drop=True)


# ---------------------------------------------------------------- Test 1
def test1_placebo_wind(f):
    print("\n[1] PLACEBO WIND SHUFFLE  (transport must collapse when wind is scrambled)")
    sin, cos = f["wind_sin"].values, f["wind_cos"].values
    vent, temp, y = f["ventilation_coef"].values, f["temperature_2m"].values, f["y"].values
    res = {}
    n_perm = 1000
    for name in ENE + SW:
        city = f[f"{name.lower()}_pm25_lag1"].values
        obs = inter_beta(y, city, sin, cos, BEAR[name], vent, temp)
        null = np.empty(n_perm)
        for i in range(n_perm):
            p = RNG.permutation(len(sin))
            null[i] = inter_beta(y, city, sin[p], cos[p], BEAR[name], vent, temp)
        # one-sided empirical p in the observed direction
        if obs >= 0:
            pval = float(np.mean(null >= obs))
        else:
            pval = float(np.mean(null <= obs))
        passed = pval < 0.05
        res[name] = {"observed": float(obs), "null_mean": float(null.mean()),
                     "null_sd": float(null.std()), "emp_p": pval, "pass": passed}
        flag = "PASS" if passed else "FAIL"
        print(f"    {name:<10} obs={obs:+.3f}  null={null.mean():+.3f}+/-{null.std():.3f}"
              f"  emp_p={pval:.3f}  [{flag}]")
    OUT["test1_placebo_wind"] = res
    ene_ok = all(res[c]["pass"] for c in ENE)
    print(f"    => E/NE transport survives falsification: "
          f"{'YES' if ene_ok else 'NO'}")


# ---------------------------------------------------------------- Test 2
def test2_bearing_sweep(f):
    print("\n[2] BEARING SWEEP  (signal must peak near each city's TRUE bearing)")
    sin, cos = f["wind_sin"].values, f["wind_cos"].values
    vent, temp, y = f["ventilation_coef"].values, f["temperature_2m"].values, f["y"].values
    bearings = np.arange(0, 360, 10)
    res, sweeps = {}, {}
    for name in ENE + SW:
        city = f[f"{name.lower()}_pm25_lag1"].values
        betas = np.array([inter_beta(y, city, sin, cos, b, vent, temp) for b in bearings])
        peak = int(bearings[np.argmax(betas)])
        true = BEAR[name]
        diff = min(abs(peak - true), 360 - abs(peak - true))
        res[name] = {"true_bearing": true, "peak_bearing": peak, "angular_err": diff,
                     "pass": diff <= 45}
        sweeps[name] = betas
        flag = "PASS" if diff <= 45 else "n/a"
        print(f"    {name:<10} true={true:>3} deg  peak={peak:>3} deg  "
              f"err={diff:>3} deg  [{flag if name in ENE else 'SW-ctrl'}]")
    OUT["test2_bearing_sweep"] = res
    _fig_sweep(bearings, sweeps)


def _fig_sweep(bearings, sweeps):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, betas in sweeps.items():
        ax.plot(bearings, betas, label=name, lw=2 if name in ENE else 1.2,
                ls="-" if name in ENE else "--")
        ax.axvline(BEAR[name], color="grey", lw=0.6, alpha=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="assumed source bearing (deg)",
           ylabel="transport interaction beta",
           title="Test 2: transport signal vs assumed direction\n"
                 "(E/NE solid; vertical lines = true bearings)")
    ax.legend(ncol=2, fontsize=8); plt.tight_layout()
    fig.savefig(FIG / "tier_a_bearing_sweep.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- Test 3
def test3_dose_response(f):
    print("\n[3] WIND-SPEED DOSE-RESPONSE  (E/NE transport should grow with wind)")
    ws = f["wind_speed_10m"].values
    med = np.median(ws)
    res = {}
    for name in ENE:
        out = {}
        for lab, mask in (("low_wind", ws <= med), ("high_wind", ws > med)):
            g = f[mask]
            out[lab] = float(inter_beta(
                g["y"].values, g[f"{name.lower()}_pm25_lag1"].values,
                g["wind_sin"].values, g["wind_cos"].values, BEAR[name],
                g["ventilation_coef"].values, g["temperature_2m"].values))
        out["pass"] = out["high_wind"] > out["low_wind"]
        res[name] = out
        print(f"    {name:<10} low_wind beta={out['low_wind']:+.3f}  "
              f"high_wind beta={out['high_wind']:+.3f}  "
              f"[{'PASS' if out['pass'] else 'FAIL'}]")
    OUT["test3_dose_response"] = res


# ---------------------------------------------------------------- Test 4
def test4_matched_pairs(f):
    print("\n[4] MATCHED PAIRS  (same temp, different boundary layer -> dispersion test)")
    d = f[["temperature_2m", "boundary_layer_height", "y"]].dropna().copy()
    d["tbin"] = (d["temperature_2m"] / 2).round() * 2   # 2C bins
    diffs, lows, highs = [], [], []
    for tb, g in d.groupby("tbin"):
        if len(g) < 20:
            continue
        bm = g["boundary_layer_height"].median()
        lo = g[g["boundary_layer_height"] <= bm]["y"]   # shallow BL (less dispersion)
        hi = g[g["boundary_layer_height"] > bm]["y"]    # deep BL (more dispersion)
        if len(lo) < 5 or len(hi) < 5:
            continue
        diffs.append((tb, lo.mean(), hi.mean(), lo.mean() - hi.mean(), len(g)))
        lows.append(lo.mean()); highs.append(hi.mean())
    arr = pd.DataFrame(diffs, columns=["tbin", "shallow_BL_pm", "deep_BL_pm", "diff", "n"])
    # paired test across temperature bins
    w = stats.wilcoxon(arr["shallow_BL_pm"], arr["deep_BL_pm"])
    mean_diff = float(arr["diff"].mean())
    res = {"mean_pm_difference_shallow_minus_deep": mean_diff,
           "wilcoxon_p": float(w.pvalue), "n_bins": int(len(arr)),
           "pass": mean_diff > 0 and w.pvalue < 0.05}
    OUT["test4_matched_pairs"] = res
    print(f"    at matched temperature, shallow-BL days are "
          f"{mean_diff:+.1f} ug/m3 vs deep-BL  (Wilcoxon p={w.pvalue:.3g}, "
          f"{len(arr)} temp bins)")
    print(f"    => dispersion drives pollution independent of temperature: "
          f"{'YES' if res['pass'] else 'NO'}")
    _fig_matched(arr)


def _fig_matched(arr):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(arr["tbin"], arr["shallow_BL_pm"], "o-", color="#c0392b",
            label="shallow boundary layer (trapped)")
    ax.plot(arr["tbin"], arr["deep_BL_pm"], "s-", color="#2980b9",
            label="deep boundary layer (ventilated)")
    ax.set(xlabel="temperature (C, 2C bins)", ylabel="mean PM2.5 (ug/m3)",
           title="Test 4: at the SAME temperature, shallow BL = more pollution\n"
                 "(isolates dispersion from heating emissions)")
    ax.legend(); plt.tight_layout()
    fig.savefig(FIG / "tier_a_matched_pairs.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- Test 5
def _block_boot(stat_fn, n, n_boot=1000, block=14):
    out = []
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in RNG.choice(starts, n_blocks)])[:n]
        out.append(stat_fn(idx))
    return np.percentile(out, [2.5, 97.5])


def test5_bootstrap(f):
    print("\n[5] BLOCK-BOOTSTRAP 95% CIs  (moving 14-day blocks)")
    sin, cos = f["wind_sin"].values, f["wind_cos"].values
    vent, temp, y = f["ventilation_coef"].values, f["temperature_2m"].values, f["y"].values
    n = len(f)
    res = {}

    # transport beta CIs
    for name in ENE + SW:
        city = f[f"{name.lower()}_pm25_lag1"].values
        lo, hi = _block_boot(
            lambda idx, c=city: inter_beta(y[idx], c[idx], sin[idx], cos[idx],
                                           BEAR[name], vent[idx], temp[idx]), n)
        excl0 = (lo > 0) or (hi < 0)
        res[f"transport_beta_{name}"] = {"ci95": [float(lo), float(hi)],
                                         "excludes_0": bool(excl0)}
        print(f"    transport beta {name:<10} 95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  {'(excludes 0)' if excl0 else '(includes 0)'}")

    # temperature: raw linear vs ventilation-controlled linear
    tz = z(temp)
    def raw_lin(idx):
        X = np.column_stack([np.ones(len(idx)), tz[idx], tz[idx]**2])
        b, *_ = np.linalg.lstsq(X, y[idx], rcond=None); return b[1]
    def ctrl_lin(idx):
        lv = np.log(vent[idx] + 1)
        Xv = np.column_stack([np.ones(len(idx)), lv])
        bv, *_ = np.linalg.lstsq(Xv, y[idx], rcond=None)
        resid = y[idx] - Xv @ bv
        X = np.column_stack([np.ones(len(idx)), tz[idx], tz[idx]**2])
        b, *_ = np.linalg.lstsq(X, resid, rcond=None); return b[1]
    for lab, fn in (("temp_raw_linear", raw_lin), ("temp_ventilation_controlled_linear", ctrl_lin)):
        lo, hi = _block_boot(fn, n)
        incl0 = (lo <= 0 <= hi)
        res[lab] = {"ci95": [float(lo), float(hi)], "includes_0": bool(incl0)}
        print(f"    {lab:<38} 95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  {'(includes 0)' if incl0 else '(excludes 0)'}")
    OUT["test5_bootstrap"] = res


# ---------------------------------------------------------------- Test 6
def test6_per_winter(f):
    print("\n[6] PER-WINTER REPLICATION  (Nov-Mar, each winter independently)")
    m = f["date"].dt.month
    wl = np.where(m >= 11, f["date"].dt.year, f["date"].dt.year - 1)
    f = f.assign(winter=wl)[m.isin([11, 12, 1, 2, 3])]
    res = {}
    for wy, g in f.groupby("winter"):
        if len(g) < 60:
            continue
        # eastern transport: mean interaction beta over E/NE cities
        ene_b = np.mean([inter_beta(
            g["y"].values, g[f"{c.lower()}_pm25_lag1"].values,
            g["wind_sin"].values, g["wind_cos"].values, BEAR[c],
            g["ventilation_coef"].values, g["temperature_2m"].values) for c in ENE])
        # cold-trapping: does ventilation-control shrink the cold (temp) effect?
        tz = z(g["temperature_2m"].values); yv = g["y"].values
        raw = np.linalg.lstsq(np.column_stack([np.ones(len(g)), tz]), yv, rcond=None)[0][1]
        lv = np.log(g["ventilation_coef"].values + 1)
        bv = np.linalg.lstsq(np.column_stack([np.ones(len(g)), lv]), yv, rcond=None)[0]
        resid = yv - np.column_stack([np.ones(len(g)), lv]) @ bv
        ctrl = np.linalg.lstsq(np.column_stack([np.ones(len(g)), tz]), resid, rcond=None)[0][1]
        res[f"{wy}-{wy+1}"] = {"ene_transport_beta": float(ene_b),
                               "temp_raw_lin": float(raw),
                               "temp_ctrl_lin": float(ctrl), "n": int(len(g))}
        print(f"    winter {wy}-{wy+1} (n={len(g):3d}): E/NE transport beta="
              f"{ene_b:+.2f}  | temp raw={raw:+.1f} -> ventilation-ctrl={ctrl:+.1f}")
    ene_consistent = all(v["ene_transport_beta"] > 0 for v in res.values())
    shrink_consistent = all(abs(v["temp_ctrl_lin"]) < abs(v["temp_raw_lin"])
                            for v in res.values())
    print(f"    => E/NE transport positive every winter: {'YES' if ene_consistent else 'NO'}")
    print(f"    => ventilation-control shrinks cold effect every winter: "
          f"{'YES' if shrink_consistent else 'NO'}")
    OUT["test6_per_winter"] = {"winters": res,
                               "ene_positive_all": bool(ene_consistent),
                               "cold_shrinks_all": bool(shrink_consistent)}


def main():
    f = load()
    print(f"Loaded {len(f)} complete-case days for validation.")
    test1_placebo_wind(f)
    test2_bearing_sweep(f)
    test3_dose_response(f)
    test4_matched_pairs(f)
    test5_bootstrap(f)
    test6_per_winter(f)
    def _jd(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        return str(o)
    (MODELS / "tier_a_results.json").write_text(json.dumps(OUT, indent=2, default=_jd))
    print(f"\nSaved models/tier_a_results.json and figures/tier_a_*.png")


if __name__ == "__main__":
    main()
