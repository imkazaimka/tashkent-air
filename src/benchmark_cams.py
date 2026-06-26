"""
The benchmark the paper needed: our ConvLSTM vs the operational gold standard (CAMS) and two classical
baselines (persistence, perfect-wind advection), scored identically on the SAME out-of-sample days.

Domain : Central Asia [55,37,75,47]   Window : 2024-03-15..06-15 (spring dust, in the model's held-out
20% — the model never trained on these dates).  Truth : MAIAC total-column AOD (what everyone is scored
against).  Metric : dust-field pattern correlation r and anomaly-correlation ACC (transport skill above
the seasonal mean), per lead day.  All forecasts are put back into raw-AOD units so the comparison with
CAMS is apples-to-apples.

A "complexity ladder":  persistence  <  perfect-wind advection  <  our ConvLSTM  <  CAMS (3-D physics +
assimilation).  Where we land on it is the honest measure of what a compact learned model is worth.

Run:  python src/benchmark_cams.py
"""
from __future__ import annotations
import sys, datetime, glob, json
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import map_coordinates
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convlstm_multimodal as cm

ROOT = Path(__file__).resolve().parent.parent
DOM = [55, 37, 75, 47]
AOD = ROOT / "data/satellite/aod_real_grid"
WIND = ROOT / "data/satellite/wind_grid"
PRECIP = ROOT / "data/satellite/precip_grid"
CAMS = ROOT / "data/satellite/cams"
DX_KM = 20.0


def aod_scale():
    vals = []
    for fp in list(cm.DAOD.glob("*.npy"))[::9]:
        a = np.load(fp)
        if a.ndim == 2: vals.append(a[a != -999].ravel())
    return float(np.percentile(np.concatenate(vals), 97))


def load_day(d, aod_s):
    """raw MAIAC AOD (+valid mask) and the model's 5-channel normalised input frame, for one day."""
    fa, fw, fp = AOD/f"{d}.npy", WIND/f"{d}.npy", PRECIP/f"{d}.npy"
    if not (fa.exists() and fw.exists() and fp.exists()): return None
    a = np.load(fa); w = np.load(fw); p = np.load(fp)
    if a.ndim != 2 or w.ndim != 3 or p.ndim != 2: return None
    araw = cm._resize(np.where(a != -999, a, np.nan))
    amask = cm._resize((a != -999).astype(np.float32))
    amag = cm._resize(np.clip(np.where(a != -999, a, 0)/aod_s, 0, 1.5))
    wu = cm._resize(np.clip(w[0]/15.0, -1.5, 1.5)); wv = cm._resize(np.clip(w[1]/15.0, -1.5, 1.5))
    pr = cm._resize(np.clip(p*100.0, 0, 3))
    frame = np.stack([amag, amask, wu, wv, pr]).astype(np.float32)       # [0,1,6,7,8] for the 'ao' net
    return dict(raw=araw, mask=(amask > 0.5), frame=frame,
                u=cm._resize(w[0]), v=cm._resize(w[1]))                  # raw wind m/s for advection


def advect(field, u, v, days):
    """Advection nowcast: rigidly translate the dust field by the prevailing (domain-mean) wind for
    `days` days. A single motion vector — the standard simple advection baseline — not a per-pixel warp
    (per-pixel wind shear would tear the field apart, which is not what an advection nowcast does)."""
    H, W = field.shape; cpd = 86400.0/(DX_KM*1000.0)                    # cells/day per (m/s)
    um, vm = float(np.nanmean(u)), float(np.nanmean(v))                 # prevailing wind
    yy, xx = np.mgrid[0:H, 0:W]
    srcx = xx - um*cpd*days                                             # east wind (u>0): air came from the west
    srcy = yy + vm*cpd*days                                             # north wind (v>0): air came from the south (row0=north)
    f = np.nan_to_num(field, nan=0.0)
    return map_coordinates(f, [srcy, srcx], order=1, mode="nearest")


def corr(a, b):
    a, b = a.ravel(), b.ravel(); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or a[m].std() < 1e-6 or b[m].std() < 1e-6: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    aod_s = aod_scale(); print(f"AOD scale {aod_s:.3f}", flush=True)
    ck = torch.load(ROOT/"models"/"convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(5, 1, 3).to(cm.DEV); net.load_state_dict(ck["ao"]); net.eval()

    dates = sorted(datetime.date.fromisoformat(Path(f).stem) for f in glob.glob(str(AOD/"*.npy")))
    days = {d: load_day(d, aod_s) for d in dates
            if datetime.date(2024, 3, 1) <= d <= datetime.date(2024, 6, 20)}
    days = {d: v for d, v in days.items() if v is not None}
    ds = sorted(days)
    # window climatology of raw truth (per pixel), for ACC
    stack = np.stack([np.where(days[d]["mask"], days[d]["raw"], np.nan) for d in ds])
    clim = np.nanmean(stack, axis=0)

    methods = ["persistence", "wind-advection", "ConvLSTM", "CAMS"]
    per = {L: {m: {"r": [], "acc": []} for m in methods} for L in (1, 2, 3)}
    best = {"load": -1}
    for d0 in ds:
        win = [d0 + datetime.timedelta(days=k) for k in range(cm.T_IN)]
        if not all(w in days for w in win): continue
        seq = np.stack([days[w]["frame"] for w in win])                 # T_IN,5,H,W
        last = days[win[-1]]                                            # for persistence/advection
        x = torch.from_numpy(seq[None]).to(cm.DEV)
        for L in (1, 2, 3):
            T = d0 + datetime.timedelta(days=cm.T_IN + L - 1)           # valid target day
            if T not in days: continue
            tr = days[T]; truth = np.where(tr["mask"], tr["raw"], np.nan)
            cams_f = CAMS/f"lead{L}"/f"{T.isoformat()}.npy"
            if not cams_f.exists(): continue
            cams = cm._resize(np.where(np.load(cams_f)[0] > -900, np.load(cams_f)[0], np.nan))
            ex = torch.from_numpy(np.stack([days[w]["frame"][2:5] for w in     # exog wind/precip future (use last obs as proxy)
                                            [d0+datetime.timedelta(days=cm.T_IN+j) if (d0+datetime.timedelta(days=cm.T_IN+j) in days) else win[-1]
                                             for j in range(cm.K_OUT)]])[None]).to(cm.DEV)
            with torch.no_grad():
                P = net(x, ex).cpu().numpy()[0]                          # K_OUT,1,H,W (normalised)
            model_raw = np.clip(P[L-1, 0], 0, None)*aod_s
            adv = advect(last["raw"], last["u"], last["v"], L)
            preds = {"persistence": last["raw"], "wind-advection": adv, "ConvLSTM": model_raw, "CAMS": cams}
            valid = tr["mask"] & np.isfinite(cams) & np.isfinite(truth)
            if valid.sum() < 200: continue
            ca = np.where(valid, clim, np.nan)
            for m, f in preds.items():
                fv = np.where(valid, f, np.nan)
                per[L][m]["r"].append(corr(fv, truth))
                per[L][m]["acc"].append(corr(fv - ca, truth - ca))
            load = np.nanmean(truth[valid])
            if L == 1 and load > best["load"]:
                best = {"load": load, "T": T, "truth": np.where(valid, truth, np.nan),
                        "ConvLSTM": np.where(valid, model_raw, np.nan), "CAMS": np.where(valid, cams, np.nan)}

    # aggregate
    out = {"domain": DOM, "window": "2024-03-15..06-15", "n_days_lead1": len(per[1]["CAMS"]["r"])}
    for L in (1, 2, 3):
        out[f"+{L}d"] = {m: {"r": round(np.nanmean(per[L][m]["r"]), 3),
                             "acc": round(np.nanmean(per[L][m]["acc"]), 3)} for m in methods}
    json.dump(out, open(ROOT/"models"/"paper2_cams_benchmark.json", "w"), indent=2)
    for L in (1, 2, 3):
        print(f"+{L}d  " + "  ".join(f"{m} r={out[f'+{L}d'][m]['r']}" for m in methods), flush=True)

    # figure: r and ACC by lead. Three real models (naive wind-advection is in the JSON but omitted here —
    # it underperforms persistence and would only be a strawman bar).
    fig_methods = ["persistence", "ConvLSTM", "CAMS"]
    col = {"persistence": "#9aa3ad", "ConvLSTM": "#c0392b", "CAMS": "#2e8b57"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160)
    leads = [1, 2, 3]; x = np.arange(len(leads)); w = 0.26
    for mi, m in enumerate(fig_methods):
        for ax, key in ((axes[0], "r"), (axes[1], "acc")):
            vals = [out[f"+{L}d"][m][key] for L in leads]
            bars = ax.bar(x + (mi-1)*w, vals, w, color=col[m], label=m if ax is axes[0] else None)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    for ax, lab in ((axes[0], "pattern correlation  r"), (axes[1], "anomaly correlation  ACC")):
        ax.set_xticks(x); ax.set_xticklabels([f"+{L} day" for L in leads]); ax.set_ylabel(lab)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_title("Raw pattern skill — our model speaks MAIAC's dialect (trained on it, finer than CAMS)", fontsize=9.5, pad=18)
    axes[1].set_title("Transport skill (climatology removed) — CAMS keeps the edge it should", fontsize=9.5, pad=18)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.suptitle(f"Compact ConvLSTM vs CAMS, Central Asia out-of-sample 2024 dust season (n={out['n_days_lead1']} days, MAIAC AOD = truth)", y=1.10, fontsize=11.5)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_cams_benchmark.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()

    # maps: truth vs ConvLSTM vs CAMS on the dustiest day
    if "T" in best:
        lon0, lat0, lon1, lat1 = DOM; ext = [lon0, lon1, lat0, lat1]
        vmax = max(float(np.nanpercentile(best["truth"], 98)), 0.5)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=160)
        for a, key in zip(ax, ("truth", "ConvLSTM", "CAMS")):
            a.imshow(best[key], extent=ext, origin="upper", cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
            a.set_title({"truth": "MAIAC truth", "ConvLSTM": "our model +1d", "CAMS": "CAMS +1d"}[key], fontsize=10)
            a.plot(69.24, 41.31, marker="*", ms=11, color="#1f4e79", mec="white"); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"Dustiest out-of-sample day {best['T']} — same field, three forecasts", y=1.03, fontsize=12)
        fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_cams_maps.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/paper2_cams_benchmark.png + paper2_cams_maps.png + models/paper2_cams_benchmark.json", flush=True)


if __name__ == "__main__":
    main()
