"""
Physics-based LOCAL bad-air engine for Tashkent — the winter combustion + stagnation model.

A transparent well-mixed box (Gifford-Hanna): combustion emissions (proxied by measured GROUND SO2)
pour into a box that clears at a rate set by the VENTILATION COEFFICIENT (wind x mixing height) — the
standard pollution-potential index — plus precip scavenging. CAMS-free, no calendar; it reasons from
the live physical state, with physical exponential decay:

  C_{t+1} = C_t * exp(-[k*VC + d + w*precip])  +  Q * SO2_recent      (VC = wind x mixing height)

Low ventilation -> weak clearing -> accumulation (the winter stagnation mechanism).
Operationally: today's PM + recent SO2 + tomorrow's weather FORECAST -> tomorrow's PM and danger.
Parameters (k, d, w, Q) calibrated to measured municipal PM2.5; evaluated on a held-out split against
persistence and the seasonal-climatology shortcut.

Run:  python src/box_model.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; OUT = ROOT / "models"; THR = 35
NAVY="#16314f"; RED="#c0392b"; ACC="#1f7a8c"; AMBER="#e8a33d"; GREY="#9aa3ad"; GREEN="#2e7d52"
DL = Path.home() / "Downloads"
STATIONS = {"chilanzar": DL / "tashkent-chilanzar-air-quality.csv",
            "embassy": DL / "tashkent-us embassy, uzbekistan-air-quality.csv",
            "yunusabad": DL / "tashkent-yunusabad-air-quality.csv"}


def ground_so2():
    """Pool measured ground SO2 across the 3 WAQI stations (drop Chilanzar-2025 sensor fault)."""
    frames = []
    for name, path in STATIONS.items():
        if not path.exists(): continue
        d = pd.read_csv(path, skipinitialspace=True); d.columns = [c.strip() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"], format="%Y/%m/%d", errors="coerce")
        d["so2"] = pd.to_numeric(d.get("so2"), errors="coerce")
        d = d.dropna(subset=["date", "so2"])[["date", "so2"]]
        if name == "chilanzar":
            d = d[d.date.dt.year != 2025]                    # diagnosed sensor fault
        frames.append(d)
    s = pd.concat(frames).groupby("date", as_index=False)["so2"].mean()
    return s


def load():
    """Full daily calendar so the box steps across CONSECUTIVE days. The WAQI ground-SO2 index is gappy
    but combustion activity is persistent, so it is interpolated over short (<=7 d) gaps; PM/weather over
    very short gaps. A `gap` flag (1 = consecutive) lets the evaluation drop any step that still bridges
    a multi-day break, so the dynamics are never scored across a discontinuity."""
    so2 = ground_so2()
    muni = pd.read_csv(ROOT / "data" / "raw" / "tashkent_municipal_pm25_daily.csv", parse_dates=["date"])[["date", "pm25_muni"]]
    wx = pd.read_csv(ROOT / "data" / "processed" / "weather_tashkent_full.csv"); wx["date"] = pd.to_datetime(wx["date"])
    full = pd.DataFrame({"date": pd.date_range(muni.date.min(), wx.date.max(), freq="D")})
    d = full.merge(muni, on="date", how="left").merge(so2, on="date", how="left").merge(wx, on="date", how="left")
    d["so2"] = d.so2.interpolate(limit=7, limit_direction="both")        # persistent combustion activity
    d["pm25_muni"] = d.pm25_muni.interpolate(limit=2)
    for c in ("wind_speed_10m", "boundary_layer_height", "precipitation"):
        d[c] = d[c].interpolate(limit=3)
    d = d.dropna(subset=["pm25_muni", "so2", "wind_speed_10m", "boundary_layer_height", "precipitation"]).reset_index(drop=True)
    d["gap"] = d.date.diff().dt.days.fillna(99).astype(int)             # 1 = consecutive day; >1 = reset
    d["so2n"] = d.so2 / d.so2.mean()                                    # relative combustion activity
    d["precip_mm"] = d.precipitation
    return d


def box(params, C, SO2, VC, P):
    """1-day-ahead box. C decays at a rate set by the VENTILATION COEFFICIENT (wind x mixing height) —
    the standard pollution-potential index — plus precip scavenging; combustion (ground SO2) is the
    source. Physical exponential decay: low ventilation -> weak clearing -> accumulation (stagnation)."""
    k, d, w, Q = params
    decay = np.exp(-(k * VC + d + w * P))                       # high ventilation/precip -> strong clearing
    emis = Q * np.r_[SO2[0], SO2[:-1]]                          # recent combustion input
    pred = np.empty_like(C); pred[0] = C[0]
    pred[1:] = np.maximum(0, C[:-1] * decay[1:] + emis[1:])
    return pred


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = load()
    print(f"days {len(d)} ({d.date.min().date()}→{d.date.max().date()}) | ground SO2 from {len(STATIONS)} stations")
    C = d.pm25_muni.values; SO2 = d.so2n.values; P = d.precip_mm.values; doy = d.date.dt.dayofyear.values
    VC = d.wind_speed_10m.values * d.boundary_layer_height.values; VC = VC / VC.mean()   # ventilation coefficient (normalised)
    cut = int(len(d) * 0.7); tr = slice(0, cut); te = slice(cut, len(d))
    gap = d.gap.values; run = np.ones(len(d), int)                # consecutive-day run length ending at t
    for i in range(1, len(d)): run[i] = run[i-1] + 1 if gap[i] == 1 else 1
    te_mask = (np.arange(len(d)) >= cut) & (run >= 2)            # held-out, consecutive-day steps only
    # calibrate on train
    def loss(p):                                         # calibrate on the 1-3 day forecast (the use case), not 1 step alone
        pen = 1e7 * sum(max(0.0, -x) for x in p)         # keep physics parameters non-negative
        dec = np.exp(-(p[0]*VC + p[1] + p[2]*P)); em = p[3]*np.r_[SO2[0], SO2[:-1]]
        tot = 0.0; fk = C.astype(float)
        for _ in range(3):                                # accumulate free-running error at leads 1,2,3
            nf = np.empty_like(fk); nf[0] = C[0]; nf[1:] = np.maximum(0, fk[:-1]*dec[1:] + em[1:]); fk = nf
            tot += np.mean((fk[tr] - C[tr]) ** 2)
        return tot / 3 + pen
    res = minimize(loss, [0.15, 0.05, 0.05, 5.0], method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-4, "fatol": 1e-3})
    pred = box(res.x, C, SO2, VC, P)
    persist = np.r_[C[0], C[:-1]]
    cmap = pd.DataFrame({"doy": doy, "C": C}).iloc[tr].groupby("doy").C.mean()
    clim = np.array([cmap.get(x, C[tr].mean()) for x in doy])
    def metr(p, name):
        pt, ct = p[te_mask], C[te_mask]; yb, pb = ct > THR, pt > THR
        r = np.corrcoef(pt, ct)[0, 1]; mae = np.mean(np.abs(pt - ct))
        rec = (yb & pb).sum() / max(yb.sum(), 1); prec = (yb & pb).sum() / max(pb.sum(), 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-6)
        print(f"  {name:30} r={r:.2f}  MAE={mae:.1f}  recall={rec*100:.0f}%  precision={prec*100:.0f}%  F1={f1:.2f}")
        return dict(r=round(float(r),2), mae=round(float(mae),1), recall=round(float(rec),2), precision=round(float(prec),2), f1=round(float(f1),2))
    print("\nTomorrow's PM2.5 (held-out test) — physics box vs the shortcuts:")
    m_clim = metr(clim, "climatology (seasonal avg)"); m_pers = metr(persist, "persistence"); m_box = metr(pred, "BOX MODEL (SO2+weather)")
    # multi-day horizon — free-running box on the weather FORECAST vs persistence (the box's real edge)
    def box_k(params, K):
        kk, dd, ww, Q = params; dec = np.exp(-(kk*VC + dd + ww*P)); em = Q*np.r_[SO2[0], SO2[:-1]]
        pr = np.full(len(C), np.nan)
        for t in range(len(C) - K):
            cc = C[t]
            for j in range(1, K + 1): cc = max(0, cc*dec[t+j] + em[t+j])
            pr[t + K] = cc
        return pr
    print("\nForecast skill by lead (held-out) — box vs persistence:")
    horizon = {}
    for K in (1, 2, 3):
        bk = box_k(res.x, K); pk = np.r_[[np.nan]*K, C[:-K]]
        m = (np.arange(len(C)) >= cut) & ~np.isnan(bk) & ~np.isnan(pk) & (run >= K + 1)
        rb = np.corrcoef(bk[m], C[m])[0,1]; rp = np.corrcoef(pk[m], C[m])[0,1]
        yb = C[m] > THR; recb = ((bk[m] > THR) & yb).sum()/max(yb.sum(),1); recp = ((pk[m] > THR) & yb).sum()/max(yb.sum(),1)
        print(f"  +{K}d | box r={rb:.2f} recall={recb*100:.0f}%   |   persistence r={rp:.2f} recall={recp*100:.0f}%")
        horizon[K] = {"box_r": round(float(rb),2), "persist_r": round(float(rp),2), "box_recall": round(float(recb),2), "persist_recall": round(float(recp),2)}

    # stagnation multiplier (data-measured)
    w = d.iloc[tr]; w = w.assign(VC=w.wind_speed_10m * w.boundary_layer_height)
    win = w[w.date.dt.month.isin([11,12,1,2])]; q = win.VC.quantile([.25,.75])
    mult = win[win.VC <= q[.25]].pm25_muni.median() / max(win[win.VC >= q[.75]].pm25_muni.median(), 1)
    print(f"\n  stagnation multiplier (low vs high ventilation, winter): {mult:.1f}x | fitted vent-k={res.x[0]:.3f} dep={res.x[1]:.3f} washout={res.x[2]:.3f} Q-emis={res.x[3]:.2f}")
    json.dump({"n": len(d), "stagnation_mult": round(float(mult),1),
               "box": m_box, "persistence": m_pers, "climatology": m_clim, "horizon": horizon,
               "params": {"vent_k": round(float(res.x[0]),3), "dep": round(float(res.x[1]),3), "washout": round(float(res.x[2]),4), "Q_emis": round(float(res.x[3]),3)}},
              open(OUT / "box_model_metrics.json", "w"), indent=2)

    # ---- figures ----
    from sklearn.metrics import confusion_matrix
    fig = plt.figure(figsize=(13, 3.8), dpi=160)
    ax1 = fig.add_subplot(1, 3, 1)
    te_d = d.iloc[te]; mask = te_d.date <= te_d.date.min() + pd.Timedelta(days=150)
    ax1.plot(te_d.date[mask], C[te][mask], color=NAVY, lw=1.4, label="measured")
    ax1.plot(te_d.date[mask], pred[te][mask], color=RED, lw=1.4, alpha=.8, label="box forecast")
    ax1.axhline(THR, color=GREY, ls="--", lw=.8); ax1.set_title("Box forecast vs measured (test)", fontsize=10); ax1.legend(fontsize=8); ax1.tick_params(axis="x", labelsize=7, rotation=30)
    ax2 = fig.add_subplot(1, 3, 2)
    names = ["Climatology", "Persistence", "Box model"]; vals = [m_clim["precision"], m_pers["precision"], m_box["precision"]]; cols = [GREY, ACC, RED]
    ax2.bar(range(3), [v*100 for v in vals], color=cols)
    for i, v in enumerate(vals): ax2.text(i, v*100+1.5, f"{v*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(names, fontsize=8); ax2.set_ylabel("dangerous-day precision"); ax2.set_ylim(0, 100); ax2.set_title("Fewer false alarms (mechanism>calendar)", fontsize=10)
    ax3 = fig.add_subplot(1, 3, 3)
    cm = confusion_matrix(C[te_mask] > THR, pred[te_mask] > THR); im = ax3.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax3.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=12, color="white" if cm[i,j]>cm.max()/2 else NAVY)
    ax3.set_xticks([0,1]); ax3.set_xticklabels(["safe","dangerous"]); ax3.set_yticks([0,1]); ax3.set_yticklabels(["safe","dangerous"]); ax3.set_xlabel("forecast"); ax3.set_ylabel("actual"); ax3.set_title(f"Box model danger forecast\nrecall {m_box['recall']*100:.0f}% precision {m_box['precision']*100:.0f}%", fontsize=9)
    for a in (ax1, ax2):
        for s in ("top","right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "box_model.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/box_model.png + models/box_model_metrics.json")


if __name__ == "__main__":
    main()
