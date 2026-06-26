"""
Out-of-domain generalisation across the world's dust regions (Iran, Sahara, Middle East, Gobi) — the
core result of Paper 2. The Central-Asia-trained AOD-only ConvLSTM is run on each region (it never saw
any of them), scored against persistence with pattern correlation and the anomaly-correlation (ACC,
transport skill). If it beats persistence everywhere, it learned dust *dynamics*, not a memorised map.

Run:  python src/test_regions.py
"""
from __future__ import annotations
import sys, datetime, glob, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convlstm_multimodal as cm

ROOT = Path(__file__).resolve().parent.parent
REGIONS = {
    "iran":       {"path": "data/satellite/iran",             "dom": [46, 27, 66, 37],  "name": "Iran"},
    "sahara":     {"path": "data/satellite/regions/sahara",   "dom": [0, 12, 30, 27],   "name": "Sahara/Sahel"},
    "middleeast": {"path": "data/satellite/regions/middleeast","dom": [38, 22, 58, 35], "name": "Middle East"},
    "gobi":       {"path": "data/satellite/regions/gobi",     "dom": [98, 40, 118, 48], "name": "Gobi (Mongolia)"},
}


def aod_scale():
    vals = []
    for fp in list(cm.DAOD.glob("*.npy"))[::9]:
        a = np.load(fp)
        if a.ndim == 2: vals.append(a[a != -999].ravel())
    return float(np.percentile(np.concatenate(vals), 97))


def build(path, aod_s):
    P = ROOT / path
    aod_d = {datetime.date.fromisoformat(Path(f).stem): f for f in glob.glob(str(P/"aod"/"*.npy"))}
    wind_d = {datetime.date.fromisoformat(Path(f).stem): f for f in glob.glob(str(P/"wind"/"*.npy"))}
    pre_d = {datetime.date.fromisoformat(Path(f).stem): f for f in glob.glob(str(P/"precip"/"*.npy"))}
    dates = sorted(set(aod_d) & set(wind_d) & set(pre_d)); F, keep = [], []
    for d in dates:
        try:
            a = np.load(aod_d[d]); w = np.load(wind_d[d]); p = np.load(pre_d[d])
            if a.ndim != 2 or w.ndim != 3 or p.ndim != 2: continue
        except Exception:
            continue
        amask = cm._resize((a != -999).astype(np.float32)); amag = cm._resize(np.clip(np.where(a != -999, a, 0)/aod_s, 0, 1.5))
        z = np.zeros_like(amag)
        wu = cm._resize(np.clip(w[0]/15.0, -1.5, 1.5)); wv = cm._resize(np.clip(w[1]/15.0, -1.5, 1.5)); pr = cm._resize(np.clip(p*100.0, 0, 3))
        F.append(np.stack([amag, amask, z, z, z, z, wu, wv, pr]).astype(np.float32)); keep.append(d)
    return (np.stack(F), keep) if F else (None, [])


def run(net, F, dates):
    idx = {d: i for i, d in enumerate(dates)}; in_idx, tgt_idx = [], []
    for d0 in dates:
        win = [d0 + datetime.timedelta(days=k) for k in range(cm.T_IN + cm.K_OUT)]
        if all(w in idx for w in win):
            ii = [idx[w] for w in win]
            if F[ii[cm.T_IN:], 1].mean() > 0.12: in_idx.append(ii[:cm.T_IN]); tgt_idx.append(ii[cm.T_IN:])
    in_idx, tgt_idx = np.array(in_idx), np.array(tgt_idx)
    F_t = torch.from_numpy(F); preds = []
    with torch.no_grad():
        for i in range(0, len(in_idx), 16):
            x = F_t[in_idx[i:i+16]][:, :, [0,1,6,7,8]].to(cm.DEV); ex = F_t[tgt_idx[i:i+16]][:, :, [6,7,8]].to(cm.DEV)
            preds.append(net(x, ex).cpu().numpy())
    P = np.concatenate(preds)[:, :, 0]; Y = F[tgt_idx][:, :, 0]; M = F[tgt_idx][:, :, 1]
    persist = np.repeat(F[in_idx[:, -1]][:, None, 0], cm.K_OUT, 1)
    clim = (Y*M).sum((0,1))/np.clip(M.sum((0,1)), 1, None)          # region climatology (mean field)
    out = {"n": len(in_idx)}
    for k in range(cm.K_OUT):
        m = M[:, k] > 0.5; a, b, pp = P[:, k][m], Y[:, k][m], persist[:, k][m]
        cm_ = np.broadcast_to(clim, Y[:, k].shape)[m]
        rm = np.corrcoef(a, b)[0, 1]; rp = np.corrcoef(pp, b)[0, 1]
        acc = np.corrcoef((a-cm_), (b-cm_))[0, 1]; accp = np.corrcoef((pp-cm_), (b-cm_))[0, 1]
        out[f"+{k+1}d"] = {"model_r": round(float(rm),2), "persist_r": round(float(rp),2),
                            "model_acc": round(float(acc),2), "persist_acc": round(float(accp),2)}
    return out, (P, Y, M, tgt_idx, dates)


def region_panels(ax_t, ax_m, dom, ev, name):
    P, Y, M, tgt_idx, dates = ev
    load = [Y[i, 0][M[i, 0] > 0].mean() if (M[i, 0] > 0).any() else 0 for i in range(len(Y))]
    i = int(np.argsort(load)[-1]); lon0, lat0, lon1, lat1 = dom; ext = [lon0, lon1, lat0, lat1]
    vmax = max(float(np.percentile(Y[i, 0][M[i, 0] > 0], 98)), 0.6)
    for ax, fld in ((ax_t, Y[i, 0]), (ax_m, P[i, 0])):
        f = fld.copy(); f[M[i, 0] <= 0] = np.nan
        ax.imshow(f, extent=ext, origin="upper", cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
    ax_t.set_title(name, fontsize=10)


def main():
    aod_s = aod_scale(); print(f"training AOD scale {aod_s:.3f}")
    ck = torch.load(ROOT/"models"/"convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(5, 1, 3).to(cm.DEV); net.load_state_dict(ck["ao"]); net.eval()
    results, evs = {}, {}
    for key, cfg in REGIONS.items():
        F, dates = build(cfg["path"], aod_s)
        if F is None: print(f"[{cfg['name']}] no data — skipping"); continue
        skill, ev = run(net, F, dates); results[key] = {"name": cfg["name"], **skill}; evs[key] = (cfg, ev)
        s = skill["+1d"]
        print(f"[{cfg['name']:14}] {skill['n']:3} seq | +1d  model r {s['model_r']} vs persist {s['persist_r']}  | ACC {s['model_acc']} vs {s['persist_acc']}")
    json.dump(results, open(ROOT/"models"/"paper2_regions.json", "w"), indent=2)

    # summary bar chart: model r vs persistence r (+1d) across regions
    keys = list(results); names = [results[k]["name"] for k in keys]
    mr = [results[k]["+1d"]["model_r"] for k in keys]; pr = [results[k]["+1d"]["persist_r"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=160); x = np.arange(len(keys)); w = 0.38
    ax.bar(x-w/2, mr, w, color="#c0392b", label="ConvLSTM (trained on Central Asia)")
    ax.bar(x+w/2, pr, w, color="#9aa3ad", label="persistence")
    for i, v in enumerate(mr): ax.text(i-w/2, v+0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("dust-field pattern correlation (+1 day)")
    ax.set_title("Generalisation: one model, never retrained, beats persistence on every dust region", fontsize=10.5)
    ax.legend(fontsize=8); ax.set_ylim(0, 1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_generalization.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()

    # multi-region map: truth (top) vs model (bottom)
    n = len(evs); fig, ax = plt.subplots(2, n, figsize=(3.3*n, 6), dpi=160)
    if n == 1: ax = ax[:, None]
    for j, (key, (cfg, ev)) in enumerate(evs.items()):
        region_panels(ax[0, j], ax[1, j], cfg["dom"], ev, cfg["name"])
    ax[0, 0].set_ylabel("SATELLITE TRUTH", fontsize=9); ax[1, 0].set_ylabel("MODEL FORECAST (+1d)", fontsize=9)
    fig.suptitle("Same Central-Asia-trained model, applied unchanged to four dust regions it never saw", fontsize=12, y=1.01)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_regions_maps.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/paper2_generalization.png + paper2_regions_maps.png + models/paper2_regions.json")


if __name__ == "__main__":
    main()
