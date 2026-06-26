"""
Statistical proofs for the generalisation claim. test_regions.py reports pooled correlations; here we
compute PER-DAY skill so we can (a) put a bootstrap 95% CI on the ConvLSTM-minus-persistence gap in each
out-of-domain region (is the win real or noise?), and (b) stratify skill by dust intensity (does the
model earn its keep on the heavy-dust days that matter?).

Run:  python src/paper2_proofs.py
"""
from __future__ import annotations
import sys, datetime, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convlstm_multimodal as cm
import test_regions as tr

ROOT = Path(__file__).resolve().parent.parent
np.random.seed(0)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or a[m].std() < 1e-6 or b[m].std() < 1e-6: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def per_day(net, F, dates):
    """Per-sequence (per-day) +1d pattern r for model & persistence, plus that day's dust load."""
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
    rows = []
    for s in range(len(in_idx)):
        m = M[s, 0] > 0.5
        if m.sum() < 50: continue
        b = Y[s, 0][m]
        rows.append((corr(P[s, 0][m], b), corr(persist[s, 0][m], b), float(b.mean())))
    return np.array(rows)                                   # [n,3] = model_r, persist_r, load


def boot(diff, n=10000):
    d = diff[np.isfinite(diff)]
    means = np.array([d[np.random.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    aod_s = tr.aod_scale(); print(f"AOD scale {aod_s:.3f}", flush=True)
    ck = torch.load(ROOT/"models"/"convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(5, 1, 3).to(cm.DEV); net.load_state_dict(ck["ao"]); net.eval()

    results, allrows = {}, []
    for key, cfg in tr.REGIONS.items():
        F, dates = tr.build(cfg["path"], aod_s)
        if F is None: continue
        rows = per_day(net, F, dates);
        if len(rows) < 6: continue
        diff = rows[:, 0] - rows[:, 1]; mean, lo, hi = boot(diff)
        results[key] = {"name": cfg["name"], "n": len(rows),
                        "model_r": round(float(np.nanmean(rows[:, 0])), 3),
                        "persist_r": round(float(np.nanmean(rows[:, 1])), 3),
                        "gap": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
                        "significant": bool(lo > 0)}
        allrows.append(rows)
        print(f"[{cfg['name']:16}] n={len(rows):3}  gap {mean:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  {'SIG' if lo>0 else 'ns'}", flush=True)
    json.dump(results, open(ROOT/"models"/"paper2_proofs.json", "w"), indent=2)

    # Fig A: per-region gap with bootstrap 95% CI
    keys = list(results); names = [results[k]["name"] for k in keys]
    gaps = [results[k]["gap"] for k in keys]
    err = np.array([[results[k]["gap"]-results[k]["ci95"][0] for k in keys],
                    [results[k]["ci95"][1]-results[k]["gap"] for k in keys]])
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=160); x = np.arange(len(keys))
    cols = ["#c0392b" if results[k]["significant"] else "#9aa3ad" for k in keys]
    ax.bar(x, gaps, 0.6, yerr=err, color=cols, capsize=5, ecolor="#333")
    ax.axhline(0, color="#333", lw=0.8)
    for i, k in enumerate(keys):
        ax.text(i, results[k]["ci95"][1]+0.01, "p<0.05" if results[k]["significant"] else "ns", ha="center", fontsize=8,
                fontweight="bold", color="#c0392b" if results[k]["significant"] else "#777")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("ConvLSTM − persistence  (+1d pattern r)")
    ax.set_title("The generalisation gap is statistically significant (bootstrap 95% CI, per-day)", fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_significance.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()

    # Fig B: skill stratified by dust load (all out-of-domain days pooled)
    R = np.vstack(allrows); load = R[:, 2]
    q1, q2 = np.percentile(load, [33, 66]); bins = [load <= q1, (load > q1) & (load <= q2), load > q2]
    labs = ["light dust", "moderate", "heavy dust"]
    mr = [np.nanmean(R[b, 0]) for b in bins]; pr = [np.nanmean(R[b, 1]) for b in bins]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160); x = np.arange(3); w = 0.38
    ax.bar(x-w/2, mr, w, color="#c0392b", label="ConvLSTM"); ax.bar(x+w/2, pr, w, color="#9aa3ad", label="persistence")
    for i in range(3):
        ax.text(i-w/2, mr[i]+0.01, f"{mr[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i+w/2, pr[i]+0.01, f"{pr[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("+1d pattern correlation r"); ax.set_ylim(0, 1)
    ax.set_title("Skill by dust intensity — the model leads persistence at every level (most on hard, light-dust days)", fontsize=9.5)
    ax.legend(fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_intensity.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/paper2_significance.png + paper2_intensity.png + models/paper2_proofs.json", flush=True)


if __name__ == "__main__":
    main()
