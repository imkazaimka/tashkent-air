"""
Roll the trained ConvLSTM out to 7 days (it was trained on 3) to see how far skill holds — for dust
(AOD) and bad air (NO2). The forecaster is a shared-weight loop, so extending the rollout is valid; steps
4-7 are pure extrapolation beyond the training horizon. Exog = OBSERVED future wind/precip (an optimistic
upper bound; a real run uses a degrading weather forecast). Held-out test split. Skill = pattern r vs
persistence (the day-0 field held fixed).

Run:  python src/test_7day.py
"""
from __future__ import annotations
import sys, datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import convlstm_multimodal as cm
LEADS = 7


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cm.K_OUT = LEADS                                   # extend the shared-weight forecaster rollout to 7 steps
    F, dates = cm.load_frames(); print(f"frames {F.shape}")
    idx = {d: i for i, d in enumerate(dates)}
    cut = int(0.8 * len(dates))                        # held-out = last 20% (chronological, as in training)
    in_idx, tgt_idx = [], []
    for d0 in dates[cut:]:
        win = [d0 + datetime.timedelta(days=k) for k in range(4 + LEADS)]
        if all(w in idx for w in win):
            ii = [idx[w] for w in win]
            if F[ii[4:], 1].mean() > 0.12:
                in_idx.append(ii[:4]); tgt_idx.append(ii[4:])
    in_idx = np.array(in_idx); tgt_idx = np.array(tgt_idx); print(f"{len(in_idx)} 11-day held-out windows")
    train_frames = np.array([idx[d] for d in dates[:cut]])         # climatology from the TRAIN period only
    def clim_field(vch, mch):
        v = F[train_frames, vch]; m = F[train_frames, mch]
        return (v * m).sum(0) / np.clip(m.sum(0), 1, None)         # static mean map [H,W]
    clim_aod, clim_no2 = clim_field(0, 1), clim_field(2, 3)
    F_t = torch.from_numpy(F)
    ck = torch.load(ROOT / "models" / "convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(9, 2, 3).to(cm.DEV); net.load_state_dict(ck["mm"]); net.eval()
    in_sel, exog_sel = [0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8]
    preds = []
    with torch.no_grad():
        for i in range(0, len(in_idx), 8):
            x = F_t[in_idx[i:i+8]][:, :, in_sel].to(cm.DEV)
            ex = F_t[tgt_idx[i:i+8]][:, :, exog_sel].to(cm.DEV)     # 7 days of (observed) wind/precip
            preds.append(net(x, ex).cpu().numpy())
    P = np.concatenate(preds)                                       # [N, 7, 2, H, W]
    last = F[in_idx[:, -1]]

    def ev(pidx, vch, mch, name, clim):
        Y = F[tgt_idx][:, :, vch]; M = F[tgt_idx][:, :, mch]; rm, rp, rc = [], [], []
        for k in range(LEADS):
            m = M[:, k] > 0.5; a, b = P[:, k, pidx][m], Y[:, k][m]
            pp = np.repeat(last[:, None, vch], LEADS, 1)[:, k][m]
            cc = np.broadcast_to(clim, Y[:, k].shape)[m]                       # static climatology map, same every lead
            rm.append(float(np.corrcoef(a, b)[0, 1])); rp.append(float(np.corrcoef(pp, b)[0, 1])); rc.append(float(np.corrcoef(cc, b)[0, 1]))
        print(f"  {name:14} MODEL      r 1-7: " + "  ".join(f"{x:.2f}" for x in rm))
        print(f"  {name:14} persistence  1-7: " + "  ".join(f"{x:.2f}" for x in rp))
        print(f"  {name:14} CLIMATOLOGY  1-7: " + "  ".join(f"{x:.2f}" for x in rc) + "   <- static average map")
        return rm, rp, rc
    print("\n7-day rollout (held-out) — pattern r by lead:")
    aod_m, aod_p, aod_c = ev(0, 0, 1, "DUST (AOD)", clim_aod)
    no2_m, no2_p, no2_c = ev(1, 2, 3, "BAD AIR (NO2)", clim_no2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=160); xs = list(range(1, LEADS + 1))
    for a, (m, p, cl, t, c) in zip(ax, [(aod_m, aod_p, aod_c, "Dust (AOD) forecast", "#c0392b"), (no2_m, no2_p, no2_c, "Bad-air (NO₂) forecast", "#16314f")]):
        a.plot(xs, m, "o-", color=c, lw=2, label="ConvLSTM"); a.plot(xs, p, "s--", color="#9aa3ad", lw=2, label="persistence")
        a.plot(xs, cl, ":", color="#e8a33d", lw=2.2, label="climatology (avg map)")
        a.axvspan(0.5, 3.5, color="#2e7d52", alpha=.07); a.text(2, a.get_ylim()[0], "trained\nhorizon", fontsize=7, color="#2e7d52", ha="center", va="bottom")
        a.axhline(0, color="k", lw=.5); a.set(xlabel="lead (days)", ylabel="pattern corr r", title=t, ylim=(-0.05, 0.85)); a.set_xticks(xs); a.legend(fontsize=8)
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("How far does skill hold? Rolling the 3-day-trained model out to 7 days (held-out)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "test_7day.png", dpi=160, bbox_inches="tight", facecolor="white")
    print("saved figures/test_7day.png")


if __name__ == "__main__":
    main()
