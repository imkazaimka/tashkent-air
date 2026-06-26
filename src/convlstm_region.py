"""
Encoder-forecaster ConvLSTM for regional aerosol MOVEMENT forecasting over Central Asia.

Learns, from a sequence of observed daily satellite AOD fields (no CAMS), where the aerosol / dust will
be 1-3 days ahead. This is the precipitation-nowcasting paradigm (Shi et al. 2015) applied to aerosol:
the model learns advection and evolution directly from data.

  input : last T=4 daily AOD fields over the domain (+ retrieval mask channel)
  output: predicted AOD field at t+1, t+2, t+3
  loss  : masked MSE (AOD has cloud gaps -> only train where the target is observed)
  honest baseline: persistence (t+k = t). The ConvLSTM must beat persistence to prove it learned motion.

Run:  python src/convlstm_region.py
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DAOD = ROOT / "data" / "satellite" / "region_aod"
FIG = ROOT / "figures"; OUT = ROOT / "models"
H, W = 56, 112            # training grid (downsampled from 160x320)
T_IN, K_OUT = 4, 3        # input days, forecast days
NAVY="#16314f"; RED="#c0392b"; ACC="#1f7a8c"; GREY="#9aa3ad"

import torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k=5):
        super().__init__(); self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class EncoderForecaster(nn.Module):
    """Encoder ConvLSTM ingests T_IN frames; forecaster ConvLSTM unrolls K_OUT predicted frames."""
    def __init__(self, in_ch=2, hid=48):
        super().__init__()
        self.enc = ConvLSTMCell(in_ch, hid); self.fc = ConvLSTMCell(1, hid)
        self.head = nn.Conv2d(hid, 1, 1); self.hid = hid
    def forward(self, x):                      # x: B,T,in_ch,H,W
        B, T, C, h_, w_ = x.shape
        hh = torch.zeros(B, self.hid, h_, w_, device=x.device); cc = torch.zeros_like(hh)
        for t in range(T):
            hh, cc = self.enc(x[:, t], hh, cc)
        outs = []; inp = torch.sigmoid(self.head(hh))      # first prediction seed
        for k in range(K_OUT):
            hh, cc = self.fc(inp, hh, cc)
            inp = torch.sigmoid(self.head(hh)); outs.append(inp)
        return torch.cat(outs, 1)              # B,K,H,W


def load_fields():
    """Return ordered (dates, mag[N,H,W], mask[N,H,W]) of the observed AOD fields."""
    items = []
    for fp in sorted(DAOD.glob("*.png")):
        try:
            a = np.asarray(Image.open(fp).convert("RGBA").resize((W, H)), float)
        except Exception:
            continue                      # skip a frame still being written by the pull
        mask = (a[..., 3] > 10).astype(np.float32)
        mag = np.where(mask > 0, (255 - a[..., 1]) / 255.0, 0).astype(np.float32)
        items.append((datetime.date.fromisoformat(fp.stem), mag, mask))
    dates = [d for d, _, _ in items]
    return dates, np.stack([m for _, m, _ in items]), np.stack([k for _, _, k in items])


def build(dates, mag, mask):
    """Consecutive-day windows: T_IN inputs -> K_OUT targets. Require all days present & consecutive."""
    idx = {d: i for i, d in enumerate(dates)}
    X, Y, M = [], [], []
    for i, d0 in enumerate(dates):
        win = [d0 + datetime.timedelta(days=k) for k in range(T_IN + K_OUT)]
        if not all(w in idx for w in win):
            continue
        ii = [idx[w] for w in win]
        inp = np.stack([np.stack([mag[j], mask[j]]) for j in ii[:T_IN]])      # T,2,H,W
        tgt = np.stack([mag[j] for j in ii[T_IN:]])                          # K,H,W
        tm = np.stack([mask[j] for j in ii[T_IN:]])                          # K,H,W
        if tm.mean() < 0.15:                                                 # skip near-empty targets
            continue
        X.append(inp); Y.append(tgt); M.append(tm)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(M, np.float32)


def masked_mse(pred, tgt, m):
    return ((pred - tgt) ** 2 * m).sum() / m.sum().clamp(min=1)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dates, mag, mask = load_fields()
    print(f"fields: {len(dates)} ({dates[0]}..{dates[-1]}) | mean coverage {mask.mean():.2f} | {DEV}")
    X, Y, M = build(dates, mag, mask)
    print(f"sequences: {len(X)} (T_IN={T_IN} -> K_OUT={K_OUT}) grid {H}x{W}")
    n = len(X); cut = int(n * 0.8)
    tr = slice(0, cut); te = slice(cut, n)
    net = EncoderForecaster().to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xtr = torch.tensor(X[tr], device=DEV); Ytr = torch.tensor(Y[tr], device=DEV); Mtr = torch.tensor(M[tr], device=DEV)
    bs = 8
    for ep in range(25):
        net.train(); perm = torch.randperm(len(Xtr)); tot = 0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = masked_mse(net(Xtr[b]), Ytr[b], Mtr[b]); loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0 or ep == 24: print(f"  epoch {ep:2d}  train masked-MSE {tot/max(1,len(Xtr)//bs):.4f}", flush=True)
    # ---- eval vs persistence, by lead time ----
    net.eval()
    Xte = torch.tensor(X[te], device=DEV)
    with torch.no_grad():
        pred = np.concatenate([net(Xte[i:i+16]).cpu().numpy() for i in range(0, len(Xte), 16)])
    Yte, Mte = Y[te], M[te]
    persist = np.repeat(X[te][:, -1, 0:1], K_OUT, axis=1)        # last input AOD, repeated
    def skill(P):
        out = []
        for k in range(K_OUT):
            m = Mte[:, k] > 0
            a, b = P[:, k][m], Yte[:, k][m]
            mse = np.mean((a - b) ** 2); r = np.corrcoef(a, b)[0, 1]
            out.append((mse, r))
        return out
    sk_cl, sk_pe = skill(pred), skill(persist)
    print("\nlead | ConvLSTM MSE / r | persistence MSE / r")
    for k in range(K_OUT):
        print(f"  +{k+1}d | {sk_cl[k][0]:.4f} / {sk_cl[k][1]:.2f}   |  {sk_pe[k][0]:.4f} / {sk_pe[k][1]:.2f}")
    json.dump({"n_seq": int(n), "grid": [H, W],
               "convlstm": [{"lead": k+1, "mse": round(float(sk_cl[k][0]),4), "r": round(float(sk_cl[k][1]),3)} for k in range(K_OUT)],
               "persistence": [{"lead": k+1, "mse": round(float(sk_pe[k][0]),4), "r": round(float(sk_pe[k][1]),3)} for k in range(K_OUT)]},
              open(OUT / "convlstm_region_metrics.json", "w"), indent=2)

    # ---- figure: skill vs lead + an example forecast ----
    fig = plt.figure(figsize=(12, 5.5), dpi=160)
    axs = fig.add_subplot(2, 4, 1)
    leads = [1, 2, 3]
    axs.plot(leads, [sk_cl[k][1] for k in range(3)], "o-", color=RED, lw=2, label="ConvLSTM")
    axs.plot(leads, [sk_pe[k][1] for k in range(3)], "s--", color=GREY, lw=2, label="persistence")
    axs.set(xlabel="lead (days)", ylabel="pattern corr r", title="Forecast skill vs lead"); axs.legend(fontsize=8); axs.set_xticks(leads)
    for s in ("top", "right"): axs.spines[s].set_visible(False)
    # pick a test example with a strong target
    ex = int(np.argmax([Mte[i].mean() * Yte[i].mean() for i in range(len(Yte))]))
    for j in range(T_IN):
        ax = fig.add_subplot(2, 4, 2 + j) if j < 3 else None
    # show last input, then pred vs actual for +1,+2,+3
    panels = [("input t", X[te][ex, -1, 0])] + \
             [(f"pred +{k+1}", pred[ex, k]) for k in range(3)] + \
             [(f"actual +{k+1}", Yte[ex, k]) for k in range(3)]
    for p, (title, fld) in enumerate(panels):
        ax = fig.add_subplot(2, 4, p + 2 if p < 3 else p + 2)
    # simpler: dedicated grid
    fig.clf()
    gs = fig.add_gridspec(2, 4)
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(leads, [sk_cl[k][1] for k in range(3)], "o-", color=RED, lw=2, label="ConvLSTM")
    axA.plot(leads, [sk_pe[k][1] for k in range(3)], "s--", color=GREY, lw=2, label="persistence")
    axA.set(xlabel="lead (days)", ylabel="pattern corr r", title="Skill vs lead"); axA.legend(fontsize=8); axA.set_xticks(leads)
    for s in ("top", "right"): axA.spines[s].set_visible(False)
    axI = fig.add_subplot(gs[1, 0]); axI.imshow(X[te][ex, -1, 0], cmap="YlOrRd", vmin=0, vmax=1); axI.set_title("last input", fontsize=8); axI.axis("off")
    for k in range(3):
        axp = fig.add_subplot(gs[0, k + 1]); axp.imshow(pred[ex, k], cmap="YlOrRd", vmin=0, vmax=1); axp.set_title(f"ConvLSTM +{k+1}d", fontsize=8); axp.axis("off")
        axt = fig.add_subplot(gs[1, k + 1]); axt.imshow(Yte[ex, k], cmap="YlOrRd", vmin=0, vmax=1); axt.set_title(f"actual +{k+1}d", fontsize=8); axt.axis("off")
    fig.suptitle("Regional aerosol movement forecast (ConvLSTM) — predicted vs actual", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "convlstm_region.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("\nsaved figures/convlstm_region.png + models/convlstm_region_metrics.json")


if __name__ == "__main__":
    main()
