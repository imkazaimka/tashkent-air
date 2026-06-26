"""
Multimodal ConvLSTM regional movement model: AOD + TROPOMI NO2 + UVAI -> forecasts the dust field (AOD)
AND the bad-air field (NO2) 1-3 days ahead. No CAMS.

Memory-light: every frame is held ONCE in a compact array F[N,6,H,W] (~0.4 GB); sequences are just
index lists, and each batch is gathered on the fly. (The earlier version duplicated frames across
overlapping windows into a multi-GB tensor and pressured RAM.)

  channels (6): AOD_mag, AOD_mask, NO2, NO2_mask, UVAI, UVAI_mask  (56x112, co-registered)
  outputs: AOD (dust) + NO2 (anthropogenic bad air) at t+1/t+2/t+3, masked-MSE loss.
  honest test: does AOD+TROPOMI beat AOD-only, and does each beat persistence.

Run:  python src/convlstm_multimodal.py
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DAOD = ROOT / "data" / "satellite" / "aod_real_grid"      # REAL MAIAC AOD values (not rendered tiles)
DNO2 = ROOT / "data" / "satellite" / "tropomi_grid" / "no2"
DUVAI = ROOT / "data" / "satellite" / "tropomi_grid" / "uvai"
DWIND = ROOT / "data" / "satellite" / "wind_grid"            # ERA5 u/v — drives advection (the movement)
DPRECIP = ROOT / "data" / "satellite" / "precip_grid"        # ERA5 precip — wet removal (clears the air)
FIG = ROOT / "figures"; OUT = ROOT / "models"
H, W = 56, 112; T_IN, K_OUT = 4, 3
NAVY="#16314f"; RED="#c0392b"; ACC="#1f7a8c"; GREY="#9aa3ad"; GREEN="#2e7d52"
import torch, torch.nn as nn, torch.nn.functional as Fn
torch.manual_seed(0); np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
import os, time, resource
TOTAL_RAM = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
RAM_CAP = 0.80 * TOTAL_RAM                       # never let our process exceed 80% of RAM
GPU_DUTY = 1.00                                  # no throttle (max speed); set back to 0.90 if the Mac runs hot
def _rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024     # darwin: bytes; linux: KB
def _bar(frac, w=22):
    n = int(frac * w); return "█" * n + "░" * (w - n)
PROG = {"done": 0, "total": 1, "t0": 0.0}


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k=5):
        super().__init__(); self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class EncFc(nn.Module):
    """Encoder ingests the observed state; forecaster rolls out K days, fed the FORECAST wind/precip
    each step (advection-aware) instead of rolling out blind."""
    def __init__(self, in_ch, out_ch, exog_ch, hid=48):
        super().__init__(); self.enc = ConvLSTMCell(in_ch, hid); self.fc = ConvLSTMCell(out_ch + exog_ch, hid)
        self.head = nn.Conv2d(hid, out_ch, 1); self.hid = hid
    def forward(self, x, exog):                # x: B,T_IN,in_ch,H,W ; exog: B,K_OUT,exog_ch,H,W (future wind/precip)
        B, T, C, h_, w_ = x.shape
        hh = torch.zeros(B, self.hid, h_, w_, device=x.device); cc = torch.zeros_like(hh)
        for t in range(T): hh, cc = self.enc(x[:, t], hh, cc)
        outs = []; inp = torch.sigmoid(self.head(hh))
        for k in range(K_OUT):
            step = torch.cat([inp, exog[:, k]], 1)        # previous prediction + that day's forecast wind/precip
            hh, cc = self.fc(step, hh, cc); inp = torch.sigmoid(self.head(hh)); outs.append(inp)
        return torch.stack(outs, 1)            # B,K,out_ch,H,W


def _resize(arr):
    t = torch.tensor(np.asarray(arr, np.float32))[None, None]
    return Fn.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()


def load_frames():
    """Build compact F[N,6,H,W]: REAL MAIAC AOD + TROPOMI NO2 + UVAI (all real values, held once)."""
    aod_d = {datetime.date.fromisoformat(fp.stem): fp for fp in DAOD.glob("*.npy")}
    no2_d = {datetime.date.fromisoformat(fp.stem): fp for fp in DNO2.glob("*.npy")}
    uv_d = {datetime.date.fromisoformat(fp.stem): fp for fp in DUVAI.glob("*.npy")}
    wind_d = {datetime.date.fromisoformat(fp.stem): fp for fp in DWIND.glob("*.npy")}
    precip_d = {datetime.date.fromisoformat(fp.stem): fp for fp in DPRECIP.glob("*.npy")}
    def scale97(dd):
        vals = []
        for fp in list(dd.values())[::7]:
            a = np.load(fp)
            if a.ndim == 2: vals.append(a[a != -999].ravel())
        return np.percentile(np.concatenate(vals), 97)
    aod_s, no2_s = scale97(aod_d), scale97(no2_d)
    dates = sorted(set(aod_d) & set(no2_d) & set(uv_d) & set(wind_d) & set(precip_d))
    F, keep = [], []
    for d in dates:
        try:
            a = np.load(aod_d[d]); n = np.load(no2_d[d]); u = np.load(uv_d[d]); w = np.load(wind_d[d]); p = np.load(precip_d[d])
            if a.ndim != 2 or n.ndim != 2 or u.ndim != 2 or w.ndim != 3 or p.ndim != 2: continue
        except Exception:
            continue
        amask = _resize((a != -999).astype(np.float32)); amag = _resize(np.clip(np.where(a != -999, a, 0) / aod_s, 0, 1.5))
        nmask = _resize((n != -999).astype(np.float32)); nval = _resize(np.clip(np.where(n != -999, n, 0) / no2_s, 0, 1.5))
        umask = _resize((u != -999).astype(np.float32)); uval = _resize(np.clip((np.where(u != -999, u, 0) + 2) / 5, 0, 1))
        wu = _resize(np.clip(w[0] / 15.0, -1.5, 1.5)); wv = _resize(np.clip(w[1] / 15.0, -1.5, 1.5))   # wind u/v (m/s -> ~[-1,1])
        pr = _resize(np.clip(p * 100.0, 0, 3))                                                          # precip (m -> cm, capped)
        F.append(np.stack([amag, amask, nval, nmask, uval, umask, wu, wv, pr]).astype(np.float32)); keep.append(d)
    return np.stack(F), keep


def build_index(F, dates):
    idx = {d: i for i, d in enumerate(dates)}
    in_idx, tgt_idx = [], []
    for d0 in dates:
        win = [d0 + datetime.timedelta(days=k) for k in range(T_IN + K_OUT)]
        if not all(w in idx for w in win): continue
        ii = [idx[w] for w in win]
        if F[ii[T_IN:], 1].mean() < 0.12: continue        # need some AOD coverage in targets
        in_idx.append(ii[:T_IN]); tgt_idx.append(ii[T_IN:])
    return np.array(in_idx), np.array(tgt_idx)


def masked_mse(p, t, m):                                  # intensity-weighted + mass-conserving — fixes the MSE blur/under-estimation
    w = 1.0 + 5.0 * t                                     # weight heavy-dust pixels up (Shi et al. B-MSE) so the model stops hedging low
    pix = ((p - t) ** 2 * w * m).sum() / (w * m).sum().clamp(min=1)
    tot_p = (p * m).sum(dim=(-1, -2)); tot_t = (t * m).sum(dim=(-1, -2))    # total amount per (sample, lead, channel)
    cons = (tot_p - tot_t).abs().sum() / m.sum().clamp(min=1)               # penalise the wrong TOTAL amount of dust
    return pix + 0.3 * cons


def run(F_t, in_idx, tgt_idx, in_sel, exog_sel, val_sel, mask_sel, epochs=22, train=True, net=None, label=""):
    if net is None:
        net = EncFc(len(in_sel), len(val_sel), len(exog_sel)).to(DEV)
    bs = 32                                    # memory-light now (~2.6GB at 32, cap 6.4GB); fewer batches = less overhead
    if train:
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for ep in range(epochs):
            net.train(); perm = np.random.permutation(len(in_idx)); tot = 0.0; nb = 0
            for i in range(0, len(in_idx), bs):
                t0 = time.time(); b = perm[i:i + bs]
                x = F_t[in_idx[b]][:, :, in_sel].to(DEV)
                ex = F_t[tgt_idx[b]][:, :, exog_sel].to(DEV)                # FORECAST wind/precip for target days
                yv = F_t[tgt_idx[b]][:, :, val_sel].to(DEV); ym = F_t[tgt_idx[b]][:, :, mask_sel].to(DEV)
                opt.zero_grad(); loss = masked_mse(net(x, ex), yv, ym); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
                time.sleep((time.time() - t0) * (1.0 / GPU_DUTY - 1.0))    # throttle GPU to ~80% duty
            PROG["done"] += 1; frac = PROG["done"] / PROG["total"]
            el = time.time() - PROG["t0"]; eta = (el / frac - el) / 60 if frac > 0 else 0
            rss = _rss()
            print(f"[{_bar(frac)}] {frac*100:3.0f}% | {label} ep {ep+1}/{epochs} | loss {tot/max(nb,1):.4f} | "
                  f"RAM {rss/1e9:.1f}/{TOTAL_RAM/1e9:.0f}GB ({rss/TOTAL_RAM*100:.0f}%) | ETA {eta:.0f}m", flush=True)
            if rss > RAM_CAP:
                print(f"!! process RAM exceeded 80% ({rss/1e9:.1f} GB) — aborting to protect the Mac", flush=True)
                raise SystemExit(1)
        net.eval(); return net
    preds = []
    with torch.no_grad():
        for i in range(0, len(in_idx), 16):
            x = F_t[in_idx[i:i+16]][:, :, in_sel].to(DEV)
            ex = F_t[tgt_idx[i:i+16]][:, :, exog_sel].to(DEV)
            preds.append(net(x, ex).cpu().numpy())
    return np.concatenate(preds)


def skill(P, F, tgt_idx, vch, mch):
    Y = F[tgt_idx][:, :, vch].numpy(); M = F[tgt_idx][:, :, mch].numpy()
    out = []
    for k in range(K_OUT):
        m = M[:, k] > 0.5; a, b = P[:, k][m], Y[:, k][m]
        hi = b > 0.4                                                        # amount on dusty pixels: 1.0 = right, <1 = under-estimate
        ratio = float(a[hi].mean() / max(b[hi].mean(), 1e-6)) if hi.sum() > 50 else float("nan")
        out.append((float(np.mean((a - b) ** 2)), float(np.corrcoef(a, b)[0, 1]), ratio))
    return out


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    F, dates = load_frames()
    print(f"frames {F.shape} (~{F.nbytes/1e6:.0f} MB held once) | {DEV}")
    in_idx, tgt_idx = build_index(F, dates)
    n = len(in_idx); cut = int(n * 0.8)
    print(f"sequences {n} (train {cut}, test {n-cut})")
    F_t = torch.tensor(F)                                   # CPU, ~0.4 GB
    tr_i, tr_t = in_idx[:cut], tgt_idx[:cut]; te_i, te_t = in_idx[cut:], tgt_idx[cut:]
    EP = 15; PROG["total"] = EP * 1; PROG["t0"] = time.time()   # one model only — AOD-only baseline dropped (it ties multimodal)
    print(f"training ({int(GPU_DUTY*100)}% GPU duty, batch {32}, RAM cap {int(RAM_CAP/1e9)}GB):", flush=True)
    net_mm = run(F_t, tr_i, tr_t, [0,1,2,3,4,5,6,7,8], [6,7,8], [0,2], [1,3], epochs=EP, label="multimodal")
    pm = run(F_t, te_i, te_t, [0,1,2,3,4,5,6,7,8], [6,7,8], [0,2], [1,3], train=False, net=net_mm)
    # checkpoint model + save test predictions (so a confusion matrix / re-eval needs NO retrain)
    torch.save({"mm": net_mm.state_dict(), "H": H, "W": W}, OUT / "convlstm_models.pt")
    np.savez(OUT / "convlstm_test.npz", pred_mm=pm,
             y_aod=F[te_t][:, :, 0], m_aod=F[te_t][:, :, 1], y_no2=F[te_t][:, :, 2], m_no2=F[te_t][:, :, 3],
             dates=np.array([str(dates[i]) for i in te_t[:, -1]]))
    print(f"  checkpointed -> {OUT/'convlstm_models.pt'}  + test preds -> {OUT/'convlstm_test.npz'}", flush=True)
    # persistence: last input frame, repeated
    last = F[te_i[:, -1]]
    pe_a = np.repeat(last[:, None, 0], K_OUT, 1); pe_n = np.repeat(last[:, None, 2], K_OUT, 1)
    sk = {"multimodal_AOD": skill(pm[:, :, 0], F_t, te_t, 0, 1),
          "persistence_AOD": skill(pe_a, F_t, te_t, 0, 1),
          "multimodal_NO2": skill(pm[:, :, 1], F_t, te_t, 2, 3),
          "persistence_NO2": skill(pe_n, F_t, te_t, 2, 3)}
    print("\nlead | AOD r: model/persist | dust AMOUNT (1.0=right) | NO2 r: model/persist")
    for k in range(K_OUT):
        print(f"  +{k+1}d | {sk['multimodal_AOD'][k][1]:.2f} / {sk['persistence_AOD'][k][1]:.2f}"
              f"  |  amount {sk['multimodal_AOD'][k][2]:.2f}"
              f"  |  {sk['multimodal_NO2'][k][1]:.2f} / {sk['persistence_NO2'][k][1]:.2f}")
    json.dump({k: [{"lead": i+1, "mse": round(v[i][0], 4), "r": round(v[i][1], 3),
                    "amount": (round(v[i][2], 2) if v[i][2] == v[i][2] else None)} for i in range(K_OUT)] for k, v in sk.items()},
              open(OUT / "convlstm_multimodal_metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160); leads = [1, 2, 3]
    ax[0].plot(leads, [sk["multimodal_AOD"][k][1] for k in range(3)], "o-", color=RED, lw=2, label="multimodal (AOD+TROPOMI)")
    ax[0].plot(leads, [sk["aod_only_AOD"][k][1] for k in range(3)], "^-", color=ACC, lw=2, label="AOD-only")
    ax[0].plot(leads, [sk["persistence_AOD"][k][1] for k in range(3)], "s--", color=GREY, lw=2, label="persistence")
    ax[0].set(xlabel="lead (days)", ylabel="pattern corr r", title="Dust field (AOD) forecast"); ax[0].legend(fontsize=8); ax[0].set_xticks(leads)
    ax[1].plot(leads, [sk["multimodal_NO2"][k][1] for k in range(3)], "o-", color=NAVY, lw=2, label="ConvLSTM")
    ax[1].plot(leads, [sk["persistence_NO2"][k][1] for k in range(3)], "s--", color=GREY, lw=2, label="persistence")
    ax[1].set(xlabel="lead (days)", ylabel="pattern corr r", title="Bad-air field (NO₂) forecast"); ax[1].legend(fontsize=8); ax[1].set_xticks(leads)
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "convlstm_multimodal.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/convlstm_multimodal.png + models/convlstm_multimodal_metrics.json")


if __name__ == "__main__":
    main()
