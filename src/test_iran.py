"""
Out-of-domain generalization test: run the Central-Asia-trained dust ConvLSTM on IRAN — a region it has
never seen, with different sources (Tigris-Euphrates, the central deserts, Sistan), terrain and climate.
Uses the saved AOD-only model and the SAME training AOD scale, so it is a true transfer test. If the
model beats persistence here, it learned transferable dust dynamics rather than memorising Central Asia.

Run:  python src/test_iran.py
"""
from __future__ import annotations
import sys, datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import convlstm_multimodal as cm                       # EncFc, _resize, T_IN, K_OUT, H, W, DEV, DAOD
IR = ROOT / "data" / "satellite" / "iran"; FIG = ROOT / "figures"
DOM = [46, 27, 66, 37]                                 # lon0, lat0, lon1, lat1


def scale97_ca():
    """The TRAINING AOD scale (97th pct of Central Asia AOD) — reuse it so Iran is fed at the model's scale."""
    vals = []
    for fp in list(cm.DAOD.glob("*.npy"))[::7]:
        a = np.load(fp)
        if a.ndim == 2: vals.append(a[a != -999].ravel())
    return float(np.percentile(np.concatenate(vals), 97))


def build_iran(aod_s):
    aod_d = {datetime.date.fromisoformat(fp.stem): fp for fp in (IR/"aod").glob("*.npy")}
    wind_d = {datetime.date.fromisoformat(fp.stem): fp for fp in (IR/"wind").glob("*.npy")}
    pre_d = {datetime.date.fromisoformat(fp.stem): fp for fp in (IR/"precip").glob("*.npy")}
    dates = sorted(set(aod_d) & set(wind_d) & set(pre_d)); F, keep = [], []
    for d in dates:
        try:
            a = np.load(aod_d[d]); w = np.load(wind_d[d]); p = np.load(pre_d[d])
            if a.ndim != 2 or w.ndim != 3 or p.ndim != 2: continue
        except Exception:
            continue
        amask = cm._resize((a != -999).astype(np.float32)); amag = cm._resize(np.clip(np.where(a != -999, a, 0)/aod_s, 0, 1.5))
        z = np.zeros_like(amag)                                                   # NO2/UVAI channels unused by AOD-only model
        wu = cm._resize(np.clip(w[0]/15.0, -1.5, 1.5)); wv = cm._resize(np.clip(w[1]/15.0, -1.5, 1.5))
        pr = cm._resize(np.clip(p*100.0, 0, 3))
        F.append(np.stack([amag, amask, z, z, z, z, wu, wv, pr]).astype(np.float32)); keep.append(d)
    return np.stack(F), keep


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    aod_s = scale97_ca(); print(f"reusing training AOD scale aod_s={aod_s:.3f}")
    F, dates = build_iran(aod_s); print(f"Iran frames {F.shape} ({dates[0]}->{dates[-1]})")
    idx = {d: i for i, d in enumerate(dates)}; in_idx, tgt_idx = [], []
    for d0 in dates:
        win = [d0 + datetime.timedelta(days=k) for k in range(cm.T_IN + cm.K_OUT)]
        if all(w in idx for w in win):
            ii = [idx[w] for w in win]
            if F[ii[cm.T_IN:], 1].mean() > 0.12:
                in_idx.append(ii[:cm.T_IN]); tgt_idx.append(ii[cm.T_IN:])
    in_idx = np.array(in_idx); tgt_idx = np.array(tgt_idx); print(f"{len(in_idx)} valid test sequences")
    F_t = torch.from_numpy(F)
    ck = torch.load(ROOT/"models"/"convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(5, 1, 3).to(cm.DEV); net.load_state_dict(ck["ao"]); net.eval()
    in_sel, exog_sel = [0, 1, 6, 7, 8], [6, 7, 8]
    preds = []
    with torch.no_grad():
        for i in range(0, len(in_idx), 16):
            x = F_t[in_idx[i:i+16]][:, :, in_sel].to(cm.DEV); ex = F_t[tgt_idx[i:i+16]][:, :, exog_sel].to(cm.DEV)
            preds.append(net(x, ex).cpu().numpy())
    P = np.concatenate(preds)[:, :, 0]
    Y = F[tgt_idx][:, :, 0]; M = F[tgt_idx][:, :, 1]
    persist = np.repeat(F[in_idx[:, -1]][:, None, 0], cm.K_OUT, 1)
    print("\nIRAN (out-of-domain) dust-field forecast  —  pattern r and amount:")
    print(f"  {'lead':5} | model r / persistence r | dust amount (1.0=right)")
    for k in range(cm.K_OUT):
        m = M[:, k] > 0.5; a, b, pp = P[:, k][m], Y[:, k][m], persist[:, k][m]
        rm = np.corrcoef(a, b)[0, 1]; rp = np.corrcoef(pp, b)[0, 1]
        hi = b > 0.4; amt = (a[hi].mean()/max(b[hi].mean(), 1e-6)) if hi.sum() > 50 else float("nan")
        print(f"  +{k+1}d   | {rm:.2f} / {rp:.2f}            | {amt:.2f}")

    # ---- visualize the dustiest held-out Iran day: truth (top) vs model forecast (bottom) ----
    load = [Y[i][M[i] > 0].mean() if (M[i] > 0).any() else 0 for i in range(len(Y))]
    i = int(np.argsort(load)[-1]); d0 = dates[tgt_idx[i, 0]]
    ext = [DOM[0], DOM[2], DOM[1], DOM[3]]; vmax = max(float(np.percentile(Y[i][M[i] > 0], 98)), 0.6)
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2), dpi=160)
    def draw(a, fld, mask, title, corr=None):
        f = fld.copy(); f[mask <= 0] = np.nan
        im = a.imshow(f, extent=ext, origin="upper", cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
        a.imshow(np.where(mask > 0, np.nan, 1), extent=ext, origin="upper", cmap="Greys", vmin=0, vmax=3, aspect="auto")
        a.plot(51.4, 35.7, marker="*", ms=13, color="#1f4e8c", mec="white", mew=.8); a.annotate("Tehran", (51.4, 35.7), (51.8, 35.9), fontsize=7, color="#1f4e8c", fontweight="bold")
        a.text(47.5, 31.5, "Tigris–\nEuphrates", fontsize=7, color="#5b4636", ha="center", style="italic")
        a.text(56, 34, "central\ndeserts", fontsize=7, color="#5b4636", ha="center", style="italic")
        a.text(61.5, 30.5, "Sistan", fontsize=7, color="#5b4636", ha="center", style="italic")
        a.set_title(title, fontsize=9.5); a.set_xticks([48, 54, 60, 66]); a.set_yticks([28, 32, 36]); a.tick_params(labelsize=6)
        if corr is not None: a.text(.97, .04, f"r = {corr:.2f}", transform=a.transAxes, ha="right", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=.85))
        return im
    for k in range(3):
        mk = M[i, k]; v = mk > 0; corr = np.corrcoef(P[i, k][v], Y[i, k][v])[0, 1]
        draw(ax[0, k], Y[i, k], mk, f"SATELLITE TRUTH  +{k+1} day")
        im = draw(ax[1, k], P[i, k], mk, f"MODEL FORECAST  +{k+1} day", corr)
    fig.suptitle(f"Out-of-domain test — IRAN, held-out dust day {d0} (model trained only on Central Asia)\nTop: satellite truth.  Bottom: ConvLSTM forecast. Does the learned dust motion transfer?", fontsize=11, y=1.02)
    fig.colorbar(im, ax=ax, fraction=.025, pad=.01).set_label("aerosol optical depth (dust)", fontsize=8)
    fig.savefig(FIG/"dust_test_iran.png", dpi=160, bbox_inches="tight", facecolor="white"); print(f"saved figures/dust_test_iran.png (event {d0})")


if __name__ == "__main__":
    main()
