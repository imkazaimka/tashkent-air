"""
Does the model predict where an EXISTING dust event MOVES (transport), separate from just knowing where
dust usually is (climatology)? Raw pattern correlation is climatology-inflated. The honest metric is the
Anomaly Correlation Coefficient (ACC): subtract the monthly climatology from prediction AND truth, then
correlate the anomalies — i.e. did it get the *deviation from normal* (the storm) right. Baseline =
persistence-of-anomaly (assume the storm stays put). If model ACC > persistence ACC, especially at +2/+3
days, it genuinely tracks movement.

Run:  python src/test_transport.py
"""
from __future__ import annotations
import sys, datetime
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import convlstm_multimodal as cm


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    F, dates = cm.load_frames(); print(f"frames {F.shape}")
    months = np.array([d.month for d in dates])
    clim = {}
    for mo in range(1, 13):
        sel = months == mo
        if sel.sum() > 0:
            v = F[sel, 0]; m = F[sel, 1]; clim[mo] = (v * m).sum(0) / np.clip(m.sum(0), 1, None)   # monthly climatology of AOD
    idx = {d: i for i, d in enumerate(dates)}
    cut = int(0.8 * len(dates)); in_idx, tgt_idx = [], []
    for d0 in dates[cut:]:
        win = [d0 + datetime.timedelta(days=k) for k in range(4 + 3)]
        if all(w in idx for w in win):
            ii = [idx[w] for w in win]
            if F[ii[4:], 1].mean() > 0.12: in_idx.append(ii[:4]); tgt_idx.append(ii[4:])
    in_idx = np.array(in_idx); tgt_idx = np.array(tgt_idx); print(f"{len(in_idx)} test sequences")
    F_t = torch.from_numpy(F)
    ck = torch.load(ROOT / "models" / "convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(9, 2, 3).to(cm.DEV); net.load_state_dict(ck["mm"]); net.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(in_idx), 16):
            x = F_t[in_idx[i:i+16]][:, :, [0,1,2,3,4,5,6,7,8]].to(cm.DEV); ex = F_t[tgt_idx[i:i+16]][:, :, [6,7,8]].to(cm.DEV)
            pred.append(net(x, ex).cpu().numpy())
    P = np.concatenate(pred)[:, :, 0]
    Y = F[tgt_idx][:, :, 0]; M = F[tgt_idx][:, :, 1]; last = F[in_idx[:, -1], 0]
    print("\nDoes it track MOVING dust? (held-out)")
    print(f"  {'lead':5} | raw pattern r | ACC (movement) | persistence ACC | gain")
    for k in range(3):
        tmon = np.array([dates[tgt_idx[i, k]].month for i in range(len(tgt_idx))])
        cl = np.stack([clim[m] for m in tmon])
        m = M[:, k] > 0.5
        raw = np.corrcoef(P[:, k][m], Y[:, k][m])[0, 1]
        ap, at = (P[:, k] - cl)[m], (Y[:, k] - cl)[m]; acc = np.corrcoef(ap, at)[0, 1]
        app = (last - cl)[m]; pacc = np.corrcoef(app, at)[0, 1]
        print(f"  +{k+1}d   | {raw:.2f}          | {acc:.2f}           | {pacc:.2f}            | {acc-pacc:+.2f}")

    # ---- visualize a MOVING event: does the predicted dust go where the real dust went? ----
    LON0, LON1, LAT0, LAT1 = 55, 75, 37, 47; H, W = 56, 112
    def centroid(f, mk):                                    # center-of-mass of the dust anomaly
        a = np.clip(f, 0, None) * (mk > 0.5)
        if a.sum() < 1e-3: return None
        r = (a.sum(1) @ np.arange(H)) / a.sum(); c = (a.sum(0) @ np.arange(W)) / a.sum()
        lon = LON0 + c / W * (LON1 - LON0); lat = LAT1 - r / H * (LAT1 - LAT0); return lon, lat
    # pick the event whose TRUE dust centroid moves the most from input to +3d
    best, bmove = None, 0
    for i in range(len(tgt_idx)):
        tmon = clim[dates[tgt_idx[i, 2]].month]
        c0 = centroid(last[i] - clim[dates[in_idx[i, -1]].month], F[in_idx[i, -1], 1])
        c3 = centroid(Y[i, 2] - tmon, M[i, 2])
        if c0 and c3:
            mv = np.hypot(c0[0]-c3[0], c0[1]-c3[1])
            if mv > bmove and (Y[i, 2][M[i,2]>0.5] > 0.6).mean() > 0.05: bmove, best = mv, i
    i = best
    import requests, io
    import matplotlib.image as mpimg
    from scipy.ndimage import zoom
    def gibs(date_str, w=900):                                # true-colour satellite base (display only)
        h = int(w * (LAT1 - LAT0) / (LON1 - LON0))
        url = (f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
               f"&FORMAT=image/png&LAYERS=VIIRS_SNPP_CorrectedReflectance_TrueColor&CRS=EPSG:4326"
               f"&BBOX={LAT0},{LON0},{LAT1},{LON1}&WIDTH={w}&HEIGHT={h}&TIME={date_str}")
        return mpimg.imread(io.BytesIO(requests.get(url, timeout=60).content))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=170); ext = [LON0, LON1, LAT0, LAT1]
    cin = centroid(last[i] - clim[dates[in_idx[i, -1]].month], F[in_idx[i, -1], 1])
    for k in range(3):
        a = ax[k]; mk = M[i, k]; clm = clim[dates[tgt_idx[i, k]].month]
        try: a.imshow(gibs(str(dates[tgt_idx[i, k]])), extent=ext, origin="upper", aspect="auto", zorder=1)
        except Exception: a.set_facecolor("#0c1b2e")
        # the dust storm = dust ABOVE normal (anomaly), smoothed — a discrete plume, not the whole field
        an = np.clip(np.nan_to_num(np.where(mk > 0.5, Y[i, k] - clm, 0.0), nan=0.0), 0, None); up = np.clip(zoom(an, 6, order=3), 0, None)
        rgba = plt.cm.YlOrRd(np.clip(up / 0.5, 0, 1)); rgba[..., 3] = np.clip(up / 0.32, 0, 0.85)
        a.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear", aspect="auto", zorder=2)
        ct = centroid(Y[i, k] - clm, mk); cp = centroid(P[i, k] - clm, mk)
        a.plot(*cin, "o", ms=11, mfc="none", mec="white", mew=2, zorder=5, label="storm start")
        if ct: a.plot(*ct, "o", ms=13, color="#00e5ff", mec="white", mew=1.2, zorder=6, label="actually went here")
        if cp: a.plot(*cp, "X", ms=14, color="#ff2d2d", mec="white", mew=1.2, zorder=6, label="model predicts")
        a.set_title(f"+{k+1} day", fontsize=12); a.set_xticks([58, 64, 70]); a.set_yticks([39, 43, 47]); a.tick_params(labelsize=7)
        if k == 0: a.legend(fontsize=8.5, loc="lower left", framealpha=.85)
    fig.suptitle(f"Where does the storm go? — {dates[tgt_idx[i,0]]}    (orange = the dust storm · ✕ where the MODEL predicts it · ● where it ACTUALLY went)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "forecast_rgb.png", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved figures/forecast_rgb.png  (storm {dates[tgt_idx[i,0]]})")


if __name__ == "__main__":
    main()
