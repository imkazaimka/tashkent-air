"""
LIVE forward forecast — the trained ConvLSTM projects the dust 1-3 days ahead, over its native Tashkent
domain, layered on the live nowcast. Honest scope, all flagged on the figure:
  * small domain only (where the model is valid), not the wide watch box
  * VIIRS (live) fed to a MAIAC-trained model — a sensor transfer
  * advection driven by an Open-Meteo wind/precip FORECAST (uniform per day, v1)
  * modest skill: direction > distance (the under-shoot we measured)

Run (after src/pull_lance.py --region watch):  python src/dust_forecast.py
"""
from __future__ import annotations
import sys, datetime, io, glob
from pathlib import Path
import numpy as np, torch, requests, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convlstm_multimodal as cm

ROOT = Path(__file__).resolve().parent.parent
MD = [55, 37, 75, 47]; H, W = 56, 112               # model native domain + grid
WD = [53, 35, 87, 48]; WSH = (90, 230)              # watch (live data) domain + grid
CITY = ("Tashkent", 69.24, 41.31)


def aod_scale():                                     # normalise by the LIVE sensor's own scale (VIIRS), not MAIAC's
    vals = []
    for f in glob.glob(str(ROOT/"data/satellite/recent_watch/aod/*.npy")):
        a = np.load(f); v = a[a > -900].ravel()
        if v.size: vals.append(v)
    return float(np.percentile(np.concatenate(vals), 97)) if vals else 1.0


def crop_regrid(A):                                  # A is model-domain-cropped already? no — crop here, regrid to HxW, keep gaps
    wlon0, wlat0, wlon1, wlat1 = WD; nlat, nlon = WSH
    c0 = int((MD[0]-wlon0)/(wlon1-wlon0)*nlon); c1 = int((MD[2]-wlon0)/(wlon1-wlon0)*nlon)
    r0 = int((wlat1-MD[3])/(wlat1-wlat0)*nlat); r1 = int((wlat1-MD[1])/(wlat1-wlat0)*nlat)
    sub = A[r0:r1, c0:c1]
    valid = np.isfinite(sub).astype(float); filled = np.where(np.isfinite(sub), sub, 0.0)
    fz = zoom(filled, (H/sub.shape[0], W/sub.shape[1]), order=1)
    vz = zoom(valid, (H/sub.shape[0], W/sub.shape[1]), order=1)
    return np.where(vz > 0.4, fz/np.maximum(vz, 1e-3), np.nan)


def recent_frames(aod_s, n=4):
    fs = sorted(glob.glob(str(ROOT/"data/satellite/recent_watch/aod/*.npy")))
    days = [(datetime.date.fromisoformat(Path(f).stem), np.load(f)) for f in fs]
    dts, mags, masks = [], [], []
    for j in range(max(0, len(days)-n), len(days)):
        d, a = days[j]; A = np.where(a > -900, a.astype(float), np.nan)
        for k in range(j-1, max(j-7, -1), -1):       # gap-fill from prior clear looks
            pr = days[k][1]; m = np.isnan(A) & (pr > -900); A[m] = pr[m]
        g = crop_regrid(A)
        mags.append(np.clip(np.where(np.isfinite(g), g/aod_s, 0.0), 0, 1.5))
        masks.append(np.isfinite(g).astype(np.float32)); dts.append(d)
    return dts, np.array(mags, np.float32), np.array(masks, np.float32)


def weather(n_fc=3):
    lat, lon = (MD[1]+MD[3])/2, (MD[0]+MD[2])/2
    j = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                     f"&hourly=wind_speed_10m,wind_direction_10m,precipitation&past_days=9&forecast_days={n_fc}"
                     f"&timezone=UTC&wind_speed_unit=ms", timeout=30).json()["hourly"]
    df = pd.DataFrame({"t": pd.to_datetime(j["time"]), "ws": j["wind_speed_10m"], "wd": j["wind_direction_10m"], "pr": j["precipitation"]})
    df["date"] = df.t.dt.date
    g = df.groupby("date").agg(ws=("ws", "mean"), wd=("wd", "mean"), pr=("pr", "sum"))
    g["u"] = -g.ws*np.sin(np.radians(g.wd)); g["v"] = -g.ws*np.cos(np.radians(g.wd))
    return g


def gibs(date_str, w=900):
    h = int(w*(MD[3]-MD[1])/(MD[2]-MD[0]))
    url = (f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
           f"&FORMAT=image/png&LAYERS=VIIRS_SNPP_CorrectedReflectance_TrueColor&CRS=EPSG:4326"
           f"&BBOX={MD[1]},{MD[0]},{MD[3]},{MD[2]}&WIDTH={w}&HEIGHT={h}&TIME={date_str}")
    return mpimg.imread(io.BytesIO(requests.get(url, timeout=60).content))


def main():
    aod_s = aod_scale()
    dts, mags, masks = recent_frames(aod_s)
    if len(dts) < 4: print(f"need 4 recent days, have {len(dts)} — run pull_lance --region watch"); return
    wx = weather(); last = dts[-1]
    fc_dates = [last + datetime.timedelta(days=k) for k in range(1, 4)]
    print(f"input days {dts} -> forecast {fc_dates}")

    def uvp(d):
        if d in wx.index: r = wx.loc[d]; return r.u, r.v, r.pr
        return 0.0, 0.0, 0.0
    F = np.zeros((7, 9, H, W), np.float32)
    for t in range(4):
        u, v, pr = uvp(dts[t])
        F[t, 0] = mags[t]; F[t, 1] = masks[t]
        F[t, 6] = np.clip(u/15, -1.5, 1.5); F[t, 7] = np.clip(v/15, -1.5, 1.5); F[t, 8] = np.clip(pr/10, 0, 3)
    for k in range(3):
        u, v, pr = uvp(fc_dates[k])
        F[4+k, 6] = np.clip(u/15, -1.5, 1.5); F[4+k, 7] = np.clip(v/15, -1.5, 1.5); F[4+k, 8] = np.clip(pr/10, 0, 3)

    Ft = torch.from_numpy(F)
    ck = torch.load(ROOT/"models"/"convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(5, 1, 3).to(cm.DEV); net.load_state_dict(ck["ao"]); net.eval()
    with torch.no_grad():
        x = Ft[None, 0:4][:, :, [0, 1, 6, 7, 8]].to(cm.DEV); ex = Ft[None, 4:7][:, :, [6, 7, 8]].to(cm.DEV)
        pred = net(x, ex).cpu().numpy()[0, :, 0]                 # [3, H, W] forecast AOD (amag units)

    ext = [MD[0], MD[2], MD[1], MD[3]]
    try: base = gibs(str(last))
    except Exception: base = None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=170)
    peak = 0.0
    for k in range(3):
        a = ax[k]
        if base is not None: a.imshow(base, extent=ext, origin="upper", aspect="auto", zorder=1)
        else: a.set_facecolor("#0c1b2e")
        up = np.clip(zoom(pred[k], 6, order=3), 0, None)
        rgba = plt.cm.YlOrRd(np.clip(up/0.9, 0, 1)); rgba[..., 3] = np.clip((up-0.45)/0.4, 0, 0.85)   # only predicted DUST shows
        a.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear", aspect="auto", zorder=2)
        a.plot(CITY[1], CITY[2], marker="*", ms=16, color="#7fd4ff", mec="white", mew=1.1, zorder=6)
        a.annotate(CITY[0], (CITY[1], CITY[2]), (CITY[1]+0.3, CITY[2]+0.3), fontsize=9, color="white", fontweight="bold", zorder=6)
        hi = pred[k] > 0.55
        if hi.sum() > 8:
            rr, cc = np.where(hi); lon = MD[0]+cc.mean()/W*(MD[2]-MD[0]); lat = MD[3]-rr.mean()/H*(MD[3]-MD[1])
            a.plot(lon, lat, "X", ms=14, color="#ff2d2d", mec="white", mew=1.2, zorder=6, label="model: dust here")
            if k == 0: a.legend(fontsize=8.5, loc="lower left", framealpha=.85)
            peak = max(peak, float(pred[k].max()))
        a.set_title(f"+{k+1} day  ({fc_dates[k]})", fontsize=11); a.set_xticks([58, 64, 70]); a.set_yticks([39, 43, 47]); a.tick_params(labelsize=7)
    head = "dust forecast" if peak > 0.55 else "outlook: no significant dust over Tashkent"
    fig.suptitle(f"LIVE 1–3 day dust outlook (model, from {last}) — {head}\nmodest skill: trust direction, not exact distance · small domain · VIIRS→model transfer", fontsize=10.5)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"forecast_live.png", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved figures/forecast_live.png  ({head})")


if __name__ == "__main__":
    main()
