"""
Real NASA true-colour satellite imagery of the dust events the model was tested on — proof that the
generalisation regions (§5) carried genuine dust storms, not retrieval noise. Pulls VIIRS/MODIS
corrected-reflectance true-colour from NASA GIBS (no auth) for each region's dustiest day, and a
satellite-vs-model strip for one event.

Run:  python src/satellite_images.py
"""
from __future__ import annotations
import sys, io, datetime, glob
from pathlib import Path
import numpy as np, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convlstm_multimodal as cm
import torch

ROOT = Path(__file__).resolve().parent.parent
REGIONS = {                                                  # dom=[lon0,lat0,lon1,lat1], dustiest day, label
    "sahara":     {"dom": [0, 12, 30, 27],   "date": "2022-06-27", "name": "Sahara / Sahel",  "city": None},
    "middleeast": {"dom": [38, 22, 58, 35],  "date": "2022-07-05", "name": "Middle East",     "city": ("Baghdad", 44.36, 33.31)},
    "iran":       {"dom": [46, 27, 66, 37],  "date": "2022-07-04", "name": "Iran",            "city": ("Tehran", 51.39, 35.69)},
    "gobi":       {"dom": [98, 40, 118, 48], "date": "2022-04-25", "name": "Gobi (Mongolia)", "city": None},
}
LAYERS = ["VIIRS_SNPP_CorrectedReflectance_TrueColor", "MODIS_Aqua_CorrectedReflectance_TrueColor",
          "MODIS_Terra_CorrectedReflectance_TrueColor"]


def gibs(dom, date, w=1100):
    lon0, lat0, lon1, lat1 = dom; h = int(w*(lat1-lat0)/(lon1-lon0))
    for layer in LAYERS:
        url = (f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
               f"&FORMAT=image/png&LAYERS={layer}&CRS=EPSG:4326"
               f"&BBOX={lat0},{lon0},{lat1},{lon1}&WIDTH={w}&HEIGHT={h}&TIME={date}")
        try:
            img = mpimg.imread(io.BytesIO(requests.get(url, timeout=60).content))
            if img.ndim == 3 and float(img[..., :3].mean()) > 0.05: return img, layer.split("_")[0]
        except Exception:
            continue
    return None, None


def grid_figure():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), dpi=160)
    for ax, (key, cfg) in zip(axes.ravel(), REGIONS.items()):
        lon0, lat0, lon1, lat1 = cfg["dom"]; ext = [lon0, lon1, lat0, lat1]
        img, src = gibs(cfg["dom"], cfg["date"])
        if img is None:
            ax.text(0.5, 0.5, "imagery unavailable", ha="center"); ax.set_axis_off(); continue
        ax.imshow(img, extent=ext, origin="upper", aspect="auto")
        if cfg["city"]:
            nm, lo, la = cfg["city"]
            ax.plot(lo, la, marker="*", ms=13, color="#7fd4ff", mec="white", mew=1.2)
            ax.text(lo+0.4, la+0.4, nm, color="white", fontsize=9, fontweight="bold")
        ax.set_title(f"{cfg['name']} · {cfg['date']}", fontsize=11)
        ax.text(0.015, 0.03, f"NASA {src} true-colour", transform=ax.transAxes, color="white",
                fontsize=7.5, alpha=0.85, bbox=dict(boxstyle="round", fc="#0008", ec="none"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Real dust storms the model was tested on — NASA true-colour imagery of each region's peak-dust day",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_satellite.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/paper2_satellite.png")


def sat_vs_model():
    """For the Iran event: real satellite image | MAIAC AOD truth | model +1d forecast, all aligned."""
    cfg = REGIONS["iran"]; dom = cfg["dom"]; lon0, lat0, lon1, lat1 = dom; ext = [lon0, lon1, lat0, lat1]
    img, src = gibs(dom, cfg["date"])
    P = Path("data/satellite/iran")
    aod_s = float(np.percentile(np.concatenate([np.load(f)[np.load(f) != -999].ravel()
              for f in list(cm.DAOD.glob("*.npy"))[::9] if np.load(f).ndim == 2]), 97))
    # build the model input for the day before, forecast the dust day
    tgt = datetime.date.fromisoformat(cfg["date"]); ins = [tgt - datetime.timedelta(days=k) for k in range(cm.T_IN, 0, -1)]
    def frame(d):
        a = np.load(P/"aod"/f"{d}.npy"); w = np.load(P/"wind"/f"{d}.npy"); pr = np.load(P/"precip"/f"{d}.npy")
        amag = cm._resize(np.clip(np.where(a != -999, a, 0)/aod_s, 0, 1.5)); amask = cm._resize((a != -999).astype(np.float32))
        wu = cm._resize(np.clip(w[0]/15, -1.5, 1.5)); wv = cm._resize(np.clip(w[1]/15, -1.5, 1.5)); p = cm._resize(np.clip(pr*100, 0, 3))
        return np.stack([amag, amask, wu, wv, p]).astype(np.float32)
    try:
        seq = np.stack([frame(d) for d in ins])
    except Exception:
        print("sat_vs_model: missing input days, skipping"); return
    ck = torch.load(ROOT/"models"/"convlstm_models.pt", map_location=cm.DEV)
    net = cm.EncFc(5, 1, 3).to(cm.DEV); net.load_state_dict(ck["ao"]); net.eval()
    ex = torch.from_numpy(seq[-1:, 2:5][None].repeat(cm.K_OUT, 1)).to(cm.DEV) if False else \
         torch.from_numpy(np.repeat(seq[-1:, 2:5], cm.K_OUT, 0)[None]).to(cm.DEV)
    with torch.no_grad():
        pred = net(torch.from_numpy(seq[None]).to(cm.DEV), ex).cpu().numpy()[0, 0, 0]
    truth = np.load(P/"aod"/f"{cfg['date']}.npy"); truth = cm._resize(np.where(truth != -999, truth, np.nan))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), dpi=160)
    if img is not None: ax[0].imshow(img, extent=ext, origin="upper", aspect="auto")
    ax[0].set_title(f"NASA {src} true-colour\n{cfg['name']} · {cfg['date']}", fontsize=10)
    vmax = float(np.nanpercentile(truth, 98))
    ax[1].imshow(truth, extent=ext, origin="upper", cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto"); ax[1].set_title("MAIAC AOD (truth)", fontsize=10)
    ax[2].imshow(np.clip(pred, 0, None)*aod_s, extent=ext, origin="upper", cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto"); ax[2].set_title("our model +1d forecast", fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("The same dust storm: how it looks from space, what the satellite measures, what the model predicts", y=1.04, fontsize=12)
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"paper2_satellite_vs_model.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/paper2_satellite_vs_model.png")


if __name__ == "__main__":
    grid_figure()
    sat_vs_model()
