"""
Animated nowcast GIF — the recent dust evolving over the last days, on the true-colour base, with a
direction arrow for the main storm. Loops in the browser. Served live by dust_server.
"""
from __future__ import annotations
import sys, datetime, io, glob
from pathlib import Path
import numpy as np, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DOM = [53, 35, 87, 48]; CITY = ("Tashkent", 69.24, 41.31)
LABELS = [("Aralkum", 59, 46), ("Karakum", 60, 38.5), ("Kyzylkum", 64, 43), ("Taklamakan", 81, 39)]
PTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def gibs(date_str, w=1000):
    lon0, lat0, lon1, lat1 = DOM; h = int(w*(lat1-lat0)/(lon1-lon0))
    url = (f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
           f"&FORMAT=image/png&LAYERS=VIIRS_SNPP_CorrectedReflectance_TrueColor&CRS=EPSG:4326"
           f"&BBOX={lat0},{lon0},{lat1},{lon1}&WIDTH={w}&HEIGHT={h}&TIME={date_str}")
    return mpimg.imread(io.BytesIO(requests.get(url, timeout=60).content))


def main():
    lon0, lat0, lon1, lat1 = DOM; ext = [lon0, lon1, lat0, lat1]
    fs = sorted(glob.glob(str(ROOT/"data/satellite/recent_watch/aod/*.npy")))
    days = [(datetime.date.fromisoformat(Path(f).stem), np.load(f)) for f in fs]
    filled = []                                                   # gap-fill each day from prior clear looks
    for j, (d, a) in enumerate(days):
        A = np.where(a > -900, a.astype(float), np.nan)
        for k in range(j-1, max(j-7, -1), -1):
            pr = days[k][1]; m = np.isnan(A) & (pr > -900); A[m] = pr[m]
        filled.append((d, A))
    use = filled[-7:]
    base = None
    try: base = gibs(str(use[-1][0]))
    except Exception: pass

    def cen(A):                                                  # AOD-weighted centre of mass — stable vs patchy peaks
        a = np.clip(np.nan_to_num(A, nan=0), 0, None); a = np.where(a > 0.3, a, 0.0)
        if a.sum() < 5: return None
        H, W = A.shape; r = (a.sum(1) @ np.arange(H)) / a.sum(); c = (a.sum(0) @ np.arange(W)) / a.sum()
        return lon0 + c/W*(lon1-lon0), lat1 - r/H*(lat1-lat0)
    cens = [cen(A) for d, A in use]
    fc = list(cens)
    for i in range(len(fc)):
        if fc[i] is None: fc[i] = next((c for c in cens[i:] if c), None) or next((c for c in reversed(cens[:i]) if c), None)
    kla, klo = 111.0, 111.0*np.cos(np.radians((lat0+lat1)/2))
    movers = []                                                 # per-frame: vector from this frame's dust centre to the NEXT frame's
    for i in range(len(fc)):
        a, b = (fc[i], fc[i+1]) if i+1 < len(fc) else (fc[i-1] if i > 0 else fc[i], fc[i])
        if fc[i] and a and b and np.hypot((b[0]-a[0])*klo, (b[1]-a[1])*kla) > 25:
            movers.append((fc[i], b[0]-a[0], b[1]-a[1]))
        else: movers.append(None)

    frames = []
    for idx, ((d, A), c) in enumerate(zip(use, fc)):
        fig, ax = plt.subplots(figsize=(9, 4.0), dpi=100)
        if base is not None: ax.imshow(base, extent=ext, origin="upper", aspect="auto", zorder=1)
        else: ax.set_facecolor("#0c1b2e")
        up = np.clip(zoom(np.clip(np.nan_to_num(A, nan=0), 0, None), 5, order=3), 0, None)
        rgba = plt.cm.YlOrRd(np.clip(up/1.3, 0, 1)); rgba[..., 3] = np.clip((up-0.22)/0.55, 0, 0.85)   # full plume, not just the peak
        ax.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear", aspect="auto", zorder=2)
        ax.plot(CITY[1], CITY[2], marker="*", ms=14, color="#7fd4ff", mec="white", mew=1, zorder=6)
        ax.text(CITY[1]+0.4, CITY[2]+0.4, CITY[0], color="white", fontsize=9, fontweight="bold", zorder=6)
        for nm, x, y in LABELS: ax.text(x, y, nm, color="white", fontsize=6.5, style="italic", alpha=.6, ha="center", zorder=3)
        mv = movers[idx]
        if mv:                                                  # arrow points to where the dust actually goes next (amplified to be visible)
            (lon, lat), dl, da = mv; ex, ey = dl*4.0, da*4.0; mag = np.hypot(ex, ey)
            if mag > 5: ex, ey = ex/mag*5, ey/mag*5
            ax.annotate("", xy=(lon+ex, lat+ey), xytext=(lon, lat), arrowprops=dict(arrowstyle="-|>", color="#ff2d2d", lw=3.5, mutation_scale=26), zorder=7)
            brg = np.degrees(np.arctan2(dl*klo, da*kla)) % 360; hd = PTS[int((brg+22.5)//45) % 8]
            ax.text(0.015, 0.04, f"dust moving {hd}", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round", fc="#c0392b", ec="none", alpha=.85), zorder=8)
        ax.set_title(f"{d}", color="#222", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white"); plt.close()
        buf.seek(0); frames.append(Image.open(buf).convert("RGB"))
    out = ROOT/"figures"/"dust_anim.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=650, loop=0)
    print(f"saved {out}  ({len(frames)} frames, per-frame direction arrows)")


if __name__ == "__main__":
    main()
