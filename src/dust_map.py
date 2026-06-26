"""
Dust map renderer — a true-colour satellite base (NASA GIBS, display only — NOT a model input) with a
SMOOTH dust overlay (upsampled, not pixelized), every tracked storm as an arrow (direction, length ∝
speed), the city, and threat highlights. PNG now; the same storm data drives the web map later.

Run:  python src/dust_map.py --source iran --date 2022-07-04 --basemap
      python src/dust_map.py                       # Central Asia, latest day, plain background
"""
from __future__ import annotations
import argparse, sys, datetime, math, io
from pathlib import Path
import numpy as np, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dust_tracker as dt

ROOT = Path(__file__).resolve().parent.parent
REGIONS = {
    "central-asia": {"dom": [55, 37, 75, 47], "path": "data/satellite/recent/aod", "city": ("Tashkent", 69.24, 41.31), "name": "Central Asia",
                     "labels": [("Kyzylkum", 63, 43), ("Aralkum", 59, 45.5), ("Fergana", 71.5, 40.5), ("Karakum", 60, 38.5)]},
    "iran":         {"dom": [46, 27, 66, 37], "path": "data/satellite/iran/aod", "city": ("Tehran", 51.40, 35.70), "name": "Iran",
                     "labels": [("Tigris–Euphrates", 47.5, 31.5), ("Dasht-e Kavir", 55, 34.5), ("Sistan", 61.5, 30.5), ("Caspian", 52, 37)]},
    "asia":         {"dom": [47, 25, 107, 50], "path": "data/satellite/recent_asia/aod", "city": ("Tashkent", 69.24, 41.31), "name": "Hormuz → Mongolia",
                     "labels": [("Tigris–Euphrates", 46.5, 32), ("Arabia", 49, 26.5), ("Karakum", 60, 39), ("Aralkum", 59, 46),
                                ("Taklamakan", 82, 39), ("Tarim", 85, 41.5), ("Gobi", 103, 44), ("Thar", 72, 27)]},
    "watch":        {"dom": [53, 35, 87, 48], "path": "data/satellite/recent_watch/aod", "city": ("Tashkent", 69.24, 41.31), "name": "Central Asia",
                     "labels": [("Aralkum", 59, 46), ("Karakum", 60, 38.5), ("Kyzylkum", 64, 43), ("Fergana", 72, 40.5), ("Taklamakan", 81, 39), ("Tarim", 84, 41.5)]},
}
GIBS = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
        "&FORMAT=image/png&LAYERS={layer}&CRS=EPSG:4326&BBOX={lat0},{lon0},{lat1},{lon1}&WIDTH={w}&HEIGHT={h}&TIME={t}")


def fetch_gibs_rgb(dom, date, layer="VIIRS_SNPP_CorrectedReflectance_TrueColor", w=1600):
    """High-res true-colour satellite image of the domain from NASA GIBS (open, ~daily). Display only."""
    lon0, lat0, lon1, lat1 = dom; h = int(w * (lat1 - lat0) / (lon1 - lon0))
    url = GIBS.format(layer=layer, lon0=lon0, lat0=lat0, lon1=lon1, lat1=lat1, w=w, h=h, t=date)
    r = requests.get(url, timeout=60); r.raise_for_status()
    return mpimg.imread(io.BytesIO(r.content))


def render(source, date=None, out=None, basemap=False):
    cfg = REGIONS[source]; dom = cfg["dom"]; lon0, lat0, lon1, lat1 = dom
    days = dt.load_days(ROOT / cfg["path"])
    if not days: print("no data — pull first."); return None
    by = {d: a for d, a in days}
    tgt = datetime.date.fromisoformat(date) if date else days[-1][0]
    if tgt not in by: print(f"no frame for {tgt}"); return None
    # gap-fill the overlay: where today is cloud/swath-blocked, use the most recent clear look (the model's
    # reconstruction idea) so the field is complete instead of full of holes.
    ti = [d for d, _ in days].index(tgt)
    A = np.where(by[tgt] > -900, by[tgt].astype(float), np.nan)
    for di in range(ti - 1, max(ti - 9, -1), -1):
        prior = days[di][1]; m = np.isnan(A) & (prior > -900); A[m] = prior[m]
    cov = np.isfinite(A).mean()
    storms = [t for t in dt.track_storms(days, dom) if len(t["pts"]) >= 3]
    sig = [t for t in storms if t["peak"] > 0.9 and (t["displacement"] > 80 or len(t["pts"]) >= 5)]   # real dust, drop flicker/snow
    here = [(t, p["b"]) for t in sig for p in t["pts"] if p["date"] == tgt]
    ext = [lon0, lon1, lat0, lat1]

    aspect = (lat1 - lat0) / (lon1 - lon0)              # keep geography undistorted for wide regions
    fig, ax = plt.subplots(figsize=(13, 13 * aspect + 1.2), dpi=170)
    if basemap:
        try:
            ax.imshow(fetch_gibs_rgb(dom, str(tgt)), extent=ext, origin="upper", aspect="auto", zorder=1)
        except Exception as e:
            print("  GIBS basemap unavailable, dark background:", str(e)[:70]); basemap = False
    if not basemap:
        ax.set_facecolor("#0c1b2e")
    # SMOOTH dust overlay: upsample 6x (cubic) + alpha by intensity so it reads as plumes, not blocks
    Af = np.clip(np.nan_to_num(A, nan=0.0), 0, None); up = np.clip(zoom(Af, 6, order=3), 0, None)
    rgba = plt.cm.YlOrRd(np.clip(up / 2.2, 0, 1)); rgba[..., 3] = np.clip((up - 0.30) / 0.70, 0, 0.82)
    ax.imshow(rgba, extent=ext, origin="upper", aspect="auto", interpolation="bilinear", zorder=2)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 2.2)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01); cb.set_label("dust thickness (AOD)", fontsize=8); cb.ax.tick_params(labelsize=7)

    tcol = "white" if basemap else "#5b4636"
    for name, x, y in cfg["labels"]:
        ax.text(x, y, name, fontsize=7, color=tcol, style="italic", ha="center", alpha=0.75, zorder=3)
    cname, clon, clat = cfg["city"]
    ax.plot(clon, clat, marker="*", ms=20, color="#7fd4ff" if basemap else "#12407a", mec="white", mew=1.2, zorder=6)
    ax.annotate(cname, (clon, clat), (clon + 0.4, clat + 0.4), fontsize=10, fontweight="bold", color="white" if basemap else "#12407a", zorder=6)

    approaching = 0
    for t, b in here:
        lon, lat, spd = b["lon"], b["lat"], t["speed"]
        if t["heading"] not in dt.PTS16: continue
        brg = math.radians(dt.PTS16.index(t["heading"]) * 22.5); L = 0.6 + spd * 0.13
        dx, dy = math.sin(brg) * L, math.cos(brg) * L
        s = {"lon": lon, "lat": lat}; c = {"lon": clon, "lat": clat}
        near = abs((dt.bearing(s, c, dom) - math.degrees(brg) + 180) % 360 - 180) < 50 and dt.km(s, c, dom) / max(spd, 1) / 24 <= 4
        col = "#ff2d2d" if near else ("#bfe3ff" if basemap else "#444"); approaching += near
        ax.annotate("", xy=(lon + dx, lat + dy), xytext=(lon, lat), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4, mutation_scale=18), zorder=5)
        ax.plot(lon, lat, "o", ms=6, color=col, mec="white", mew=0.8, zorder=5)
        ax.text(lon, lat - 0.5, f"{t['heading']} {spd:.0f}km/h", fontsize=7, color=col, ha="center", fontweight="bold", zorder=5)

    status = f"{len(here)} storm(s) tracked" + (f"  ·  {approaching} APPROACHING {cname}" if approaching else "")
    if not here: status = "no active dust storms — clear"
    ax.set_title(f"DUST MAP · {cfg['name']} · {tgt}    ({status})", fontsize=11)
    ax.set_xlabel("longitude °E", fontsize=8); ax.set_ylabel("latitude °N", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    out = out or (ROOT / "figures" / f"dust_map_{source}{'_rgb' if basemap else ''}.png")
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"saved {out}  ({status})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="central-asia", choices=list(REGIONS))
    ap.add_argument("--date", default=None); ap.add_argument("--basemap", action="store_true", help="true-colour GIBS satellite base")
    a = ap.parse_args()
    render(a.source, a.date, basemap=a.basemap)
