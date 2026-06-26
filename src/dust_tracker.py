"""
Dust object-tracker — the classical-CV engine (NO neural net, NO training). Segments each daily AOD frame
into distinct dust blobs, matches each to the next day's nearest blob, and measures displacement →
direction + speed. Reusable core for the `dust_watch` terminal tool and (later) the website backend.
"""
from __future__ import annotations
import datetime
from pathlib import Path
import numpy as np
from scipy import ndimage

THR, MIN_PIX, GATE_KM = 0.55, 12, 700               # dust threshold (AOD), min blob size (px), max day-to-day match (km)
PTS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def km_per_deg(dom):
    return 111.0, 111.0 * np.cos(np.radians((dom[1] + dom[3]) / 2))   # (per °lat, per °lon)


def to_lonlat(r, c, H, W, dom):
    lon0, lat0, lon1, lat1 = dom
    return lon0 + c / W * (lon1 - lon0), lat1 - r / H * (lat1 - lat0)


def blobs(a, dom):
    """Distinct dust blobs in one frame: centroid (lon,lat), size, peak AOD."""
    lab, n = ndimage.label((a > THR) & (a > -900)); out = []
    for k in range(1, n + 1):
        pix = lab == k
        if pix.sum() >= MIN_PIX:
            r, c = np.array(np.where(pix)).mean(1); H, W = a.shape
            lon, lat = to_lonlat(r, c, H, W, dom)
            out.append({"lon": lon, "lat": lat, "size": int(pix.sum()), "peak": float(a[pix].max())})
    return out


def km(a, b, dom):
    kla, klo = km_per_deg(dom)
    return float(np.hypot((a["lon"] - b["lon"]) * klo, (a["lat"] - b["lat"]) * kla))


def bearing(a, b, dom):
    kla, klo = km_per_deg(dom)
    return np.degrees(np.arctan2((b["lon"] - a["lon"]) * klo, (b["lat"] - a["lat"]) * kla)) % 360


def heading16(brg):
    return PTS16[int((brg + 11.25) // 22.5) % 16]


def track_storms(days, dom):
    """days = list of (date, aod[H,W]). Returns tracks, each annotated with heading/speed/peak/loc."""
    tracks, prev = [], []
    for d, a in days:
        cur = blobs(a, dom); links, used = [], set()
        for b in cur:
            best, bd = None, GATE_KM
            for j, (pb, _) in enumerate(prev):
                if j in used: continue
                dd = km(pb, b, dom)
                if dd < bd: bd, best = dd, j
            if best is not None:
                tr = prev[best][1]; used.add(best)
            else:
                tr = {"pts": []}; tracks.append(tr)
            tr["pts"].append({"date": d, "b": b}); links.append((b, tr))
        prev = links
    for t in tracks:
        pts = t["pts"]; sb, ss = [], []
        for p0, p1 in zip(pts[:-1], pts[1:]):
            dd = (p1["date"] - p0["date"]).days
            if dd >= 1:
                ss.append(km(p0["b"], p1["b"], dom) / (dd * 24)); sb.append(bearing(p0["b"], p1["b"], dom))
        t["speed"] = float(np.mean(ss)) if ss else 0.0
        t["heading"] = heading16((np.degrees(np.arctan2(np.mean(np.sin(np.radians(sb))), np.mean(np.cos(np.radians(sb))))) % 360)) if sb else "-"
        t["displacement"] = km(pts[0]["b"], pts[-1]["b"], dom)
        t["peak"] = max(p["b"]["peak"] for p in pts)
        t["start"], t["last"] = pts[0]["date"], pts[-1]["date"]
        t["loc"] = (pts[-1]["b"]["lon"], pts[-1]["b"]["lat"])
    return tracks


def load_days(path):
    fs = sorted(Path(path).glob("*.npy"))
    return [(datetime.date.fromisoformat(f.stem), np.load(f)) for f in fs]


if __name__ == "__main__":
    dom = [46, 27, 66, 37]
    ts = [t for t in track_storms(load_days(Path(__file__).resolve().parent.parent / "data/satellite/iran/aod"), dom) if len(t["pts"]) >= 3]
    print(f"{len(ts)} storms tracked")
    for t in sorted(ts, key=lambda t: -len(t["pts"]))[:6]:
        print(f"  {t['start']}→{t['last']}  {t['heading']:>3} {t['speed']:4.0f} km/h  peak {t['peak']:.1f}")
