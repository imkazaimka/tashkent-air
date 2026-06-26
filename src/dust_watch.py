"""
DUST WATCH — terminal dust-storm monitor for the Tashkent / Central Asia warning system.

Scans the most recent satellite AOD, tracks every dust storm (classical CV — no model needed for the
observed report), and prints its direction, speed, intensity, and whether it is heading toward the city.
The same engine (dust_tracker) will back the website later; this is the terminal front-end.

Run:  python src/dust_watch.py                  # Central Asia (Tashkent), most recent data
      python src/dust_watch.py --source iran     # demo on real Iran summer-2022 storms
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dust_tracker as dt

BOLD = "\033[1m" if sys.stdout.isatty() else ""        # colour only on a real terminal, not when piped
RST = "\033[0m" if sys.stdout.isatty() else ""
ROOT = Path(__file__).resolve().parent.parent
REGIONS = {
    "central-asia": {"dom": [55, 37, 75, 47], "path": "data/satellite/recent/aod", "city": ("Tashkent", 69.24, 41.31), "name": "Central Asia"},
    "iran":         {"dom": [46, 27, 66, 37], "path": "data/satellite/iran/aod",   "city": ("Tehran", 51.40, 35.70),   "name": "Iran"},
}
W = 66


def intensity(p):
    return "SEVERE" if p > 3 else "HEAVY" if p > 1.5 else "MODERATE" if p > 0.8 else "light"


def threat(storm, city, dom):
    """Is the city in the storm's path? Returns (distance_km, eta_hours) or None."""
    cname, clon, clat = city
    s = {"lon": storm["loc"][0], "lat": storm["loc"][1]}; c = {"lon": clon, "lat": clat}
    if storm["heading"] not in dt.PTS16 or storm["speed"] < 1: return None
    dist = dt.km(s, c, dom); to_city = dt.bearing(s, c, dom); head = dt.PTS16.index(storm["heading"]) * 22.5
    if abs((to_city - head + 180) % 360 - 180) < 50:                 # city within ~50° of the storm's heading
        return dist, dist / storm["speed"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Dust-storm terminal monitor")
    ap.add_argument("--source", default="central-asia", choices=list(REGIONS))
    ap.add_argument("--active-days", type=int, default=7, help="how many recent days count as 'active'")
    args = ap.parse_args()
    cfg = REGIONS[args.source]; dom = cfg["dom"]; city = cfg["city"]
    days = dt.load_days(ROOT / cfg["path"])
    if not days:
        print(f"no data at {cfg['path']} — pull it first."); return
    first, latest = days[0][0], days[-1][0]
    storms = [t for t in dt.track_storms(days, dom) if len(t["pts"]) >= 3]
    active = [t for t in storms if (latest - t["last"]).days <= args.active_days]
    moving = sorted([t for t in storms if t["displacement"] > 200], key=lambda t: -t["displacement"])

    p = print
    p("═" * W)
    p(f"  DUST WATCH · {cfg['name']:<24}              as of {latest}")
    p("═" * W)
    p(f"  scanned    {first} → {latest}   ({len(days)} days of satellite AOD)")
    p(f"  reference  {city[0]}  ({city[1]:.2f}°E, {city[2]:.2f}°N)")
    p("")
    if active:
        p(f"  STATUS   ⚠  ACTIVE — {len(active)} storm(s) in the last {args.active_days} days")
    else:
        recent = np.concatenate([d[1][d[1] > -900] for d in days[-args.active_days:]])
        p(f"  STATUS   ✓  CLEAR — no active dust storms over {city[0]}")
        p(f"            background AOD ~{recent.mean():.2f}  (normal)")
    p("═" * W)

    if active:
        p("  ACTIVE STORMS")
        p("  " + "─" * (W - 4))
        for t in sorted(active, key=lambda t: -t["peak"]):
            lon, lat = t["loc"]
            p(f"  •  {lat:4.1f}°N {lon:5.1f}°E    heading {t['heading']:>3} @ {t['speed']:3.0f} km/h    {intensity(t['peak'])} (AOD {t['peak']:.1f})")
            th = threat(t, city, dom)
            if th and th[1] / 24 <= 4:                          # within the useful warning window
                dist, eta = th
                p(f"     {BOLD}⚠ APPROACHING {city[0]}{RST} — {dist:.0f} km away, ETA ~{eta/24:.1f} days")
            elif th:
                p(f"     → toward {city[0]} but distant (~{th[0]:.0f} km, {th[1]/24:.0f} days out)")
            else:
                p(f"     → not heading toward {city[0]}")
        p("")

    if moving:
        p(f"  ALL MOVING STORMS THIS PERIOD  (travelled > 200 km)")
        p("  " + "─" * (W - 4))
        p(f"     {'period':<22}{'head':>5}{'speed':>10}{'travel':>9}{'peak':>7}")
        for t in moving[:10]:
            p(f"     {str(t['start'])+'→'+str(t['last']):<22}{t['heading']:>5}{t['speed']:7.0f} km/h{t['displacement']:6.0f} km{t['peak']:7.1f}")
        p("")
    elif active:
        p("")

    p("═" * W)
    p("  source: observed satellite motion (reliable speed).")
    p("  next:   ConvLSTM extends each storm 1–3 days forward (forecast heading).")
    p("═" * W)


if __name__ == "__main__":
    main()
