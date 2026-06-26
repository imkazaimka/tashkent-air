"""
NASA LANCE near-real-time AOD pull — the live data feed for the dust watch (VIIRS Deep-Blue, good over
bright desert = good for dust). Grids each swath into a regular lon/lat grid so dust_tracker / dust_map /
dust_watch use it unchanged.

Regions:  central-asia  (Tashkent box)            asia  (Hormuz -> Mongolia: ME + Central Asia + Gobi)

Auth: NASA Earthdata creds in .env (gitignored):  EARTHDATA_USERNAME=...  EARTHDATA_PASSWORD=...
      (free account: https://urs.earthdata.nasa.gov/users/new)

Run:  python src/pull_lance.py --region asia --days 6
"""
from __future__ import annotations
import os, argparse, datetime, tempfile, shutil, itertools, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SHORTS = ["AERDB_L2_VIIRS_SNPP_NRT", "AERDB_L2_VIIRS_NOAA20_NRT"]   # NRT (~3h after overpass), two satellites = more passes
AOD_VARS = ["Aerosol_Optical_Thickness_550_Land_Best_Estimate", "Aerosol_Optical_Thickness_550_Land", "Aerosol_Optical_Thickness_550_Land_Ocean"]
REGIONS = {
    "central-asia": {"dom": [55, 37, 75, 47], "shape": (56, 112),  "path": "data/satellite/recent/aod"},
    "asia":         {"dom": [47, 25, 107, 50], "shape": (125, 300), "path": "data/satellite/recent_asia/aod"},
    "watch":        {"dom": [53, 35, 87, 48], "shape": (90, 230),   "path": "data/satellite/recent_watch/aod"},
}


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


def getvar(ds, names):
    for grp in itertools.chain([ds], ds.groups.values()):
        for nm in names:
            if nm in grp.variables:
                return np.ma.filled(grp.variables[nm][:], np.nan).astype(np.float64)
    return None


def grid(lat, lon, aod, dom, shape):
    lon0, lat0, lon1, lat1 = dom; nlat, nlon = shape
    v = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(aod) & (aod > 0) & (aod < 5) \
        & (lon >= lon0) & (lon < lon1) & (lat >= lat0) & (lat < lat1)
    if v.sum() < 20: return None
    la, lo, ao = lat[v].ravel(), lon[v].ravel(), aod[v].ravel()
    ci = ((lo - lon0) / (lon1 - lon0) * nlon).astype(int).clip(0, nlon - 1)
    ri = ((lat1 - la) / (lat1 - lat0) * nlat).astype(int).clip(0, nlat - 1)
    s = np.zeros(shape); c = np.zeros(shape)
    np.add.at(s, (ri, ci), ao); np.add.at(c, (ri, ci), 1)
    g = np.full(shape, -999.0, np.float32); m = c > 0; g[m] = (s[m] / c[m]).astype(np.float32)
    return g


def main():
    import earthaccess, netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="central-asia", choices=list(REGIONS))
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()
    cfg = REGIONS[args.region]; dom = cfg["dom"]; shape = cfg["shape"]
    out = ROOT / cfg["path"]; out.mkdir(parents=True, exist_ok=True)
    load_env()
    if not (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")):
        print("Missing EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env."); return
    earthaccess.login(strategy="environment")
    print(f"region {args.region}  dom {dom}  grid {shape}  -> {out}", flush=True)
    today = datetime.date.today()
    for k in range(1, args.days + 1):                       # start at 1 day back (today rarely processed)
        d = today - datetime.timedelta(days=k)
        if (out / f"{d}.npy").exists():
            print(f"  {d}: cached"); continue
        res = []
        for sh in SHORTS:
            try:
                res += earthaccess.search_data(short_name=sh, temporal=(d.isoformat(), (d + datetime.timedelta(days=1)).isoformat()),
                                               bounding_box=tuple(dom), count=200)
            except Exception as e:
                print(f"  {d} {sh}: err {str(e)[:40]}")
        if not res:
            print(f"  {d}: no granules"); continue
        def gtime(g):
            try: return g["umm"]["TemporalExtent"]["RangeDateTime"]["EndingDateTime"]
            except Exception: return ""
        res = sorted(res, key=gtime, reverse=True)[:16]      # FRESHEST granules only (live) + bound the download
        tmp = Path(tempfile.mkdtemp())
        try:
            files = earthaccess.download(res, str(tmp))
            la, lo, ao = [], [], []
            for f in files:
                try:
                    ds = netCDF4.Dataset(str(f))
                    aod = getvar(ds, AOD_VARS); lat = getvar(ds, ["Latitude"]); lon = getvar(ds, ["Longitude"]); ds.close()
                    if aod is not None and lat is not None and aod.shape == lat.shape:
                        la.append(lat); lo.append(lon); ao.append(aod)
                except Exception:
                    pass
            if ao:
                g = grid(np.concatenate([x.ravel() for x in la]), np.concatenate([x.ravel() for x in lo]),
                         np.concatenate([x.ravel() for x in ao]), dom, shape)
                if g is not None:
                    np.save(out / f"{d}.npy", g); print(f"  {d}: {len(files)} granules -> {(g>-900).mean()*100:.0f}% coverage", flush=True)
                else:
                    print(f"  {d}: no valid AOD over domain")
            else:
                print(f"  {d}: unreadable granules")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
