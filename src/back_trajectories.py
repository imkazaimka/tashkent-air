"""
B4 — independent transport proof via kinematic back-trajectories.

Uses ERA5 surface winds (independent of CAMS) to trace where the air over
Tashkent came from on the dirtiest vs cleanest REAL-sensor days. If H-A is right,
high-pollution air should originate from the E/NE (Fergana valley / Kazakh-Kyrgyz
sector). This owes nothing to CAMS's transport scheme — it's pure meteorology +
the real embassy sensor.

Caveats: a 2D kinematic trajectory on a coarse (~2 deg) 10 m-wind grid — it shows
synoptic origin direction, not a full 3D HYSPLIT dispersion run.

Run:  python src/back_trajectories.py   (fetches + caches a regional wind grid)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"
GRID_CACHE = C.RAW / "wind_grid.parquet"
LATS = np.arange(37, 46, 2.0)        # 37..45
LONS = np.arange(63, 80, 2.0)        # 63..79
GSTART, GEND = "2022-10-01", "2025-03-01"
TASH = (C.TASHKENT["lat"], C.TASHKENT["lon"])
HOURS_BACK = 48


def fetch_grid() -> pd.DataFrame:
    if GRID_CACHE.exists():
        print(f"Using cached {GRID_CACHE.name}")
        return pd.read_parquet(GRID_CACHE)
    print(f"Fetching {len(LATS)*len(LONS)} grid points of ERA5 10m wind ...")
    frames = []
    for la in LATS:
        for lo in LONS:
            for attempt in range(4):
                try:
                    r = requests.get(C.ARCHIVE_URL, params={
                        "latitude": float(la), "longitude": float(lo),
                        "hourly": "wind_speed_10m,wind_direction_10m",
                        "start_date": GSTART, "end_date": GEND,
                        "timezone": "UTC"}, timeout=60)
                    r.raise_for_status()
                    h = r.json()["hourly"]
                    d = pd.DataFrame({"time": pd.to_datetime(h["time"]),
                                      "spd": h["wind_speed_10m"],
                                      "dir": h["wind_direction_10m"]})
                    d["lat"] = la; d["lon"] = lo
                    frames.append(d)
                    break
                except Exception as e:
                    time.sleep(2 * (attempt + 1))
            time.sleep(0.2)
    g = pd.concat(frames, ignore_index=True)
    g.to_parquet(GRID_CACHE)
    print(f"  cached {GRID_CACHE.name} ({len(g)} rows)")
    return g


def build_uv(g: pd.DataFrame):
    """Return times, U[t,lat,lon], V[t,lat,lon] (velocity components, m/s)."""
    g = g.copy()
    rad = np.radians(g["dir"])
    g["u"] = -g["spd"] * np.sin(rad)      # eastward
    g["v"] = -g["spd"] * np.cos(rad)      # northward
    times = np.sort(g["time"].unique())
    tix = {t: i for i, t in enumerate(times)}
    U = np.full((len(times), len(LATS), len(LONS)), np.nan)
    V = np.full_like(U, np.nan)
    li = {la: i for i, la in enumerate(LATS)}; lj = {lo: j for j, lo in enumerate(LONS)}
    for r in g.itertuples():
        U[tix[r.time], li[r.lat], lj[r.lon]] = r.u
        V[tix[r.time], li[r.lat], lj[r.lon]] = r.v
    # fill any grid NaNs by nan-mean over space per time
    for A in (U, V):
        for t in range(A.shape[0]):
            m = np.nanmean(A[t])
            A[t] = np.where(np.isnan(A[t]), m, A[t])
    return pd.DatetimeIndex(times), U, V


def back_traj(t0, times, U, V):
    """Integrate one 48h backward trajectory from Tashkent starting at t0."""
    lat, lon = TASH
    path = [(lat, lon)]
    idx = times.get_indexer([t0], method="nearest")[0]
    for k in range(HOURS_BACK):
        ti = max(idx - k, 0)
        fu = RegularGridInterpolator((LATS, LONS), U[ti], bounds_error=False,
                                     fill_value=None)
        fv = RegularGridInterpolator((LATS, LONS), V[ti], bounds_error=False,
                                     fill_value=None)
        la = min(max(lat, LATS[0]), LATS[-1]); lo = min(max(lon, LONS[0]), LONS[-1])
        u = float(fu([la, lo])[0]); v = float(fv([la, lo])[0])
        lat -= (v * 3600) / 110540.0
        lon -= (u * 3600) / (111320.0 * np.cos(np.radians(lat)))
        path.append((lat, lon))
    return np.array(path)


def bearing_to(lat, lon):
    """Compass bearing from Tashkent to a point (origin direction)."""
    dlat = lat - TASH[0]
    dlon = (lon - TASH[1]) * np.cos(np.radians(TASH[0]))
    return (np.degrees(np.arctan2(dlon, dlat))) % 360


def _fig_trajectories(paths, origins, bh, bl):
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold",
                         "axes.titlesize": 12, "axes.edgecolor": "#9aa5b1",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0]); axp = fig.add_subplot(gs[1], projection="polar")

    for p in paths["low"]:
        ax.plot(p[:, 1], p[:, 0], color="#5dade2", alpha=0.14, lw=1, zorder=1)
    for p in paths["high"]:
        ax.plot(p[:, 1], p[:, 0], color="#e74c3c", alpha=0.16, lw=1, zorder=2)
    mh = np.mean(np.stack(paths["high"]), axis=0)
    ml = np.mean(np.stack(paths["low"]), axis=0)
    ax.plot(ml[:, 1], ml[:, 0], color="#1f6fb2", lw=3.5, zorder=4,
            label="mean path — cleanest days")
    ax.plot(mh[:, 1], mh[:, 0], color="#b21f2d", lw=3.5, zorder=5,
            label="mean path — dirtiest days")
    for m, col in ((mh, "#b21f2d"), (ml, "#1f6fb2")):
        ax.scatter(m[-1, 1], m[-1, 0], color=col, s=80, zorder=6,
                   edgecolor="white", lw=1.4)

    for c in C.REGIONAL_CITIES:
        ax.plot(c["lon"], c["lat"], "^", color="#2c3e50", ms=7, zorder=7)
        ax.annotate(c["name"], (c["lon"], c["lat"]), xytext=(4, 5),
                    textcoords="offset points", fontsize=8.5, color="#2c3e50", zorder=8)
    ax.plot(TASH[1], TASH[0], "*", color="#f1c40f", ms=26, mec="#7d6608",
            mew=1.3, zorder=9, label="Tashkent")
    ax.set(xlim=(58, 80), ylim=(36, 46), xlabel="longitude (°E)",
           ylabel="latitude (°N)",
           title="(a)  48-hour kinematic back-trajectories (ERA5 winds)")
    ax.set_aspect(1.25); ax.grid(alpha=0.3, zorder=0)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=9)

    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    bins = np.arange(0, 361, 45)
    for label, col, name in (("low", "#1f6fb2", "cleanest"), ("high", "#b21f2d", "dirtiest")):
        h, _ = np.histogram(origins[label], bins=bins)
        axp.bar(np.radians(bins[:-1] + 22.5), h, width=np.radians(40), color=col,
                alpha=0.6, edgecolor="white", label=name)
    axp.set_title(f"(b)  Air-mass origin direction\ndirtiest ≈ {bh:.0f}° (E/NE)   "
                  f"cleanest ≈ {bl:.0f}° (NW)", fontsize=11, pad=20)
    axp.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), fontsize=8.5)

    fig.suptitle("Where Tashkent's air originates: dirtiest vs cleanest days",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.savefig(FIG / "back_trajectories.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    g = fetch_grid()
    times, U, V = build_uv(g)
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    gt = gt[(gt["date"] >= GSTART) & (gt["date"] <= GEND)].dropna()
    hi = gt.nlargest(30, "pm25_ground")["date"]
    lo = gt.nsmallest(30, "pm25_ground")["date"]

    paths = {"high": [], "low": []}
    origins = {"high": [], "low": []}
    for label, days in (("high", hi), ("low", lo)):
        for d in days:
            t0 = pd.Timestamp(d) + pd.Timedelta(hours=12)
            p = back_traj(t0, times, U, V)
            paths[label].append(p)
            origins[label].append(bearing_to(p[-1, 0], p[-1, 1]))

    # circular mean origin bearing
    def circ_mean(b):
        a = np.radians(b)
        return np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360
    bh, bl = circ_mean(origins["high"]), circ_mean(origins["low"])
    # share of high-pollution origins in the E/NE quadrant (0-110 deg)
    ene_share = np.mean([(0 <= b <= 110) for b in origins["high"]])

    print(f"High-pollution days: mean 48h air-origin bearing = {bh:.0f} deg "
          f"(E/NE quadrant share = {ene_share*100:.0f}%)")
    print(f"Low-pollution  days: mean 48h air-origin bearing = {bl:.0f} deg")
    print(f"=> dirtiest air originates from the "
          f"{'E/NE' if 0 <= bh <= 110 else 'other'} sector "
          f"(independent of CAMS): "
          f"{'CONSISTENT with H-A' if 0 <= bh <= 110 else 'NOT consistent'}")

    _fig_trajectories(paths, origins, bh, bl)
    print("Saved figures/back_trajectories.png")


if __name__ == "__main__":
    main()
