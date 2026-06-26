"""
Real-time smog-cause diagnosis — when the air is (forecast to be) dangerous, say WHY.

Each source has a tracer fingerprint validated in Section 4 of the paper:
  • Mazut/coal heating  = SO₂ + CO elevated, and cold (heating demand)
  • Dust storm          = coarse aerosol (high PM10/PM2.5) + CAMS dust
  • Traffic             = NO₂ elevated *relative to sulfur* (vehicles emit NO₂ without SO₂)
  • Imported (transport)= high upwind/regional PM2.5 arriving on an easterly wind
A stagnation flag ("trapped air") is added when the boundary layer is shallow and winds are calm.

For any day, each fingerprint is scored by how elevated its tracers are versus the historical record
(percentile rank); the dominant score is the diagnosed cause, with the score margin as confidence.
The deployed tool feeds it live API values (CAMS forecast gases/dust + ground sensors + ERA5 weather).

Validation (dirty days, >35 µg/m³): winter → 81% heating / 14% traffic / 4% dust; non-winter → 36% dust
/ 32% heating — consistent with the paper's seasonal attribution.

Run:  python src/diagnose_cause.py        # prints the seasonal validation
Use:  from diagnose_cause import Diagnoser; Diagnoser().diagnose(day_dict) -> {cause, confidence, trapped}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

SOURCES = ["Mazut/coal heating", "Dust storm", "Traffic", "Imported (transport)"]


def _features(d):
    d = d.copy()
    d["coarse"] = d["pm10"] / d["pm2_5"].clip(lower=1)
    d["hdd"] = (18 - d["temperature_2m"]).clip(lower=0)
    d["no2_rel"] = d["nitrogen_dioxide"] / (d["cams_so2"] + 1)
    d["east"] = (d[["fergana_pm25", "almaty_pm25", "bishkek_pm25"]].mean(axis=1)
                 * np.cos(np.deg2rad(d["wind_direction_10m"]) - np.deg2rad(70))).clip(lower=0)
    return d


class Diagnoser:
    """Fits percentile references on the historical record; diagnoses any day's dominant cause."""
    REF_COLS = ["cams_so2", "carbon_monoxide", "hdd", "dust", "coarse", "no2_rel", "east",
                "wind_speed_10m", "boundary_layer_height"]

    def __init__(self):
        muni = pd.read_csv(C.RAW / "tashkent_municipal_pm25_daily.csv", parse_dates=["date"])
        dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
        so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"]).rename(columns={"so2": "cams_so2"})
        self.ref = _features(dm.merge(so2, on="date", how="left"))
        self._sorted = {c: np.sort(self.ref[c].dropna().values) for c in self.REF_COLS}
        self.hist = self.ref.merge(muni[["date", "pm25_muni"]], on="date")

    def _pct(self, col, val):
        a = self._sorted[col]
        return np.searchsorted(a, val) / len(a)

    def _scores(self, r):
        p = {c: self._pct(c, r[c]) for c in self.REF_COLS}
        return {
            "Mazut/coal heating": (p["cams_so2"] + p["carbon_monoxide"] + p["hdd"]) / 3,
            "Dust storm": (p["dust"] + p["coarse"]) / 2,
            "Traffic": p["no2_rel"],
            "Imported (transport)": (p["east"] + p["wind_speed_10m"]) / 2,
        }, p

    def diagnose(self, day: dict) -> dict:
        """day: dict with raw tracer/weather values (live API). Returns dominant cause + confidence."""
        r = _features(pd.DataFrame([day])).iloc[0]
        sc, p = self._scores(r)
        order = sorted(sc, key=sc.get, reverse=True)
        margin = sc[order[0]] - sc[order[1]]
        trapped = p["boundary_layer_height"] < 0.25 and p["wind_speed_10m"] < 0.35
        return {"cause": order[0],
                "confidence": "high" if margin > 0.15 else "moderate",
                "trapped_air": bool(trapped),
                "scores": {k: round(v, 2) for k, v in sc.items()}}


def main():
    dg = Diagnoser()
    h = _features(dg.hist)
    h["cause"] = [dg._scores(r)[0] for _, r in h.iterrows()]
    h["cause"] = h.apply(lambda r: max(dg._scores(r)[0], key=dg._scores(r)[0].get), axis=1)
    dirty = h[h.pm25_muni > 35].copy()
    dirty["season"] = np.where(dirty.date.dt.month.isin([11, 12, 1, 2, 3]), "winter", "non-winter")
    print(f"cause of dirty days (>35), by season (n={len(dirty)}):")
    print((pd.crosstab(dirty.season, dirty.cause, normalize="index") * 100).round(0).astype(int))


if __name__ == "__main__":
    main()
