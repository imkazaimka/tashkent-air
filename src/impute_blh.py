"""
Fill the boundary-layer-height (BLH) gap in the dataset.

ERA5 on Open-Meteo is missing BLH for ~Jan–Jun 2024 (181 days) plus a few recent
archive-lag days. BLH is the core of the `trapping_index` feature, so we can't
drop it. No Open-Meteo model serves that window, so we reconstruct it from the
surface meteorology we DO have for every hour.

Why HOURLY (not daily): BLH has a very strong, regular diurnal cycle — near zero
at night, peaking at midday with solar heating. Modelling at hourly resolution
with `hour-of-day + shortwave_radiation` captures that cycle far more tightly
than a daily-mean model, then we aggregate the filled hours to a daily mean.

Honesty guarantees:
  * Every filled day is flagged in a new `blh_imputed` column (1 = contains
    modelled hours).
  * The measured daily value is preserved in `boundary_layer_height_era5`.
  * Quality is validated on a HELD-OUT contiguous Jan–Jun block (2023) that
    mirrors the real gap, before anything is written.

At inference time BLH comes live from the Open-Meteo forecast API (no gap), so
this imputation only affects the historical training/eval window.

Run:  python src/impute_blh.py        (after collect.py)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

TARGET = "boundary_layer_height"
# Hourly predictors available for EVERY hour (no BLH-derived leakage).
PREDICTORS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "precipitation", "shortwave_radiation",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


def add_features(h: pd.DataFrame) -> pd.DataFrame:
    t = h.index
    hour, doy = t.hour, t.dayofyear
    h["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    h["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    h["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    h["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return h


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def validate(h: pd.DataFrame) -> None:
    """Hold out a contiguous Jan–Jun 2023 block (same shape as the real gap),
    train on the rest, predict, aggregate to daily mean, and report recovery."""
    have = h[h[TARGET].notna()]
    blk = (have.index >= "2023-01-01") & (have.index <= "2023-06-30")
    if blk.sum() < 24 * 30:
        print("  [validation] not enough 2023 H1 hours; skipping")
        return
    train, test = have[~blk], have[blk]
    m = make_model()
    m.fit(train[PREDICTORS], train[TARGET])
    pred = pd.Series(m.predict(test[PREDICTORS]).clip(min=0), index=test.index)

    # compare DAILY MEANS (what actually feeds the model)
    pday = pred.resample("1D").mean()
    aday = test[TARGET].resample("1D").mean()
    j = pd.concat([aday, pday], axis=1, keys=["actual", "pred"]).dropna()
    mae = mean_absolute_error(j["actual"], j["pred"])
    r2 = r2_score(j["actual"], j["pred"])
    print(f"  [validation] held-out Jan–Jun 2023, daily means ({len(j)} days):")
    print(f"      MAE  = {mae:6.1f} m  ({mae / j['actual'].mean() * 100:.0f}% of "
          f"{j['actual'].mean():.0f} m mean)")
    print(f"      R^2  = {r2:6.3f}")
    print(f"      bias = {j['pred'].mean() - j['actual'].mean():+6.1f} m")


def main() -> None:
    wx_path = C.RAW / "tashkent_weather_hourly.parquet"
    h = pd.read_parquet(wx_path)
    h = add_features(h)

    missing = h[TARGET].isna()
    feats_ok = h[PREDICTORS].notna().all(axis=1)
    fillable = missing & feats_ok
    print(f"Hourly BLH gap: {int(missing.sum())} hours; "
          f"{int(fillable.sum())} have full predictors and are fillable.")

    validate(h)

    # Train on all available hours, predict the fillable gap hours.
    have = h[h[TARGET].notna()]
    model = make_model()
    model.fit(have[PREDICTORS], have[TARGET])
    h_filled = h[TARGET].copy()
    h_filled.loc[fillable] = model.predict(h.loc[fillable, PREDICTORS]).clip(min=0)

    # Aggregate to daily means. Wholesale assignment keeps this idempotent
    # regardless of any prior run's state in daily_merged.csv.
    measured_daily = h[TARGET].resample("1D").mean()        # NaN where all hrs gone
    filled_daily = h_filled.resample("1D").mean()           # measured + imputed
    # a day is "imputed" only if it had a gap AND we could fill it
    imputed_day = (missing.resample("1D").max().astype(bool)
                   & filled_daily.notna())
    for s in (measured_daily, filled_daily, imputed_day):
        s.index = s.index.normalize()

    # --- climatology bias-correction -------------------------------------
    # The surface-met model has no upper-air stability info, so it carries a
    # systematic monthly offset (validation showed ~+130 m). Anchor each imputed
    # month's MEAN to the measured climatology for that month (from other years),
    # preserving the model's day-to-day weather-driven variation.
    meas_clim = measured_daily.dropna().groupby(
        lambda d: d.month).mean()  # measured monthly climatology
    for m in range(1, 13):
        day_mask = imputed_day & (filled_daily.index.month == m)
        if day_mask.any() and m in meas_clim.index:
            shift = meas_clim[m] - filled_daily[day_mask].mean()
            filled_daily.loc[day_mask] = (filled_daily[day_mask] + shift).clip(lower=0)
            print(f"  bias-corr month {m:02d}: shift {shift:+6.1f} m "
                  f"-> target clim {meas_clim[m]:.0f} m")

    # Merge into the daily table (overwrite, not patch).
    path = C.PROCESSED / "daily_merged.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df["boundary_layer_height_era5"] = measured_daily.reindex(df.index)  # measured
    df[TARGET] = filled_daily.reindex(df.index)                          # filled
    df["blh_imputed"] = imputed_day.reindex(df.index).fillna(False).astype(int)

    still = int(df[TARGET].isna().sum())
    df.reset_index().to_csv(path, index=False)

    n_imp = int(df["blh_imputed"].sum())
    imp_vals = df.loc[df["blh_imputed"] == 1, TARGET]
    print(f"\nFilled {n_imp} days (still missing: {still} — recent archive-lag, "
          f"self-heals next collect).")
    print(f"  imputed daily BLH: mean {imp_vals.mean():.0f} m  "
          f"min {imp_vals.min():.0f}  max {imp_vals.max():.0f}")
    print(f"  measured daily BLH mean for reference: "
          f"{df.loc[df['blh_imputed'] == 0, TARGET].mean():.0f} m")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
