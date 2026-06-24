"""
Phase 2 — feature engineering for the Tashkent PM2.5 forecaster.

Framing (important — resolves a plan inconsistency):
  Each row is ONE TARGET DAY D. We predict pm2_5(D) using
    * SAME-DAY (D) features that are forecastable the evening before:
      weather, derived meteorology, calendar.
    * LAGGED (<= D-1) pollution features from actuals:
      PM2.5 lags / rolling stats / episode streak, regional-city lags.
  This matches Phase 5 inference (weather from tomorrow's forecast, pollution
  from yesterday's actuals) and avoids the train/inference mismatch you'd get by
  training on "today's weather -> tomorrow's PM2.5".

Leakage guards:
  * The target's same-day siblings (pm10, NO2, dust, us_aqi, ...) are DROPPED —
    they come from the same CAMS model and are only available as forecasts.
  * All rolling/streak stats are computed on shift(1) (through D-1).
  * An assertion checks pm25_lag1(D) == pm2_5(D-1) and that the raw target is
    not among the features.

Output: data/processed/features.csv  (model-ready, one row per target day)

Run:  python src/features.py        (after collect.py + impute_blh.py)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

TARGET = "pm2_5"
WARMUP = 7  # rows dropped at the start (longest lag)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    f = pd.DataFrame({"date": df["date"]})
    pm = df[TARGET]

    # ---- target (same-day actual) ----
    f["y"] = pm

    # ---- PM2.5 history (<= D-1) ----
    for k in (1, 2, 3, 7):
        f[f"pm25_lag{k}"] = pm.shift(k)
    f["pm25_roll3_mean"] = pm.shift(1).rolling(3).mean()
    f["pm25_roll7_mean"] = pm.shift(1).rolling(7).mean()
    f["pm25_roll7_std"] = pm.shift(1).rolling(7).std()
    f["pm25_diff1"] = pm.shift(1) - pm.shift(2)          # yesterday's momentum

    # episode streak: consecutive days pm>threshold ending at D-1
    exceed = (pm > C.PM25_THRESHOLD).astype(int)
    grp = (exceed != exceed.shift()).cumsum()
    run = exceed.groupby(grp).cumcount() + 1
    streak_incl = run.where(exceed == 1, 0)
    f["episode_streak"] = streak_incl.shift(1).fillna(0)

    # ---- meteorology (same-day D, forecastable) ----
    f["temperature_2m"] = df["temperature_2m"]
    f["relative_humidity_2m"] = df["relative_humidity_2m"]
    f["surface_pressure"] = df["surface_pressure"]
    f["wind_speed_10m"] = df["wind_speed_10m"]
    f["boundary_layer_height"] = df["boundary_layer_height"]
    f["shortwave_radiation"] = df["shortwave_radiation"]
    f["blh_imputed"] = df.get("blh_imputed", 0)           # 0 at inference always

    # circular wind encoding (avoids 0/360 discontinuity)
    wd = np.radians(df["wind_direction_10m"])
    f["wind_sin"] = np.sin(wd)
    f["wind_cos"] = np.cos(wd)

    # ventilation / trapping (the core physical signal)
    ventilation = df["boundary_layer_height"] * df["wind_speed_10m"]
    f["ventilation_coef"] = ventilation
    f["trapping_index"] = 1.0 / (ventilation + 1e-6)
    # gap-robust stagnation proxy (no BLH): calm + high pressure
    inv_wind = 1.0 / (df["wind_speed_10m"] + 0.5)
    f["inv_wind"] = inv_wind
    f["stagnation_proxy"] = inv_wind * df["surface_pressure"]

    # precipitation washout
    f["precip"] = df["precipitation"]                     # same-day (forecast)
    f["precip_lag1"] = df["precipitation"].shift(1)
    f["precip_sum_3d"] = df["precipitation"].rolling(3).sum()

    # ---- regional transport (lagged pollution x same-day wind alignment) ----
    for city in C.REGIONAL_CITIES:
        col = f"{city['name'].lower()}_pm25"
        if col not in df.columns:
            continue
        lag1 = df[col].shift(1)
        f[f"{col}_lag1"] = lag1
        f[f"{col}_lag2"] = df[col].shift(2)
        # wind blowing FROM the city toward Tashkent aligns with its bearing
        toward = np.cos(np.radians(df["wind_direction_10m"] - city["bearing"]))
        f[f"{city['name'].lower()}_transport"] = lag1 * toward * df["wind_speed_10m"]

    # ---- calendar (same-day D) ----
    doy = df["date"].dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["is_heating_season"] = df["date"].dt.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    f["weekday"] = df["date"].dt.dayofweek
    f["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

    return f


def assign_split(f: pd.DataFrame) -> pd.Series:
    d = f["date"]
    s = pd.Series("test", index=f.index)
    s[d <= C.SPLIT["train_end"]] = "train"
    s[(d > C.SPLIT["train_end"]) & (d <= C.SPLIT["val_end"])] = "val"
    return s


def leakage_checks(df_raw: pd.DataFrame, f: pd.DataFrame) -> None:
    assert TARGET not in f.columns, "raw target leaked into features!"
    # pm25_lag1(D) must equal pm2_5(D-1)
    raw = df_raw.sort_values("date").reset_index(drop=True)
    chk = f.sort_values("date").reset_index(drop=True)
    a = chk["pm25_lag1"].iloc[WARMUP:].values
    b = raw[TARGET].shift(1).iloc[WARMUP:].values
    assert np.allclose(a, b, equal_nan=True), "pm25_lag1 misaligned!"
    # no feature may be identical to the target
    for col in f.columns:
        if col in ("date", "y"):
            continue
        if f[col].equals(f["y"]):
            raise AssertionError(f"feature {col} is identical to target")
    print("  leakage checks: PASSED")


def main() -> None:
    src = C.PROCESSED / "daily_merged.csv"
    df = pd.read_csv(src, parse_dates=["date"])
    print(f"Loaded {src.name}: {df.shape[0]} days")

    f = build(df)
    leakage_checks(df, f)

    # drop warm-up rows (NaN lags) and any row missing the target
    f = f.iloc[WARMUP:].copy()
    f = f[f["y"].notna()]
    f["split"] = assign_split(f)

    out = C.PROCESSED / "features.csv"
    f.to_csv(out, index=False)

    feat_cols = [c for c in f.columns if c not in ("date", "y", "split")]
    print(f"\nSaved {out}")
    print(f"  rows: {len(f)}  features: {len(feat_cols)}")
    print(f"  range: {f['date'].min().date()} -> {f['date'].max().date()}")
    print("  split sizes:", f["split"].value_counts().reindex(
        ["train", "val", "test"]).to_dict())

    # persistence baseline (the bar to beat): predict y(D) = pm25_lag1
    mae_persist = (f["y"] - f["pm25_lag1"]).abs().mean()
    print(f"\n  persistence baseline MAE (y vs pm25_lag1): {mae_persist:.2f} ug/m3")

    # quick sanity: top correlations with target
    corr = f[feat_cols + ["y"]].corr(numeric_only=True)["y"].drop("y")
    top = corr.abs().sort_values(ascending=False).head(8)
    print("  top |corr| with target:")
    for k in top.index:
        print(f"    {k:24s} {corr[k]:+.3f}")

    nan_feat = f[feat_cols].isna().sum()
    nan_feat = nan_feat[nan_feat > 0]
    if len(nan_feat):
        print("  features with NaNs (LightGBM handles natively):")
        for k, v in nan_feat.items():
            print(f"    {k:24s} {int(v)}")


if __name__ == "__main__":
    main()
