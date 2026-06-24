"""Central configuration: locations, date range, variables, paths."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# --- Target location ---
TASHKENT = {"name": "Tashkent", "lat": 41.2995, "lon": 69.2401}
TIMEZONE = "Asia/Tashkent"

# --- History window ---
# Air-quality (CAMS) on Open-Meteo begins ~Sep 2022; weather (ERA5) goes back further.
# We start AQ-driven collection at 2022-08-01 and drop leading null rows.
START_DATE = "2022-08-01"

# --- Air-quality variables (hourly -> daily) ---
AQ_VARS = [
    "pm2_5", "pm10", "us_aqi",
    "nitrogen_dioxide", "ozone", "carbon_monoxide", "dust",
]

# --- Weather variables (hourly -> daily) ---
# NOTE: temperature_850hPa is NOT available in the ERA5 archive for this point,
# so the explicit inversion feature is dropped; boundary_layer_height carries the
# atmospheric-trapping signal instead (available in both archive and forecast).
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "boundary_layer_height",
    "precipitation", "shortwave_radiation",
]

# --- Regional cities for transport features (pm2_5 only) ---
REGIONAL_CITIES = [
    {"name": "Fergana",   "lat": 40.39, "lon": 71.79, "bearing": 90},
    {"name": "Almaty",    "lat": 43.26, "lon": 76.93, "bearing": 45},
    {"name": "Bishkek",   "lat": 42.87, "lon": 74.57, "bearing": 50},
    {"name": "Samarkand", "lat": 39.65, "lon": 66.96, "bearing": 225},
    {"name": "Dushanbe",  "lat": 38.56, "lon": 68.79, "bearing": 180},
    {"name": "Ashgabat",  "lat": 37.96, "lon": 58.33, "bearing": 240},
]

# --- WAQI ground station (validation) ---
# NOTE: the plan's "@7396" is wrong (it resolves to a sensor in Illinois, USA).
# Real Tashkent stations: @14722 Chilanzar (LIVE), @11219 US Embassy (stale since
# ~Feb 2026), @14723 Yunusabad (stale). Chilanzar is the only one reporting now.
WAQI_STATION = "@14722"          # Tashkent Chilanzar (live)
WAQI_STATION_FALLBACK = "@11219"  # Tashkent US Embassy (currently not reporting)

# --- API endpoints ---
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WAQI_URL = "https://api.waqi.info/feed"

# --- Daily aggregation rules ---
# precipitation accumulates (sum); everything else is a daily mean.
# wind_direction is handled separately via a vector (resultant) mean.
SUM_VARS = {"precipitation"}

# --- Phase 2/3 modeling constants ---
PM25_THRESHOLD = 35.0      # "exceedance" day (≈ EPA Moderate->USG boundary 35.4)
EPISODE_THRESHOLD = 55.0   # pollution "episode" (≈ EPA USG->Unhealthy boundary 55.4)

# Same-day columns that must NOT be features: they are model-siblings of the
# target (only available as CAMS forecasts at inference -> circular).
LEAK_COLS = ["pm10", "us_aqi", "nitrogen_dioxide", "ozone",
             "carbon_monoxide", "dust", "boundary_layer_height_era5",
             "wind_direction_10m"]

# Time-based split boundaries (validation must include a full winter).
SPLIT = {
    "train_end": "2024-09-30",   # extended vs plan: we have ~46 months of data
    "val_end":   "2025-06-30",   # includes winter 2024-25
    # test = val_end -> present (hold out, touch once)
}
