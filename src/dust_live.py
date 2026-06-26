"""
Live dust-watch cycle: refresh data -> track storms -> render map -> print report. Run on a schedule
(cron / launchd) for an auto-updating monitor.

  */30 * * * *  cd /Users/.../TashkentAir && EE_PROJECT=civil-sentry-379101 python3 src/dust_live.py >> logs/dust.log 2>&1

LATENCY NOTE — freshness is bound by the AOD SOURCE, not this loop:
  * current source  : MAIAC via Earth Engine  -> ~5-day lag  (so "live" = latest ~5 days ago)
  * for ~3-hour NRT : switch the AOD pull to NASA LANCE (MODIS/VIIRS NRT aerosol) or geostationary
                      Meteosat-IODC. Same tracker/map downstream; only the pull changes.
"""
from __future__ import annotations
import subprocess, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(name, args):
    print(f"  · {name} …", flush=True)
    subprocess.run([sys.executable] + args, cwd=ROOT, check=False)


def main():
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] dust-watch cycle", flush=True)
    step("refresh satellite data (LANCE NRT ~3h)", ["src/pull_lance.py"])   # needs EARTHDATA_* in .env
    step("render map", ["src/dust_map.py", "--source", "central-asia"])
    step("report", ["src/dust_watch.py", "--source", "central-asia"])
    print("  done.\n", flush=True)


if __name__ == "__main__":
    main()
