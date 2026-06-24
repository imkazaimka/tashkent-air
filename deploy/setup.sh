#!/usr/bin/env bash
# One-shot setup for the Tashkent air-quality alert bot on a Linux host (Ubuntu/Debian VPS).
# Run from the repo root *after* you have copied data/ and .env onto the host (see DEPLOY.md).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> 1/4  Python venv + dependencies"
python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "==> 2/4  Checking the token and the training data are present"
[ -f .env ] || { echo "!! Missing .env — it must contain TELEGRAM_BOT_TOKEN=... (scp it from your Mac)"; exit 1; }
grep -q '^TELEGRAM_BOT_TOKEN=' .env || { echo "!! .env has no TELEGRAM_BOT_TOKEN"; exit 1; }
if [ ! -f data/processed/daily_merged.csv ] || [ ! -f data/raw/openaq_embassy_pm25_daily.csv ]; then
  echo "!! Missing cached data/ (it is gitignored). Either:"
  echo "     scp -r data user@host:$ROOT/        # copy from your Mac (simplest)"
  echo "   or regenerate on the host (needs OPENAQ_TOKEN in .env):"
  echo "     .venv/bin/python src/collect.py && .venv/bin/python src/features.py"
  exit 1
fi

echo "==> 3/4  Verifying the bot token reaches Telegram"
.venv/bin/python - <<'PY'
import os, requests
from dotenv import load_dotenv
load_dotenv(".env")
r = requests.get(f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/getMe", timeout=20).json()
print("    Bot:", "@" + r["result"]["username"] if r.get("ok") else r)
assert r.get("ok"), "Bot token rejected by Telegram"
PY

echo "==> 4/4  Dry run (forecast + danger tier, sends nothing)"
.venv/bin/python src/telegram_alert.py --status | grep -E "Subscribers:|Worst upcoming:" || true

cat <<EOF

Setup OK. Now schedule it (cron — runs forever in the background):

  crontab -e   # then add these two lines (adjust path + morning hour for Tashkent, UTC+5):

  */15 * * * * cd $ROOT && .venv/bin/python src/telegram_alert.py --sync  >> \$HOME/alert.log 2>&1
  0 3 * * *    cd $ROOT && .venv/bin/python src/telegram_alert.py         >> \$HOME/alert.log 2>&1
                # 03:00 UTC = 08:00 Tashkent

Or use the systemd timers in deploy/ (see DEPLOY.md). Test a live send: .venv/bin/python src/telegram_alert.py --force
EOF
