# Deploying the alert bot on a Linux host

This works on **any** Linux VPS — FastHosts, Hetzner, DigitalOcean, Oracle Cloud Free Tier, a
Raspberry Pi, etc. The bot is a small daily job (`python src/telegram_alert.py`) that keeps a
little state on disk (the subscriber list + de-dup tier), so any always-on box with a disk is fine.

## 1. Provision and connect
Create the smallest instance (1 vCPU / 1 GB RAM is plenty), then SSH in and install the basics:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
git clone https://github.com/imkazaimka/tashkent-air.git TashkentAir
cd TashkentAir
```

## 2. Copy the two things that are NOT in git
The bot token and the cached training data are deliberately kept out of the repo. Copy them from
your Mac (run this **on your Mac**, replacing `user@host`):

```bash
scp .env user@host:~/TashkentAir/.env                 # contains TELEGRAM_BOT_TOKEN=...
scp -r data user@host:~/TashkentAir/data              # cached daily_merged.csv + embassy CSV (training data)
```

*(Alternative to copying `data/`: regenerate it on the host with
`python src/collect.py && python src/features.py` — but that needs `OPENAQ_TOKEN` in `.env`.)*

## 3. Install and verify
```bash
bash deploy/setup.sh
```
This builds a virtualenv, installs deps, checks the token reaches Telegram, and does a dry run
(forecast + danger tier, sends nothing). A live send to all subscribers:
`.venv/bin/python src/telegram_alert.py --force`.

## 4. Schedule it — pick ONE

### Option A — cron (simplest)
```bash
crontab -e
```
Add (adjust the path and the hour — the host is probably on **UTC**, Tashkent is **UTC+5**):
```cron
*/15 * * * * cd ~/TashkentAir && .venv/bin/python src/telegram_alert.py --sync  >> $HOME/alert.log 2>&1
0 3 * * *    cd ~/TashkentAir && .venv/bin/python src/telegram_alert.py         >> $HOME/alert.log 2>&1
#            03:00 UTC = 08:00 in Tashkent
```
The `--sync` line every 15 min catches new `/start` subscribers promptly (Telegram only keeps
updates for 24 h); the daily line does the forecast and alerts only if a dangerous day is coming.

### Option B — systemd timers (more robust; survives reboots, logs to journalctl)
Edit the `User=`/`WorkingDirectory=` paths in the four unit files in this folder, then:
```bash
sudo cp deploy/tashkentair-*.service deploy/tashkentair-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tashkentair-alert.timer tashkentair-sync.timer
systemctl list-timers | grep tashkentair          # confirm they're scheduled
journalctl -u tashkentair-alert.service --no-pager # view a run
```

## 5. Keep the model fresh (optional but recommended)
The forecast trains on the cached history each run; over months that history goes stale. To keep it
current, add `OPENAQ_TOKEN` to `.env` and refresh the data weekly:
```cron
0 2 * * 1 cd ~/TashkentAir && .venv/bin/python src/collect.py && .venv/bin/python src/features.py >> $HOME/alert.log 2>&1
```

## Notes
- **Reaching people:** the bot can only message users who pressed **Start** at
  https://t.me/airqualitytash_bot. Share that link; each person subscribes once (`/stop` to leave).
- **State files** (`data/telegram_subscribers.json`, `data/telegram_alert_state.json`) are created
  automatically and are gitignored — back them up if you rebuild the host.
- **Security:** never commit `.env`. The token is read only from there.
