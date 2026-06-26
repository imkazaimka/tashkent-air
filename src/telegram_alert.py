"""
Live danger-alert broadcaster (Telegram) — the warning side of the project.

It runs the same 7-day forecast as forecast_week.py and, if a DANGEROUS PM2.5 level is
coming within the week, broadcasts a warning to everyone subscribed to the bot
(@airqualitytash_bot). It is meant to run once a day (cron).

Who gets notified: a Telegram bot can only message users who have started a chat with it,
so "everyone" = everyone who has sent /start to the bot. We poll getUpdates, store each
subscriber's chat_id in data/telegram_subscribers.json, and /stop removes them.

No spam: we remember the last alert tier (data/telegram_alert_state.json) and only send a
NEW message when the threat first appears or gets WORSE, plus a single "all-clear" when it
passes. A week of bad air = one onset alert, not seven daily ones.

Danger tiers (worst of the next 7 days):
  2  SEVERE   — best-estimate PM2.5 >= 150 (Very Unhealthy+), or P(>100) >= 30%
  1  WARNING  — best-estimate PM2.5 >=  55 (Unhealthy),       or P(>40)  >= 60%
  0  none

Usage:
  python src/telegram_alert.py            # sync subscribers, forecast, alert if needed (cron)
  python src/telegram_alert.py --sync     # only pick up new /start subscribers
  python src/telegram_alert.py --status    # show subscribers + outlook + tier, send nothing
  python src/telegram_alert.py --force     # broadcast the current outlook to all (demo/manual)
  python src/telegram_alert.py --test      # send a one-line test ping to all subscribers
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
load_dotenv(C.ROOT / ".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API = f"https://api.telegram.org/bot{TOKEN}"
SUBS = C.ROOT / "data" / "telegram_subscribers.json"
STATE = C.ROOT / "data" / "telegram_alert_state.json"
WARN_PM, SEVERE_PM = 55, 150          # PM2.5 µg/m³ best-estimate thresholds
WARN_P_BAD, SEVERE_P_VBAD = 0.60, 0.30


# ---------------- small json store helpers ----------------
def _load(p, default):
    try:
        return json.load(open(p))
    except Exception:
        return default

def _save(p, obj):
    json.dump(obj, open(p, "w"), indent=2, ensure_ascii=False)


# ---------------- Telegram I/O ----------------
def tg(method, **params):
    r = requests.post(f"{API}/{method}", json=params, timeout=30)
    return r.json()

def send(chat_id, text):
    return tg("sendMessage", chat_id=chat_id, text=text, parse_mode="HTML",
              disable_web_page_preview=True)

def sync_subscribers():
    """Pick up new /start and /stop messages; maintain the subscriber list."""
    subs = _load(SUBS, {})
    st = _load(STATE, {})
    offset = st.get("last_update_id", 0) + 1
    upd = tg("getUpdates", offset=offset, timeout=0, allowed_updates=["message"])
    added, removed = 0, 0
    for u in upd.get("result", []):
        st["last_update_id"] = max(st.get("last_update_id", 0), u["update_id"])
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        text = (msg.get("text") or "").strip().lower()
        if not cid:
            continue
        name = (chat.get("first_name", "") + " " + chat.get("last_name", "")).strip() or chat.get("username", "")
        if text.startswith("/start"):
            if str(cid) not in subs:
                subs[str(cid)] = {"name": name}; added += 1
            send(cid, "✅ <b>Subscribed</b> to Tashkent air-quality alerts.\n"
                      "You'll get a message when a dangerous PM2.5 level is forecast within the week.\n"
                      "Send /stop to unsubscribe.\n\n"
                      "✅ <b>Подписка оформлена</b> на оповещения о качестве воздуха в Ташкенте.\n"
                      "Вы получите сообщение, когда на неделе ожидается опасный уровень PM2.5.\n"
                      "Отправьте /stop, чтобы отписаться.")
        elif text.startswith("/stop"):
            if subs.pop(str(cid), None) is not None:
                removed += 1
            send(cid, "🚫 Unsubscribed. Send /start to resubscribe.\n"
                      "🚫 Вы отписались. Отправьте /start, чтобы подписаться снова.")
    _save(SUBS, subs); _save(STATE, st)
    return subs, added, removed

def broadcast(text):
    """Send to every subscriber; drop anyone who has blocked the bot."""
    subs = _load(SUBS, {})
    ok, dropped = 0, 0
    for cid in list(subs):
        r = send(int(cid), text)
        if r.get("ok"):
            ok += 1
        elif r.get("error_code") in (403, 400):     # blocked / chat gone
            subs.pop(cid, None); dropped += 1
        time.sleep(0.05)                             # stay under Telegram's rate limit
    if dropped:
        _save(SUBS, subs)
    return ok, dropped


# ---------------- danger logic ----------------
def assess(rows):
    """Return (tier, worst_row) for the next 7 days."""
    tier = 0; worst = None
    for r in rows:
        sev = 2 if (r["pm25"] >= SEVERE_PM or r["p_vbad"] >= SEVERE_P_VBAD) else \
              1 if (r["pm25"] >= WARN_PM or r["p_bad"] >= WARN_P_BAD) else 0
        if sev > tier:
            tier = sev
        if worst is None or r["pm25"] > worst["pm25"]:
            worst = r
    return tier, worst

def weekday(datestr):
    import datetime
    return datetime.date.fromisoformat(datestr).strftime("%a %d %b")

CAUSE_RU = {"Mazut/coal heating": "мазут/уголь (отопление)", "Dust storm": "пыльная буря",
            "Traffic": "транспорт", "Imported (transport)": "перенос из других регионов"}

def cause_lines(worst):
    """Diagnose the dominant cause of the worst day (if its tracers are in the row) → (EN, RU) lines."""
    need = ("cams_so2", "carbon_monoxide", "nitrogen_dioxide", "dust", "pm10", "pm2_5",
            "temperature_2m", "wind_speed_10m", "boundary_layer_height", "wind_direction_10m",
            "fergana_pm25", "almaty_pm25", "bishkek_pm25")
    if not all(k in worst for k in need):
        return "", ""
    try:
        from diagnose_cause import Diagnoser
        d = Diagnoser().diagnose(worst)
        te = " (trapped by still air)" if d["trapped_air"] else ""
        tr = " (застой воздуха)" if d["trapped_air"] else ""
        return (f"🔎 Likely cause: <b>{d['cause']}</b>{te}\n",
                f"🔎 Вероятная причина: <b>{CAUSE_RU.get(d['cause'], d['cause'])}</b>{tr}\n")
    except Exception:
        return "", ""

def alert_text(tier, worst, rows):
    head = "🚨 <b>SEVERE air-quality warning — Tashkent</b>" if tier == 2 else \
           "⚠️ <b>Air-quality warning — Tashkent</b>"
    cause_en, cause_ru = cause_lines(worst)
    advice_en = ("Stay indoors, keep windows closed, use a mask/purifier if you go out."
                 if tier == 2 else
                 "Limit time outdoors; sensitive groups (children, elderly, asthma) stay in.")
    advice_ru = ("Оставайтесь дома, закройте окна, при выходе — маска/очиститель."
                 if tier == 2 else
                 "Сократите время на улице; чувствительным группам (дети, пожилые, астма) — дома.")
    strip = " · ".join(f"{weekday(r['date'])[:6]} {int(r['pm25'])}" for r in rows)
    return (
        f"{head}\n\n"
        f"A dangerous level is forecast:\n"
        f"📅 <b>{weekday(worst['date'])}</b>: PM2.5 ≈ <b>{int(worst['pm25'])} µg/m³</b> ({worst['level']})\n"
        f"Chance air is BAD (&gt;40): <b>{int(worst['p_bad']*100)}%</b> · "
        f"VERY BAD (&gt;100): <b>{int(worst['p_vbad']*100)}%</b>\n"
        f"{cause_en}"
        f"➡️ {advice_en}\n\n"
        f"🚨 <b>Опасный уровень воздуха — Ташкент</b>\n"
        f"📅 <b>{weekday(worst['date'])}</b>: PM2.5 ≈ <b>{int(worst['pm25'])} мкг/м³</b>\n"
        f"{cause_ru}"
        f"➡️ {advice_ru}\n\n"
        f"<i>7-day PM2.5: {strip}</i>"
    )

def all_clear_text():
    return ("✅ <b>Air quality back to normal — Tashkent</b>\n"
            "No dangerous PM2.5 days forecast in the coming week.\n\n"
            "✅ <b>Воздух снова в норме — Ташкент</b>\n"
            "Опасных уровней PM2.5 на ближайшую неделю не прогнозируется.")


# ---------------- main ----------------
def main():
    if not TOKEN:
        sys.exit("TELEGRAM_BOT_TOKEN not set in .env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    subs, added, removed = sync_subscribers()
    print(f"Subscribers: {len(subs)} (+{added} new, -{removed} left)")
    if a.sync:
        return
    if a.test:
        ok, dr = broadcast("🔔 Test alert from the Tashkent air-quality bot. "
                           "Alerts are live.\n🔔 Тестовое оповещение. Система работает.")
        print(f"Test sent to {ok} subscriber(s); dropped {dr}.")
        return

    from forecast_week import compute_outlook
    rows = compute_outlook()
    tier, worst = assess(rows)
    print(f"Worst upcoming: {worst['date']} PM2.5 {worst['pm25']:.0f} ({worst['level']}) → tier {tier}")

    if a.status:
        print(json.dumps(rows, indent=2)); return

    st = _load(STATE, {}); last = st.get("last_tier", 0)
    if a.force:
        ok, dr = broadcast(alert_text(max(tier, 1), worst, rows))
        print(f"Forced broadcast to {ok} subscriber(s); dropped {dr}."); return

    if tier > last:                                  # onset or escalation → warn
        ok, dr = broadcast(alert_text(tier, worst, rows))
        print(f"ALERT (tier {last}→{tier}) sent to {ok} subscriber(s); dropped {dr}.")
    elif tier == 0 and last > 0:                      # threat passed → all-clear
        ok, dr = broadcast(all_clear_text())
        print(f"All-clear sent to {ok} subscriber(s); dropped {dr}.")
    else:
        print(f"No notification (tier {tier}, last {last}).")
    st["last_tier"] = tier; _save(STATE, st)


if __name__ == "__main__":
    main()
