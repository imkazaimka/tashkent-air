"""
Live dust-watch dashboard — a tiny self-contained web server. It re-renders the nowcast + the model
forecast on a schedule and serves an auto-refreshing page, so it updates itself in the browser (no
screenshots). The DATA refreshes only as fast as satellites allow (~after each daytime overpass); the
PAGE refreshes every 2 min and always shows the freshest render.

Run:  EE_PROJECT=civil-sentry-379101 python src/dust_server.py
      then open  http://localhost:8000

Needs EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env for the live pull.
"""
from __future__ import annotations
import sys, time, threading, subprocess, datetime, http.server, socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; PORT = 8000; INTERVAL = 1200          # re-render every 20 min
STATE = {"updated": "starting…", "report": "rendering…"}


def sh(args, t=600):
    try: return subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True, timeout=t)
    except Exception as e: return type("R", (), {"stdout": f"({args[1]} failed: {e})", "returncode": 1})()


def render_cycle():
    first = True
    while True:
        try:
            if not first: sh(["src/pull_lance.py", "--region", "watch", "--days", "6"])   # skip pull on first tick (use existing)
            sh(["src/dust_map.py", "--source", "watch", "--basemap"])
            sh(["src/dust_anim.py"])
            sh(["src/dust_forecast.py"])
            STATE["report"] = sh(["src/dust_watch.py", "--source", "watch"]).stdout
            STATE["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            STATE["report"] = f"render error: {e}"
        first = False
        time.sleep(INTERVAL)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Tashkent Dust Watch</title>
<meta http-equiv=refresh content=120>
<style>body{{background:#0b1320;color:#dfe7f3;font-family:system-ui,Segoe UI,sans-serif;margin:0;padding:26px;text-align:center}}
h1{{margin:0 0 2px;font-size:26px}} .t{{color:#7fa8d8;font-size:13px;margin-bottom:20px}}
h2{{color:#9fc1ec;font-weight:600;margin:22px 0 6px;font-size:17px}}
img{{max-width:1150px;width:100%;border-radius:12px;box-shadow:0 6px 28px #000a;margin:6px 0 10px}}
pre{{text-align:left;max-width:780px;margin:12px auto 0;background:#111c2e;padding:16px;border-radius:10px;font-size:12px;line-height:1.45;overflow:auto;color:#cbd8ee}}</style>
</head><body>
<h1>🌫️ Tashkent Dust Watch — live</h1>
<div class=t>updated {updated} · page auto-refreshes every 2 min</div>
<h2>Now — last 7 days, looping (→ shows the dust's direction)</h2><img src="/fig/dust_anim.gif?t={ts}" alt="nowcast animation">
<h2>1–3 day outlook — model forecast</h2><img src="/fig/forecast_live.png?t={ts}" alt="forecast">
<pre>{report}</pre>
</body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/fig/"):
            fp = FIG / self.path.split("/fig/")[1].split("?")[0]
            if fp.exists() and fp.suffix in (".png", ".gif"):
                self._send(fp.read_bytes(), "image/gif" if fp.suffix == ".gif" else "image/png")
            else: self.send_error(404)
            return
        html = PAGE.format(updated=STATE["updated"], report=STATE["report"] or "rendering…", ts=int(time.time()))
        self._send(html.encode(), "text/html; charset=utf-8")


def main():
    threading.Thread(target=render_cycle, daemon=True).start()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    print(f"Dust Watch live → http://localhost:{PORT}   (Ctrl-C to stop)", flush=True)
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), H) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
