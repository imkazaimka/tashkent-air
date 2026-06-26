# Tashkent Air Quality — Attribution, Forecasting & a Live Dust Watch

A reproducible study of bad air in **Tashkent, Uzbekistan**, and the tools built around it. It grew
from two questions — *where does the bad air come from?* and *can we warn people before it's
dangerous?* — into a small operational system. This README documents **what was built, what was used,
and what was honestly found**, including the things that **didn't** work (which is most of the
satellite story).

**Author:** Imronbek Shoniyozov · University of Essex

---

## The one thing to understand first: two seasons, two different problems

Tashkent's air is bad for **two unrelated reasons**, and they need different tools:

| Season | The threat | Where it lives | What sees it |
|---|---|---|---|
| **Winter** | combustion **PM2.5** (coal/mazut/wood heating) | at the **surface**, under cold inversions | **ground sensors** — satellites are blind to it |
| **Summer** | **dust** (transported from the Karakum / regional deserts) | a lofted **column** | **satellites** — ground sensors don't see it coming |

Almost every honest finding below follows from this split. The dangerous days (winter PM2.5) are a
**ground** problem; the dust (summer) is a **satellite** problem; and **the satellite cannot see the
surface PM2.5 that actually harms people** (AOD ≈ 0 correlation with surface PM2.5 in winter).

---

## What we built

1. **An attribution study** (`src/` stats pipeline → *Paper 1*) — what the winter pollution is and
   where it comes from, from measured ground data.
2. **A winter PM2.5 forecaster** (→ *Paper 1*) — two lineages:
   - a **physics box model** (`box_model.py`): ground SO₂ → emission, wind × mixing-height →
     ventilation, precip → washout. CAMS-free, interpretable.
   - a **gradient-boosted forecaster** (`train_*.py`) → the 5-day dangerous-days early-warning.
3. **A satellite dust-watch** (→ *Paper 2*, [`Tashkent-Paper2-Dust-Forecasting.md`](Tashkent-Paper2-Dust-Forecasting.md))
   — the multimodal ConvLSTM dust forecaster, characterised and transferred across four world dust
   regions, plus a live regional dust monitor:
   - `pull_lance.py` — NASA **LANCE** near-real-time VIIRS AOD feed.
   - `dust_tracker.py` — classical CV (connected-components + centroid matching) → storm heading/speed.
   - `dust_map.py` / `dust_anim.py` — true-colour (GIBS) maps + an animated nowcast with a direction arrow.
   - `dust_forecast.py` — the **ConvLSTM** projecting dust 1–3 days ahead (the *only* place the trained
     net runs in production).
   - `dust_watch.py` — terminal report. `dust_server.py` — a **live auto-refreshing web dashboard**.
4. **Delivery:** a Telegram alert bot (`telegram_alert.py`) and the web dashboard.

---

## What we used

- **Ground data:** measured PM2.5 (Tashkent municipal open-data dataset 133 + US-Embassy/OpenAQ) and
  SO₂/CO/NO₂ (WAQI historical, 3 stations — *unvalidated, not redistributed*).
- **Satellite:** MAIAC AOD 550 nm (MODIS, via Earth Engine), TROPOMI NO₂/UVAI/SO₂ (Sentinel-5P),
  **VIIRS Deep-Blue AOD** (NASA LANCE NRT), GIBS true-colour imagery (display only).
- **Weather:** ERA5 / Open-Meteo (wind, boundary-layer height, precipitation) — reanalysis + forecast.
- **Methods:** hypothesis testing, PCA, k-means, multiple regression + variance decomposition,
  gradient boosting; a Gifford-Hanna box model; a ConvLSTM encoder-forecaster; classical CV tracking;
  temporal gap-fill.
- **Stack:** Python, PyTorch (MPS on an M2), scikit-learn/LightGBM, Earth Engine, `earthaccess`+netCDF4,
  scipy, matplotlib, Pillow.

---

## What we found (honestly)

**The winter PM2.5 problem (the part that works):**
- **~70% of winter PM2.5 is local combustion** (heating), ~16% traffic-and-other, ~30% natural
  background — confirmed across six methods + measured ground SO₂ (2.0× winter) + TROPOMI SO₂.
  Dust is *not* the winter cause (the dustiest days are among the cleanest).
- **CAMS reads ~½ the real surface PM2.5** and misses ~80% of dangerous days (verified at 9 sensors).
- **The box model forecasts surface PM2.5:** ties persistence at +1 day (better precision, ~75%),
  beats it multi-day; stagnation multiplier ~2×.
- **The GBT early-warning catches 72% of dangerous days (leave-one-winter-out, AUC 0.84) → 91% at a
  recall-favouring operating point.** Winter dangerous days are **92% heating** by the cause classifier.

**The satellite/dust problem (mostly honest limits):**
- **Satellite AOD ≈ uncorrelated with surface PM2.5 in winter (r ≈ 0)** — satellites cannot warn on
  Tashkent's dangerous days. The column isn't the surface.
- **The ConvLSTM is climatology-limited.** It beats persistence, but a 7-day rollout vs a climatology
  baseline shows most of its apparent skill *is* climatology; genuine forecast skill is modest
  (dust +0.13–0.19 above climatology; NO₂ almost nil).
- **Multimodal fusion adds nothing over AOD-only** for dust (a null result).
- **Architecture doesn't matter here:** linear/MLP/RF/GBT/ensemble all within 0.02 AUC →
  **data-limited (3 winters), not architecture-limited.**
- **What the model *can* do:** track dust **transport** (anomaly-correlation ~0.3, ~2× persistence) —
  direction is reliable, **distance is under-shot** (damped advection). It **generalises to Iran**
  (the learned dynamics transfer), which also means the skill is universal, not Tashkent-specific.
- **Nothing here is methodologically novel** — ConvLSTM, multimodal, advection, the box model,
  gap-fill and tracking are all established, and CAMS/Aurora forecast regional dust better. The only
  un-built thing was the *deployed, open, local system* itself.

---

## What it's genuinely good at

Past the limits, these are real, evidenced, and defensible:

1. **Forecasting the *dangerous winter days* — and beating the alternatives.** The ground-SO₂ + physics
   forecaster catches **72% of dangerous days out-of-sample, 91% at the warning operating point**, vs
   **~17% for raw CAMS** and ~76% for persistence — while being CAMS-free and interpretable. This is the
   part you could actually act on, and it targets the air that actually harms people.
2. **Pinning down the winter source — rigorously.** Six *independent* methods (PCA, clustering,
   regression + variance decomposition, a gradient-boosted classifier, measured ground SO₂, and an
   independent TROPOMI retrieval) all land on the same answer — **local heating combustion, ~70%** —
   and the gas-crisis mechanism explains *why*. That convergence is hard to argue with.
3. **Reading dust *transport* direction — and generalising.** Strip out climatology and the satellite
   model genuinely tracks where existing dust is heading (anomaly-correlation **~0.3, ~2× persistence**),
   and it **transfers to Iran — a region it never trained on** — so it learned real dust *dynamics*, not
   a memorised map. "Which way is the dust going" it answers honestly.
4. **Catching the bias in a trusted global product.** It shows, at nine sensors across five cities, that
   **CAMS under-reads measured surface PM2.5 by ~2×** and misses most dangerous days — a concrete,
   reproducible correction, not a vibe.
5. **Knowing its own limits — measurably.** Every headline number is recomputable from source, and every
   boundary is *tested* rather than assumed (the 7-day climatology check, the architecture bake-off, the
   AOD-vs-surface-PM correlation). In a field that routinely overclaims, that rigor is the quiet strength.
6. **A complete, live system on a laptop.** NRT satellite pull → classical tracking → ConvLSTM forecast →
   self-updating dashboard, end to end on an 8 GB M2 — for a city and a region that had no such tool.

**The single sentence:** it is *very* good at **telling Tashkent, ahead of time and from honest local
data, when the winter air will be dangerous** — and reasonably good at **showing which way regional dust
is drifting** — while being unusually honest about everything it can't do.

---

## What works / what doesn't (scope)

- ✅ **Winter dangerous-days warning** (ground PM2.5 forecaster) — the real, usable contribution.
- ✅ **Summer regional dust *awareness*** (the live dust-watch) — useful as a "is there dust, roughly
  where + which way" view, at satellite-paced latency.
- ◐ **Dust *forecast* (1–3 day)** — trust direction, not timing/intensity; near-climatology when calm.
- ❌ **Real-time fast haboobs** — satellite latency (hours-to-a-day, none overnight) is too slow.
- ❌ **Winter air from satellites** — physically impossible (surface PM invisible to AOD).
- ❌ **Beating CAMS/Aurora** on dust — they have 3-D physics + assimilation; this doesn't.

In one line: **a genuinely working winter-PM2.5 early-warning + a summer-dust awareness dashboard —
honest about being awareness, not a replacement for operational forecast centres.**

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env                      # add your own free tokens (see below)

# --- attribution + winter PM2.5 forecaster ---
python src/collect.py                     # public ground/weather data
python src/train_ground_truth.py          # calibrated forecaster
python src/box_model.py                   # physics box model (ground-SO₂ driven)
python src/verify_claims.py               # recompute every headline number from source

# --- live satellite dust-watch (needs Earthdata creds in .env) ---
python src/pull_lance.py --region watch   # NRT VIIRS AOD
python src/dust_map.py --source watch --basemap   # true-colour map + dust overlay
python src/dust_anim.py                    # animated nowcast + direction arrow
python src/dust_forecast.py               # ConvLSTM 1–3 day outlook
python src/dust_server.py                 # live dashboard → http://localhost:8000
```

## Live tools

- **Telegram bot** — [@airqualitytash_bot](https://t.me/airqualitytash_bot): alerts subscribers when a
  dangerous PM2.5 day is forecast (`telegram_alert.py`, run daily by cron).
- **Web dashboard** — `dust_server.py`: self-refreshing nowcast GIF + forecast + report.

## API keys (all free, in git-ignored `.env`)

```
WAQI_TOKEN=...           # aqicn.org (free)
OPENAQ_TOKEN=...         # openaq.org (free)
EARTHDATA_USERNAME=...   # urs.earthdata.nasa.gov  — for the LANCE NRT dust feed
EARTHDATA_PASSWORD=...
TELEGRAM_BOT_TOKEN=...   # @BotFather, for the alert bot
```
Open-Meteo (weather/forecast) and GIBS (imagery) need no key.

## Honest notes

- **Research / prototype, not a certified service.** The winter forecaster is the dependable part; the
  dust-watch is awareness, not a guarantee.
- **Satellite "live" is satellite-paced** — fresh ~3–5 h after a daytime overpass, older overnight; no
  passive instrument sees through cloud or measures dust in the dark.
- **WAQI ground data are unvalidated and not redistributed** (used under their Data Use Statement);
  `data/` and trained models are regenerated by the scripts and not committed.
