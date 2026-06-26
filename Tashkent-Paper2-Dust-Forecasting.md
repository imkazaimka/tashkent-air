# A Transferable Multimodal ConvLSTM for Short-Range Dust Forecasting — and an Honest Account of Its Ceiling

**Paper 2 of the Tashkent Air-Quality study**
*(the satellite / forecasting half; Paper 1 covers attribution of the winter surface PM2.5 problem)*

**Author:** Imronbek Shoniyozov · University of Essex
**Code & figures:** https://github.com/imkazaimka/tashkent-air
**All images in this paper:** https://github.com/imkazaimka/tashkent-air/tree/main/figures

---

## Abstract

We build a multimodal convolutional-LSTM (ConvLSTM) encoder–forecaster that ingests satellite aerosol
optical depth (AOD) together with NO₂, UV-aerosol-index, and reanalysis wind and precipitation, and
predicts the dust field 1–3 days ahead over Central Asia. The model is genuinely better than
persistence in-domain (dust-field pattern correlation **r = 0.50 vs 0.38 at +1 day**), and — the central
result — **the same network, never retrained, beats persistence on every dust region we tested it on:
Iran, the Sahara/Sahel, the Arabian/Iraqi Middle East, and the Gobi (Mongolia)** — three continents it
never saw in training. That transfer is the evidence that it learned dust *dynamics* rather than a
memorised Central-Asian map.

We are equally explicit about what it is **not**. Multimodal fusion adds **nothing** over AOD-alone for
dust (a clean null result). A 7-day rollout against a climatology baseline shows most of the apparent
skill *is* climatology; genuine forecast skill above climatology is modest and decays to ≈0 by day 3 in
the harder regions. Nothing here is methodologically novel, and operational systems (CAMS, Aurora) with
3-D physics and data assimilation forecast dust better. The contribution is therefore not a new method
but a **characterised, transferable, laptop-scale** model and a deployed open dust-watch — together with
a measurement of exactly where its skill begins and ends.

---

## 1. Scope: which problem this paper solves

Tashkent's air is bad for two unrelated reasons, and they need different instruments:

| Season | Threat | Where it lives | What can see it |
|---|---|---|---|
| **Winter** | combustion **PM2.5** (heating) | the **surface**, under inversions | **ground sensors** — satellites are blind |
| **Summer** | **dust** (regional deserts) | a lofted **column** | **satellites** — ground sensors don't see it coming |

**Paper 1** is the winter/surface/attribution problem (where the bad air comes from, the ground-SO₂
physics warning, the World-Bank reassessment). **This paper (Paper 2)** is the summer/column/forecasting
problem: *given satellites, how far ahead and how well can a compact deep model push dust, and where does
its skill stop?* The two papers share a data pipeline but answer different questions, because — as Paper 1
establishes — **column AOD is ≈ uncorrelated with surface PM2.5 in winter (r ≈ 0)**: the satellite
cannot warn on the days that actually harm people. So the satellite model is held to the only honest
standard available to it — **dust** — and tested to destruction.

---

## 2. Data

| Stream | Product | Use |
|---|---|---|
| **AOD (training/target)** | MODIS **MAIAC** AOD 550 nm (`MCD19A2`, ×0.001), via Google Earth Engine | dust field, the variable forecast |
| **NO₂ / UVAI / SO₂** | Sentinel-5P **TROPOMI** | multimodal channels |
| **Wind / precip** | **ERA5-Land** daily (10 m u/v, total precipitation) | advection + washout drivers |
| **Live AOD** | NASA **LANCE** NRT VIIRS Deep-Blue (`AERDB_L2_VIIRS_SNPP/NOAA20_NRT`) | the deployed now-/forecast |
| **Imagery** | NASA **GIBS** VIIRS true-colour (WMS) | display basemap only |
| **Live wind forecast** | **Open-Meteo** | the 1–3 day operational push |

Everything is resampled to a common **20 km EPSG:4326 grid** and cropped to model size **56 × 112**
(2:1, ~lon × lat). Training domain is Central Asia **[55, 37, 75, 47]** (lon/lat). The generalisation
regions (§5) are pulled with the *identical* recipe so the only thing that changes between training and
test is the geography.

**Channel recipe (9 channels per day):** `[AOD_norm, AOD_validmask, NO₂, UVAI, SO₂, (reserved), wind_u,
wind_v, precip]`, each clipped/normalised to ~[0, 1.5]. A critical, non-obvious detail: **AOD is
normalised by the 97th percentile of its *own* sensor** (MAIAC training scale ≈ 0.395; VIIRS by VIIRS) —
normalising VIIRS by MAIAC's scale saturates the input and paints a false "dust blanket." The AOD-only
model uses channels `[0, 1, 6, 7, 8]`.

---

## 3. Method

### 3.1 Architecture
A ConvLSTM **encoder–forecaster** (`EncFc`): a convolutional-LSTM encoder consumes **T_IN = 4** past
daily frames into a spatial hidden state; a convolutional-LSTM forecaster then rolls out **K_OUT = 3**
future frames. It is **advection-aware** — at each forecast step it is *fed that day's future wind and
precipitation* (the exogenous channels `[6, 7, 8]`), so the network can move dust with the flow instead
of inventing it. Trained with an MSE objective on the AOD channel; PyTorch on Apple-MPS (an 8 GB M2
laptop — no datacentre GPU).

### 3.2 Two model variants
- **Multimodal** — all 9 channels (AOD + NO₂ + UVAI + SO₂ + wind + precip).
- **AOD-only** — 5 channels (AOD + mask + wind + precip).

Both share the `EncFc(in=5, out=1, exog=3)` forecaster head; the comparison isolates *what the extra
satellite chemistry buys.*

### 3.3 Baselines & metrics
- **Persistence** — "tomorrow looks like today," the standard hard-to-beat short-range baseline.
- **Climatology** — the per-pixel seasonal mean field; the standard that separates *forecasting* from
  *knowing the average*.
- **Pattern correlation (r)** — spatial correlation between predicted and observed dust field (over
  valid pixels), per lead day.
- **Anomaly Correlation Coefficient (ACC)** — correlation *after subtracting climatology*. This is the
  honest transport metric: it strips out "the desert is always dusty" and asks **did the model get the
  day's deviation — the actual moving plume — right?**

Sequences are kept only when the target window carries real dust (valid-mask mean > 0.12), so we score
on dust days, not empty sky.

---

## 4. In-domain results (Central Asia)

### 4.1 The model beats persistence
On 1,080 held-out Central-Asia sequences, the ConvLSTM beats persistence at every lead:

| Lead | ConvLSTM r | Persistence r |
|---|---|---|
| +1 day | **0.50** | 0.38 |
| +2 day | **0.44** | 0.32 |
| +3 day | **0.38** | 0.24 |

![Regional ConvLSTM vs persistence](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/convlstm_region.png)

### 4.2 Multimodal fusion adds nothing for dust — a clean null
Giving the model NO₂, UVAI and SO₂ on top of AOD does **not** help it forecast dust:

| Lead | Multimodal r | AOD-only r | Persistence r |
|---|---|---|---|
| +1 day | 0.547 | **0.555** | 0.31 |
| +2 day | 0.556 | **0.562** | 0.32 |
| +3 day | 0.519 | **0.525** | 0.28 |

The two curves are within 0.01 of each other and the AOD-only model is *marginally better*. For the NO₂
target the model is strongly better than persistence (r ≈ 0.75 vs 0.43–0.58) — but that is because NO₂ is
a smoother, more persistent field, not because fusion unlocked anything. **The extra modalities are not
informative for next-day dust.** We report this rather than bury it.

![Multimodal vs AOD-only](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/convlstm_multimodal.png)

### 4.3 The climatology ceiling
A 7-day rollout scored against the climatology baseline is the decisive test. Beyond ~3 days the model
flattens toward climatology: most of its apparent multi-day "skill" **is** the seasonal mean, and the
genuine forecast skill above climatology is modest (dust ≈ +0.13–0.19 r over climatology, shrinking with
lead). This is the honest ceiling — the model is **climatology-limited**, which §5 shows is a property of
the *data*, not the architecture.

![7-day rollout vs climatology](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/test_7day.png)

### 4.4 Transport skill (ACC)
Strip out climatology and the model still tracks where *existing* dust is heading better than persistence
— anomaly-correlation ≈ 0.3, roughly 2× persistence at +1 day. The signature is consistent across the
study: **direction is reliable; distance is under-shot** (damped advection — the model is conservative
about how far a plume travels). "Which way is the dust going" it answers honestly; "exactly how far by
Thursday" it does not.

![Transport / ACC](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/test_transport.png)

---

## 5. The central result — it generalises across three continents

This is the experiment the paper is built around. We took the **Central-Asia-trained AOD model,
unchanged**, and ran it over four dust regions it never saw, pulled with the identical recipe over each
region's own dust season:

| Region | Domain (lon/lat) | Season | n seq |
|---|---|---|---|
| **Iran** | [46, 27, 66, 37] | summer | 86 |
| **Sahara / Sahel** | [0, 12, 30, 27] (Bodélé) | summer | 41 |
| **Middle East** | [38, 22, 58, 35] (Arabian/Iraqi Shamal) | summer | 41 |
| **Gobi (Mongolia)** | [98, 40, 118, 48] | spring | 41 |

**Result: it beats persistence on the dust-field pattern correlation in every region, at every lead.**

![Generalisation across regions](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/paper2_generalization.png)

| Region | +1d model r | +1d persist r | +1d ACC (model / persist) |
|---|---|---|---|
| **Iran** | **0.69** | 0.55 | 0.50 / 0.42 |
| **Sahara / Sahel** | **0.53** | 0.30 | 0.19 / 0.12 |
| **Middle East** | **0.73** | 0.59 | 0.29 / 0.28 |
| **Gobi (Mongolia)** | **0.34** | 0.19 | 0.17 / 0.08 |

The lift over persistence is *largest where dust is least persistent* — Sahara (+77%) and Gobi (+79% on
r) — exactly where a learned mover should help most. Side-by-side truth-vs-forecast maps show the model
reproducing the broad plume geometry in every region (slightly smoothed — the same damped-advection
signature as in-domain):

![Truth vs forecast across four regions](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/paper2_regions_maps.png)

And the original single-region transfer test (Iran, in detail) that motivated the sweep:

![Iran generalisation detail](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/dust_test_iran.png)

**The honest caveat on ACC.** Pattern correlation (climatology + anomaly) improves *everywhere, at every
lead.* The pure **transport-anomaly** skill (ACC) is clearest at **+1 day** and across all leads in **Iran
and the Sahara**; in the **Middle East and Gobi the ACC edge washes out by +2–3 days** (model ≈
persistence ≈ ≈0 anomaly skill). So the transfer claim is strong for *next-day structure* and for
*direction*, and deliberately not overstated for multi-day plume *distance*. Full multi-lead numbers are
in [`models/paper2_regions.json`](https://github.com/imkazaimka/tashkent-air/blob/main/models/paper2_regions.json).

**Why this matters:** a model that beats persistence on the Bodélé depression and the Mongolian Gobi
*without ever being trained there* has not memorised a map — it has learned a transferable approximation
of how a dust field advects and decays. That is the one genuinely re-usable thing in the project.

---

## 6. The deployed system

The trained network is wired into a small live dust-watch (the only place it runs in production):

- `pull_lance.py` — NASA LANCE NRT VIIRS AOD (freshest 16 granules; ~3 h after a daytime overpass).
- `dust_map.py` / `dust_anim.py` — GIBS true-colour maps + an animated 7-day nowcast with a per-frame
  direction arrow (classical CV: connected-components + AOD-weighted centroid tracking; **0% of the
  trained net** — this layer is deliberately model-free and robust).
- `dust_forecast.py` — **the ConvLSTM**, projecting dust 1–3 days ahead on the live feed (VIIRS
  self-normalised, Open-Meteo future wind as the exogenous push). *This is the only production use of the
  trained model.*
- `dust_server.py` — a self-refreshing web dashboard; `dust_watch.py` — a terminal report.

![Live forecast on true-colour](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/forecast_rgb.png)
![Live forecast](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/forecast_live.png)
![Live dust map](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/dust_map_watch_rgb.png)
![Animated nowcast](https://raw.githubusercontent.com/imkazaimka/tashkent-air/main/figures/dust_anim.gif)

---

## 7. What the model is genuinely good at

1. **Transferable next-day dust structure.** One model, never retrained, beats persistence on the dust
   field across **Iran, the Sahara, the Middle East and the Gobi** — three continents — by **+0.12 to
   +0.23 r at +1 day**. The learned dynamics are universal, not Tashkent-specific.
2. **Reading dust *direction*.** With climatology removed it still tracks where existing dust is heading
   (ACC ≈ 2× persistence at +1 day). Direction is the reliable output.
3. **A complete, honest, laptop-scale pipeline.** NRT satellite pull → classical tracking → ConvLSTM
   forecast → self-updating dashboard, end-to-end on an 8 GB M2, for a region that had no such open tool.
4. **Knowing its own ceiling, measurably.** The fusion null (§4.2), the climatology rollout (§4.3) and
   the per-region ACC breakdown (§5) are *tests*, not assertions — every boundary is quantified and
   reproducible from source.

**One sentence:** it is a small, transferable model that honestly answers *"is there dust, where, and
which way is it drifting over the next day or two"* on a laptop — and is candid that it cannot do more.

---

## 8. Limitations (stated plainly)

- **Multimodal fusion is a null** for dust — AOD alone is as good (§4.2).
- **Climatology-limited** — genuine skill above the seasonal mean is modest and decays to ≈0 by day 3 in
  the harder regions; this is a **data limit** (a few dust seasons), not an architecture limit. Within
  that limit, swapping architectures does not move the needle.
- **Damped advection** — plume *direction* is trustworthy, plume *distance/intensity* is under-shot.
- **Satellite latency** — fresh ~3–5 h after a daytime overpass, nothing overnight (no passive
  instrument sees dust in the dark or through thick cloud); unusable for fast haboobs.
- **Cannot see surface PM2.5** — column AOD ≈ uncorrelated with winter surface PM2.5 (r ≈ 0); the
  dangerous-days problem is Paper 1's ground model, not this one.
- **Not novel, not operational-grade** — ConvLSTM, advection-aware rollout, CV tracking and gap-fill are
  all established; CAMS and Aurora forecast regional dust better with 3-D physics + assimilation. The
  contribution is the *characterised, transferable, open, local* system, not a new method.

---

## 9. Reproducibility & where the images are

**Repository:** https://github.com/imkazaimka/tashkent-air
**All figures (browse):** https://github.com/imkazaimka/tashkent-air/tree/main/figures
**Regional results JSON:** [`models/paper2_regions.json`](https://github.com/imkazaimka/tashkent-air/blob/main/models/paper2_regions.json)

```bash
# pull the four dust regions (identical recipe to training)
EE_PROJECT=civil-sentry-379101 python src/pull_dustregions.py     # Sahara, Middle East, Gobi
EE_PROJECT=civil-sentry-379101 python src/pull_iran.py            # Iran

# reproduce the central generalisation result (Fig. paper2_generalization / paper2_regions_maps)
python src/test_regions.py

# in-domain skill, fusion null, climatology rollout, transport
python src/convlstm_region.py        # Fig. convlstm_region
python src/convlstm_multimodal.py    # Fig. convlstm_multimodal  (the fusion null)
python src/test_7day.py              # Fig. test_7day            (the climatology ceiling)
python src/test_transport.py         # Fig. test_transport / forecast_rgb

# the live system
python src/dust_server.py            # → http://localhost:8000
```

### Figure index
| Figure | File | Shows |
|---|---|---|
| 1 | `convlstm_region.png` | in-domain ConvLSTM vs persistence (Central Asia) |
| 2 | `convlstm_multimodal.png` | multimodal vs AOD-only — the fusion null |
| 3 | `test_7day.png` | 7-day rollout vs climatology — the ceiling |
| 4 | `test_transport.png` | transport / anomaly-correlation skill |
| 5 | **`paper2_generalization.png`** | **one model beats persistence on every region** |
| 6 | **`paper2_regions_maps.png`** | **truth vs forecast across four regions** |
| 7 | `dust_test_iran.png` | Iran transfer, in detail |
| 8 | `forecast_rgb.png` / `forecast_live.png` | live 1–3 day forecast |
| 9 | `dust_map_watch_rgb.png` | live dust map |
| 10 | `dust_anim.gif` | animated nowcast with direction arrow |

*Data note: raw satellite arrays and any ground data are regenerated by the scripts and not committed;
WAQI ground data are used under their Data-Use Statement and not redistributed.*
