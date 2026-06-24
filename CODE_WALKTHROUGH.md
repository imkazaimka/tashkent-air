# Code Walkthrough — Building the Tashkent PM2.5 Study Step by Step

A **code-along companion** to `Tashkent-PM25-ML-Study.md`. Where the paper explains *what we
found*, this explains *how the code does it* — annotated snippets from each script,
in the order you'd run them. Aimed at programmers who want to learn the data-science
moves by reading real, working code.

> Run order (each re-derives from raw data, so all are safe to re-run):
> ```
> collect → impute_blh → features → train → research → tier_a_validation
> → fetch_openaq → validate_ground_truth → train_ground_truth
> → train_episode_classifier → shap_analysis → back_trajectories → figures_paper
> ```

Every script starts the same way — load central config so paths, locations, and
constants live in one place ([config.py](config.py)):

```python
import config as C          # C.TASHKENT, C.REGIONAL_CITIES, C.PROCESSED, C.SPLIT ...
```

> 🧑‍🏫 **Lesson — one source of truth.** Hard-coding the same latitude in ten files
> guarantees they'll drift apart. Put constants in one module and import it.

---

## 1. Collecting the data — [`src/collect.py`](src/collect.py)

We pull from free APIs (no key) and aggregate hourly → daily. Two details matter.

**Robust HTTP with backoff** — networks fail; retry politely:

```python
def get_json(url, params, retries=4, pause=2.0):
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(pause * (2 ** attempt))   # exponential backoff: 2s, 4s, 8s...
    raise RuntimeError(f"Failed after {retries} attempts: {url}")
```

**Correct daily aggregation of wind direction** — you *cannot* average angles
naïvely (the mean of 350° and 10° is 180°, the exact opposite of the truth). Convert
to vector components first:

```python
rad = np.radians(df["wind_direction_10m"])
u = -spd * np.sin(rad)        # east-west component
v = -spd * np.cos(rad)        # north-south component
u_d, v_d = u.resample("1D").mean(), v.resample("1D").mean()
daily["wind_direction_10m"] = (np.degrees(np.arctan2(-u_d, -v_d)) % 360)
```

> 🧑‍🏫 **Lesson — circular quantities.** Anything cyclic (wind direction, time of
> day, day of year) needs `sin`/`cos` or vector handling, never plain arithmetic.

→ Produces `data/processed/daily_merged.csv`. (Paper §2, Figure 1.)

---

## 2. Filling a 6-month gap honestly — [`src/impute_blh.py`](src/impute_blh.py)

Boundary-layer height was missing for Jan–Jun 2024. We predict it from the weather we
*do* have, at **hourly** resolution (its daily cycle is very regular), then validate
the fill on a held-out block that mimics the real gap:

```python
def validate(h):
    have = h[h[TARGET].notna()]
    blk = (have.index >= "2023-01-01") & (have.index <= "2023-06-30")  # fake gap
    m = make_model(); m.fit(have[~blk][PREDICTORS], have[~blk][TARGET])
    pred = m.predict(have[blk][PREDICTORS])
    print("held-out R²:", r2_score(have[blk][TARGET], pred))     # ~0.46
```

We then **flag** every filled value and keep the measured-only copy:

```python
df["boundary_layer_height_era5"] = measured_daily   # provenance: measured only
df["blh_imputed"] = imputed_day.astype(int)         # 1 = this value is modelled
```

> 🧑‍🏫 **Lesson — never impute silently.** A flag column lets you later prove your
> conclusions don't hinge on guesses (ours don't — all synthetic data is in `train`).

→ Paper §2 "data-quality notes."

---

## 3. Feature engineering — [`src/features.py`](src/features.py)

This is where domain knowledge becomes numbers, and where leakage is prevented.

**Target-day alignment (fixing a subtle leak).** Each row predicts day `D` using
day-`D` weather (forecastable) and pollution lags from `D-1` and earlier:

```python
f["y"] = pm                                  # target = today's actual PM2.5
for k in (1, 2, 3, 7):
    f[f"pm25_lag{k}"] = pm.shift(k)          # only the PAST
f["pm25_roll7_mean"] = pm.shift(1).rolling(7).mean()   # rolling stat ends YESTERDAY
```

**The physics features** — encode the science directly:

```python
ventilation = df["boundary_layer_height"] * df["wind_speed_10m"]
f["trapping_index"] = 1.0 / (ventilation + 1e-6)        # high = stagnant air
wd = np.radians(df["wind_direction_10m"])
f["wind_sin"], f["wind_cos"] = np.sin(wd), np.cos(wd)   # circular encoding
# transport: upwind pollution × whether wind blows FROM that city × wind speed
toward = np.cos(np.radians(df["wind_direction_10m"] - city["bearing"]))
f[f"{city}_transport"] = lag1 * toward * df["wind_speed_10m"]
```

**Automated leakage checks** — assert the pipeline didn't cheat:

```python
assert TARGET not in f.columns                       # raw answer not a feature
assert np.allclose(f["pm25_lag1"][7:], raw[TARGET].shift(1)[7:], equal_nan=True)
```

> 🧑‍🏫 **Lesson — encode the answer's *unavailability*.** We dropped same-day sibling
> pollutants (pm10, dust) because they come from the same model as the target —
> using them would be circular. Ask of every feature: "would I really have this at
> prediction time?"

→ Produces `features.csv` (48 features). Paper §3.

---

## 4. The baseline & the forecaster — [`src/train.py`](src/train.py)

**Always build the dumb baseline first:**

```python
persist = test["pm25_lag1"]                  # "tomorrow = today"
print("persistence MAE:", mean_absolute_error(y_test, persist))   # 3.99
```

**Then the model**, with early stopping on a validation set:

```python
model = lgb.train(params, lgb.Dataset(Xtr, ytr),
                  valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(50)])   # stop when val stops improving
```

> 🧑‍🏫 **Lesson — early stopping prevents overfitting.** The model keeps adding trees
> only while the *validation* score improves, then stops — you don't guess the tree
> count. Result: it beat persistence by just 1.8% on MAE but 11.5% on RMSE. Honest
> reporting means showing both. Paper §3.

---

## 5. Testing the hypotheses — [`src/research.py`](src/research.py)

**H‑A, the wind-direction quasi-experiment.** The whole transport claim rests on one
interaction coefficient, with autocorrelation-robust standard errors:

```python
al = cos*np.cos(radians(bearing)) + sin*np.sin(radians(bearing))  # inbound alignment
d = pd.DataFrame({"y": y, "city": city_pm25, "align": al,
                  "inter": city_pm25 * al, "vent": ventilation, "temp": temp})
ols = sm.OLS(d["y"], sm.add_constant(zscore(d[[...]]))
            ).fit(cov_type="HAC", cov_kwds={"maxlags": 7})   # robust SEs for time series
# the coefficient on "inter" is the transport signal
```

> 🧑‍🏫 **Lesson — HAC standard errors.** Time-series rows aren't independent (today
> resembles yesterday). HAC (Newey–West) errors widen the uncertainty so you don't
> declare false significance. Naïve p-values on serial data lie.

**H‑B, controlling for a confounder** — regress out dispersion, then test temperature
on the residual:

```python
resid = y - sm.OLS(y, sm.add_constant(np.log(vent+1))).fit().predict(...)
qr = sm.OLS(resid, sm.add_constant([z, z**2])).fit(cov_type="HAC", ...)
# raw temp slope −5.65 → residual slope −0.28 (p=0.49): the cold effect was dispersion
```

**H‑C, attribution by ablation** — how much R² does each feature group uniquely add?

```python
r2_drop = r2_score(yte, lgbm(Xtr[without_group], ytr).predict(Xte[without_group]))
delta = r2_full - r2_drop          # unique contribution of that group
```

→ Paper §§4–6, Figures 6, 8, 9.

---

## 6. Trying to break it — [`src/tier_a_validation.py`](src/tier_a_validation.py)

**Permutation / placebo test** — scramble the wind, rebuild a null distribution:

```python
obs = inter_beta(y, city, sin, cos, bearing, vent, temp)
null = [inter_beta(y, city, *shuffle(sin, cos), bearing, vent, temp)
        for _ in range(1000)]
emp_p = np.mean(np.array(null) >= obs)     # how often noise beats the real signal → 0.000
```

**Bearing sweep** — does the signal peak at the city's *true* compass direction?

```python
betas = [inter_beta(y, city, sin, cos, b, vent, temp) for b in range(0, 360, 10)]
peak = bearings[np.argmax(betas)]          # Bishkek → 50°, true bearing 50°
```

**Matched pairs** — same temperature, different mixing height:

```python
for tbin, g in d.groupby((d.temp/2).round()*2):       # 2°C bins
    lo = g[g.blh <= g.blh.median()].pm.mean()          # trapped air
    hi = g[g.blh >  g.blh.median()].pm.mean()          # ventilated
    diffs.append(lo - hi)                              # mean +3.4 µg/m³ (p=4e-4)
```

**Block bootstrap** — confidence intervals that respect autocorrelation:

```python
def _block_boot(stat_fn, n, n_boot=1000, block=14):
    return np.percentile([stat_fn(resample_blocks(n, block)) for _ in range(n_boot)],
                         [2.5, 97.5])     # ventilation-controlled temp CI includes 0
```

> 🧑‍🏫 **Lesson — these four tools are model-agnostic.** Permutation, sweep, matched
> pairs and bootstrap need no fancy library — just resampling and counting. They are
> how you turn "interesting correlation" into "credible finding." Paper §9.

---

## 7. The reality check — [`src/fetch_openaq.py`](src/fetch_openaq.py) + [`src/validate_ground_truth.py`](src/validate_ground_truth.py)

Pull the real sensor (paginated API), then compare to CAMS:

```python
pear = stats.pearsonr(cams, ground)          # r = 0.57
bias = cams.mean() - ground.mean()           # −21.6 (CAMS = 46% of reality)
recall = (gb & cb).sum() / gb.sum()          # 0.19 — misses 81% of real episodes
```

The punchline: re-run the *same* hypothesis tests with `ground` as the target and
confirm the conclusions hold.

> 🧑‍🏫 **Lesson — a model that fits its own (model) data can still be wrong about the
> world.** Ground truth is the only cure. Paper §7, Figure 10.

---

## 8. Fixing the model — [`src/train_ground_truth.py`](src/train_ground_truth.py)

Reframe: CAMS becomes an *input*; the real sensor is the target; model **log** PM2.5
because of the skew:

```python
ylog = lambda s: np.log1p(s.clip(lower=0))           # tame the heavy tail
model = fit_lgbm(Xtr, ylog(ytr_real), Xva, ylog(yva_real))
pred = np.expm1(model.predict(Xte))                  # invert the log for real units
```

Compare against *every* sensible baseline (raw CAMS, ×2 rescale, persistence) so the
ML model has to earn its place. Episode recall went 0.17 → 0.79. Paper §8, Figure 11.

---

## 9. Episode warnings & explanation — [`train_episode_classifier.py`](src/train_episode_classifier.py), [`shap_analysis.py`](src/shap_analysis.py)

**Classification with class imbalance** (bad days are rarer):

```python
pos_w = (ytr == 0).sum() / (ytr == 1).sum()          # up-weight the rare class
clf = lgb.LGBMClassifier(scale_pos_weight=pos_w, ...)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1]) # 0.86 for >55 µg/m³
```

**Onset recall** — the hardest, most useful metric (first day of a spike):

```python
onset = exceed & ~exceed.shift(1, fill_value=False)  # True only on day 1 of a run
onset_recall = ((proba >= t) & onset).sum() / onset.sum()   # 0.77
```

**SHAP** — open the black box:

```python
sv = shap.TreeExplainer(booster).shap_values(X)
imp = np.abs(sv).mean(axis=0)            # mean |contribution| per feature
# grouped: dispersion 30%, regional 22%, season 20% — confirms H-C on REAL data
```

> 🧑‍🏫 **Lesson — pick the metric that matches the goal.** For health alerts, *onset
> recall* matters more than overall accuracy: catching the *start* of an episode is
> what saves lungs. Paper §8–9, Figures 12, 16.

---

## 10. An independent proof — [`src/back_trajectories.py`](src/back_trajectories.py)

Using only wind (no CAMS), integrate the air parcel backward in time:

```python
u = -spd*np.sin(radians(dir));  v = -spd*np.cos(radians(dir))   # wind vector
for k in range(48):                                # step back 48 hours
    lat -= (v*3600)/110540.0                        # ~metres-per-degree conversions
    lon -= (u*3600)/(111320.0*np.cos(radians(lat)))
# dirtiest days → air origin 63° (E/NE, 73%); cleanest → 320° (NW)
```

> 🧑‍🏫 **Lesson — triangulate.** A conclusion reached by three independent methods
> (statistics, falsification, physics) is far stronger than one. Paper §9, Figure 15.

---

## 11. Publication figures — [`src/figures_paper.py`](src/figures_paper.py)

One shared style block keeps every figure consistent (fonts, grid, colours), then
each function builds one chart — histogram, wind-rose (polar bar of PM2.5 by wind
sector), confounding scatter, correlation heatmap, ROC + confusion matrix, transport
compass.

```python
plt.rcParams.update({"axes.spines.top": False, "axes.grid": True, ...})  # set once
ax = fig.add_subplot(111, projection="polar")   # wind-rose / compass
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)   # compass orientation
```

> 🧑‍🏫 **Lesson — a figure is an argument.** Figure 5 (wind-rose) and Figure 6
> (compass) make the directional-transport finding obvious at a glance — often more
> persuasive than the table behind them.

---

## How the code maps to the paper

| Paper section | Script(s) | Figure(s) |
|---------------|-----------|-----------|
| §2 Data | `collect`, `impute_blh`, `figures_paper` | 1–4 |
| §3 Methods | `features`, `train` | — |
| §4 H-A transport | `research` | 5, 6 |
| §5 H-B temperature | `research`, `figures_paper` | 7, 8 |
| §6 H-C attribution | `research` | 9 |
| §7 Validation | `fetch_openaq`, `validate_ground_truth` | 10 |
| §8 Models | `train_ground_truth`, `train_episode_classifier` | 11, 12 |
| §9 Robustness | `tier_a_validation`, `back_trajectories`, `shap_analysis` | 13–16 |

That's the whole study — from a raw API call to a falsified, ground-truth-validated,
explainable conclusion. Clone it, change the coordinates in `config.py`, and you have
an air-quality study for *your* city.
