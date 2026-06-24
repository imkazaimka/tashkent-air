"""
Reproducibility ledger — recompute every headline number in the paper from the cached
data and saved model metrics, and check it against the value the paper states.

This is the "proof for everything" layer: each claim below is regenerated from source
(a CSV, a parquet, or a saved *_metrics.json), compared to the figure printed in the
paper, and marked PASS / FAIL. It writes data/claims_ledger.csv and prints the table.

Run:  python src/verify_claims.py
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# ---- load every source once -------------------------------------------------
dm  = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
so2 = pd.read_csv(C.RAW / "cams_so2_daily.csv", parse_dates=["date"])
emb = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
tro = pd.read_csv(C.ROOT / "data" / "tropomi_cams_monthly.csv", parse_dates=["month"])
GTM = json.load(open(C.ROOT / "models" / "ground_truth_model_metrics.json"))
EPI = json.load(open(C.ROOT / "models" / "episode_classifier_metrics.json"))
IMP = json.load(open(C.ROOT / "models" / "improvements.json"))

d   = dm.merge(emb, on="date", how="left").merge(so2, on="date", how="left")  # study-window join
g   = d.dropna(subset=["pm25_ground"])                                        # days with a real sensor
gs  = d.dropna(subset=["pm25_ground", "so2"])                                 # + SO2 present (tracer-vs-real)
sw  = dm.merge(so2, on="date").dropna(subset=["so2"])                         # all days with SO2 (× weather)
mon = lambda df, col, ms: df[df.date.dt.month.isin(ms)][col].mean()
WIN, SUM = [11, 12, 1, 2, 3], [6, 7, 8]

# ---- claims: (group, statement, paper_value, recomputed, source) ------------
CLAIMS = []
def claim(group, stmt, paper, got, source, tol=0.05, absolute=False):
    try:
        gotv = float(got)
        ok = (abs(gotv - paper) <= tol) if absolute else (abs(gotv - paper) <= tol * max(abs(paper), 1e-9))
    except Exception:
        gotv, ok = got, (str(got) == str(paper))
    CLAIMS.append({"group": group, "claim": stmt, "paper": paper, "recomputed": gotv,
                   "status": "PASS" if ok else "FAIL", "source": source})

# Data & provenance
claim("Data", "daily records in study window", 1419, len(dm), "data/processed/daily_merged.csv", tol=0, absolute=True)
claim("Data", "real-sensor PM2.5 obs (in window)", 767, g.pm25_ground.notna().sum(), "openaq_embassy_pm25_daily.csv ∩ window", tol=0, absolute=True)
claim("Data", "real-sensor PM2.5 mean (µg/m³)", 40.4, g.pm25_ground.mean(), "descriptive_stats.py", tol=0.2, absolute=True)
claim("Data", "real-sensor PM2.5 max (µg/m³)", 289, g.pm25_ground.max(), "descriptive_stats.py", tol=0, absolute=True)
claim("Data", "BLH days model-imputed (flagged)", 183, int(dm.blh_imputed.sum()), "impute_blh.py", tol=0, absolute=True)

# CAMS bias (raw global model vs reality)
b = GTM["baseline_raw_cams"]; rc = GTM["baseline_cams_rescaled"]
claim("CAMS bias", "CAMS under-scale factor (≈2×)", 1.95, rc["rescale"][0], "ground_truth_model_metrics.json", tol=0.05, absolute=True)
claim("CAMS bias", "raw CAMS R² (worse than the mean)", -0.16, b["R2"], "ground_truth_model_metrics.json", tol=0.02, absolute=True)
claim("CAMS bias", "raw CAMS bad-day recall (misses ~83%)", 0.165, b["exceed35"]["recall"], "ground_truth_model_metrics.json", tol=0.02, absolute=True)

# Ground-truth model
gd = GTM["ground_model_deployable"]; gl = GTM["ground_model_plus_live_sensor"]; pe = GTM["baseline_persistence"]
claim("GT model", "deployable R² (real sensor)", 0.42, gd["R2"], "ground_truth_model_metrics.json", tol=0.01, absolute=True)
claim("GT model", "R² with yesterday's live reading", 0.52, gl["R2"], "ground_truth_model_metrics.json", tol=0.01, absolute=True)
claim("GT model", "bad-day precision @35", 0.78, gd["exceed35"]["precision"], "ground_truth_model_metrics.json", tol=0.02, absolute=True)
claim("GT model", "bad-day recall @35", 0.79, gd["exceed35"]["recall"], "ground_truth_model_metrics.json", tol=0.02, absolute=True)
claim("GT model", "deployable MAE (µg/m³)", 16.8, gd["MAE"], "ground_truth_model_metrics.json", tol=0.2, absolute=True)
claim("GT model", "persistence baseline R²", 0.40, pe["R2"], "ground_truth_model_metrics.json", tol=0.02, absolute=True)

# Warning / episode classifier
claim("Warning", "episode ROC-AUC @35", 0.873, EPI["thr_35"]["roc_auc"], "episode_classifier_metrics.json", tol=0.01, absolute=True)
claim("Warning", "episode PR-AUC (AP) @35", 0.849, IMP["classifier35_ap"], "improvements.json", tol=0.01, absolute=True)
claim("Warning", "episode ROC-AUC @55", 0.856, EPI["thr_55"]["roc_auc"], "episode_classifier_metrics.json", tol=0.01, absolute=True)
claim("Warning", "cross-winter recall (0.86±0.08)", 0.86, IMP["winter_cv"]["recall_mean"], "improvements.json (LOWO-CV)", tol=0.01, absolute=True)
claim("Warning", "cross-winter precision (0.80±0.04)", 0.80, IMP["winter_cv"]["precision_mean"], "improvements.json (LOWO-CV)", tol=0.01, absolute=True)

# Attribution / tracers (vs REAL ground PM2.5)
claim("Tracers", "SO₂ ↔ ground PM2.5  (r)", 0.71, stats.pearsonr(gs.so2, gs.pm25_ground)[0], "cams_so2 vs embassy", tol=0.02, absolute=True)
claim("Tracers", "NO₂ ↔ ground PM2.5  (r)", 0.60, stats.pearsonr(g.nitrogen_dioxide, g.pm25_ground)[0], "daily_merged vs embassy", tol=0.03, absolute=True)
claim("Tracers", "SO₂ on warm days >20 °C (µg/m³)", 3.6, sw.loc[sw.temperature_2m > 20, "so2"].mean(), "cams_so2 × temp (all days)", tol=0.2, absolute=True)
claim("Tracers", "SO₂ on deep-cold days <−5 °C (µg/m³)", 12.6, sw.loc[sw.temperature_2m < -5, "so2"].mean(), "cams_so2 × temp (all days)", tol=0.3, absolute=True)
claim("Tracers", "SO₂ winter÷summer (ground days)", 3.0, mon(gs, "so2", WIN) / mon(gs, "so2", SUM), "mazut_hypothesis.py [5]", tol=0.15, absolute=True)
claim("Tracers", "SO₂ enrichment, worst-10% ÷ low PM days", 3.3, gs.loc[gs.pm25_ground > gs.pm25_ground.quantile(.9), "so2"].mean() / gs.loc[gs.pm25_ground < gs.pm25_ground.quantile(.5), "so2"].mean(), "mazut_hypothesis.py [2]", tol=0.2, absolute=True)

# TROPOMI satellite cross-check
r_tro = stats.pearsonr(tro.cams_so2, tro.tropomi_so2)[0]
claim("Satellite", "TROPOMI ↔ CAMS SO₂ monthly (r)", 0.59, r_tro, "tropomi_cams_monthly.csv", tol=0.02, absolute=True)
claim("Satellite", "TROPOMI SO₂ winter÷summer", 5.16, mon(tro.rename(columns={'month':'date'}), "tropomi_so2", WIN) / mon(tro.rename(columns={'month':'date'}), "tropomi_so2", SUM), "tropomi_cams_monthly.csv", tol=0.2, absolute=True)
claim("Satellite", "CAMS SO₂ winter÷summer", 2.42, mon(tro.rename(columns={'month':'date'}), "cams_so2", WIN) / mon(tro.rename(columns={'month':'date'}), "cams_so2", SUM), "tropomi_cams_monthly.csv", tol=0.15, absolute=True)


def main():
    out = C.ROOT / "data" / "claims_ledger.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "claim", "paper", "recomputed", "status", "source"])
        w.writeheader()
        for c in CLAIMS:
            row = dict(c); row["recomputed"] = round(c["recomputed"], 3) if isinstance(c["recomputed"], float) else c["recomputed"]
            w.writerow(row)
    npass = sum(c["status"] == "PASS" for c in CLAIMS)
    grp = None
    print(f"{'CLAIM':<42}{'PAPER':>9}{'RECOMP':>9}  STATUS")
    print("-" * 72)
    for c in CLAIMS:
        if c["group"] != grp:
            grp = c["group"]; print(f"[{grp}]")
        rv = f"{c['recomputed']:.3f}" if isinstance(c["recomputed"], float) else str(c["recomputed"])
        print(f"  {c['claim']:<40}{c['paper']:>9}{rv:>9}  {c['status']}")
    print("-" * 72)
    print(f"{npass}/{len(CLAIMS)} claims reproduce from source → data/claims_ledger.csv")
    if npass != len(CLAIMS):
        print("FAILS:", [c["claim"] for c in CLAIMS if c["status"] == "FAIL"])


if __name__ == "__main__":
    main()
