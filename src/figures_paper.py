"""
Publication-quality figures for the research paper.

Generates a cohesive, professionally-styled set of data visualisations that were
missing from the analysis scripts: distributions, the full record, seasonal
boxplots, a pollution wind-rose, the temperature/boundary-layer confounding
scatter, a correlation heatmap, classifier ROC + confusion matrix, and a
transport compass.

Run:  python src/figures_paper.py
Outputs: figures/fig_*.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.ROOT / "figures"
INK = "#1b2a4a"; ACCENT = "#c0392b"; COOL = "#2980b9"; GREEN = "#16a085"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "font.family": "sans-serif",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#9aa5b1",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d8dee6", "grid.alpha": 0.7,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load():
    dm = pd.read_csv(C.PROCESSED / "daily_merged.csv", parse_dates=["date"])
    ft = pd.read_csv(C.PROCESSED / "features.csv", parse_dates=["date"])
    gt = pd.read_csv(C.RAW / "openaq_embassy_pm25_daily.csv", parse_dates=["date"])
    # the embassy CSV holds pre-period history (back to ~2019); restrict the ground truth
    # to the study window so every figure matches the analysis and Table 3.
    gt = gt[(gt["date"] >= dm["date"].min()) & (gt["date"] <= dm["date"].max())].reset_index(drop=True)
    return dm, ft, gt


# ----------------------------------------------------------------- 1. distribution
def fig_distribution(dm, gt):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    g = gt["pm25_ground"].dropna()
    ax[0].hist(g, bins=60, color=COOL, alpha=0.85, edgecolor="white")
    ax[0].axvline(g.mean(), color=ACCENT, lw=2, label=f"mean {g.mean():.0f}")
    ax[0].axvline(g.median(), color=INK, lw=2, ls="--", label=f"median {g.median():.0f}")
    ax[0].axvline(35, color="#e67e22", lw=1.5, ls=":", label="35 (unhealthy)")
    ax[0].set(title="A  Real PM2.5 is heavy-tailed (right-skewed)",
              xlabel="PM2.5 (µg/m³)", ylabel="days"); ax[0].legend()
    ax[1].hist(np.log1p(g), bins=60, color=GREEN, alpha=0.85, edgecolor="white")
    ax[1].set(title="B  …but roughly bell-shaped in log space\n(why we model log PM2.5)",
              xlabel="log(1 + PM2.5)", ylabel="days")
    fig.tight_layout()
    fig.savefig(FIG / "fig_distribution.png"); plt.close(fig)


# ----------------------------------------------------------------- 2. full record
def fig_timeseries(dm, gt):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    CAP = 150  # cap the y-axis so the seasonal rhythm is readable; flag the few episode days
    ax.plot(dm["date"], dm["pm2_5"], color=COOL, lw=0.7, alpha=0.6, label="CAMS model")
    ax.plot(dm["date"], dm["pm2_5"].rolling(30, min_periods=10).mean(),
            color=INK, lw=1.6, label="CAMS 30-day avg")
    inr = gt[gt["pm25_ground"] <= CAP]
    hi = gt[gt["pm25_ground"] > CAP]
    ax.scatter(inr["date"], inr["pm25_ground"], s=5, color=ACCENT, alpha=0.5, label="real sensor")
    ax.scatter(hi["date"], np.full(len(hi), CAP + 3), s=24, marker="^", color="#7b241c",
               zorder=5, label=f"{len(hi)} episode days >{CAP} (peak {gt['pm25_ground'].max():.0f})")
    ax.axhline(35, color="#e67e22", ls=":", lw=1)
    ax.set(title="The full record: model vs reality, 2022–2026",
           ylabel="PM2.5 (µg/m³)", xlabel="date", ylim=(0, CAP + 12))
    ax.legend(ncol=4, loc="upper right", fontsize=8.5)
    fig.savefig(FIG / "fig_timeseries_full.png"); plt.close(fig)


# ----------------------------------------------------------------- 3. monthly box
def fig_monthly_box(dm):
    dm = dm.assign(m=dm["date"].dt.month)
    data = [dm[dm["m"] == k]["pm2_5"].dropna().values for k in range(1, 13)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=2))
    for i, box in enumerate(bp["boxes"]):
        heat = cm.RdYlBu_r((dm.groupby("m")["pm2_5"].mean().iloc[i] - 8) / 25)
        box.set(facecolor=heat, alpha=0.85, edgecolor="#9aa5b1")
    ax.set(title="Seasonal cycle: PM2.5 distribution by month (winter is the problem)",
           xlabel="month", ylabel="PM2.5 (µg/m³)",
           xticklabels=["J","F","M","A","M","J","J","A","S","O","N","D"])
    fig.savefig(FIG / "fig_monthly_box.png"); plt.close(fig)


# ----------------------------------------------------------------- 4. wind rose
def fig_windrose(dm, gt):
    d = dm.merge(gt, on="date").dropna(subset=["wind_direction_10m", "pm25_ground"])
    sectors = np.arange(0, 360, 22.5)
    labels = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    means = []
    for s in sectors:
        lo, hi = (s - 11.25) % 360, (s + 11.25) % 360
        wd = d["wind_direction_10m"]
        m = (wd >= lo) & (wd < hi) if lo < hi else (wd >= lo) | (wd < hi)
        means.append(d.loc[m, "pm25_ground"].mean())
    means = np.array(means)
    theta = np.radians(sectors)
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    colors = cm.RdYlBu_r((means - np.nanmin(means)) / (np.nanmax(means) - np.nanmin(means)))
    ax.bar(theta, means, width=np.radians(20), color=colors, edgecolor="white", alpha=0.9)
    ax.set_xticks(theta); ax.set_xticklabels(labels)
    ax.set_title("Pollution wind-rose: mean real PM2.5 by wind origin\n"
                 "(dirtiest air arrives from the E/NE)", fontweight="bold", color=INK, pad=20)
    fig.savefig(FIG / "fig_windrose.png"); plt.close(fig)


# ----------------------------------------------------------------- 5. temp/BLH scatter
def fig_temp_blh(dm):
    d = dm.dropna(subset=["temperature_2m", "pm2_5", "boundary_layer_height"])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sc = ax.scatter(d["temperature_2m"], d["pm2_5"], c=d["boundary_layer_height"],
                    cmap="viridis", s=14, alpha=0.7, vmax=900)
    cb = plt.colorbar(sc); cb.set_label("boundary-layer height (m)", color=INK)
    ax.set(title="The confounder, visualised: cold + low mixing-lid = dirty\n"
                 "(dark points = trapped air, cluster top-left)",
           xlabel="temperature (°C)", ylabel="PM2.5 (µg/m³)")
    fig.savefig(FIG / "fig_temp_blh_scatter.png"); plt.close(fig)


# ----------------------------------------------------------------- 6. corr heatmap
def fig_corr(dm, ft):
    cols = {"PM2.5": dm["pm2_5"], "temp": dm["temperature_2m"],
            "humidity": dm["relative_humidity_2m"], "pressure": dm["surface_pressure"],
            "wind spd": dm["wind_speed_10m"], "BL height": dm["boundary_layer_height"],
            "radiation": dm["shortwave_radiation"], "precip": dm["precipitation"],
            "trapping": ft.set_index("date")["trapping_index"].reindex(dm["date"]).values,
            "Samarkand": dm["samarkand_pm25"], "Fergana": dm["fergana_pm25"]}
    M = pd.DataFrame(cols).corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(M))); ax.set_yticks(range(len(M)))
    ax.set_xticklabels(M.columns, rotation=45, ha="right"); ax.set_yticklabels(M.columns)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.iloc[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(M.iloc[i,j]) > 0.5 else INK, fontsize=8)
    ax.grid(False); plt.colorbar(im, fraction=0.046)
    ax.set_title("Correlation heatmap of key daily variables", color=INK)
    fig.savefig(FIG / "fig_corr_heatmap.png"); plt.close(fig)


# ----------------------------------------------------------------- 7. classifier eval
def fig_classifier(dm, ft, gt):
    DROP = ["pm25_lag1","pm25_lag2","pm25_lag3","pm25_lag7","pm25_roll3_mean",
            "pm25_roll7_mean","pm25_roll7_std","pm25_diff1","episode_streak"]
    cams = dm[["date","pm2_5","pm10","dust","nitrogen_dioxide","ozone","carbon_monoxide"]
              ].rename(columns=lambda c: "cams_"+c if c != "date" else c)
    df = (ft.drop(columns=[c for c in DROP if c in ft.columns])
          .merge(cams, on="date").merge(gt, on="date")
          .dropna(subset=["pm25_ground","cams_pm2_5"]).sort_values("date"))
    cols = [c for c in df.columns if c not in (["date","y","split","pm25_ground","ground_lag1"])]
    tr = df[df["date"] < "2024-06-01"]; te = df[df["date"] >= "2024-06-01"]
    ytr = (tr["pm25_ground"] > 55).astype(int); yte = (te["pm25_ground"] > 55).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                             scale_pos_weight=(ytr==0).sum()/max((ytr==1).sum(),1),
                             random_state=42, verbose=-1)
    clf.fit(tr[cols], ytr)
    proba = clf.predict_proba(te[cols])[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba); auc = roc_auc_score(yte, proba)
    cmat = confusion_matrix(yte, (proba >= 0.5).astype(int))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"classifier (AUC={auc:.2f})")
    ax[0].plot([0,1],[0,1],"--",color="#9aa5b1",label="random (AUC=0.50)")
    ax[0].fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax[0].set(title="Episode classifier (>55 µg/m³): ROC curve",
              xlabel="false-positive rate", ylabel="true-positive rate"); ax[0].legend()
    ax[1].imshow(cmat, cmap="Blues"); ax[1].grid(False)
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cmat[i,j], ha="center", va="center", fontsize=16,
                       color="white" if cmat[i,j] > cmat.max()/2 else INK)
    ax[1].set(title="Confusion matrix", xticks=[0,1], yticks=[0,1],
              xticklabels=["calm","episode"], yticklabels=["calm","episode"],
              xlabel="predicted", ylabel="actual")
    fig.savefig(FIG / "fig_classifier_eval.png"); plt.close(fig)


# ----------------------------------------------------------------- 8. transport compass
def fig_compass():
    res = json.load(open(C.ROOT / "models" / "research_results.json"))["H_A"]["wind_interaction"]
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    for c in C.REGIONAL_CITIES:
        b = res[c["name"]]["beta_interaction"]; th = np.radians(c["bearing"])
        col = ACCENT if b > 0 else COOL
        ax.annotate("", xy=(th, abs(b)), xytext=(th, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.text(th, abs(b)+0.4, f"{c['name']}\n{b:+.1f}", ha="center", fontsize=9, color=col)
    ax.set_ylim(0, 6); ax.set_yticks([2,4])
    ax.set_title("Transport compass: where dirty air really comes from\n"
                 "red = genuine inbound transport, blue = shared weather (not transport)",
                 fontweight="bold", color=INK, pad=24)
    fig.savefig(FIG / "fig_transport_compass.png"); plt.close(fig)


def main():
    dm, ft, gt = load()
    fig_distribution(dm, gt); print("  fig_distribution")
    fig_timeseries(dm, gt); print("  fig_timeseries_full")
    fig_monthly_box(dm); print("  fig_monthly_box")
    fig_windrose(dm, gt); print("  fig_windrose")
    fig_temp_blh(dm); print("  fig_temp_blh_scatter")
    fig_corr(dm, ft); print("  fig_corr_heatmap")
    fig_classifier(dm, ft, gt); print("  fig_classifier_eval")
    fig_compass(); print("  fig_transport_compass")
    print("Done -> figures/fig_*.png")


if __name__ == "__main__":
    main()
