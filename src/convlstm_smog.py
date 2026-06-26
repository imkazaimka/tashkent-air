"""
Hybrid ConvLSTM + tabular smog forecast over Tashkent.

Task ("is smog coming?"): from the last 5 daily satellite frames (D-5..D-1) plus tabular context,
predict whether day D is a smog day (US-Embassy measured PM2.5 > 35 µg/m³).

Modalities (co-registered 64x64):  true-color RGB (3) + MAIAC AOD (3) + AOD retrieval mask (1) = 7 ch.
Tabular (operationally available): ERA5 weather for day D (temp, RH, pressure, wind, mixing depth,
precip), recent ground PM2.5 (embassy lag-1/2/3/7), and season (doy sin/cos) = 12 features.

A single network runs image-only / tabular-only / hybrid via flags, so the ablation isolates exactly
what each modality adds. Honest evaluation: leave-one-winter-out (7 winters, 2018-2025), pooled
out-of-fold. Baselines: LightGBM on tabular (the "cheap-data" reference), persistence, base rate.

Key questions: (1) does the AOD channel beat RGB alone? (2) does imagery add anything over the
tabular ground+weather data?

Run:  python src/convlstm_smog.py
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FRGB = ROOT / "data" / "satellite" / "modis"; FAOD = ROOT / "data" / "satellite" / "aod"
MAN = ROOT / "data" / "satellite" / "modis_manifest.csv"
WX = ROOT / "data" / "processed" / "weather_era5_2018_2025.csv"
FIGDIR = ROOT / "figures"; OUT = ROOT / "models"
SIZE, L, THR = 64, 5, 35
NAVY="#16314f"; ACC="#1f7a8c"; RED="#c0392b"; AMBER="#e8a33d"; GREY="#9aa3ad"; GREEN="#2e7d52"

import torch, torch.nn as nn
torch.manual_seed(42); np.random.seed(42)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
WXCOLS = ["temperature_2m","relative_humidity_2m","surface_pressure","wind_speed_10m","boundary_layer_height","precipitation"]


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k=3):
        super().__init__(); self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class Hybrid(nn.Module):
    def __init__(self, in_ch, n_tab, hid=32, use_img=True, use_tab=True):
        super().__init__(); self.use_img, self.use_tab = use_img, use_tab
        fdim = 0
        if use_img:
            self.cell = ConvLSTMCell(in_ch, hid)
            self.ihead = nn.Sequential(nn.Conv2d(hid, 16, 3, padding=1), nn.ReLU(),
                                       nn.AdaptiveAvgPool2d(1), nn.Flatten()); fdim += 16
        if use_tab:
            self.thead = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()); fdim += 16
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(fdim, 1))
    def forward(self, x, t):
        feats = []
        if self.use_img:
            B, T, C, H, W = x.shape
            h = torch.zeros(B, self.cell.hid, H, W, device=x.device); c = torch.zeros_like(h)
            for k in range(T): h, c = self.cell(x[:, k], h, c)
            feats.append(self.ihead(h))
        if self.use_tab:
            feats.append(self.thead(t))
        return self.cls(torch.cat(feats, 1)).squeeze(1)


def _img(fp, mode): return np.asarray(Image.open(fp).convert(mode).resize((SIZE, SIZE)), float)


def load():
    rgb = {datetime.date.fromisoformat(fp.stem): _img(fp, "RGB")/255. for fp in FRGB.glob("*.jpg")}
    aod = {}
    for fp in FAOD.glob("*.png"):
        a = _img(fp, "RGBA")/255.; aod[datetime.date.fromisoformat(fp.stem)] = (a[..., :3], a[..., 3:4])
    emb = pd.read_csv(ROOT/"data"/"raw"/"openaq_embassy_pm25_daily.csv")
    dc = [c for c in emb.columns if "date" in c.lower()][0]; pc = [c for c in emb.columns if "pm" in c.lower()][0]
    emb["d"] = pd.to_datetime(emb[dc]).dt.date; emb["pm"] = pd.to_numeric(emb[pc], errors="coerce")
    lab = {r.d: r.pm for _, r in emb.dropna(subset=["pm"]).iterrows()}
    wx = pd.read_csv(WX); wx["d"] = pd.to_datetime(wx["date"]).dt.date
    wxmap = {r.d: [r[c] for c in WXCOLS] for _, r in wx.iterrows()}
    return rgb, aod, lab, wxmap


def wyear(d): return d.year if d.month >= 10 else d.year - 1


def frame7(d, rgb, aod):
    r = rgb[d].transpose(2, 0, 1)
    if d in aod:
        a, m = aod[d]; a, m = a.transpose(2, 0, 1), m.transpose(2, 0, 1)
    else:
        a, m = np.zeros((3, SIZE, SIZE)), np.zeros((1, SIZE, SIZE))
    return np.concatenate([r, a, m], 0)


def build():
    rgb, aod, lab, wxmap = load()
    med = np.log1p(np.median(list(lab.values())))
    X, T, y, fold, tgt = [], [], [], [], []
    for d in sorted(lab):
        win = [d - datetime.timedelta(days=k) for k in range(L, 0, -1)]
        if not all(w in rgb for w in win) or d not in wxmap:
            continue
        X.append(np.stack([frame7(w, rgb, aod) for w in win]))
        lags = [np.log1p(lab[d - datetime.timedelta(days=k)]) if (d - datetime.timedelta(days=k)) in lab else med
                for k in (1, 2, 3, 7)]
        doy = d.timetuple().tm_yday
        T.append(wxmap[d] + lags + [np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)])
        y.append(1 if lab[d] > THR else 0); fold.append(wyear(d)); tgt.append(d)
    return (np.array(X, np.float32), np.array(T, np.float32), np.array(y),
            np.array(fold), np.array(tgt, dtype=object))


def train_fold(Xtr, Ttr, ytr, Xte, Tte, use_img, use_tab, epochs=30):
    mu, sd = Ttr.mean(0), Ttr.std(0) + 1e-6
    Ttr, Tte = (Ttr - mu)/sd, (Tte - mu)/sd
    net = Hybrid(Xtr.shape[2], Ttr.shape[1], use_img=use_img, use_tab=use_tab).to(DEV)
    pw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], device=DEV, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, device=DEV); Tt = torch.tensor(Ttr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV); n = len(Xtr); bs = 32
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            lossf(net(Xt[idx], Tt[idx]), yt[idx]).backward(); opt.step()
    net.eval(); out = []
    Xe = torch.tensor(Xte, device=DEV); Te = torch.tensor(Tte, dtype=torch.float32, device=DEV)
    with torch.no_grad():
        for i in range(0, len(Xte), 64): out.append(torch.sigmoid(net(Xe[i:i+64], Te[i:i+64])).cpu().numpy())
    return np.concatenate(out)


def lowo(X, T, y, fold, use_img, use_tab, label):
    oof = np.full(len(X), np.nan)
    for wy in sorted(set(fold)):
        te = fold == wy; tr = ~te
        if te.sum() < 10 or y[tr].sum() < 5 or y[te].sum() < 2: continue
        oof[te] = train_fold(X[tr], T[tr], y[tr], X[te], T[te], use_img, use_tab)
        print(f"    [{label}] winter {wy}: n_te {int(te.sum())} smog {int(y[te].sum())}", flush=True)
    return oof


def main():
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
    import lightgbm as lgb
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X, T, y, fold, tgt = build()
    print(f"samples {len(X)} | img {X.shape[2]}ch | tab {T.shape[1]} | smog {y.mean():.2f} | winters {sorted(set(fold))} | {DEV}")
    res_oof = {}
    print("  [1/4] RGB-only ConvLSTM");      res_oof["RGB"]    = lowo(X[:, :, :3], T, y, fold, True, False, "RGB")
    print("  [2/4] RGB+AOD ConvLSTM");       res_oof["RGB+AOD"]= lowo(X, T, y, fold, True, False, "RGB+AOD")
    print("  [3/4] Tabular-only MLP");       res_oof["TAB"]    = lowo(X, T, y, fold, False, True, "TAB")
    print("  [4/4] Hybrid (img+tabular)");   res_oof["HYBRID"] = lowo(X, T, y, fold, True, True, "HYBRID")
    # LightGBM tabular reference + baselines
    oof_gbt = np.full(len(X), np.nan)
    for wy in sorted(set(fold)):
        te = fold == wy; tr = ~te
        if te.sum() < 10 or y[tr].sum() < 5: continue
        g = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, random_state=42, verbose=-1).fit(T[tr], y[tr])
        oof_gbt[te] = g.predict_proba(T[te])[:, 1]
    m = ~np.isnan(res_oof["HYBRID"]) & ~np.isnan(oof_gbt); Y = y[m]
    labmap = {d: yy for d, yy in zip(tgt, y)}
    persist = np.array([labmap.get(d - datetime.timedelta(days=1), 0) for d in tgt[m]], float)
    A = lambda P: roc_auc_score(Y, P[m] if len(P) == len(y) else P)
    res = {"n": int(m.sum()), "smog_rate": round(float(Y.mean()), 3),
           "auc_base": 0.5, "auc_persistence": round(roc_auc_score(Y, persist), 3),
           "auc_rgb": round(A(res_oof["RGB"]), 3), "auc_rgb_aod": round(A(res_oof["RGB+AOD"]), 3),
           "auc_tab_mlp": round(A(res_oof["TAB"]), 3), "auc_tab_lgbm": round(roc_auc_score(Y, oof_gbt[m]), 3),
           "auc_hybrid": round(A(res_oof["HYBRID"]), 3),
           "prauc_hybrid": round(average_precision_score(Y, res_oof["HYBRID"][m]), 3)}
    fpr, tpr, thr = roc_curve(Y, res_oof["HYBRID"][m]); idx = np.where(tpr >= 0.90)[0][0]
    cm = confusion_matrix(Y, (res_oof["HYBRID"][m] >= thr[idx]).astype(int))
    res.update({"op_recall": round(tpr[idx], 2), "op_falsealarm": round(fpr[idx], 2),
                "op_precision": round(cm[1, 1]/max(cm[:, 1].sum(), 1), 2)})
    json.dump(res, open(OUT/"convlstm_metrics.json", "w"), indent=2)
    np.save(OUT/"convlstm_oof.npy", np.column_stack([res_oof["RGB"][m], res_oof["RGB+AOD"][m], res_oof["TAB"][m], oof_gbt[m], res_oof["HYBRID"][m], Y]))
    print("\n=== smog forecast (leave-one-winter-out, 2018-2025) ===")
    for k, v in res.items(): print(f"  {k}: {v}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=160)
    for P, c, lb in [(res_oof["RGB+AOD"][m], GREY, f"image (RGB+AOD) {res['auc_rgb_aod']:.2f}"),
                      (oof_gbt[m], ACC, f"tabular {res['auc_tab_lgbm']:.2f}"),
                      (res_oof["HYBRID"][m], RED, f"hybrid {res['auc_hybrid']:.2f}")]:
        f_, t_, _ = roc_curve(Y, P); ax[0].plot(f_, t_, color=c, lw=2, label=lb)
    ax[0].plot([0, 1], [0, 1], ":", color="#bbb"); ax[0].set(xlabel="false-alarm rate", ylabel="catch rate", title="Smog forecast — ROC"); ax[0].legend(fontsize=8, loc="lower right")
    names = ["Base", "Persist", "RGB", "RGB+\nAOD", "Tabular", "Hybrid"]
    aucs = [0.5, res["auc_persistence"], res["auc_rgb"], res["auc_rgb_aod"], res["auc_tab_lgbm"], res["auc_hybrid"]]
    cols = [GREY, ACC, "#9bb0bf", "#6b8aa0", GREEN, RED]
    ax[1].bar(range(6), aucs, color=cols)
    for i, a in enumerate(aucs): ax[1].text(i, a + .008, f"{a:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax[1].set_xticks(range(6)); ax[1].set_xticklabels(names, fontsize=7.5); ax[1].set_ylim(0.45, max(aucs)+.07); ax[1].set_ylabel("ROC-AUC"); ax[1].set_title("What each modality adds")
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGDIR/"convlstm_roc.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("saved figures/convlstm_roc.png + models/convlstm_metrics.json")


if __name__ == "__main__":
    main()
