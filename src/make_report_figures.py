"""Make the report figures (model diagram, CTR by hour, CTR by site/device) from outputs/ctr_by_*.csv.

Run from the repo root:  python src/make_report_figures.py outputs outputs
"""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
src, out = sys.argv[1], sys.argv[2]
hour = pd.read_csv(f"{src}/ctr_by_hour_of_day.csv").sort_values("hour_of_day")
site = pd.read_csv(f"{src}/ctr_by_site_category.csv", dtype={"site_category": str})
dev = pd.read_csv(f"{src}/ctr_by_device_type.csv", dtype={"device_type": str})
overall = hour.clicks.sum() / hour.impressions.sum()

# Figure 2: CTR by hour of day
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(hour.hour_of_day, hour.ctr * 100, color="#4C72B0")
ax.axhline(overall * 100, color="grey", ls="--", lw=1, label=f"Overall CTR ({overall:.1%})")
ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hour of day (0 = midnight)")
ax.set_ylabel("Click-through rate (%)"); ax.set_ylim(0, 22)
ax.set_title(f"CTR by hour of day (training sample, {hour.impressions.sum():,} impressions)")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(f"{out}/ctr_by_hour.png", dpi=150)

# Figure 3: CTR by site category and device type
s = site[site.impressions >= 400].sort_values("ctr")
d = dev[dev.impressions >= 500].sort_values("ctr")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [3, 2]})
a1.barh([f"{c[:6]} ({n:,})" for c, n in zip(s.site_category, s.impressions)], s.ctr * 100, color="#4C72B0")
a1.set_xlabel("Click-through rate (%)"); a1.set_title("By site category (400+ impressions)")
a2.barh([f"Type {c} ({n:,})" for c, n in zip(d.device_type, d.impressions)], d.ctr * 100, color="#DD8452")
a2.set_xlabel("Click-through rate (%)"); a2.set_title("By device type")
for a in (a1, a2): a.axvline(overall * 100, color="grey", ls="--", lw=1)
fig.suptitle("CTR varies far more by site and device than by hour (impressions in brackets)", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{out}/ctr_by_site_device.png", dpi=150)

# Figure 1: model diagram
fig, ax = plt.subplots(figsize=(8, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color, ec="#333", lw=1))
    ax.text(x + w/2, y + h - 3.2, title, ha="center", va="top", fontsize=9.5, weight="bold")
    ax.text(x + w/2, y + h/2 - 2.5, sub, ha="center", va="center", fontsize=8)
def arrow(x0, x1, y=21):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
box(1, 6, 17, 30, "21 input\nfeatures", "site, app, device,\nbanner position,\nhour of day,\nday of week,\nC14–C21 ...", "#EAF0F8")
box(23, 6, 17, 30, "Embeddings", "each feature value\n→ 8 learned numbers\n\n21 × 8 = 168\nnumbers per ad", "#EAF0F8")
box(45, 6, 15, 30, "Dense layer", "128 units\nReLU\n20% dropout", "#FDEBD8")
box(65, 6, 15, 30, "Dense layer", "64 units\nReLU\n20% dropout", "#FDEBD8")
box(85, 6, 14, 30, "Sigmoid\noutput", "click\nprobability\n0 to 1", "#E3F2E1")
for a, b in [(18.8, 22.2), (40.8, 44.2), (60.8, 64.2), (80.8, 84.2)]: arrow(a, b)
ax.text(50, 40.5, "Neural network: one ad impression in, one click probability out", ha="center", fontsize=10.5, weight="bold")
ax.text(50, 1, "Trained with Adam on log loss; stops when validation loss stops improving", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(); fig.savefig(f"{out}/model_diagram.png", dpi=150)
