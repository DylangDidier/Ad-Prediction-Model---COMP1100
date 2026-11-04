"""Permutation feature importance for the CTR neural network.

For each feature, shuffle that one column across the test day and measure how
much worse the model gets (drop in AUC, rise in log loss). A feature the model
relies on hurts a lot when shuffled; a feature it ignores barely matters.
Features that carry the same information (site_id and site_domain, say) can
cover for each other when shuffled one at a time, so related features are
also shuffled together as groups.

Run from PyCharm (right-click > Run) or from the repo root, after
ctr_mid_report.py has run once so the sample is cached:
    python src/feature_importance.py

Uses the same sample, split, features and seed as ctr_mid_report.py, so it
trains the same model. The trained model is saved in data/cache/ and reused.

Output: outputs/feature_importance.csv, outputs/feature_importance_groups.csv,
        outputs/feature_importance.png
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

import ctr_mid_report as base
from ctr_mid_report import OUT, CACHE_DIR, ROOT, SEED, log

N_REPEATS = 5            # shuffles per feature; the chart shows the spread
USE_MODEL_CACHE = True   # reuse the trained model instead of training again

# related features shuffled together; anything not listed is its own group
GROUPS = {
    "Site (id, domain, category)": ["site_id", "site_domain", "site_category"],
    "App (id, domain, category)": ["app_id", "app_domain", "app_category"],
    "Device (model, type, connection)": ["device_model", "device_type", "device_conn_type"],
    "Banner position": ["banner_pos"],
    "Time (hour of day, day of week)": ["hour_of_day", "day_of_week"],
    "Anonymous C1, C14-C21": ["C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"],
}


def scores(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return roc_auc_score(y, p), log_loss(y, p)


def get_model(X_tr, y_tr, X_va, y_va, cardinalities):
    path = CACHE_DIR / f"ctrnet_frac{base.SAMPLE_FRAC}_seed{SEED}.pt"
    if USE_MODEL_CACHE and path.exists():
        saved = torch.load(path)
        if saved["cardinalities"] == cardinalities:
            log(f"  loading trained model from {path.relative_to(ROOT)}")
            model = base.CTRNet(cardinalities)
            model.load_state_dict(saved["state"])
            return model
    model, _ = base.train_model(X_tr, y_tr, X_va, y_va, cardinalities)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"cardinalities": cardinalities, "state": model.state_dict()}, path)
    log(f"  saved trained model to {path.relative_to(ROOT)}")
    return model


def permutation_importance(model, X, y, cols, groups, rng):
    """Shuffle each group's columns together (same row order) N_REPEATS times."""
    auc0, ll0 = scores(y, base.predict(model, X))
    rows = []
    for name, members in groups.items():
        idx = [cols.index(c) for c in members]
        auc_drops, ll_rises = [], []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
            auc, ll = scores(y, base.predict(model, Xp))
            auc_drops.append(auc0 - auc)
            ll_rises.append(ll - ll0)
        rows.append({"feature": name, "columns": " ".join(members),
                     "auc_drop": np.mean(auc_drops), "auc_drop_std": np.std(auc_drops),
                     "log_loss_rise": np.mean(ll_rises), "log_loss_rise_std": np.std(ll_rises)})
        log(f"  {name:34s} AUC drop {np.mean(auc_drops):.4f}")
    out = pd.DataFrame(rows).sort_values("auc_drop", ascending=False, ignore_index=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out["test_auc"], out["test_log_loss"] = auc0, ll0
    return out


def plot(single, groups, auc0):
    color = "#4C72B0"  # same blue as the other report figures
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, df, title in [(a1, single, "Each feature shuffled alone"),
                          (a2, groups, "Related features shuffled together")]:
        d = df.iloc[::-1]
        ax.barh(d["feature"], d["auc_drop"], xerr=d["auc_drop_std"], color=color,
                height=0.7, error_kw={"ecolor": "#555", "lw": 0.8, "capsize": 2})
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("Drop in test AUC when shuffled")
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="x", color="#ddd", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        top = df.head(3)
        for f, v, e in zip(top["feature"], top["auc_drop"], top["auc_drop_std"]):
            ax.text(v + e + ax.get_xlim()[1] * 0.01, f, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle(f"Which inputs the neural network relies on (test-day AUC {auc0:.3f}; "
                 f"bars = mean of {N_REPEATS} shuffles, lines = spread)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)

    log("1. Loading the sample and building features (same as ctr_mid_report.py)")
    df = base.add_time_features(base.load_sample())
    train, val, test = base.split(df)
    cols = [c for c in df.columns if c not in base.DROP_COLS]
    vocab = base.fit_vocab(train, cols)
    cardinalities = [len(m) + 1 for m in vocab.values()]
    X_tr, X_va, X_te = (base.encode(d, vocab) for d in (train, val, test))
    y_tr, y_va, y_te = (d["click"].to_numpy(np.float32) for d in (train, val, test))

    log("2. Getting the trained model")
    model = get_model(X_tr, y_tr, X_va, y_va, cardinalities)

    y = y_te.astype(int)
    rng = np.random.default_rng(SEED)
    log(f"3. Shuffling each of the {len(cols)} features on the test day, {N_REPEATS} times each")
    single = permutation_importance(model, X_te, y, cols, {c: [c] for c in cols}, rng)
    single["distinct_values"] = [cardinalities[cols.index(c)] for c in single["feature"]]
    single.to_csv(OUT / "feature_importance.csv", index=False)

    log("4. Shuffling related features together")
    missing = {c for m in GROUPS.values() for c in m} - set(cols)
    if missing:
        raise ValueError(f"GROUPS lists columns the model doesn't use: {missing}")
    grouped = dict(GROUPS)
    for c in cols:
        if not any(c in m for m in GROUPS.values()):
            grouped[c] = [c]
    groups = permutation_importance(model, X_te, y, cols, grouped, rng)
    groups.to_csv(OUT / "feature_importance_groups.csv", index=False)

    plot(single, groups, single["test_auc"].iloc[0])
    show = ["rank", "feature", "auc_drop", "auc_drop_std", "log_loss_rise"]
    print(single[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(groups[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"Done in {(time.time() - start) / 60:.1f} min. Files are in {OUT}")


if __name__ == "__main__":
    main()
