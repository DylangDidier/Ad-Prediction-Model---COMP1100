"""Predict ad clicks on Avazu with an embedding neural network and compare it
with the historical-average CTR baseline.

Run from PyCharm (right-click > Run) or from the repo root:
    python src/ctr_mid_report.py

Input:  data/raw/train.gz  (Kaggle Avazu CTR Prediction, left zipped)
Output: outputs/  (results.csv, loss_curve.png, click_capture.png,
        failure_check.csv, and ctr_by_*.csv tables for Excel)
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

# ---------------------------------------------------------------- knobs
SAMPLE_FRAC = 0.005      # 0.005 is about 200k rows; 0.0125 is about 500k
CHUNK_SIZE = 1_000_000   # rows read from train.gz at a time
TEST_DAY = "141030"      # YYMMDD of the held-out test day (2014-10-30)
VAL_FRAC = 0.10          # share of the non-test rows kept aside for validation
MIN_COUNT = 10           # category values seen fewer times become "other"
EMB_DIM = 8              # numbers per embedding
HIDDEN = (128, 64)       # dense layer sizes
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 20
PATIENCE = 2             # stop after this many epochs without val improvement
SMOOTHING = 20           # pseudo-impressions pulling small baseline groups to the mean
TOP_SHARE = 0.10         # "top 10%" for click capture
RARE_SITE_COUNT = 20     # sites seen fewer times than this in training count as rare
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.gz"
OUT = ROOT / "outputs"

DROP_COLS = ["id", "click", "hour", "device_id", "device_ip"]
BASELINE_KEYS = ["site_category", "device_type", "banner_pos"]
EXCEL_TABLES = ["hour_of_day", "day_of_week", "device_type", "site_category",
                "app_category", "banner_pos", "device_conn_type"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. sample
def load_sample(path=None):
    """Read train.gz in chunks and keep SAMPLE_FRAC of each chunk."""
    path = path or DATA_PATH
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")  # an unzipped train.csv works too
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download train.gz from the Kaggle Avazu "
            "competition and put it in data/raw/ (train.gz or train.csv).")
    rng = np.random.default_rng(SEED)
    parts = []
    reader = pd.read_csv(path, dtype=str, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader, 1):
        keep = rng.random(len(chunk)) < SAMPLE_FRAC
        parts.append(chunk[keep])
        if i % 5 == 0:
            log(f"  read {i * CHUNK_SIZE:,} rows, kept {sum(map(len, parts)):,}")
    df = pd.concat(parts, ignore_index=True)
    df["click"] = df["click"].astype(int)
    return df


def split(df):
    """Last day is the test set; 10% of the earlier days is validation."""
    is_test = df["hour"].str[:6] == TEST_DAY
    test = df[is_test].reset_index(drop=True)
    rest = df[~is_test]
    val_mask = np.random.default_rng(SEED).random(len(rest)) < VAL_FRAC
    train = rest[~val_mask].reset_index(drop=True)
    val = rest[val_mask].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------- 2. features
def add_time_features(df):
    ts = pd.to_datetime(df["hour"], format="%y%m%d%H")
    df = df.copy()
    df["hour_of_day"] = ts.dt.hour.astype(str)
    df["day_of_week"] = ts.dt.dayofweek.astype(str)  # 0 = Monday
    return df


def fit_vocab(train, cols):
    """Map each value seen at least MIN_COUNT times to an ID; 0 means other/unseen."""
    vocab = {}
    for c in cols:
        counts = train[c].value_counts()
        kept = counts[counts >= MIN_COUNT].index
        vocab[c] = {v: i + 1 for i, v in enumerate(kept)}
    return vocab


def encode(df, vocab):
    return np.stack([df[c].map(m).fillna(0).astype(np.int64).to_numpy()
                     for c, m in vocab.items()], axis=1)


# ---------------------------------------------------------------- 3. baseline
def baseline_predict(train, test):
    """Past CTR for each (site category, device type, banner position) group,
    smoothed toward the overall CTR so tiny groups don't get extreme rates."""
    global_ctr = train["click"].mean()
    g = train.groupby(BASELINE_KEYS)["click"].agg(["sum", "count"])
    g["ctr"] = (g["sum"] + SMOOTHING * global_ctr) / (g["count"] + SMOOTHING)
    pred = test.join(g["ctr"], on=BASELINE_KEYS)["ctr"].fillna(global_ctr)
    return pred.to_numpy()


# ---------------------------------------------------------------- 4. model
class CTRNet(nn.Module):
    """One embedding per feature -> concatenate -> dense layers -> click logit."""

    def __init__(self, cardinalities):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(n, EMB_DIM) for n in cardinalities)
        layers, width = [], EMB_DIM * len(cardinalities)
        for h in HIDDEN:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        z = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.mlp(z).squeeze(1)  # logit; sigmoid gives click probability


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = [torch.sigmoid(model(torch.from_numpy(X[i:i + 8192])))
               for i in range(0, len(X), 8192)]
    return torch.cat(out).numpy()


def mean_loss(model, X, y, loss_fn):
    p = np.clip(predict(model, X), 1e-7, 1 - 1e-7)
    return float(loss_fn(torch.logit(torch.from_numpy(p)), torch.from_numpy(y)))


def train_model(X_tr, y_tr, X_va, y_va, cardinalities):
    torch.manual_seed(SEED)
    model = CTRNet(cardinalities)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        tr_loss = total / len(Xt)
        va_loss = mean_loss(model, X_va, y_va, loss_fn)
        history.append((epoch, tr_loss, va_loss))
        log(f"  epoch {epoch:2d}  train loss {tr_loss:.4f}  val loss {va_loss:.4f}")
        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"  early stop: no val improvement for {PATIENCE} epochs")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"])


# ---------------------------------------------------------------- 5. evaluate
def click_capture(y, score, share=TOP_SHARE):
    """Share of all clicks that land in the top `share` of impressions by score."""
    k = max(1, int(round(len(y) * share)))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].sum() / y.sum()


def metrics(name, y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"method": name,
            "auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, p),
            f"top{int(TOP_SHARE * 100)}pct_click_capture": click_capture(y, p)}


def plot_loss(hist):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="Training")
    ax.plot(hist["epoch"], hist["val_loss"], marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss (binary cross-entropy)")
    ax.set_title("Neural network training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", dpi=150)
    plt.close(fig)


def plot_capture(y, curves):
    """Cumulative gains: % of clicks caught vs % of impressions targeted."""
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, len(y) + 1) / len(y) * 100
    for name, score in curves.items():
        order = np.argsort(-score, kind="stable")
        ax.plot(xs, np.cumsum(y[order]) / y.sum() * 100, label=name)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Random")
    ax.axvline(TOP_SHARE * 100, color="grey", linewidth=0.8)
    ax.set_xlabel("Impressions targeted, highest score first (%)")
    ax.set_ylabel("Clicks captured (%)")
    ax.set_title("Clicks captured on the test day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "click_capture.png", dpi=150)
    plt.close(fig)


def write_excel_tables(train):
    for col in EXCEL_TABLES:
        t = (train.groupby(col)["click"].agg(impressions="count", clicks="sum")
             .reset_index())
        t["ctr"] = t["clicks"] / t["impressions"]
        t.sort_values("impressions", ascending=False).to_csv(
            OUT / f"ctr_by_{col}.csv", index=False)


# ---------------------------------------------------------------- 6. failure check
def failure_check(train, test, p_model, p_base):
    counts = train["site_id"].value_counts()
    seen = test["site_id"].map(counts).fillna(0)
    groups = {f"common sites (>= {RARE_SITE_COUNT} train rows)": seen >= RARE_SITE_COUNT,
              f"rare sites (1-{RARE_SITE_COUNT - 1} train rows)": (seen > 0) & (seen < RARE_SITE_COUNT),
              "new sites (not in training)": seen == 0}
    rows = []
    y = test["click"].to_numpy()
    for name, mask in groups.items():
        m = mask.to_numpy()
        row = {"group": name, "test_rows": int(m.sum()),
               "clicks": int(y[m].sum()), "ctr": y[m].mean() if m.any() else np.nan}
        both_classes = m.any() and 0 < y[m].sum() < m.sum()
        row["model_auc"] = roc_auc_score(y[m], p_model[m]) if both_classes else np.nan
        row["baseline_auc"] = roc_auc_score(y[m], p_base[m]) if both_classes else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)

    log(f"1. Sampling {SAMPLE_FRAC:.2%} of the Avazu training file (takes several minutes)")
    df = add_time_features(load_sample())
    train, val, test = split(df)
    log(f"   train {len(train):,}  val {len(val):,}  test {len(test):,} rows; "
        f"train CTR {train['click'].mean():.3f}, test CTR {test['click'].mean():.3f}")
    if len(test) == 0:
        raise ValueError(f"No rows from test day {TEST_DAY}; is this the full train.gz?")

    log("2. Building features")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    vocab = fit_vocab(train, feature_cols)
    cardinalities = [len(m) + 1 for m in vocab.values()]
    X_tr, X_va, X_te = (encode(d, vocab) for d in (train, val, test))
    y_tr, y_va, y_te = (d["click"].to_numpy(np.float32) for d in (train, val, test))
    log(f"   {len(feature_cols)} features, {sum(cardinalities):,} embedding rows")
    write_excel_tables(train)

    log("3. Baseline: historical CTR by " + ", ".join(BASELINE_KEYS))
    # the baseline may use all past days, including the validation rows
    past = pd.concat([train, val], ignore_index=True)
    p_base = baseline_predict(past, test)

    log("4. Training the neural network")
    model, hist = train_model(X_tr, y_tr, X_va, y_va, cardinalities)
    plot_loss(hist)
    hist.to_csv(OUT / "loss_history.csv", index=False)
    p_model = predict(model, X_te)

    log("5. Evaluating on the test day")
    y = y_te.astype(int)
    p_const = np.full(len(y), past["click"].mean())
    results = pd.DataFrame([
        metrics("Neural network", y, p_model),
        metrics("Historical-average baseline", y, p_base),
        {"method": "Overall average (no ranking)", "auc": 0.5,
         "log_loss": log_loss(y, p_const), f"top{int(TOP_SHARE * 100)}pct_click_capture": TOP_SHARE},
    ])
    results["test_rows"] = len(y)
    results["test_clicks"] = int(y.sum())
    results.to_csv(OUT / "results.csv", index=False)
    plot_capture(y, {"Neural network": p_model, "Historical-average baseline": p_base})
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    log("6. Failure check: common vs rare vs new sites")
    fc = failure_check(train, test, p_model, p_base)
    fc.to_csv(OUT / "failure_check.csv", index=False)
    print(fc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    log(f"Done in {(time.time() - start) / 60:.1f} min. Files are in {OUT}")


if __name__ == "__main__":
    main()
