"""
run_benchmark.py
================
Full comparison experiment: four attention variants on synthetic tree data.

Models compared
---------------
1. Standard      – O(N²) softmax attention, NO tree structure (vanilla baseline)
2. Naive Tree    – O(N²) exact tree attention with polynomial distance weighting
                   (this is the ground-truth reference for #3 and #4)
3. Tree          – sub-quadratic tree attention via binomial DP + Performer features
4. FFT-Tree      – same as Tree but polynomial shift done with O(d log d) FFT

Outputs (saved to ./benchmark_results/)
----------------------------------------
* training_curves.png  – 2×2 grid: train loss / val loss / val acc / val F1
* epoch_time.png        – bar chart of average epoch wall-clock time
* scaling.png           – log-log plot of forward-pass time vs tree size N
* summary.csv           – numeric results table

Usage
-----
    python run_benchmark.py
"""

import os
import time
import csv
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Force non-interactive Matplotlib backend so plots save cleanly without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Local imports ──────────────────────────────────────────────────────────────
from _tree_attention import (
    SyntheticTreeDataset,
    TreeAttentionModel,
)
from FFT_tree import FFTTreeAttentionModel
from baseline import StandardAttentionModel, NaiveTreeAttentionModel


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data source -------------------------------------------------------------
# Set USE_REAL_DATA = True to train on PY150k AST trees instead of synthetic
# random trees.  Requires `pip install datasets` and a HuggingFace connection.
USE_REAL_DATA = True

# --- Dataset -----------------------------------------------------------------
NUM_TRAIN   = 500      # training samples
NUM_VAL     = 100       # validation samples
NUM_NODES   = 60       # nodes per tree (synthetic mode only; ignored for real data)
D_FEAT      = 8        # raw node feature dimension (synthetic); overridden for real data
NUM_CLASSES = 4        # classification targets
# Real-data overrides (used when USE_REAL_DATA = True)
REAL_MAX_SAMPLES  = 600   # total samples to load from PY150k
REAL_MAX_NODES    = 200   # cap AST size per sample
REAL_NUM_CLASSES  = 200    # vocabulary size (top-K gt tokens)
REAL_CACHE_PATH   = "./processed_data/py150k_cache.pkl"

# --- Model -------------------------------------------------------------------
D_INPUT     = D_FEAT  # updated automatically in main() when USE_REAL_DATA=True
D_MODEL     = 64
D_QK        = 32
D_V         = 32
POLY_DEGREE = 8        # polynomial degree for tree attention models
FEATURE_DIM = 32       # random-feature dimension for linearised models

# --- Training ----------------------------------------------------------------
NUM_EPOCHS = 20
LR         = 1e-3

# --- Scaling experiment ------------------------------------------------------
SCALE_NODE_SIZES = [20, 100, 500,1000,10000]   # N values for scaling plot
SCALE_N_RUNS     = 10                       # forward passes per N for timing

# --- Output ------------------------------------------------------------------
SAVE_DIR = os.path.join(
    os.path.dirname(__file__), "benchmark_results2"
)
os.makedirs(SAVE_DIR, exist_ok=True)

# Consistent colour palette across all plots
PALETTE = {
    "Standard":   "#2196F3",   # blue
    "Naive Tree": "#FF9800",   # orange
    "Tree":       "#4CAF50",   # green
    "FFT-Tree":   "#9C27B0",   # purple
}


# ============================================================================
# Model factory
# ============================================================================

def build_model(name: str, d_input: int = D_INPUT, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Construct a fresh model by name."""
    base = dict(
        d_input=d_input, d_model=D_MODEL, d_qk=D_QK, d_v=D_V,
        num_classes=num_classes,
    )
    tree_extra = dict(poly_degree=POLY_DEGREE)
    lin_extra  = dict(feature_map_type="random", feature_dim=FEATURE_DIM)

    if name == "Standard":
        return StandardAttentionModel(**base)
    elif name == "Naive Tree":
        return NaiveTreeAttentionModel(**base, **tree_extra)
    elif name == "Tree":
        return TreeAttentionModel(**base, **tree_extra, **lin_extra)
    elif name == "FFT-Tree":
        return FFTTreeAttentionModel(**base, **tree_extra, **lin_extra)
    else:
        raise ValueError(f"Unknown model name: {name!r}")


# ============================================================================
# Training loop
# ============================================================================

def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    device: str = DEVICE,
) -> dict:
    """
    Train *model* for *num_epochs* epochs, evaluating on *val_dataset* each epoch.

    Returns a history dict with keys:
        train_loss, val_loss, val_acc, val_f1, epoch_time_s
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss":   [],
        "val_loss":     [],
        "val_acc":      [],
        "val_f1":       [],
        "epoch_time_s": [],
    }

    # Early stopping state — must be outside the epoch loop
    patience      = 5
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(num_epochs):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        t_start    = time.perf_counter()
        total_loss = 0.0

        for features, tree, label in train_dataset:
            features = features.to(device)
            label_t  = torch.tensor(label, device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(features, tree)
            loss   = criterion(logits, label_t.unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.perf_counter() - t_start
        avg_loss   = total_loss / len(train_dataset)
        history["train_loss"].append(avg_loss)
        history["epoch_time_s"].append(epoch_time)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        preds, targets = [], []

        with torch.no_grad():
            for features, tree, label in val_dataset:
                features = features.to(device)
                label_t  = torch.tensor(label, device=device, dtype=torch.long)
                logits   = model(features, tree)
                val_loss += criterion(logits, label_t.unsqueeze(0)).item()
                pred = logits.argmax(dim=1).item()
                preds.append(pred)
                targets.append(label)
                if pred == label:
                    correct += 1

        history["val_loss"].append(val_loss / len(val_dataset))
        history["val_acc"].append(correct / len(val_dataset))
        history["val_f1"].append(
            f1_score(targets, preds, average="macro", zero_division=0)
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    epoch {epoch+1:3d}/{num_epochs} | "
                f"train_loss={avg_loss:.4f}  "
                f"val_loss={history['val_loss'][-1]:.4f}  "
                f"val_acc={history['val_acc'][-1]:.3f}  "
                f"val_f1={history['val_f1'][-1]:.3f}  "
                f"({epoch_time:.2f}s)"
            )

        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    return history


# ============================================================================
# Scaling experiment
# ============================================================================

def _forward_time_ms(model: nn.Module, dataset, device: str, n_runs: int) -> float:
    """Return mean forward-pass time (ms) over at most *n_runs* samples."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (features, tree, _) in enumerate(dataset):
            if i >= n_runs:
                break
            features = features.to(device)
            t0 = time.perf_counter()
            model(features, tree)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1_000)
    return float(np.mean(times)) if times else float("nan")


def run_scaling_experiment(
    model_names: list,
    node_sizes:  list,
    n_runs:      int = SCALE_N_RUNS,
    device:      str = DEVICE,
) -> dict:
    """
    For each model and each tree size N, measure the mean forward-pass time.

    Returns
    -------
    results : dict  name -> List[float]  (ms, same order as node_sizes)
    """
    results = {name: [] for name in model_names}

    for N in node_sizes:
        print(f"  N = {N}:")
        ds = SyntheticTreeDataset(n_runs, N, D_FEAT, NUM_CLASSES)
        for name in model_names:
            model = build_model(name).to(device)
            t_ms  = _forward_time_ms(model, ds, device, n_runs)
            results[name].append(t_ms)
            print(f"    {name:<12}: {t_ms:7.2f} ms")

    return results


# ============================================================================
# Plotting helpers
# ============================================================================

def _savefig(fig, fname: str):
    path = os.path.join(SAVE_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_training_curves(histories: dict):
    """2×2 grid: train loss / val loss / val accuracy / val F1."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        ("train_loss", "Train Loss",       axes[0, 0]),
        ("val_loss",   "Validation Loss",  axes[0, 1]),
        ("val_acc",    "Validation Acc",   axes[1, 0]),
        ("val_f1",     "Validation F1",    axes[1, 1]),
    ]

    for key, ylabel, ax in panels:
        for name, hist in histories.items():
            ax.plot(
                hist[key],
                label=name,
                color=PALETTE.get(name),
                linewidth=2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Model Comparison — Training Curves\n"
        f"(N={NUM_NODES} nodes, poly_degree={POLY_DEGREE}, "
        f"feature_dim={FEATURE_DIM})",
        fontsize=12,
    )
    plt.tight_layout()
    _savefig(fig, "training_curves.png")


def plot_epoch_time(histories: dict):
    """Horizontal bar chart of average epoch time for each model."""
    names     = list(histories.keys())
    avg_times = [np.mean(histories[n]["epoch_time_s"]) for n in names]
    colors    = [PALETTE.get(n, "#607D8B") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, avg_times, color=colors, edgecolor="white")
    for bar, t in zip(bars, avg_times):
        ax.text(
            t + 0.005 * max(avg_times),
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s",
            va="center", fontsize=10,
        )
    ax.set_xlabel("Average epoch time (s)")
    ax.set_title(
        f"Epoch Wall-Clock Time\n"
        f"(N={NUM_NODES}, {NUM_TRAIN} train samples, {NUM_EPOCHS} epochs)"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "epoch_time.png")


def plot_scaling(scaling_results: dict, node_sizes: list):
    """Log-log plot of forward-pass time vs N with O(N) and O(N²) reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = np.array(node_sizes, dtype=float)

    for name, times in scaling_results.items():
        ax.plot(
            node_sizes, times,
            marker="o",
            label=name,
            color=PALETTE.get(name),
            linewidth=2,
            markersize=5,
        )

    # Reference lines anchored to the Tree model's first data point
    tree_times = scaling_results.get("Tree", list(scaling_results.values())[0])
    if tree_times and not math.isnan(tree_times[0]):
        base = tree_times[0] / node_sizes[0]
        ax.plot(
            node_sizes,
            base * ns,
            "--", color="gray", alpha=0.55, linewidth=1.5,
            label="O(N) reference",
        )
        ax.plot(
            node_sizes,
            base * (ns ** 2) / node_sizes[0],
            ":",  color="gray", alpha=0.55, linewidth=1.5,
            label="O(N²) reference",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes  N  (log scale)")
    ax.set_ylabel("Forward-pass time (ms, log scale)")
    ax.set_title(
        f"Scaling Behaviour: Forward-Pass Time vs Tree Size\n"
        f"(poly_degree={POLY_DEGREE}, feature_dim={FEATURE_DIM})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    _savefig(fig, "scaling.png")


# ============================================================================
# Summary table
# ============================================================================

def print_summary(histories: dict):
    header = (
        f"{'Model':<14} | {'FinalValAcc':>11} | {'FinalValF1':>10} | "
        f"{'BestValAcc':>10} | {'AvgEpochT(s)':>12}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("BENCHMARK SUMMARY")
    print(sep)
    print(header)
    print(sep)

    rows = []
    for name, hist in histories.items():
        acc_final  = hist["val_acc"][-1]
        f1_final   = hist["val_f1"][-1]
        acc_best   = max(hist["val_acc"])
        avg_t      = np.mean(hist["epoch_time_s"])
        print(
            f"{name:<14} | {acc_final:>11.4f} | {f1_final:>10.4f} | "
            f"{acc_best:>10.4f} | {avg_t:>12.3f}"
        )
        rows.append({
            "model":          name,
            "final_val_acc":  f"{acc_final:.4f}",
            "final_val_f1":   f"{f1_final:.4f}",
            "best_val_acc":   f"{acc_best:.4f}",
            "avg_epoch_time": f"{avg_t:.3f}",
        })

    print(sep)
    return rows


def save_summary_csv(rows: list, scaling_results: dict, node_sizes: list):
    path = os.path.join(SAVE_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "final_val_acc", "final_val_f1",
                        "best_val_acc", "avg_epoch_time"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Also append scaling table to the same CSV
    with open(path, "a", newline="") as f:
        f.write("\n\nScaling (forward-pass ms)\n")
        header = ["model"] + [f"N={n}" for n in node_sizes]
        writer = csv.writer(f)
        writer.writerow(header)
        for name, times in scaling_results.items():
            writer.writerow([name] + [f"{t:.2f}" for t in times])

    print(f"  Saved → {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Tree Attention Benchmark")
    print("=" * 60)
    print(f"Device      : {DEVICE}")
    print(f"Dataset     : {NUM_TRAIN} train / {NUM_VAL} val")
    print(f"Tree size   : N = {NUM_NODES} nodes")
    print(f"Classes     : {NUM_CLASSES}")
    print(f"Poly degree : {POLY_DEGREE}")
    print(f"Feature dim : {FEATURE_DIM}  (for linearised models)")
    print(f"Epochs      : {NUM_EPOCHS}")
    print(f"Output dir  : {SAVE_DIR}")
    print()

    # ── Reproducibility ──────────────────────────────────────────────────────
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Build datasets ───────────────────────────────────────────────────────
    # d_input and num_classes may differ between synthetic and real-data modes;
    # we compute them here and pass explicitly to build_model via local vars.
    d_input     = D_INPUT
    num_classes = NUM_CLASSES

    if USE_REAL_DATA:
        from load_python150 import process_py150k, train_val_test_split, D_NODE_FEAT
        print("Dataset     : PY150k (real AST trees)")
        all_data, feat_dim = process_py150k(
            split="train",
            max_samples=REAL_MAX_SAMPLES,
            num_classes=REAL_NUM_CLASSES,
            max_nodes=REAL_MAX_NODES,
            cache_path=REAL_CACHE_PATH,
        )
        train_data, val_data, _ = train_val_test_split(
            all_data,
            train_ratio=NUM_TRAIN / REAL_MAX_SAMPLES,
            val_ratio=NUM_VAL   / REAL_MAX_SAMPLES,
        )
        train_ds    = train_data
        val_ds      = val_data
        d_input     = feat_dim          # D_NODE_FEAT ≈ 100
        num_classes = REAL_NUM_CLASSES
        print(f"  Loaded  : {len(train_ds)} train / {len(val_ds)} val")
        print(f"  d_input : {D_INPUT}  (AST node-type one-hot)")
        print(f"  classes : {NUM_CLASSES}  (top-K gt tokens + OOV)")
    else:
        print("Dataset     : Synthetic random trees")
        train_ds = SyntheticTreeDataset(NUM_TRAIN, NUM_NODES, D_FEAT, NUM_CLASSES)
        val_ds   = SyntheticTreeDataset(NUM_VAL,   NUM_NODES, D_FEAT, NUM_CLASSES)

    # ── Train all four models ─────────────────────────────────────────────────
    MODEL_NAMES = ["Standard", "Naive Tree", "Tree", "FFT-Tree"]
    histories   = {}

    for name in MODEL_NAMES:
        print(f"\n{'─'*55}")
        print(f"  Training: {name}")
        model  = build_model(name, d_input=d_input, num_classes=num_classes)
        n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params  : {n_param:,}")
        print(f"{'─'*55}")
        histories[name] = train_model(model, train_ds, val_ds)

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = print_summary(histories)

    # ── Training-curve plots ──────────────────────────────────────────────────
    print("\nGenerating training-curve plots …")
    plot_training_curves(histories)
    plot_epoch_time(histories)

    # ── Scaling experiment ────────────────────────────────────────────────────
    print(f"\nRunning scaling experiment  N ∈ {SCALE_NODE_SIZES} …")
    scaling_results = run_scaling_experiment(
        model_names=MODEL_NAMES,
        node_sizes=SCALE_NODE_SIZES,
    )
    plot_scaling(scaling_results, SCALE_NODE_SIZES)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    print("\nSaving summary CSV …")
    save_summary_csv(rows, scaling_results, SCALE_NODE_SIZES)

    print(f"\n✓  All outputs written to  {SAVE_DIR}/")
    print("  training_curves.png – learning curves for all four models")
    print("  epoch_time.png      – epoch wall-clock comparison")
    print("  scaling.png         – forward-pass time vs tree size (log-log)")
    print("  summary.csv         – numeric results")


if __name__ == "__main__":
    main()
