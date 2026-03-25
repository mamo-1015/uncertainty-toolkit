"""
visualizations.py
-----------------
All diagnostic plots for Part 3 of the assignment.

Each function is standalone — it accepts numpy/tensor arrays and saves
a publication-quality figure to ``output_dir``.

Files saved by generate_all()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. scatter_ua_ue.png                — Ue vs Ua scatter, correct vs wrong
2. uncertainty_histograms.png       — histograms, same y-axis scale
3. uncertainty_histograms_twinx.png — histograms, independent y-axis per side
4. per_class_breakdown.png          — bar chart, same y-axis scale
5. per_class_breakdown_twinx.png    — bar chart, independent y-axis per metric
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_CORRECT_COLOR = "#2196F3"
_WRONG_COLOR   = "#F44336"
_UE_COLOR      = "#9C27B0"
_UA_COLOR      = "#FF9800"
_GRID_KW       = dict(alpha=0.3, linewidth=0.6)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Scatter
# ---------------------------------------------------------------------------

def scatter_ua_ue(
    ue: "array-like",
    ua: "array-like",
    correct: "array-like",
    output_dir: str | Path,
    title: str = "Epistemic vs Aleatoric Uncertainty",
    alpha: float = 0.35,
    s: float = 8.0,
) -> Path:
    """
    Scatter plot of Ua (x) vs Ue (y), coloured by correct / incorrect.
    Saved as scatter_ua_ue.png.
    """
    ue, ua, correct = to_numpy(ue), to_numpy(ua), to_numpy(correct).astype(bool)

    fig, ax = plt.subplots(figsize=(8, 7))
    for mask, label, color, zorder in [
        (~correct, "Misclassified", _WRONG_COLOR,  2),
        (correct,  "Correct",       _CORRECT_COLOR, 3),
    ]:
        ax.scatter(ua[mask], ue[mask], c=color, label=label,
                   alpha=alpha, s=s, zorder=zorder, linewidths=0)

    ax.set_xlabel("Aleatoric Uncertainty  $U_a$", fontsize=13)
    ax.set_ylabel("Epistemic Uncertainty  $U_e$", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, **_GRID_KW)
    fig.tight_layout()

    out = Path(output_dir) / "scatter_ua_ue.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 2a. Histograms — same scale
# ---------------------------------------------------------------------------

def uncertainty_histograms(
    ue: "array-like",
    ua: "array-like",
    correct: "array-like",
    output_dir: str | Path,
    bins: int = 60,
) -> Path:
    """
    Two-panel density histogram where BOTH panels share the same y-axis scale.

    Honest magnitude comparison — if Ua spans a wider range its bars will
    appear flatter than Ue.  Use this to compare absolute density heights.
    Saved as uncertainty_histograms.png.
    """
    ue, ua, correct = to_numpy(ue), to_numpy(ua), to_numpy(correct).astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    max_density = 0.0
    configs = [
        (axes[0], ue, "Epistemic Uncertainty  $U_e$", "Epistemic"),
        (axes[1], ua, "Aleatoric Uncertainty  $U_a$",  "Aleatoric"),
    ]

    # First pass — draw histograms and track max density
    for ax, values, xlabel, metric_name in configs:
        shared_range = (float(values.min()), float(values.max()))
        n_c, _, _ = ax.hist(values[correct],  bins=bins, range=shared_range,
                            density=True, color=_CORRECT_COLOR, alpha=0.65,
                            label="Correct", edgecolor="none")
        n_w, _, _ = ax.hist(values[~correct], bins=bins, range=shared_range,
                            density=True, color=_WRONG_COLOR, alpha=0.65,
                            label="Misclassified", edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{metric_name} Uncertainty Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, **_GRID_KW)
        max_density = max(max_density, float(n_c.max()), float(n_w.max()))

    # Second pass — enforce same y-limit on both panels
    for ax in axes:
        ax.set_ylim(0, max_density * 1.12)

    fig.suptitle(
        "Uncertainty Distributions: Correct vs Misclassified  [same scale]",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out = Path(output_dir) / "uncertainty_histograms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 2b. Histograms — dual scale (twinx)
# ---------------------------------------------------------------------------

def uncertainty_histograms_twinx(
    ue: "array-like",
    ua: "array-like",
    correct: "array-like",
    output_dir: str | Path,
    bins: int = 60,
) -> Path:
    """
    Two-panel density histogram where each panel has INDEPENDENT y-axes:
      Left y-axis  (blue ticks)  — Correct distribution 
      Right y-axis (red ticks)   — Misclassified distribution 

   
    """
    ue, ua, correct = to_numpy(ue), to_numpy(ua), to_numpy(correct).astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, values, xlabel, metric_name in [
        (axes[0], ue, "Epistemic Uncertainty  $U_e$", "Epistemic"),
        (axes[1], ua, "Aleatoric Uncertainty  $U_a$",  "Aleatoric"),
    ]:
        shared_range = (float(values.min()), float(values.max()))

        # Left y-axis — Correct
        n_c, _, _ = ax.hist(values[correct], bins=bins, range=shared_range,
                            density=True, color=_CORRECT_COLOR, alpha=0.65,
                            edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Correct", fontsize=11, color=_CORRECT_COLOR)
        ax.tick_params(axis="y", labelcolor=_CORRECT_COLOR)
        ax.set_ylim(0, float(n_c.max()) * 1.25)

        # Right y-axis — Misclassified
        ax2 = ax.twinx()
        n_w, _, _ = ax2.hist(values[~correct], bins=bins, range=shared_range,
                             density=True, color=_WRONG_COLOR, alpha=0.65,
                             edgecolor="none")
        ax2.set_ylabel("Misclassified", fontsize=11, color=_WRONG_COLOR)
        ax2.tick_params(axis="y", labelcolor=_WRONG_COLOR)
        ax2.set_ylim(0, float(n_w.max()) * 1.25)

        handles = [
            mpatches.Patch(color=_CORRECT_COLOR, alpha=0.65, label="Correct"),
            mpatches.Patch(color=_WRONG_COLOR,   alpha=0.65, label="Misclassified"),
        ]
        ax.legend(handles=handles, fontsize=10)
        ax.set_title(
            f"{metric_name} Uncertainty Distribution\n(independent y-axes)",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, **_GRID_KW)

    fig.suptitle(
        "Uncertainty Distributions: Correct vs Misclassified  [dual scale]",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out = Path(output_dir) / "uncertainty_histograms_dual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 3a. Per-class bar chart — same scale
# ---------------------------------------------------------------------------

def per_class_breakdown(
    ue: "array-like",
    ua: "array-like",
    labels: "array-like",
    output_dir: str | Path,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Grouped bar chart of mean Ue and Ua per class — SAME y-axis scale.

    Both metrics share one y-axis so the absolute magnitude difference
    between Ue and Ua is visible.
    Saved as per_class_breakdown.png.
    """
    ue, ua, labels = to_numpy(ue), to_numpy(ua), to_numpy(labels).astype(int)
    classes = np.unique(labels)
    n_classes = len(classes)
    if class_names is None:
        class_names = [str(c) for c in classes]

    mean_ue = np.array([ue[labels == c].mean() for c in classes])
    mean_ua = np.array([ua[labels == c].mean() for c in classes])

    x, bar_w = np.arange(n_classes), 0.38
    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.1), 6))

    bars_ue = ax.bar(x - bar_w / 2, mean_ue, bar_w, label="Epistemic $U_e$",
                     color=_UE_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_ua = ax.bar(x + bar_w / 2, mean_ua, bar_w, label="Aleatoric $U_a$",
                     color=_UA_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)

    y_max = max(mean_ue.max(), mean_ua.max())
    for bar in (*bars_ue, *bars_ua):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.01,
                f"{h:.4f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Mean Uncertainty", fontsize=12)
    ax.set_title("Per-Class Mean Uncertainty", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", **_GRID_KW)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path(output_dir) / "per_class_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 3b. Per-class bar chart — dual scale (twinx)
# ---------------------------------------------------------------------------

def per_class_breakdown_twinx(
    ue: "array-like",
    ua: "array-like",
    labels: "array-like",
    output_dir: str | Path,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Grouped bar chart of mean Ue and Ua per class — DUAL y-axis (twinx).

    Left y-axis  (purple ticks) — Epistemic Ue bars
    Right y-axis (orange ticks) — Aleatoric Ua bars

    Each axis is independently scaled so the per-class pattern of both
    signals is clearly readable even when their absolute values differ
    greatly (e.g. Ue ≈ 0.02 and Ua ≈ 0.6).
    Saved as per_class_breakdown_twinx.png.
    """
    ue, ua, labels = to_numpy(ue), to_numpy(ua), to_numpy(labels).astype(int)
    classes = np.unique(labels)
    n_classes = len(classes)
    if class_names is None:
        class_names = [str(c) for c in classes]

    mean_ue = np.array([ue[labels == c].mean() for c in classes])
    mean_ua = np.array([ua[labels == c].mean() for c in classes])

    x, bar_w = np.arange(n_classes), 0.38
    fig, ax_ue = plt.subplots(figsize=(max(10, n_classes * 1.1), 6))

    # Left axis — Ue
    bars_ue = ax_ue.bar(x - bar_w / 2, mean_ue, bar_w,
                        color=_UE_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_ue.set_ylabel("Mean Epistemic Uncertainty  $U_e$", fontsize=12, color=_UE_COLOR)
    ax_ue.tick_params(axis="y", labelcolor=_UE_COLOR)
    ax_ue.set_ylim(0, mean_ue.max() * 1.30)

    # Right axis — Ua
    ax_ua = ax_ue.twinx()
    bars_ua = ax_ua.bar(x + bar_w / 2, mean_ua, bar_w,
                        color=_UA_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_ua.set_ylabel("Mean Aleatoric Uncertainty  $U_a$", fontsize=12, color=_UA_COLOR)
    ax_ua.tick_params(axis="y", labelcolor=_UA_COLOR)
    ax_ua.set_ylim(0, mean_ua.max() * 1.30)

    # Value labels
    for bar, val in zip(bars_ue, mean_ue):
        ax_ue.text(bar.get_x() + bar.get_width() / 2, val + mean_ue.max() * 0.01,
                   f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, color=_UE_COLOR)
    for bar, val in zip(bars_ua, mean_ua):
        ax_ua.text(bar.get_x() + bar.get_width() / 2, val + mean_ua.max() * 0.01,
                   f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, color=_UA_COLOR)

    handles = [
        mpatches.Patch(color=_UE_COLOR, alpha=0.85, label="Epistemic $U_e$  (left axis)"),
        mpatches.Patch(color=_UA_COLOR, alpha=0.85, label="Aleatoric $U_a$  (right axis)"),
    ]
    ax_ue.legend(handles=handles, fontsize=10, loc="upper left")
    ax_ue.set_xticks(x)
    ax_ue.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax_ue.set_title("Per-Class Mean Uncertainty  (dual scale)", fontsize=14, fontweight="bold")
    ax_ue.grid(True, axis="y", **_GRID_KW)
    ax_ue.set_axisbelow(True)
    fig.tight_layout()

    out = Path(output_dir) / "per_class_breakdown_dual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Convenience — generate all five plots
# ---------------------------------------------------------------------------

def generate_all(
    ue: "array-like",
    ua: "array-like",
    correct: "array-like",
    labels: "array-like",
    output_dir: str | Path,
    class_names: Optional[Sequence[str]] = None,
) -> list[Path]:
    """
    Generate all five diagnostic plots and return their paths.

    Saved files
    -----------
    scatter_ua_ue.png
    uncertainty_histograms.png          ← same scale
    uncertainty_histograms_twinx.png    ← dual scale
    per_class_breakdown.png             ← same scale
    per_class_breakdown_twinx.png       ← dual scale
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return [
        scatter_ua_ue(ue, ua, correct, output_dir),
        # uncertainty_histograms(ue, ua, correct, output_dir),
        uncertainty_histograms_twinx(ue, ua, correct, output_dir),
        # per_class_breakdown(ue, ua, labels, output_dir, class_names),
        per_class_breakdown_twinx(ue, ua, labels, output_dir, class_names),
    ]


