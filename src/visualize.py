"""
src/visualize.py
All visualization functions for wildfire risk prediction.

Includes:
  - Geospatial risk maps (GeoTIFF → Folium/Plotly)
  - Embedding PCA plots (visualize what the foundation model learned)
  - Temporal dryness trend maps (year-over-year embedding drift)
  - Model evaluation plots (ROC, PR, calibration)
  - SHAP summary and dependence plots
  - Feature importance comparison (XGBoost vs LightGBM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RISK_THRESHOLDS, RISK_COLORS, OUTPUTS_DIR


# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ─────────────────────────────────────────────────────────────────
# Risk Map
# ─────────────────────────────────────────────────────────────────

def plot_risk_map(
    df: pd.DataFrame,
    prob_col: str = "fire_prob",
    title: str = "Wildfire Risk Map",
    save_path: Optional[Path] = None,
    figsize: Tuple = (14, 10),
) -> plt.Figure:
    """
    Plot predicted fire probability as a geospatial heatmap.
    
    Uses a custom 4-tier colormap: green (low) → yellow → orange → purple (extreme)
    matching RISK_COLORS from config.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap matching risk tiers
    colors_list = ["#4caf50", "#8bc34a", "#ffeb3b", "#ff9800", "#f44336", "#7b1fa2"]
    cmap = mcolors.LinearSegmentedColormap.from_list("wildfire_risk", colors_list, N=256)
    
    scatter = ax.scatter(
        df["lon"], df["lat"],
        c=df[prob_col],
        cmap=cmap,
        vmin=0, vmax=1,
        s=1,
        alpha=0.7,
        rasterized=True,
    )
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fire Probability", fontsize=11)
    
    # Add risk tier boundaries on colorbar
    for tier, (lo, hi) in RISK_THRESHOLDS.items():
        cbar.ax.axhline(lo, color="white", linewidth=0.8, alpha=0.7)
        cbar.ax.text(1.5, (lo + hi) / 2, tier.upper(), va="center",
                     fontsize=8, color="white", fontweight="bold",
                     transform=cbar.ax.get_yaxis_transform())
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_aspect("equal")
    
    _add_risk_stats(ax, df, prob_col)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Risk map saved → {save_path}")
    return fig


def plot_risk_tier_map(
    df: pd.DataFrame,
    tier_col: str = "risk_tier",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Categorical risk tier map with 4 discrete colors."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    tier_order = ["low", "medium", "high", "extreme"]
    
    for tier in tier_order:
        mask = df[tier_col] == tier
        if mask.any():
            ax.scatter(
                df.loc[mask, "lon"], df.loc[mask, "lat"],
                c=RISK_COLORS[tier],
                s=1, alpha=0.6, label=f"{tier.title()} ({mask.sum():,})",
                rasterized=True,
            )
    
    ax.legend(loc="lower right", frameon=True, markerscale=5)
    ax.set_title("Wildfire Risk Tiers", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_interactive_risk_map(
    df: pd.DataFrame,
    prob_col: str = "fire_prob",
    save_html: Optional[Path] = None,
):
    """
    Create an interactive Folium map with risk overlay.
    Opens in browser; can be embedded in a dashboard.
    """
    import folium
    from folium.plugins import HeatMap
    
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB dark_matter",
    )
    
    # Heatmap layer
    heat_data = df[[prob_col, "lat", "lon"]].sample(min(50000, len(df)))
    heat_list = [[row["lat"], row["lon"], row[prob_col]]
                 for _, row in heat_data.iterrows()]
    
    HeatMap(
        heat_list,
        radius=8,
        blur=6,
        gradient={0.0: "green", 0.35: "yellow", 0.6: "orange", 0.85: "red", 1.0: "purple"},
        name="Wildfire Risk",
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    if save_html:
        m.save(str(save_html))
        logger.info(f"Interactive map saved → {save_html}")
    
    return m


def _add_risk_stats(ax: plt.Axes, df: pd.DataFrame, prob_col: str):
    """Add risk tier breakdown text to map."""
    stats = []
    for tier, (lo, hi) in RISK_THRESHOLDS.items():
        count = ((df[prob_col] >= lo) & (df[prob_col] < hi)).sum()
        pct = 100 * count / len(df)
        stats.append(f"{tier.title()}: {pct:.1f}%")
    
    ax.text(
        0.02, 0.98, "\n".join(stats),
        transform=ax.transAxes,
        fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )


# ─────────────────────────────────────────────────────────────────
# Temporal Dryness Trends
# ─────────────────────────────────────────────────────────────────

def plot_dryness_trend(
    df: pd.DataFrame,
    drift_col: str = "drift_norm_3yr",
    year: int = 2024,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Map embedding drift magnitude as a proxy for vegetation dryness change.
    
    High drift = pixel has changed significantly in the foundation model's
    latent space over the past 3 years. Combined with direction (drift_mean),
    this identifies pixels transitioning from dense/moist to dry/sparse.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Drift magnitude
    sc1 = axes[0].scatter(
        df["lon"], df["lat"],
        c=df[drift_col],
        cmap="YlOrRd",
        s=1, alpha=0.7, rasterized=True,
    )
    plt.colorbar(sc1, ax=axes[0], label="Embedding Drift Magnitude")
    axes[0].set_title(f"Vegetation Change Intensity ({year-3}–{year})")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    
    # Panel 2: Drift direction (positive = drying)
    direction_col = drift_col.replace("drift_norm", "drift_mean")
    if direction_col in df.columns:
        sc2 = axes[1].scatter(
            df["lon"], df["lat"],
            c=df[direction_col],
            cmap="RdYlGn_r",  # Red = drying, Green = greening
            s=1, alpha=0.7, rasterized=True,
            vmin=-0.5, vmax=0.5,
        )
        plt.colorbar(sc2, ax=axes[1], label="Direction (+ = drying, - = greening)")
        axes[1].set_title(f"Drying/Greening Trend ({year-3}–{year})")
        axes[1].set_xlabel("Longitude")
    
    plt.suptitle("Temporal Embedding Analysis — Vegetation Dryness", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_annual_embedding_pca(
    pca_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize how the embedding space evolves across years using PCA.
    
    Overlaid scatterplots of PC1 vs PC2 for each year.
    If labels provided, fire/non-fire pixels are colored differently.
    
    This answers: "Is the landscape getting drier over time?"
    If the fire pixel cluster shifts toward the 2023-2024 non-fire cluster,
    drying is happening at landscape scale.
    """
    n_years = len(pca_dict)
    years = sorted(pca_dict.keys())
    
    fig, axes = plt.subplots(2, (n_years + 1) // 2, figsize=(4 * ((n_years + 1) // 2), 8))
    axes = axes.ravel()
    
    all_pca = np.vstack([pca_dict[y] for y in years])
    x_lim = (np.percentile(all_pca[:, 0], 1), np.percentile(all_pca[:, 0], 99))
    y_lim = (np.percentile(all_pca[:, 1], 1), np.percentile(all_pca[:, 1], 99))
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        pca_data = pca_dict[year]
        
        if labels_dict and year in labels_dict:
            labels = labels_dict[year]
            fire_mask = labels == 1
            ax.scatter(pca_data[~fire_mask, 0], pca_data[~fire_mask, 1],
                      c="#4caf50", s=0.5, alpha=0.3, label="No fire", rasterized=True)
            ax.scatter(pca_data[fire_mask, 0], pca_data[fire_mask, 1],
                      c="#f44336", s=2, alpha=0.8, label="Fire", rasterized=True)
            if idx == 0:
                ax.legend(markerscale=5, fontsize=8)
        else:
            ax.scatter(pca_data[:, 0], pca_data[:, 1],
                      c="#2196F3", s=0.5, alpha=0.4, rasterized=True)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(str(year), fontsize=12)
        ax.set_xlabel("PC1" if idx >= n_years // 2 else "")
        ax.set_ylabel("PC2" if idx % 2 == 0 else "")
    
    # Hide unused axes
    for i in range(len(years), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("AEF Embedding Space by Year (PCA)\nRed=fire pixels, Green=non-fire",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────
# Model Evaluation Plots
# ─────────────────────────────────────────────────────────────────

def plot_evaluation_suite(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cv_results: pd.DataFrame = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    4-panel evaluation dashboard:
      1. ROC curve
      2. Precision-Recall curve
      3. Calibration curve (reliability diagram)
      4. Spatial CV scores (if cv_results provided)
    """
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ── Panel 1: ROC ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = np.trapz(tpr, fpr)
    ax1.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.1, color="#1565C0")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    
    # ── Panel 2: Precision-Recall ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision, recall)
    baseline = y_true.mean()
    ax2.plot(recall, precision, color="#7B1FA2", lw=2, label=f"PR-AUC = {abs(pr_auc):.4f}")
    ax2.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline = {baseline:.3f}")
    ax2.fill_between(recall, precision, alpha=0.1, color="#7B1FA2")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (primary metric)")
    ax2.legend()
    
    # ── Panel 3: Calibration ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=15)
    ax3.plot(prob_pred, prob_true, "s-", color="#E65100", lw=2, ms=6, label="Model")
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax3.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color="#E65100")
    ax3.set_xlabel("Predicted Probability")
    ax3.set_ylabel("Observed Frequency")
    ax3.set_title("Calibration (Reliability Diagram)")
    ax3.legend()
    
    # ── Panel 4: Spatial CV ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if cv_results is not None:
        metrics = ["auc_roc", "pr_auc", "brier_score"]
        x = np.arange(len(cv_results))
        width = 0.25
        colors = ["#1565C0", "#7B1FA2", "#E65100"]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax4.bar(x + i * width, cv_results[metric], width,
                   label=metric.replace("_", " ").upper(), color=color, alpha=0.8)
        
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([f"Fold {i+1}" for i in range(len(cv_results))], fontsize=9)
        ax4.set_title("Spatial CV — Per-fold Metrics")
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1.1)
    else:
        ax4.text(0.5, 0.5, "Run spatial_cross_validate()\nto see fold results",
                ha="center", va="center", transform=ax4.transAxes, fontsize=11,
                color="gray")
        ax4.set_title("Spatial CV Results")
    
    plt.suptitle("Wildfire Risk Model — Evaluation Dashboard", fontsize=15, fontweight="bold", y=1.01)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Evaluation suite saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────
# SHAP Plots
# ─────────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 25,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """SHAP beeswarm summary plot — shows which features drive predictions most."""
    import shap as _shap
    
    fig, ax = plt.subplots(figsize=(10, 12))
    _shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Importance — Wildfire Risk Model", fontsize=13, pad=15)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_group_contributions(
    group_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart showing SHAP contribution by feature group.
    Shows how much of the model relies on the foundation model vs other signals.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["#1565C0", "#E65100", "#7B1FA2", "#2E7D32", "#C62828", "#546E7A"]
    
    bars = ax.barh(
        group_df["feature_group"],
        group_df["pct_contribution"],
        color=colors[:len(group_df)],
        edgecolor="white",
        linewidth=0.5,
    )
    
    for bar, val in zip(bars, group_df["pct_contribution"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    
    ax.set_xlabel("% of SHAP Contribution")
    ax.set_title("What Drives Wildfire Risk Predictions?\n(by feature group)", fontsize=13)
    ax.set_xlim(0, group_df["pct_contribution"].max() * 1.15)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_fire_history_vs_risk(
    df: pd.DataFrame,
    prob_col: str = "fire_prob",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Violin plot: Distribution of predicted fire probability by historical fire frequency.
    Validates that the model correctly assigns higher risk to pixels with fire history.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for ax, window in zip(axes, [3, 5, 7]):
        col = f"fire_count_{window}yr"
        if col not in df.columns:
            continue
        
        df_plot = df.copy()
        df_plot["fire_bucket"] = pd.cut(
            df_plot[col], bins=[-0.1, 0, 1, 3, 10, 999],
            labels=["0", "1", "2-3", "4-10", "10+"]
        )
        
        groups = [df_plot[df_plot["fire_bucket"] == b][prob_col].dropna().values
                  for b in ["0", "1", "2-3", "4-10", "10+"]]
        groups = [g for g in groups if len(g) > 10]
        
        ax.violinplot(groups, showmedians=True)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(["0", "1", "2-3", "4-10", "10+"][:len(groups)])
        ax.set_xlabel(f"Fire detections (past {window} yrs)")
        ax.set_title(f"{window}-year fire history")
        ax.set_ylabel("Predicted Fire Probability" if window == 3 else "")
    
    plt.suptitle("Predicted Risk vs Historical Fire Frequency", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
