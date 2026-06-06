"""
Module: Publication-Grade Visualizations
==========================================
Generates 9 high-quality, journal-ready plots for the smart grid pipeline.

Plots produced:
  1.  Training curves       — loss, MAE, 90% coverage over epochs
  2.  Probabilistic forecast — actual vs [q05, q50, q95] over 1 week
  3.  Anomaly timeline       — reconstruction error + flagged windows
  4.  Error distribution     — normal vs anomaly histogram (KDE)
  5.  SHAP importance        — horizontal bar chart of top features
  6.  Load heatmap           — hour × day-of-week average consumption
  7.  Adaptive gate          — gate values + corrected windows over time
  8.  Quantile calibration   — predicted vs actual coverage (reliability diagram)
  9.  Latent t-SNE           — 2D embedding coloured by anomaly/normal

All plots use a consistent IEEE/Nature-inspired style:
  - Font: DejaVu Sans (universally available, clean)
  - Colour palette: accessible, print-safe
  - 300 DPI output as both PNG and PDF
  - Tight layout, proper axis labels, legends, and captions
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                      # headless rendering (works on Colab/Windows)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Global style — applied once at import time
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":    "#1A237E",   # deep navy
    "secondary":  "#E53935",   # crimson
    "accent":     "#00897B",   # teal
    "warn":       "#FB8C00",   # amber
    "light":      "#E8EAF6",   # lavender tint
    "grid":       "#CFD8DC",   # cool grey
    "normal":     "#43A047",   # green
    "anomaly":    "#E53935",   # red
    "band":       "#90CAF9",   # light blue (uncertainty band)
}

def _apply_style():
    plt.rcParams.update({
        "font.family":          "DejaVu Sans",
        "font.size":            10,
        "axes.titlesize":       12,
        "axes.titleweight":     "bold",
        "axes.labelsize":       10,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.grid.which":      "major",
        "grid.color":           PALETTE["grid"],
        "grid.linewidth":       0.6,
        "grid.linestyle":       "--",
        "legend.fontsize":      9,
        "legend.framealpha":    0.85,
        "legend.edgecolor":     PALETTE["grid"],
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "lines.linewidth":      1.5,
    })

_apply_style()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, plots_dir: str, name: str):
    """Save as high-resolution PNG and PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(plots_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


# ---------------------------------------------------------------------------
# Main Visualizer Class
# ---------------------------------------------------------------------------

class SmartGridVisualizer:

    def __init__(self, plots_dir: str = "plots", target_col: str = "Global_active_power"):
        self.plots_dir  = plots_dir
        self.target_col = target_col
        os.makedirs(plots_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Training Curves
    # -----------------------------------------------------------------------

    def plot_training_curves(self, history: list):
        """
        3-panel figure:
          (a) Pinball training loss
          (b) Validation MAE
          (c) 90% prediction interval coverage
        """
        epochs      = [h["epoch"]           for h in history]
        train_loss  = [h["train_loss"]       for h in history]
        val_mae     = [h["mae"]              for h in history]
        coverage    = [h["90pct_coverage"]   for h in history]
        best_ep     = epochs[int(np.argmin(val_mae))]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle("Model Training Diagnostics", fontsize=13, fontweight="bold", y=1.02)

        # (a) Loss
        axes[0].plot(epochs, train_loss, color=PALETTE["primary"], lw=2)
        axes[0].fill_between(epochs, train_loss, alpha=0.12, color=PALETTE["primary"])
        axes[0].set_title("(a) Training Loss (Pinball)")
        axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss")

        # (b) Val MAE
        axes[1].plot(epochs, val_mae, color=PALETTE["secondary"], lw=2)
        axes[1].axvline(best_ep, ls=":", color=PALETTE["accent"], lw=1.5,
                        label=f"Best @ ep {best_ep}")
        axes[1].fill_between(epochs, val_mae, alpha=0.12, color=PALETTE["secondary"])
        axes[1].set_title("(b) Validation MAE")
        axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("MAE (kW)")
        axes[1].legend()

        # (c) Coverage
        axes[2].plot(epochs, [c * 100 for c in coverage], color=PALETTE["accent"], lw=2)
        axes[2].axhline(90, ls="--", color=PALETTE["warn"], lw=1.5, label="Target 90%")
        axes[2].set_ylim(0, 105)
        axes[2].set_title("(c) 90% PI Coverage")
        axes[2].set_xlabel("Epoch");  axes[2].set_ylabel("Coverage (%)")
        axes[2].legend()

        fig.tight_layout()
        _save(fig, self.plots_dir, "01_training_curves")

    # -----------------------------------------------------------------------
    # 2. Probabilistic Forecast
    # -----------------------------------------------------------------------

    def plot_probabilistic_forecast(self, timestamps, actuals, q05, q50, q95):
        """
        Actual load vs [q05–q95] uncertainty band + q50 median forecast.
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        x = np.arange(len(timestamps))
        ax.fill_between(x, q05, q95, alpha=0.25, color=PALETTE["band"],
                        label="90% Prediction Interval")
        ax.plot(x, q50,     color=PALETTE["primary"],   lw=1.8, label="Median Forecast (q50)")
        ax.plot(x, actuals, color=PALETTE["secondary"],  lw=1.2, ls="--", alpha=0.85,
                label="Actual Load")

        # X-axis: show daily ticks
        step = max(1, len(x) // 14)
        tick_pos   = x[::step]
        tick_labels = [str(timestamps[i])[:13] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7.5)

        mae  = np.abs(q50 - actuals).mean()
        cov  = ((actuals >= q05) & (actuals <= q95)).mean() * 100
        ax.set_title(
            f"Probabilistic Load Forecast — MAE = {mae:.3f} kW  |  90% PI Coverage = {cov:.1f}%",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Time");  ax.set_ylabel("Active Power (kW)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, self.plots_dir, "02_probabilistic_forecast")

    # -----------------------------------------------------------------------
    # 3. Anomaly Detection Timeline
    # -----------------------------------------------------------------------

    def plot_anomaly_timeline(self, timestamps, recon_error, combined_flag,
                               threshold=None, detector_threshold=None):
        """
        Reconstruction error over time with anomaly windows highlighted.
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        x      = np.arange(len(recon_error))
        normal = combined_flag == 0
        anomal = combined_flag == 1

        # Shaded anomaly regions
        in_block = False
        for i in range(len(combined_flag)):
            if anomal[i] and not in_block:
                start = i;  in_block = True
            elif not anomal[i] and in_block:
                ax.axvspan(start, i, alpha=0.18, color=PALETTE["anomaly"], lw=0)
                in_block = False
        if in_block:
            ax.axvspan(start, len(combined_flag), alpha=0.18, color=PALETTE["anomaly"], lw=0)

        ax.plot(x, recon_error, color=PALETTE["primary"], lw=1.0, alpha=0.8,
                label="Reconstruction Error")

        # Threshold line
        thresh = threshold or float(np.percentile(recon_error[normal], 99)) if normal.any() else recon_error.max()
        ax.axhline(thresh, ls="--", color=PALETTE["warn"], lw=1.6,
                   label=f"Threshold = {thresh:.4f}")

        # Scatter anomalies
        ax.scatter(x[anomal], recon_error[anomal], color=PALETTE["anomaly"],
                   s=12, zorder=5, alpha=0.6, label=f"Anomaly ({anomal.sum():,})")

        # X-ticks
        step = max(1, len(x) // 12)
        tick_pos    = x[::step]
        tick_labels = [str(timestamps[i])[:13] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7.5)

        pct = 100 * anomal.mean()
        ax.set_title(f"Anomaly Detection Timeline  —  {anomal.sum():,} windows flagged ({pct:.1f}%)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time");  ax.set_ylabel("Reconstruction MSE")
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, self.plots_dir, "03_anomaly_timeline")

    # -----------------------------------------------------------------------
    # 4. Reconstruction Error Distribution
    # -----------------------------------------------------------------------

    def plot_recon_error_distribution(self, recon_error, combined_flag):
        """
        Overlapping KDE + histogram for normal vs anomalous windows.
        """
        from scipy.stats import gaussian_kde

        normal = recon_error[combined_flag == 0]
        anomal = recon_error[combined_flag == 1]

        fig, ax = plt.subplots(figsize=(9, 5))

        def _plot_dist(data, color, label):
            if len(data) < 2:
                return
            bins = np.linspace(recon_error.min(), np.percentile(recon_error, 99.5), 60)
            ax.hist(data, bins=bins, density=True, alpha=0.28, color=color, edgecolor="none")
            kde = gaussian_kde(data, bw_method=0.3)
            xs  = np.linspace(data.min(), data.max(), 400)
            ax.plot(xs, kde(xs), color=color, lw=2.2, label=label)

        _plot_dist(normal, PALETTE["normal"],  f"Normal  (n={len(normal):,})")
        _plot_dist(anomal, PALETTE["anomaly"], f"Anomaly (n={len(anomal):,})")

        thresh = float(np.percentile(normal, 99)) if len(normal) > 1 else recon_error.max()
        ax.axvline(thresh, ls="--", color=PALETTE["warn"], lw=1.8,
                   label=f"p99 Threshold = {thresh:.4f}")

        ax.set_title("Reconstruction Error Distribution: Normal vs Anomalous Windows",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Reconstruction MSE");  ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        _save(fig, self.plots_dir, "04_error_distribution")

    # -----------------------------------------------------------------------
    # 5. SHAP Feature Importance
    # -----------------------------------------------------------------------

    def plot_shap_importance(self, mean_shap: np.ndarray, feature_names: list, top_n: int = 15):
        """
        Horizontal bar chart of mean |SHAP| values for the top-N features.
        """
        if len(mean_shap) != len(feature_names):
            return

        sorted_idx  = np.argsort(mean_shap)[::-1][:top_n]
        sorted_vals = mean_shap[sorted_idx]
        sorted_feat = [feature_names[i] for i in sorted_idx]

        # Colour bars by magnitude
        norm      = plt.Normalize(sorted_vals.min(), sorted_vals.max())
        cmap      = plt.cm.get_cmap("Blues")
        bar_colors = [cmap(norm(v) * 0.7 + 0.3) for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.42)))
        bars = ax.barh(range(len(sorted_feat)), sorted_vals[::-1],
                       color=bar_colors[::-1], edgecolor="none", height=0.65)

        ax.set_yticks(range(len(sorted_feat)))
        ax.set_yticklabels(sorted_feat[::-1], fontsize=9)

        for bar, val in zip(bars, sorted_vals[::-1]):
            ax.text(bar.get_width() + sorted_vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax.set_title(f"SHAP Feature Importance — Mean |SHAP| for Anomaly Detection\n"
                     f"(Top {top_n} of {len(feature_names)} features)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|  (contribution to anomaly score)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
        _save(fig, self.plots_dir, "05_shap_importance")

    # -----------------------------------------------------------------------
    # 6. Load Profile Heatmap (hour × day-of-week)
    # -----------------------------------------------------------------------

    def plot_load_heatmap(self, df: pd.DataFrame):
        """
        Average consumption heatmap — rows = hour of day, cols = day of week.
        """
        if self.target_col not in df.columns:
            print(f"  [SKIP] load heatmap — '{self.target_col}' not in df")
            return

        pivot = df.groupby([df.index.hour, df.index.dayofweek])[self.target_col] \
                  .mean().unstack(fill_value=0)
        pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot.index.name = "Hour of Day"

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Avg. Active Power (kW)", fontsize=9)

        ax.set_xticks(range(7))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

        # Annotate cells
        for r in range(24):
            for c in range(7):
                ax.text(c, r, f"{pivot.values[r, c]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if pivot.values[r, c] > pivot.values.max() * 0.6 else "black")

        ax.set_title("Average Load Profile — Hour of Day × Day of Week",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Day of Week");  ax.set_ylabel("Hour of Day")
        fig.tight_layout()
        _save(fig, self.plots_dir, "06_load_heatmap")

    # -----------------------------------------------------------------------
    # 7. Adaptive Gate Values
    # -----------------------------------------------------------------------

    def plot_adaptive_gate(self, timestamps, gate_vals, corrected):
        """
        Gate values over time (1 = fully trusted, 0 = anomalous/corrected).
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        x = np.arange(len(gate_vals))

        ax.fill_between(x, gate_vals, alpha=0.18, color=PALETTE["primary"])
        ax.plot(x, gate_vals, color=PALETTE["primary"], lw=1.5, label="Gate Value")

        if corrected.any():
            ax.scatter(x[corrected], gate_vals[corrected],
                       color=PALETTE["anomaly"], s=20, zorder=5, alpha=0.7,
                       label=f"Corrected ({corrected.sum():,})")

        ax.axhline(0.5, ls=":", color=PALETTE["warn"], lw=1.2, alpha=0.8,
                   label="Trust threshold (0.5)")
        ax.set_ylim(-0.05, 1.10)

        step = max(1, len(x) // 12)
        tick_pos    = x[::step]
        tick_labels = [str(timestamps[i])[:13] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7.5)

        ax.set_title("Adaptive Feedback Gate — Input Trust Level Over Time",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time");  ax.set_ylabel("Gate Value  [0 = anomaly, 1 = normal]")
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save(fig, self.plots_dir, "07_adaptive_gate")

    # -----------------------------------------------------------------------
    # 8. Quantile Calibration (Reliability Diagram)
    # -----------------------------------------------------------------------

    def plot_quantile_calibration(self, actuals, q05, q50, q95):
        """
        Expected quantile coverage vs actual empirical coverage.
        A perfectly calibrated model lies on the diagonal.
        """
        quantile_levels = np.linspace(0.01, 0.99, 49)
        empirical       = []

        for q_level in quantile_levels:
            # Build symmetric PI centred on q50
            lo_q = 0.5 - q_level / 2
            hi_q = 0.5 + q_level / 2
            # Linearly interpolate between q05/q50/q95
            lo = q05 + (q50 - q05) * (lo_q - 0.05) / (0.50 - 0.05)
            hi = q50 + (q95 - q50) * (hi_q - 0.50) / (0.95 - 0.50)
            lo = np.clip(lo, -np.inf, q50)
            hi = np.clip(hi, q50, np.inf)
            coverage = float(((actuals >= lo) & (actuals <= hi)).mean())
            empirical.append(coverage)

        fig, ax = plt.subplots(figsize=(7, 7))

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["grid"], lw=1.8, label="Perfect Calibration")

        # Shaded band ±5%
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                         alpha=0.10, color=PALETTE["accent"], label="±5% Band")

        ax.plot(quantile_levels, empirical, "o-", color=PALETTE["primary"],
                lw=2, ms=4, label="Model")

        # Annotate the three main quantiles
        for nom, emp_col in zip([0.1, 0.5, 0.9],
                                [empirical[4], empirical[24], empirical[44]]):
            ax.annotate(f"  {nom:.0%} → {emp_col:.1%}",
                        xy=(nom, emp_col), fontsize=8,
                        color=PALETTE["secondary"])

        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
        ax.set_title("Quantile Calibration (Reliability Diagram)\n"
                     "Ideal: empirical coverage = nominal coverage",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Nominal Coverage (Expected)")
        ax.set_ylabel("Empirical Coverage (Actual)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(fig, self.plots_dir, "08_quantile_calibration")

    # -----------------------------------------------------------------------
    # 9. Latent Space t-SNE
    # -----------------------------------------------------------------------

    def plot_latent_tsne(self, latents: np.ndarray, combined_flag: np.ndarray,
                          max_points: int = 2000):
        """
        2-D t-SNE projection of autoencoder latent vectors,
        coloured by normal (green) / anomaly (red).
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("  [SKIP] t-SNE — sklearn not available")
            return

        n = min(max_points, len(latents))
        idx = np.random.choice(len(latents), n, replace=False)
        Z   = latents[idx]
        lbl = combined_flag[idx]

        # Normalise latents before t-SNE
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)

        tsne    = TSNE(n_components=2, perplexity=min(30, n // 5),
                       random_state=42, n_iter=1000, verbose=0)
        Z_2d    = tsne.fit_transform(Z)

        fig, ax = plt.subplots(figsize=(8, 7))

        for flag, color, label, marker, size in [
            (0, PALETTE["normal"],  "Normal",  "o", 14),
            (1, PALETTE["anomaly"], "Anomaly", "^", 30),
        ]:
            mask = lbl == flag
            if mask.any():
                ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                           c=color, label=f"{label} (n={mask.sum():,})",
                           s=size, alpha=0.55, marker=marker, edgecolors="none")

        ax.set_title("Autoencoder Latent Space — t-SNE Projection\n"
                     "Anomalies form distinct clusters away from the normal manifold",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(markerscale=1.5)
        ax.set_xticks([]);  ax.set_yticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        fig.tight_layout()
        _save(fig, self.plots_dir, "09_latent_tsne")


# ---------------------------------------------------------------------------
# Smoke-test (run standalone to verify all plots generate without error)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.data_utils import generate_synthetic_dataset

    np.random.seed(0)
    n = 500
    df  = generate_synthetic_dataset(n_hours=n)
    idx = pd.date_range("2023-01-01", periods=n, freq="H")

    viz = SmartGridVisualizer(plots_dir="test_plots")

    # Fake training history
    history = [{"epoch": e, "train_loss": 0.2 - e * 0.003,
                "mae": 0.3 - e * 0.002, "90pct_coverage": 0.85 + e * 0.001,
                "predictions": None, "targets": None} for e in range(1, 21)]

    q50 = np.random.rand(n) * 2 + 1
    q05 = q50 - 0.4
    q95 = q50 + 0.4
    act = q50 + np.random.randn(n) * 0.2

    flags = np.zeros(n, dtype=int)
    flags[np.random.choice(n, 20, replace=False)] = 1
    errors = np.abs(np.random.randn(n)) * 0.05
    errors[flags == 1] += 0.3

    latents = np.random.randn(n, 8)
    latents[flags == 1] += 3

    feat_names = [f"feature_{i}" for i in range(12)]
    shap_vals  = np.abs(np.random.randn(12))

    viz.plot_training_curves(history)
    viz.plot_probabilistic_forecast(idx, act, q05, q50, q95)
    viz.plot_anomaly_timeline(idx, errors, flags)
    viz.plot_recon_error_distribution(errors, flags)
    viz.plot_shap_importance(shap_vals, feat_names)
    viz.plot_load_heatmap(df)
    viz.plot_adaptive_gate(idx, np.clip(1 - errors * 2, 0, 1), flags.astype(bool))
    viz.plot_quantile_calibration(act, q05, q50, q95)
    viz.plot_latent_tsne(latents, flags)

    print("\nAll 9 plots generated in ./test_plots/")


    # -----------------------------------------------------------------------
    # 10. Metrics Summary Bar Chart
    # -----------------------------------------------------------------------

    def plot_metrics_summary(self, best_metrics: dict):
        """
        Clean publication-grade bar chart summarising all 5 forecasting metrics.
        Two panels: error metrics (MAE, RMSE, MAPE) and quality metrics (R², Coverage).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Forecasting Performance — Best Epoch Metrics Summary",
                     fontsize=13, fontweight="bold", y=1.02)

        # ── Panel A: Error metrics (lower is better) ───────────────────
        ax = axes[0]
        labels = ["MAE (kW)", "RMSE (kW)", "MAPE (%)"]
        values = [
            best_metrics.get("mae",  float("nan")),
            best_metrics.get("rmse", float("nan")),
            best_metrics.get("mape", float("nan")),
        ]
        colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warn"]]
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="none")

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(values) * 0.02,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        ax.set_title("(a) Error Metrics  [lower is better]", fontsize=11)
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.25)
        ax.spines["left"].set_linewidth(1.2)

        # ── Panel B: Quality metrics (higher is better) ───────────────
        ax = axes[1]
        labels2 = ["R²", "90% PI Coverage"]
        values2 = [
            best_metrics.get("r2",             float("nan")),
            best_metrics.get("90pct_coverage", float("nan")),
        ]
        colors2 = [PALETTE["accent"], PALETTE["normal"]]
        bars2 = ax.bar(labels2, values2, color=colors2, width=0.4, edgecolor="none")

        for bar, val in zip(bars2, values2):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        ax.axhline(1.0, ls="--", color=PALETTE["grid"], lw=1.2, label="Perfect = 1.0")
        ax.axhline(0.9, ls=":",  color=PALETTE["warn"], lw=1.2, label="Target 90% PI")
        ax.set_title("(b) Quality Metrics  [higher is better]", fontsize=11)
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)

        fig.tight_layout()
        _save(fig, self.plots_dir, "10_metrics_summary")
