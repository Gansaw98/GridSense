"""
Main Pipeline: Smart Grid Forecasting + Anomaly Detection
All files are in the same flat directory — no subfolders.
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Flat imports (all files in same folder) ────────────────────────────────
from data_utils import load_uci_dataset, generate_synthetic_dataset, train_val_test_split
from feature_engineering import SmartGridFeatureEngineer
from probabilistic_transformer import (
    ProbabilisticLSTMTransformer,
    SmartGridDataset,
    ForecastTrainer,
)
from anomaly_detector import SmartGridAnomalyDetector
from explainability import AnomalyExplainer
from adaptive_feedback import AnomalyGate, AdaptiveFeedbackLoop
from visualizations import SmartGridVisualizer   # ← was missing


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "target_col":          "Global_active_power",
    "seq_len":             48,
    "horizon":             1,
    "holiday_dates":       ["2023-01-01", "2023-12-25", "2023-07-04"],
    "d_model":             128,
    "n_heads":             4,
    "n_lstm_layers":       2,
    "n_tf_layers":         2,
    "dropout":             0.1,
    "quantiles":           [0.05, 0.50, 0.95],
    "forecast_epochs":     50,
    "early_stop_patience": 8,
    "ae_epochs":           40,
    "batch_size":          64,
    "lr":                  1e-3,
    "latent_dim":          32,
    "iso_contamination":   0.02,
    "recon_percentile":    99.0,
    "iso_threshold":       0.65,
    "correction_mode":     "impute",
    "top_k_features":      3,
    "max_explanations":    5,
    "plots_dir":           "plots",
    "plot_forecast_hours": 168,
}


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

def step1_prepare_data(data_path="household_power_consumption.txt"):
    print("\n" + "="*60)
    print("STEP 1: Data Loading & Feature Engineering")
    print("="*60)

    if data_path and os.path.exists(data_path):
        print(f"  Loading UCI dataset from: {data_path}")
        df = load_uci_dataset(data_path)
    else:
        print("  WARNING: Dataset not found. Generating synthetic data...")
        df = generate_synthetic_dataset(n_hours=8760)

    print(f"  Dataset shape: {df.shape}  |  {df.index[0]} -> {df.index[-1]}")

    eng = SmartGridFeatureEngineer(
        target_col=CONFIG["target_col"],
        lag_hours=[1, 2, 3, 6, 12, 24, 48, 168],
        rolling_windows=[6, 12, 24, 48],
        holiday_dates=CONFIG["holiday_dates"],
    )
    X_all = eng.fit_transform(df)
    print(f"  Engineered features: {X_all.shape[1]}  (shape: {X_all.shape})")

    X_train, X_val, X_test = train_val_test_split(X_all)
    print(f"  Train / Val / Test: {X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}")

    target_idx = eng.feature_names.index(CONFIG["target_col"])
    return X_train, X_val, X_test, eng, target_idx, df


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

def step2_train_forecaster(X_train, X_val, target_idx):
    print("\n" + "="*60)
    print("STEP 2: Probabilistic LSTM-Transformer Training")
    print("="*60)

    n_features   = X_train.shape[1]
    seq_len      = CONFIG["seq_len"]
    horizon      = CONFIG["horizon"]
    patience     = CONFIG["early_stop_patience"]

    train_ds     = SmartGridDataset(X_train, target_idx, seq_len, horizon)
    val_ds       = SmartGridDataset(X_val,   target_idx, seq_len, horizon)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)

    model = ProbabilisticLSTMTransformer(
        n_features=n_features,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_lstm_layers=CONFIG["n_lstm_layers"],
        n_tf_layers=CONFIG["n_tf_layers"],
        dropout=CONFIG["dropout"],
        quantiles=CONFIG["quantiles"],
        horizon=horizon,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer     = ForecastTrainer(model, lr=CONFIG["lr"])
    history     = []
    best_mae    = float("inf")
    patience_ct = 0

    for ep in range(1, CONFIG["forecast_epochs"] + 1):
        tr_loss     = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        history.append({"epoch": ep, "train_loss": tr_loss, **val_metrics})

        if val_metrics["mae"] < best_mae:
            best_mae    = val_metrics["mae"]
            patience_ct = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_ct += 1

        if ep % 5 == 0:
            print(
                f"  Ep {ep:3d} | loss={tr_loss:.4f} | "
                f"MAE={val_metrics['mae']:.4f} | "
                f"RMSE={val_metrics['rmse']:.4f} | "
                f"MAPE={val_metrics['mape']:.2f}% | "
                f"R²={val_metrics['r2']:.4f} | "
                f"90%cov={val_metrics['90pct_coverage']:.2%} | "
                f"patience={patience_ct}/{patience}"
            )

        if patience_ct >= patience:
            print(f"\n  Early stopping at epoch {ep}.")
            break

    model.load_state_dict(torch.load("best_model.pt", map_location=trainer.device))

    # ── Print best-epoch metrics ───────────────────────────────────────
    best_h = history[int(np.argmin([h["mae"] for h in history]))]
    print(f"\n  {'='*50}")
    print(f"  BEST EPOCH FORECASTING METRICS (Epoch {best_h['epoch']})")
    print(f"  {'='*50}")
    print(f"  MAE             : {best_h['mae']:.4f} kW")
    print(f"  RMSE            : {best_h['rmse']:.4f} kW")
    print(f"  MAPE            : {best_h['mape']:.2f} %")
    print(f"  R²              : {best_h['r2']:.4f}")
    print(f"  90% PI Coverage : {best_h['90pct_coverage']:.2%}")
    print(f"  {'='*50}")

    # ── Save metrics to CSV ───────────────────────────────────────────
    metrics_df = pd.DataFrame([{
        "Metric": "MAE (kW)",        "Value": round(best_h["mae"],  4)},
        {"Metric": "RMSE (kW)",       "Value": round(best_h["rmse"], 4)},
        {"Metric": "MAPE (%)",        "Value": round(best_h["mape"], 2)},
        {"Metric": "R²",              "Value": round(best_h["r2"],   4)},
        {"Metric": "90% PI Coverage", "Value": round(best_h["90pct_coverage"], 4)},
    ])
    metrics_df.to_csv("forecasting_metrics.csv", index=False)
    print("  Saved: forecasting_metrics.csv")

    return model, trainer, history


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------

def step3_train_anomaly_detector(X_train, n_features):
    print("\n" + "="*60)
    print("STEP 3: Anomaly Detector (AE + Isolation Forest + Contrastive)")
    print("="*60)

    detector = SmartGridAnomalyDetector(
        n_features=n_features,
        latent_dim=CONFIG["latent_dim"],
        seq_len=CONFIG["seq_len"],
        iso_contamination=CONFIG["iso_contamination"],
        recon_percentile=CONFIG["recon_percentile"],
    )
    detector.fit(X_train, epochs=CONFIG["ae_epochs"])
    return detector


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

def step4_detect(detector, X_test):
    print("\n" + "="*60)
    print("STEP 4: Anomaly Detection on Test Set")
    print("="*60)

    results = detector.detect(X_test)
    n_anom  = int(results["combined_flag"].sum())
    n_total = len(results["combined_flag"])
    print(f"  Anomalies detected: {n_anom} / {n_total}  ({100*n_anom/n_total:.1f}%)")
    print(f"  Mean reconstruction error : {results['recon_error'].mean():.5f}")
    print(f"  Mean isolation score      : {results['iso_score'].mean():.4f}")
    return results


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------

def step5_explain(detector, results, X_test, feature_names, seq_len):
    print("\n" + "="*60)
    print("STEP 5: SHAP Explainability for Flagged Anomalies")
    print("="*60)

    n_windows = len(X_test) - seq_len + 1
    X_windows = np.array([X_test[i: i + seq_len] for i in range(n_windows)])

    normal_mask = results["combined_flag"][:n_windows] == 0
    background  = X_windows[normal_mask][:50]

    explainer = AnomalyExplainer(
        detector=detector,
        feature_names=feature_names,
        background_X=background,
        top_k=CONFIG["top_k_features"],
    )

    reports = explainer.explain_batch(
        detect_results={k: v[:n_windows] for k, v in results.items() if isinstance(v, np.ndarray)},
        X_windows=X_windows,
        max_explanations=CONFIG["max_explanations"],
    )

    print(f"\n  Explanations for top {len(reports)} anomalies:\n")
    for r in reports:
        print(f"  Window {r['window_index']}:")
        print("  " + r["summary"].replace("\n", "\n  "))
        print()

    df_exp = explainer.to_dataframe(reports)
    df_exp.to_csv("anomaly_explanations.csv", index=False)
    print("  Saved: anomaly_explanations.csv")
    return reports, explainer


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------

def step6_adaptive_inference(model, detector, results, X_test, seq_len):
    print("\n" + "="*60)
    print("STEP 6: Adaptive Feedback Loop Inference")
    print("="*60)

    n_features = X_test.shape[1]
    gate       = AnomalyGate(n_features=n_features)
    controller = AdaptiveFeedbackLoop(
        forecast_model=model,
        anomaly_gate=gate,
        iso_threshold=CONFIG["iso_threshold"],
        recon_threshold=float(detector._recon_threshold),
        correction_mode=CONFIG["correction_mode"],
    )

    predictions     = []
    corrected_count = 0
    n_steps         = min(len(results["combined_flag"]), len(X_test) - seq_len)

    for step in range(n_steps):
        x_seq         = X_test[step: step + seq_len]
        single_result = {k: v[step:step+1] for k, v in results.items()
                         if isinstance(v, np.ndarray) and v.ndim >= 1}
        pred = controller.predict(x_seq, single_result, window_idx=0)
        predictions.append(pred)
        if pred["corrected"]:
            corrected_count += 1

    print(f"  Total predictions : {len(predictions)}")
    print(f"  Corrected windows : {corrected_count}  ({100*corrected_count/max(len(predictions),1):.1f}%)")
    for k, v in controller.get_stats().items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return predictions


# ---------------------------------------------------------------------------
# Step 7 — Publication Plots
# ---------------------------------------------------------------------------

def step7_visualize(df, eng, history, results, reports, predictions,
                    X_test, target_idx, explainer):
    print("\n" + "="*60)
    print("STEP 7: Generating Publication-Grade Plots")
    print("="*60)

    os.makedirs(CONFIG["plots_dir"], exist_ok=True)
    viz = SmartGridVisualizer(plots_dir=CONFIG["plots_dir"],
                              target_col=CONFIG["target_col"])

    n_h  = CONFIG["plot_forecast_hours"]
    seq  = CONFIG["seq_len"]
    q05  = np.array([p["q05"] for p in predictions[:n_h]])
    q50  = np.array([p["q50"] for p in predictions[:n_h]])
    q95  = np.array([p["q95"] for p in predictions[:n_h]])
    actuals         = X_test[seq: seq + n_h, target_idx]
    test_timestamps = df.index[-len(X_test):]
    pred_timestamps = test_timestamps[seq: seq + n_h]
    n_windows       = len(results["combined_flag"])
    anom_timestamps = test_timestamps[:n_windows]

    viz.plot_training_curves(history)
    viz.plot_probabilistic_forecast(pred_timestamps, actuals, q05, q50, q95)
    viz.plot_anomaly_timeline(anom_timestamps, results["recon_error"],
                              results["combined_flag"])
    viz.plot_recon_error_distribution(results["recon_error"], results["combined_flag"])

    if reports:
        all_shap  = np.array([r["shap_values"] for r in reports])
        mean_shap = np.abs(all_shap).mean(axis=0)
        viz.plot_shap_importance(mean_shap, eng.feature_names)

    viz.plot_load_heatmap(df)

    gate_vals = np.array([p["gate_value"] for p in predictions[:n_h]])
    corrected = np.array([p["corrected"]  for p in predictions[:n_h]])
    viz.plot_adaptive_gate(pred_timestamps, gate_vals, corrected)
    viz.plot_quantile_calibration(actuals, q05, q50, q95)
    viz.plot_latent_tsne(results["latents"], results["combined_flag"])

    # Metrics summary bar chart
    best_h = history[int(np.argmin([h["mae"] for h in history]))]
    viz.plot_metrics_summary(best_h)

    print(f"\n  All plots saved to ./{CONFIG['plots_dir']}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",            type=str, default=None)
    parser.add_argument("--forecast-epochs", type=int, default=CONFIG["forecast_epochs"])
    parser.add_argument("--ae-epochs",       type=int, default=CONFIG["ae_epochs"])
    args = parser.parse_args()
    CONFIG["forecast_epochs"] = args.forecast_epochs
    CONFIG["ae_epochs"]       = args.ae_epochs

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║    SMART GRID: Probabilistic Forecasting + XAI Pipeline  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── FIX: use args.data if provided, otherwise fall back to default ──
    data_path = args.data or "household_power_consumption.txt"

    X_train, X_val, X_test, eng, target_idx, df = step1_prepare_data(data_path)
    model, trainer, history = step2_train_forecaster(X_train, X_val, target_idx)
    detector = step3_train_anomaly_detector(X_train, X_train.shape[1])
    results  = step4_detect(detector, X_test)
    reports, explainer = step5_explain(detector, results, X_test,
                                       eng.feature_names, CONFIG["seq_len"])
    predictions = step6_adaptive_inference(model, detector, results,
                                           X_test, CONFIG["seq_len"])
    step7_visualize(df, eng, history, results, reports, predictions,
                    X_test, target_idx, explainer)

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("Outputs: best_model.pt | anomaly_explanations.csv | plots/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
