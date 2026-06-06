"""
Module 5: Adaptive Feedback Loop
- Monitors Isolation Forest confidence scores in real-time
- When high-confidence anomaly detected, filters the corrupted window
  from the forecasting context via soft attention masking
- Adjusts the forecaster's effective input weights adaptively
- Provides a "corrected" prediction that ignores the anomalous signal
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings


# ---------------------------------------------------------------------------
# Anomaly-Aware Input Gate
# ---------------------------------------------------------------------------

class AnomalyGate(nn.Module):
    """
    Learnable soft gate applied to the input sequence.
    
    When the anomaly score for a window is high, the gate
    suppresses that window's contribution to the forecasting context.
    
    Gate value g ∈ [0,1]:
        g = 0 → window fully masked (anomaly)
        g = 1 → window fully trusted (normal)
    
    The gate is a lightweight MLP that takes anomaly signals as input.
    """

    def __init__(self, n_features: int, hidden_dim: int = 32):
        super().__init__()
        # Takes (recon_error, iso_score, rolling_mean_error) as input → scalar gate
        self.gate_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, seq_len, n_features) — input sequence
        recon_errors: torch.Tensor,  # (B,) per-window reconstruction error
        iso_scores: torch.Tensor,    # (B,) per-window isolation forest score
        rolling_mean: torch.Tensor,  # (B,) rolling mean of recent errors
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_gated:    (B, seq_len, n_features) — gated input
            gate_values:(B,) gate weights [0,1]
        """
        # Stack anomaly signals as gate input
        gate_input = torch.stack([recon_errors, iso_scores, rolling_mean], dim=1)  # (B, 3)
        gate = self.gate_net(gate_input)        # (B, 1)
        gate_values = gate.squeeze(-1)          # (B,)

        # Broadcast gate over seq_len and features
        x_gated = x * gate.unsqueeze(1)        # (B, seq_len, n_features)
        return x_gated, gate_values


# ---------------------------------------------------------------------------
# Adaptive Feedback Controller
# ---------------------------------------------------------------------------

class AdaptiveFeedbackLoop:
    """
    Real-time adaptive controller that:
    
    1. Maintains a rolling buffer of anomaly signals.
    2. Computes a trust-adjusted input for the forecasting model.
    3. Provides an alternative "clean" prediction when anomaly confidence
       is high, by:
         a) Masking the anomalous window from the context, OR
         b) Imputing with a seasonal prototype from the rolling buffer.
    4. Optionally recalibrates the forecaster's output based on recent
       prediction errors (online learning step).
    
    Args:
        forecast_model:     ProbabilisticLSTMTransformer instance
        anomaly_gate:       AnomalyGate module
        buffer_size:        Rolling window for anomaly history
        iso_threshold:      Isolation Forest score threshold to trigger correction
        recon_threshold:    Reconstruction error threshold to trigger correction
        correction_mode:    'mask' | 'impute' — how to handle anomalous windows
    """

    def __init__(
        self,
        forecast_model,
        anomaly_gate: AnomalyGate,
        buffer_size: int = 168,       # 1 week of hourly data
        iso_threshold: float = 0.7,
        recon_threshold: Optional[float] = None,
        correction_mode: str = "impute",
        device: Optional[str] = None,
    ):
        self.forecast_model = forecast_model
        self.gate = anomaly_gate
        self.buffer_size = buffer_size
        self.iso_threshold = iso_threshold
        self.recon_threshold = recon_threshold
        self.correction_mode = correction_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Rolling buffers
        self._error_buffer: deque = deque(maxlen=buffer_size)
        self._score_buffer: deque = deque(maxlen=buffer_size)
        self._window_buffer: deque = deque(maxlen=buffer_size)  # stores clean windows

        # Online calibration
        self._calibration_bias: float = 0.0
        self._calibration_scale: float = 1.0
        self._recent_errors: deque = deque(maxlen=48)           # last 48 pred errors

    # ------------------------------------------------------------------
    # Core prediction with adaptive correction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x_seq: np.ndarray,            # (seq_len, n_features) — current window
        detect_result: Dict,           # output from SmartGridAnomalyDetector.detect()
        window_idx: int = 0,
    ) -> Dict:
        """
        Make an anomaly-aware forecast.
        
        Returns:
            {
              'q05': float,  'q50': float,  'q95': float,
              'gate_value':  float   — how much this window was trusted
              'corrected':   bool    — whether correction was applied
              'method':      str     — 'direct' | 'masked' | 'imputed'
            }
        """
        iso_score = float(detect_result["iso_score"][window_idx])
        recon_err = float(detect_result["recon_error"][window_idx])

        # Update rolling buffers
        self._error_buffer.append(recon_err)
        self._score_buffer.append(iso_score)

        rolling_mean_err = np.mean(self._error_buffer)
        is_anomalous = self._is_anomaly(iso_score, recon_err)

        # Decide correction strategy
        if is_anomalous and len(self._window_buffer) >= 24:
            if self.correction_mode == "mask":
                x_corrected = self._mask_correction(x_seq)
                method = "masked"
            else:
                x_corrected = self._impute_correction(x_seq)
                method = "imputed"
            corrected = True
        else:
            x_corrected = x_seq
            method = "direct"
            corrected = False
            self._window_buffer.append(x_seq.copy())

        # Run forecaster on (possibly corrected) input
        x_tensor = torch.tensor(x_corrected, dtype=torch.float32).unsqueeze(0).to(self.device)
        preds = self.forecast_model(x_tensor)         # (1, horizon, 3)
        preds = preds.cpu().numpy()[0, 0]             # (3,) — q05, q50, q95

        # Apply calibration bias
        preds_calibrated = preds * self._calibration_scale + self._calibration_bias

        # Compute gate value (informational)
        gate_val = self._compute_gate_value(iso_score, recon_err, rolling_mean_err)

        return {
            "q05": float(preds_calibrated[0]),
            "q50": float(preds_calibrated[1]),
            "q95": float(preds_calibrated[2]),
            "gate_value": gate_val,
            "corrected": corrected,
            "method": method,
            "iso_score": iso_score,
            "recon_error": recon_err,
        }

    def update_calibration(self, true_value: float, predicted_q50: float):
        """
        Online calibration: update bias/scale using exponential smoothing
        based on recent median prediction errors.
        
        Call this after observing the true ground-truth value.
        """
        error = true_value - predicted_q50
        self._recent_errors.append(error)

        if len(self._recent_errors) >= 12:
            recent = np.array(self._recent_errors)
            # Exponential smoothing of bias
            alpha = 0.1
            new_bias = float(np.median(recent))
            self._calibration_bias = (1 - alpha) * self._calibration_bias + alpha * new_bias

            # Scale calibration (ratio of actual to predicted std)
            if np.std(recent) > 1e-6:
                scale_adj = 1.0 + alpha * (np.sign(new_bias) * 0.05)
                self._calibration_scale = np.clip(scale_adj, 0.8, 1.2)

    # ------------------------------------------------------------------
    # Correction strategies
    # ------------------------------------------------------------------

    def _mask_correction(self, x_seq: np.ndarray) -> np.ndarray:
        """
        Masking strategy: zero out the most recent anomalous segment
        (last 25% of the window), keeping the earlier context.
        """
        masked = x_seq.copy()
        cutoff = int(len(x_seq) * 0.75)
        masked[cutoff:] = masked[cutoff - 1]   # repeat last "clean" step
        return masked

    def _impute_correction(self, x_seq: np.ndarray) -> np.ndarray:
        """
        Imputation strategy: replace the anomalous window with the most
        contextually similar historical clean window from the buffer.
        
        Similarity = cosine similarity of mean feature vectors.
        """
        clean_windows = list(self._window_buffer)
        if not clean_windows:
            return self._mask_correction(x_seq)

        query_vec = x_seq.mean(axis=0)         # (n_features,)
        similarities = []
        for w in clean_windows:
            ref_vec = w.mean(axis=0)
            cos_sim = np.dot(query_vec, ref_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(ref_vec) + 1e-9
            )
            similarities.append(cos_sim)

        best_idx = int(np.argmax(similarities))
        best_window = clean_windows[best_idx]

        # Blend: 30% current (to keep some original signal) + 70% clean reference
        blended = 0.3 * x_seq + 0.7 * best_window
        return blended

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_anomaly(self, iso_score: float, recon_err: float) -> bool:
        above_iso = iso_score >= self.iso_threshold
        above_recon = (
            recon_err >= self.recon_threshold
            if self.recon_threshold is not None
            else False
        )
        return above_iso or above_recon

    @staticmethod
    def _compute_gate_value(iso_score: float, recon_err: float, rolling_mean: float) -> float:
        """Soft gate value ∈ [0,1] — 1 = fully trusted, 0 = anomalous."""
        anomaly_signal = 0.5 * iso_score + 0.3 * min(recon_err / (rolling_mean + 1e-9), 1.0)
        return float(np.clip(1.0 - anomaly_signal, 0.0, 1.0))

    def get_stats(self) -> Dict:
        """Runtime statistics for monitoring."""
        return {
            "buffer_fill": len(self._window_buffer),
            "mean_recon_error": float(np.mean(self._error_buffer)) if self._error_buffer else 0.0,
            "mean_iso_score": float(np.mean(self._score_buffer)) if self._score_buffer else 0.0,
            "calibration_bias": self._calibration_bias,
            "calibration_scale": self._calibration_scale,
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models.probabilistic_transformer import ProbabilisticLSTMTransformer
    from models.anomaly_detector import SmartGridAnomalyDetector

    F, T = 16, 48
    model = ProbabilisticLSTMTransformer(n_features=F, d_model=32, n_heads=2)
    gate = AnomalyGate(n_features=F)

    detector = SmartGridAnomalyDetector(n_features=F, seq_len=T)
    X_dummy = np.random.randn(200, F).astype(np.float32)
    detector.fit(X_dummy, epochs=2)

    controller = AdaptiveFeedbackLoop(
        forecast_model=model,
        anomaly_gate=gate,
        iso_threshold=0.5,
    )

    # Simulate 5 prediction steps
    for step in range(5):
        x_seq = np.random.randn(T, F).astype(np.float32)
        results = detector.detect(x_seq.reshape(-1, F))
        # Create a dummy detect_result with a single entry
        single_result = {k: v[:1] for k, v in results.items() if isinstance(v, np.ndarray) and v.ndim > 0}
        pred = controller.predict(x_seq, single_result, window_idx=0)
        print(
            f"Step {step+1}: q50={pred['q50']:.4f}  gate={pred['gate_value']:.3f} "
            f"corrected={pred['corrected']} ({pred['method']})"
        )

    print("\n✅ Adaptive feedback loop OK")
    print("Stats:", controller.get_stats())
