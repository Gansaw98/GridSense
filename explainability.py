"""
Module 4: Explainability (XAI)
- SHAP-based feature attribution for anomaly detection
- Highlights top-K features driving each anomaly flag
- LIME fallback for model-agnostic explanation
- Human-readable report generation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# SHAP Explainer Wrapper
# ---------------------------------------------------------------------------

class AnomalyExplainer:
    """
    Explains WHY a window was flagged as anomalous using SHAP.
    
    Uses KernelExplainer (model-agnostic) on a reconstruction-error scorer,
    so it works with any AE / detection model.
    
    Usage:
        explainer = AnomalyExplainer(detector, feature_names, background_X)
        report = explainer.explain(anomaly_window, top_k=3)
    """

    def __init__(
        self,
        detector,           # SmartGridAnomalyDetector with .detect()
        feature_names: List[str],
        background_X: np.ndarray,
        n_background: int = 50,
        top_k: int = 3,
    ):
        self.detector = detector
        self.feature_names = feature_names
        self.top_k = top_k
        self._shap_available = self._try_import_shap()

        # Sub-sample background for KernelSHAP
        idx = np.random.choice(len(background_X), min(n_background, len(background_X)), replace=False)
        self.background = background_X[idx]

        if self._shap_available:
            self._build_explainer()

    def _try_import_shap(self) -> bool:
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            warnings.warn(
                "shap not installed. Install with: pip install shap\n"
                "Falling back to gradient-based attribution."
            )
            return False

    def _build_explainer(self):
        import shap

        def scorer(X_flat: np.ndarray) -> np.ndarray:
            """
            SHAP calls this with (n_samples, seq_len * n_features).
            Must return exactly one scalar per input sample.
            """
            n_feat = len(self.feature_names)
            seq_len = X_flat.shape[1] // n_feat
            scores = []
            for row in X_flat:
                window = row.reshape(seq_len, n_feat)   # (seq_len, n_feat)
                results = self.detector.detect(window)  # detect on flat 2D
                scores.append(float(results["recon_error"].mean()))
            return np.array(scores)                     # (n_samples,) — one value per row

        bg_flat = self.background.reshape(len(self.background), -1)
        self._shap_explainer = shap.KernelExplainer(scorer, bg_flat)

    def _infer_shape(self, X_flat: np.ndarray) -> Tuple[int, int]:
        n_feat = len(self.feature_names)
        seq_len = X_flat.shape[1] // n_feat
        return seq_len, n_feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        window: np.ndarray,          # (seq_len, n_features) — ONE anomalous window
        recon_error: float,
        iso_score: float,
    ) -> Dict:
        """
        Generate an explanation for a flagged anomaly window.
        
        Returns:
            {
              'top_features': [(feature_name, mean_abs_shap), ...],
              'shap_values':  (n_features,) array,
              'summary':      str — human-readable explanation,
              'recon_error':  float,
              'iso_score':    float,
            }
        """
        if self._shap_available:
            attribution = self._shap_attribution(window)
        else:
            attribution = self._gradient_free_attribution(window)

        # Aggregate over sequence dimension → per-feature importance
        if attribution.ndim == 2:
            feat_importance = np.abs(attribution).mean(axis=0)  # (n_features,)
        else:
            feat_importance = np.abs(attribution)

        top_idx = np.argsort(feat_importance)[::-1][: self.top_k]
        top_features = [
            (self.feature_names[i], float(feat_importance[i]))
            for i in top_idx
        ]

        summary = self._build_summary(top_features, recon_error, iso_score)

        return {
            "top_features": top_features,
            "shap_values": feat_importance,
            "summary": summary,
            "recon_error": recon_error,
            "iso_score": iso_score,
        }

    def explain_batch(
        self,
        detect_results: Dict,
        X_windows: np.ndarray,
        timestamps=None,
        max_explanations: int = 10,
    ) -> List[Dict]:
        """
        Explain all flagged windows (up to max_explanations).
        
        Args:
            detect_results: output of SmartGridAnomalyDetector.detect()
            X_windows:      (N, seq_len, n_features)
            timestamps:     optional list of timestamp strings per window
        """
        anomaly_indices = np.where(detect_results["combined_flag"] == 1)[0]
        reports = []

        for rank, idx in enumerate(anomaly_indices[:max_explanations]):
            window = X_windows[idx] if X_windows.ndim == 3 else X_windows
            report = self.explain(
                window,
                recon_error=float(detect_results["recon_error"][idx]),
                iso_score=float(detect_results["iso_score"][idx]),
            )
            report["window_index"] = int(idx)
            report["timestamp"] = str(timestamps[idx]) if timestamps is not None else f"window_{idx}"
            reports.append(report)

        return reports

    # ------------------------------------------------------------------
    # Attribution methods
    # ------------------------------------------------------------------

    def _shap_attribution(self, window: np.ndarray) -> np.ndarray:
        """Run KernelSHAP on a flattened window."""
        import shap
        flat = window.reshape(1, -1)
        shap_vals = self._shap_explainer.shap_values(flat, nsamples=100, silent=True)
        # shap_vals: (1, seq_len * n_feat) — reshape back
        seq_len, n_feat = window.shape
        return np.array(shap_vals).reshape(seq_len, n_feat)

    def _gradient_free_attribution(self, window: np.ndarray) -> np.ndarray:
        """
        Model-agnostic fallback: permutation-based feature importance.
        Measures how much reconstruction error increases when each feature
        column is shuffled (replaced with background noise).
        """
        seq_len, n_feat = window.shape
        base_results = self.detector.detect(window.reshape(-1, n_feat))
        base_err = float(base_results["recon_error"].mean())

        importances = np.zeros(n_feat)
        for f in range(n_feat):
            perturbed = window.copy()
            perturbed[:, f] = np.random.choice(
                self.background.reshape(-1, n_feat)[:, f], size=seq_len
            )
            r = self.detector.detect(perturbed.reshape(-1, n_feat))
            importances[f] = float(r["recon_error"].mean()) - base_err

        return importances  # (n_feat,)

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(top_features: List[Tuple], recon_error: float, iso_score: float) -> str:
        severity = "HIGH" if recon_error > 0.5 or iso_score > 0.8 else "MEDIUM" if recon_error > 0.2 else "LOW"
        lines = [
            f"[ANOMALY — Severity: {severity}]",
            f"  Reconstruction Error : {recon_error:.4f}",
            f"  Isolation Score      : {iso_score:.4f}",
            f"  Top Contributing Features:",
        ]
        for rank, (feat, score) in enumerate(top_features, 1):
            lines.append(f"    {rank}. {feat:<35s}  SHAP={score:.4f}")
        return "\n".join(lines)

    def to_dataframe(self, reports: List[Dict]) -> pd.DataFrame:
        """Convert a list of explanation reports to a tidy DataFrame."""
        rows = []
        for r in reports:
            base = {
                "timestamp": r.get("timestamp"),
                "window_index": r.get("window_index"),
                "recon_error": r["recon_error"],
                "iso_score": r["iso_score"],
            }
            for rank, (feat, score) in enumerate(r["top_features"], 1):
                base[f"top{rank}_feature"] = feat
                base[f"top{rank}_shap"] = score
            rows.append(base)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models.anomaly_detector import SmartGridAnomalyDetector

    N, F = 300, 16
    feature_names = [f"feat_{i}" for i in range(F)]
    X = np.random.randn(N, F).astype(np.float32)

    detector = SmartGridAnomalyDetector(n_features=F, latent_dim=16, seq_len=24)
    detector.fit(X, epochs=3)
    results = detector.detect(X)

    explainer = AnomalyExplainer(
        detector, feature_names, X.reshape(-1, 24, F)[:20],
    )

    # Build fake 3D windows for explainer
    seq_len = 24
    X_windows = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len + 1)])
    reports = explainer.explain_batch(results, X_windows, max_explanations=3)

    for r in reports:
        print(r["summary"])
        print()

    df = explainer.to_dataframe(reports)
    print(df.head())
