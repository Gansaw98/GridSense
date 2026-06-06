"""
Module 3: Anomaly Detection
- LSTM Autoencoder for reconstruction-error-based anomaly scoring
- Contrastive Learning: seasonal/contextual prototype matching
- Isolation Forest as a complementary ensemble detector
- Adaptive threshold (rolling percentile)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Dataset for Autoencoder (returns sequences only — reconstruction task)
# ---------------------------------------------------------------------------

class AEDataset(Dataset):
    def __init__(self, X: np.ndarray, seq_len: int = 48):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len]   # (seq_len, n_features)


# ---------------------------------------------------------------------------
# Autoencoder Model
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """
    BiLSTM Encoder → bottleneck → LSTM Decoder.
    Trained to reconstruct normal sequences; anomalies → high reconstruction error.
    """

    def __init__(self, n_features: int, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.LSTM(
            n_features, latent_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.enc_proj = nn.Linear(latent_dim * 2, latent_dim)  # BiLSTM → single latent

        # Decoder
        self.decoder = nn.LSTM(
            latent_dim, latent_dim, num_layers=2,
            batch_first=True, dropout=dropout
        )
        self.output_proj = nn.Linear(latent_dim, n_features)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: (B, seq_len, latent_dim)  — full sequence latent
            z_pooled: (B, latent_dim)    — mean-pooled for contrastive use
        """
        enc_out, _ = self.encoder(x)          # (B, T, 2*latent)
        z = self.enc_proj(enc_out)            # (B, T, latent)
        return z, z.mean(dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        dec_out, _ = self.decoder(z)          # (B, T, latent)
        return self.output_proj(dec_out)       # (B, T, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, z_pooled = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z_pooled

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample mean squared reconstruction error."""
        x_hat, _ = self(x)
        return F.mse_loss(x_hat, x, reduction='none').mean(dim=[1, 2])  # (B,)


# ---------------------------------------------------------------------------
# Contrastive Learning Loss (NT-Xent style)
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Seasonal Contrastive Loss.
    
    Positive pairs  : same (season, day-type) within a small time window
    Negative pairs  : different season or day-type
    
    Forces the latent space to cluster by context (e.g., 'winter weekday'),
    making anomalies (deviating from their cluster) stand out more clearly.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      (B, latent_dim) normalized embeddings
            labels: (B,) integer context label (0=spring-weekday, 1=winter-weekday, …)
        """
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.T) / self.tau           # (B, B)
        B = z.size(0)

        # Build mask: 1 if same label (positive pair), 0 otherwise
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = label_eq & self_mask

        # Remove self-similarity from denominator
        sim_exp = torch.exp(sim) * self_mask.float()
        log_prob = sim - torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-9)

        # Average over positives
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / (pos_mask.float().sum(dim=1) + 1e-9)
        return loss.mean()


def get_context_label(timestamps) -> np.ndarray:
    """
    Assign a coarse context label to each timestamp:
      season (0–3) × day-type (weekday/weekend) → 8 classes
    """
    import pandas as pd
    ts = pd.DatetimeIndex(timestamps)
    month = ts.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]),
         month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [0, 1, 2, 3]
    )
    is_weekend = (ts.dayofweek >= 5).astype(int)
    return season * 2 + is_weekend  # 8 classes


# ---------------------------------------------------------------------------
# Autoencoder Trainer (Reconstruction + Contrastive)
# ---------------------------------------------------------------------------

class AETrainer:

    def __init__(
        self,
        model: LSTMAutoencoder,
        lr: float = 1e-3,
        contrastive_weight: float = 0.3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.recon_loss = nn.MSELoss()
        self.contra_loss = ContrastiveLoss()
        self.alpha = contrastive_weight

    def train_epoch(self, loader: DataLoader, label_loader=None) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            x = batch.to(self.device)
            x_hat, z = self.model(x)
            loss = self.recon_loss(x_hat, x)
            total += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total / len(loader)

    def train_contrastive_epoch(
        self,
        loader: DataLoader,
        context_labels: np.ndarray,
        seq_len: int,
    ) -> float:
        """Train with both reconstruction and contrastive objectives."""
        self.model.train()
        ctx_tensor = torch.tensor(context_labels, dtype=torch.long)
        total = 0.0
        for i, batch in enumerate(loader):
            x = batch.to(self.device)
            x_hat, z = self.model(x)
            r_loss = self.recon_loss(x_hat, x)

            # Align context labels to batch
            idx_start = i * loader.batch_size
            idx_end = min(idx_start + x.size(0), len(ctx_tensor))
            labels = ctx_tensor[idx_start:idx_end].to(self.device)
            if labels.shape[0] == x.size(0):
                c_loss = self.contra_loss(z, labels)
                loss = r_loss + self.alpha * c_loss
            else:
                loss = r_loss

            total += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total / len(loader)


# ---------------------------------------------------------------------------
# Isolation Forest Wrapper
# ---------------------------------------------------------------------------

class IsoForestDetector:
    """
    Isolation Forest trained on AE latent representations.
    Provides an anomaly score independent from reconstruction error.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.isoforest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, latent_vectors: np.ndarray):
        self.isoforest.fit(latent_vectors)
        self._fitted = True

    def score(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].
        Higher = more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        raw = self.isoforest.score_samples(latent_vectors)    # negative, lower = anomaly
        return 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    def predict(self, latent_vectors: np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal."""
        return (self.isoforest.predict(latent_vectors) == -1).astype(int)


# ---------------------------------------------------------------------------
# Unified Anomaly Detector
# ---------------------------------------------------------------------------

class SmartGridAnomalyDetector:
    """
    Combines:
      - LSTMAutoencoder reconstruction error
      - Isolation Forest on latent embeddings
    into a single anomaly pipeline with adaptive thresholding.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 32,
        seq_len: int = 48,
        iso_contamination: float = 0.02,  # FIXED
        recon_percentile: float = 99.0,  # FIXED
    ):
        self.ae = LSTMAutoencoder(n_features, latent_dim)
        self.iso = IsoForestDetector(iso_contamination)
        self.seq_len = seq_len
        self.recon_percentile = recon_percentile
        self._recon_threshold: Optional[float] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ae.to(self.device)

    @torch.no_grad()
    def _extract_latents_and_errors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.ae.eval()
        dataset = AEDataset(X, self.seq_len)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        errors, latents = [], []
        for batch in loader:
            batch = batch.to(self.device)
            x_hat, z = self.ae(batch)
            err = F.mse_loss(x_hat, batch, reduction='none').mean(dim=[1, 2])
            errors.append(err.cpu().numpy())
            latents.append(z.cpu().numpy())
        return np.concatenate(latents), np.concatenate(errors)

    def fit(self, X_train: np.ndarray, epochs: int = 30, lr: float = 1e-3):
        trainer = AETrainer(self.ae, lr=lr, device=self.device)
        dataset = AEDataset(X_train, self.seq_len)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        print("Training Autoencoder...")
        for ep in range(1, epochs + 1):
            loss = trainer.train_epoch(loader)
            if ep % 10 == 0:
                print(f"  AE Epoch {ep:3d} | recon_loss={loss:.6f}")

        latents, errors = self._extract_latents_and_errors(X_train)
        self._recon_threshold = float(np.percentile(errors, self.recon_percentile))
        print(f"  Reconstruction threshold (p{self.recon_percentile}): {self._recon_threshold:.6f}")

        print("Training Isolation Forest on latent space...")
        self.iso.fit(latents)
        print("✅ Anomaly detector ready.")

    def detect(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns a dict:
          'recon_error'    : (N,) reconstruction MSE per window
          'iso_score'      : (N,) isolation forest anomaly score [0,1]
          'recon_flag'     : (N,) 1 if recon_error > threshold
          'iso_flag'       : (N,) 1 if iso forest says anomaly
          'combined_flag'  : (N,) 1 if either detector flags
          'latents'        : (N, latent_dim) — for SHAP / visualization
        """
        latents, errors = self._extract_latents_and_errors(X)
        iso_scores = self.iso.score(latents)
        iso_flags = self.iso.predict(latents)
        recon_flags = (errors > self._recon_threshold).astype(int)
        combined = ((recon_flags == 1) | (iso_flags == 1)).astype(int)
        return {
            "recon_error": errors,
            "iso_score": iso_scores,
            "recon_flag": recon_flags,
            "iso_flag": iso_flags,
            "combined_flag": combined,
            "latents": latents,
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N, F = 500, 16
    X_train = np.random.randn(N, F).astype(np.float32)
    detector = SmartGridAnomalyDetector(n_features=F, latent_dim=16, seq_len=24)
    detector.fit(X_train, epochs=5)
    results = detector.detect(X_train)
    n_anom = results["combined_flag"].sum()
    print(f"✅ Detected {n_anom} anomalies out of {len(results['combined_flag'])} windows")
