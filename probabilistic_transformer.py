"""
Module 2: Probabilistic LSTM-Transformer Model
- BiLSTM encoder to capture local temporal dependencies
- Cross-Attention Transformer to attend over BiLSTM hidden states
- Quantile output head (q=0.05, 0.50, 0.95) for uncertainty estimation
- Pinball (Quantile) Loss training criterion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SmartGridDataset(Dataset):
    """Sliding-window dataset returning (X_seq, y_target) pairs."""

    def __init__(self, X: np.ndarray, target_idx: int, seq_len: int = 48, horizon: int = 1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]                          # (seq_len, n_features)
        y = self.X[idx + self.seq_len: idx + self.seq_len + self.horizon, self.target_idx]  # (horizon,)
        return x_seq, y


# ---------------------------------------------------------------------------
# Pinball Loss (Quantile Loss)
# ---------------------------------------------------------------------------

class PinballLoss(nn.Module):
    """
    Quantile / Pinball loss for probabilistic forecasting.
    
    L_q(y, ŷ) = q * (y - ŷ)   if y >= ŷ
              = (1-q) * (ŷ - y) if y < ŷ
    """

    def __init__(self, quantiles: List[float] = [0.05, 0.50, 0.95]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)  # (Q,)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:   (B, horizon, Q)  — one prediction per quantile
            targets: (B, horizon)
        """
        q = self.quantiles.to(preds.device)              # (Q,)
        targets = targets.unsqueeze(-1)                  # (B, horizon, 1)
        errors = targets - preds                         # (B, horizon, Q)
        loss = torch.max(q * errors, (q - 1) * errors)  # element-wise pinball
        return loss.mean()


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Cross-Attention Layer
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention: Transformer queries attend over BiLSTM key/values.
    
    This lets the Transformer explicitly fuse short-range LSTM context
    with its long-range self-attention, capturing multi-scale patterns.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # from Transformer: (B, T, d_model)
        key_value: torch.Tensor  # from BiLSTM:    (B, T, d_model)
    ) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out))


# ---------------------------------------------------------------------------
# Main Model: BiLSTM + Cross-Attention Transformer
# ---------------------------------------------------------------------------

class ProbabilisticLSTMTransformer(nn.Module):
    """
    Hybrid architecture:
      BiLSTM encoder → positional encoding →
      Transformer self-attention →
      Cross-attention over BiLSTM hidden states →
      Quantile output head
    
    Args:
        n_features:    Number of input features
        d_model:       Internal representation dimension
        n_heads:       Number of attention heads
        n_lstm_layers: Stacked BiLSTM layers
        n_tf_layers:   Transformer encoder layers
        dropout:       Dropout rate
        quantiles:     Forecast quantiles [q_lo, q_mid, q_hi]
        horizon:       Multi-step forecast horizon
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        n_tf_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = [0.05, 0.50, 0.95],
        horizon: int = 1,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.horizon = horizon
        Q = len(quantiles)

        # --- Input projection ---
        self.input_proj = nn.Linear(n_features, d_model)

        # --- BiLSTM encoder ---
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,       # bidirectional → d_model total
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # --- Positional Encoding for Transformer ---
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # --- Transformer Self-Attention Encoder ---
        tf_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=n_tf_layers)

        # --- Cross-Attention (Transformer → BiLSTM) ---
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)

        # --- Quantile output head ---
        # Pool over sequence, then project to (horizon × Q)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * Q),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, n_features)
        Returns:
            preds: (B, horizon, Q)  — quantile forecasts
        """
        B, T, _ = x.shape
        Q = len(self.quantiles)

        # 1. Project inputs
        z = self.input_proj(x)                  # (B, T, d)

        # 2. BiLSTM encoding
        lstm_out, _ = self.bilstm(z)            # (B, T, d)
        lstm_out = self.lstm_norm(lstm_out)

        # 3. Transformer (with positional encoding)
        tf_in = self.pos_enc(lstm_out)
        tf_out = self.transformer(tf_in)        # (B, T, d)

        # 4. Cross-attention: TF queries over LSTM key/values
        fused = self.cross_attn(tf_out, lstm_out)  # (B, T, d)

        # 5. Global average pooling → head
        pooled = fused.mean(dim=1)              # (B, d)
        out = self.head(pooled)                 # (B, horizon*Q)
        return out.view(B, self.horizon, Q)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ForecastTrainer:
    """Training loop for ProbabilisticLSTMTransformer."""

    def __init__(
        self,
        model: ProbabilisticLSTMTransformer,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = PinballLoss(model.quantiles)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            preds = self.model(X_batch)          # (B, horizon, Q)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())
        preds_arr    = np.concatenate(all_preds)    # (N, horizon, Q)
        targets_arr  = np.concatenate(all_targets)  # (N, horizon)
        median_preds = preds_arr[:, :, 1]           # q=0.50

        # MAE
        mae = float(np.abs(median_preds - targets_arr).mean())

        # RMSE
        rmse = float(np.sqrt(np.mean((median_preds - targets_arr) ** 2)))

        # MAPE - skip near-zero targets to avoid division explosion
        mask = np.abs(targets_arr) > 1e-6
        mape = float(
            np.mean(np.abs((median_preds[mask] - targets_arr[mask])
                           / targets_arr[mask])) * 100
        ) if mask.any() else float("nan")

        # R-squared
        ss_res = np.sum((targets_arr - median_preds) ** 2)
        ss_tot = np.sum((targets_arr - targets_arr.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-9))

        # 90% PI coverage
        coverage_90 = float(
            ((targets_arr > preds_arr[:, :, 0]) &
             (targets_arr < preds_arr[:, :, 2])).mean()
        )

        return {
            "mae":            mae,
            "rmse":           rmse,
            "mape":           mape,
            "r2":             r2,
            "90pct_coverage": coverage_90,
            "predictions":    preds_arr,
            "targets":        targets_arr,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        print(f"Training on {self.device} for {epochs} epochs")
        history = []
        best_val = float("inf")
        for ep in range(1, epochs + 1):
            tr_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            history.append({"epoch": ep, "train_loss": tr_loss, **val_metrics})
            if val_metrics["mae"] < best_val:
                best_val = val_metrics["mae"]
                torch.save(self.model.state_dict(), "best_model.pt")
            if ep % 10 == 0:
                print(
                    f"Ep {ep:3d} | train_loss={tr_loss:.4f} | "
                    f"val_mae={val_metrics['mae']:.4f} | "
                    f"90% coverage={val_metrics['90pct_coverage']:.2%}"
                )
        return history


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, F = 16, 48, 32
    model = ProbabilisticLSTMTransformer(n_features=F, d_model=64, n_heads=4)
    x = torch.randn(B, T, F)
    out = model(x)
    print(f"✅ Model output shape: {out.shape}")   # (16, 1, 3)

    loss_fn = PinballLoss([0.05, 0.50, 0.95])
    y = torch.randn(B, 1)
    loss = loss_fn(out, y)
    print(f"   Pinball loss: {loss.item():.4f}")
