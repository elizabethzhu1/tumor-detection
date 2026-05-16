"""
Bottleneck classifier for tumor type from 96-d mutation spectra.

Architecture, following the proposal:
    Input (96)
      -> Linear (96 -> K), Softplus applied to *weights* to keep them >= 0
      -> Softmax over the K-dim bottleneck output (so each sample's hidden
         representation is a probability distribution over K latent processes)
      -> Linear (K -> num_classes)
      -> Cross-entropy loss

The first weight matrix's rows are interpretable as "what 96-d pattern does
each latent unit respond to". Comparing those rows to COSMIC signatures
(via cosine similarity) is the central interpretability test.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckClassifier(nn.Module):
    def __init__(self, n_features=96, K=8, n_classes=4):
        super().__init__()
        self.K = K
        # Raw parameter; we apply softplus during forward to enforce >= 0.
        self.W1_raw = nn.Parameter(torch.randn(K, n_features) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(K))
        self.fc2 = nn.Linear(K, n_classes)

    def signature_weights(self):
        """The nonneg first-layer weights, shape (K, 96). For interpretation."""
        return F.softplus(self.W1_raw)

    def forward(self, x):
        # x: (batch, 96), already probability-normalized
        W1 = self.signature_weights()                 # (K, 96)
        h = F.linear(x, W1, self.b1)                  # (batch, K)
        h = F.softmax(h, dim=-1)                      # bottleneck as distribution
        logits = self.fc2(h)
        return logits, h


def train_model(X_train, y_train, X_val, y_val, K=8, n_classes=4,
                epochs=200, lr=1e-2, weight_decay=1e-4, verbose=True):
    """Train the bottleneck classifier with simple full-batch Adam."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BottleneckClassifier(n_features=X_train.shape[1], K=K,
                                 n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_acc = 0
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(Xt)
        loss = F.cross_entropy(logits, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits, _ = model(Xv)
            v_loss = F.cross_entropy(v_logits, yv).item()
            v_acc = (v_logits.argmax(1) == yv).float().mean().item()
            t_acc = (logits.argmax(1) == yt).float().mean().item()

        history.append({"epoch": epoch, "train_loss": loss.item(),
                        "val_loss": v_loss, "train_acc": t_acc, "val_acc": v_acc})

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}: loss {loss.item():.4f} "
                  f"val_loss {v_loss:.4f} train_acc {t_acc:.3f} val_acc {v_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X):
    device = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    logits, h = model(Xt)
    return logits.argmax(1).cpu().numpy(), h.cpu().numpy()


if __name__ == "__main__":
    # Smoke test.
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(96) * 0.5, size=200).astype(np.float32)
    y = rng.integers(0, 4, size=200)
    Xv = rng.dirichlet(np.ones(96) * 0.5, size=50).astype(np.float32)
    yv = rng.integers(0, 4, size=50)
    model, hist = train_model(X, y, Xv, yv, K=6, epochs=30, verbose=True)
    preds, h = predict(model, Xv)
    print(f"Predictions shape: {preds.shape}, bottleneck shape: {h.shape}")
    print(f"Signature weights shape: {model.signature_weights().shape}")