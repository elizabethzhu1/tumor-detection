"""Train a basic neural network on processed 96-channel spectra counts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import classification_metrics
from maf_features import (
    DEFAULT_DRIVER_GENES,
    driver_feature_names,
    load_driver_genes,
    load_or_build_driver_gene_features,
    load_or_build_maf_summary_features,
    load_or_build_top_mutated_gene_features,
)
from mutation_categories import CATEGORIES
from data_processing_helpers import DEFAULT_PROCESSED_DATA, encode_spectra, load_spectra_counts
from cosmic import fit_exposures, load_cosmic_signatures


def build_features(
    spectra_df,
    include_sbs96=True,
    include_sbs96_log_counts=False,
    include_mutation_burden=False,
    driver_features_df=None,
    maf_summary_df=None,
    top_gene_features_df=None,
    cosmic_exposures_df=None,
):
    """Build model features from processed spectra counts."""
    X, y, class_names = encode_spectra(spectra_df)
    feature_names = list(CATEGORIES)
    if not include_sbs96:
        X = np.empty((len(spectra_df), 0), dtype=np.float32)
        feature_names = []
    if include_sbs96_log_counts:
        counts = spectra_df[CATEGORIES].to_numpy(dtype=np.float32)
        log_counts = np.log1p(counts).astype(np.float32)
        X = np.concatenate([X, log_counts], axis=1)
        feature_names.extend([f"log1p_count_{category}" for category in CATEGORIES])
    if include_mutation_burden:
        counts = spectra_df[CATEGORIES].to_numpy(dtype=np.float32)
        mutation_burden = np.log1p(counts.sum(axis=1, keepdims=True)).astype(np.float32)
        X = np.concatenate([X, mutation_burden], axis=1)
        feature_names.append("log1p_total_snv_count")
    if driver_features_df is not None:
        driver_df = driver_features_df.set_index("sample_id")
        driver_columns = [column for column in driver_df.columns if column.endswith("_mutated")]
        missing_samples = set(spectra_df["sample_id"]) - set(driver_df.index)
        if missing_samples:
            raise ValueError(
                f"Driver features missing {len(missing_samples)} spectra samples. "
                "Rebuild the driver feature cache from the same MAF directory."
            )
        driver_matrix = (
            driver_df.reindex(spectra_df["sample_id"].to_numpy())[driver_columns]
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        X = np.concatenate([X, driver_matrix], axis=1)
        feature_names.extend(driver_columns)
    if maf_summary_df is not None:
        summary_df = maf_summary_df.set_index("sample_id")
        summary_columns = [column for column in summary_df.columns if column != "sample_id"]
        missing_samples = set(spectra_df["sample_id"]) - set(summary_df.index)
        if missing_samples:
            raise ValueError(
                f"MAF summary features missing {len(missing_samples)} spectra samples. "
                "Rebuild the MAF summary feature cache from the same MAF directory."
            )
        summary_matrix = (
            summary_df.reindex(spectra_df["sample_id"].to_numpy())[summary_columns]
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        X = np.concatenate([X, summary_matrix], axis=1)
        feature_names.extend(summary_columns)
    if top_gene_features_df is not None:
        top_gene_df = top_gene_features_df.set_index("sample_id")
        top_gene_columns = [column for column in top_gene_df.columns if column != "sample_id"]
        missing_samples = set(spectra_df["sample_id"]) - set(top_gene_df.index)
        if missing_samples:
            raise ValueError(
                f"Top mutated gene features missing {len(missing_samples)} spectra samples. "
                "Rebuild the top gene feature cache from the same MAF directory."
            )
        top_gene_matrix = (
            top_gene_df.reindex(spectra_df["sample_id"].to_numpy())[top_gene_columns]
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        X = np.concatenate([X, top_gene_matrix], axis=1)
        feature_names.extend(top_gene_columns)
    if cosmic_exposures_df is not None:
        cosmic_df = cosmic_exposures_df.set_index("sample_id")
        cosmic_columns = [column for column in cosmic_df.columns if column != "sample_id"]
        missing_samples = set(spectra_df["sample_id"]) - set(cosmic_df.index)
        if missing_samples:
            raise ValueError(
                f"COSMIC exposure features missing {len(missing_samples)} spectra samples. "
                "Rebuild the COSMIC exposure feature cache from the same processed data."
            )
        cosmic_matrix = (
            cosmic_df.reindex(spectra_df["sample_id"].to_numpy())[cosmic_columns]
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        X = np.concatenate([X, cosmic_matrix], axis=1)
        feature_names.extend(cosmic_columns)
    if X.shape[1] == 0:
        raise ValueError(
            "No model features selected. Enable SBS96, SBS96 log counts, mutation burden, or driver genes."
        )
    return X, y, class_names, feature_names


class BasicSpectraNN(nn.Module):
    """Configurable MLP classifier for tumor type from normalized SBS spectra."""

    def __init__(self, n_features, hidden_dim, n_classes, dropout, num_layers,
                 bottleneck_dim=None, normalization="batch"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if normalization not in {"batch", "layer", "none"}:
            raise ValueError("normalization must be one of: batch, layer, none.")

        layers = []
        in_dim = n_features
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim is not None:
            if bottleneck_dim < 1:
                raise ValueError("bottleneck_dim must be at least 1.")
            bottleneck_layers = [nn.Linear(hidden_dim, bottleneck_dim)]
            if normalization == "batch":
                bottleneck_layers.append(nn.BatchNorm1d(bottleneck_dim))
            elif normalization == "layer":
                bottleneck_layers.append(nn.LayerNorm(bottleneck_dim))
            bottleneck_layers.append(nn.ReLU())
            self.bottleneck = nn.Sequential(*bottleneck_layers)
            in_dim = bottleneck_dim
        else:
            self.bottleneck = None
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        h = self.feature_extractor(x)
        if self.bottleneck is not None:
            h = self.bottleneck(h)
        return self.classifier(h)

    def bottleneck_activations(self, x):
        """Return bottleneck activations for visualization."""
        if self.bottleneck is None:
            raise ValueError("Model was created without --bottleneck-dim.")
        h = self.feature_extractor(x)
        return self.bottleneck(h)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    balanced_sampling=False,
    seed=0,
):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    sampler = None
    shuffle = True
    if balanced_sampling:
        class_counts = np.bincount(y_train)
        class_counts = np.maximum(class_counts, 1)
        sample_weights = 1.0 / class_counts[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_class_weights(y_train, n_classes):
    """Inverse-frequency weights normalized by class count."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(y_train) / (n_classes * counts)
    return weights.astype(np.float32)


def monitor_score(row, monitor):
    if monitor == "val_loss":
        return -row["val_loss"]
    if monitor == "val_acc":
        return row["val_acc"]
    raise ValueError(f"Unsupported monitor: {monitor}")


def l1_penalty(model):
    """L1 parameter penalty for sparse/smaller weights."""
    return sum(param.abs().sum() for param in model.parameters())


def fit_standardizer(X_train):
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    scale = X_train.std(axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


def apply_standardizer(X, mean, scale):
    return ((X - mean) / scale).astype(np.float32)


@torch.no_grad()
def evaluate(model, X, y, device, label_smoothing=0.0):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    logits = model(Xt)
    loss = F.cross_entropy(logits, yt, label_smoothing=label_smoothing).item()
    preds = logits.argmax(dim=1).cpu().numpy()
    accuracy = float((preds == y).mean())
    return loss, accuracy, preds


@torch.no_grad()
def top_k_accuracy(model, X, y, device, k_values=(2, 3)):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(Xt)
    max_k = min(max(k_values), logits.shape[1])
    topk = logits.topk(max_k, dim=1).indices.cpu().numpy()
    out = {}
    for k in k_values:
        effective_k = min(k, logits.shape[1])
        out[f"top_{k}_accuracy"] = float(
            np.any(topk[:, :effective_k] == y[:, None], axis=1).mean()
        )
    return out


def load_or_build_cosmic_exposure_features(
    spectra_df,
    cosmic_path,
    cache_path,
    force=False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    counts = spectra_df[CATEGORIES].to_numpy(dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    spectra_probs = counts / row_sums
    cosmic_matrix, cosmic_names = load_cosmic_signatures(cosmic_path)
    exposures = fit_exposures(spectra_probs, cosmic_matrix)
    exposure_df = pd.DataFrame(
        exposures,
        columns=[f"cosmic_exposure_{name}" for name in cosmic_names],
    )
    exposure_df.insert(0, "sample_id", spectra_df["sample_id"].to_numpy())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_df.to_csv(cache_path, index=False)
    return exposure_df


def train_model(X_train, y_train, X_val, y_val, args, n_classes):
    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = BasicSpectraNN(
        n_features=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        n_classes=n_classes,
        dropout=args.dropout,
        num_layers=args.num_layers,
        bottleneck_dim=args.bottleneck_dim,
        normalization=args.normalization,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    class_weights = None
    if not args.no_class_weights:
        class_weights = torch.tensor(
            compute_class_weights(y_train, n_classes),
            dtype=torch.float32,
            device=device,
        )
    train_loader, _ = make_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        args.batch_size,
        balanced_sampling=args.balanced_sampler,
        seed=args.seed,
    )

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(
                model(xb),
                yb,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            if args.l1_lambda > 0:
                loss = loss + args.l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss, train_acc, _ = evaluate(
            model, X_train, y_train, device, label_smoothing=args.label_smoothing)
        val_loss, val_acc, _ = evaluate(
            model, X_val, y_val, device, label_smoothing=args.label_smoothing)
        row = {
            "epoch": epoch,
            "batch_train_loss": float(np.mean(epoch_losses)),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)

        score = monitor_score(row, args.monitor)
        if score > best_score + args.early_stopping_min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if args.print_every > 0 and (epoch % args.print_every == 0 or epoch == args.epochs - 1):
            print(
                f"epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}",
                flush=True,
            )

        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            if args.print_every > 0:
                print(
                    f"early stopping at epoch {epoch:03d}; "
                    f"best_epoch={best_epoch:03d} monitor={args.monitor}",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, str(device), best_epoch, best_score


def _polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return matrix / row_sums


def _color(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))
    r0, g0, b0 = 239, 246, 255
    r1, g1, b1 = 37, 99, 235
    r = round(r0 + t * (r1 - r0))
    g = round(g0 + t * (g1 - g0))
    b = round(b0 + t * (b1 - b0))
    return f"rgb({r},{g},{b})"


def save_heatmap_svg(matrix, row_labels, col_labels, out_path, title):
    """Save a simple dependency-free heatmap SVG."""
    if out_path is None:
        return
    matrix = np.asarray(matrix, dtype=np.float64)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = matrix.shape
    cell_w = 26 if n_cols > 32 else 48
    cell_h = 28
    left = 150
    top = 60
    width = left + n_cols * cell_w + 30
    height = top + n_rows * cell_h + 70
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))

    rects = []
    for i in range(n_rows):
        y = top + i * cell_h
        rects.append(
            f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" '
            f'font-size="12" font-family="Arial" fill="#111827">{row_labels[i]}</text>'
        )
        for j in range(n_cols):
            x = left + j * cell_w
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{_color(matrix[i, j], vmin, vmax)}" stroke="#ffffff" />'
            )

    col_text = []
    label_step = max(1, n_cols // 24)
    for j, label in enumerate(col_labels):
        if j % label_step != 0 and j != n_cols - 1:
            continue
        x = left + j * cell_w + cell_w / 2
        col_text.append(
            f'<text x="{x:.1f}" y="{top + n_rows * cell_h + 18}" '
            f'text-anchor="end" font-size="10" font-family="Arial" fill="#374151" '
            f'transform="rotate(-45 {x:.1f} {top + n_rows * cell_h + 18})">{label}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="18" font-family="Arial" fill="#111827">{title}</text>
  {"".join(rects)}
  {"".join(col_text)}
</svg>
'''
    out_path.write_text(svg)


def save_accuracy_plot(history, out_path):
    """Save train/validation accuracy curves as an SVG."""
    if out_path is None:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    if not epochs:
        return

    width = 820
    height = 500
    left = 70
    right = 30
    top = 35
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_epoch = max(max(epochs), 1)

    def xy(epoch, acc):
        x = left + (epoch / max_epoch) * plot_w
        y = top + (1.0 - acc) * plot_h
        return x, y

    train_points = [xy(epoch, acc) for epoch, acc in zip(epochs, train_acc)]
    val_points = [xy(epoch, acc) for epoch, acc in zip(epochs, val_acc)]
    grid_lines = []
    tick_labels = []
    for i in range(6):
        acc = i / 5
        y = top + (1.0 - acc) * plot_h
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#d8dee9" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#4b5563">{acc:.1f}</text>'
        )
    grid_svg = "\n  ".join(grid_lines)
    tick_svg = "\n  ".join(tick_labels)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="22" text-anchor="middle" font-size="18" font-family="Arial" fill="#111827">Train vs Validation Accuracy</text>
  {grid_svg}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
  {tick_svg}
  <text x="{left + plot_w / 2:.0f}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="Arial" fill="#111827">Epoch</text>
  <text x="18" y="{top + plot_h / 2:.0f}" text-anchor="middle" font-size="13" font-family="Arial" fill="#111827" transform="rotate(-90 18 {top + plot_h / 2:.0f})">Accuracy</text>
  <polyline points="{_polyline(train_points)}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <polyline points="{_polyline(val_points)}" fill="none" stroke="#dc2626" stroke-width="3"/>
  <rect x="{left + 12}" y="{top + 12}" width="160" height="52" fill="white" stroke="#d1d5db"/>
  <line x1="{left + 26}" y1="{top + 30}" x2="{left + 58}" y2="{top + 30}" stroke="#2563eb" stroke-width="3"/>
  <text x="{left + 68}" y="{top + 34}" font-size="13" font-family="Arial" fill="#111827">Train accuracy</text>
  <line x1="{left + 26}" y1="{top + 50}" x2="{left + 58}" y2="{top + 50}" stroke="#dc2626" stroke-width="3"/>
  <text x="{left + 68}" y="{top + 54}" font-size="13" font-family="Arial" fill="#111827">Validation accuracy</text>
</svg>
'''
    out_path.write_text(svg)


@torch.no_grad()
def extract_bottleneck_activations(model, X, device):
    """Return bottleneck activation matrix, shape n_samples x bottleneck_dim."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model.bottleneck_activations(Xt).cpu().numpy()


def bottleneck_input_attributions(model, X, device, max_samples=1024):
    """
    Approximate each bottleneck unit's input pattern with mean absolute gradient.

    This is an attribution map, not a direct COSMIC-style learned signature.
    """
    if model.bottleneck is None:
        return None
    model.eval()
    X_sample = X[:min(len(X), max_samples)]
    Xt = torch.tensor(X_sample, dtype=torch.float32, device=device, requires_grad=True)
    bottleneck = model.bottleneck_activations(Xt)
    rows = []
    for unit_idx in range(bottleneck.shape[1]):
        model.zero_grad(set_to_none=True)
        if Xt.grad is not None:
            Xt.grad.zero_()
        bottleneck[:, unit_idx].sum().backward(retain_graph=True)
        rows.append(Xt.grad.detach().abs().mean(dim=0).cpu().numpy().copy())
    return _normalize_rows(np.vstack(rows))


def bottleneck_input_attributions_by_class(model, X, y, device, max_samples_per_class=512):
    """
    Approximate each bottleneck unit's input pattern separately within each class.

    Returns an array with shape n_classes x bottleneck_dim x n_features.
    """
    if model.bottleneck is None:
        return None
    model.eval()
    class_rows = []
    for class_idx in range(int(np.max(y)) + 1):
        class_indices = np.flatnonzero(y == class_idx)[:max_samples_per_class]
        if len(class_indices) == 0:
            class_rows.append(np.zeros((model.bottleneck_dim, X.shape[1]), dtype=np.float64))
            continue
        Xt = torch.tensor(
            X[class_indices],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        bottleneck = model.bottleneck_activations(Xt)
        unit_rows = []
        for unit_idx in range(bottleneck.shape[1]):
            model.zero_grad(set_to_none=True)
            if Xt.grad is not None:
                Xt.grad.zero_()
            bottleneck[:, unit_idx].sum().backward(retain_graph=True)
            unit_rows.append(Xt.grad.detach().abs().mean(dim=0).cpu().numpy().copy())
        class_rows.append(_normalize_rows(np.vstack(unit_rows)))
    return np.stack(class_rows, axis=0)


def class_logit_input_attributions(model, X, y, n_classes, device, max_samples_per_class=512):
    """
    Approximate one input-attribution pattern per output class from its logit.

    Each class row averages absolute gradients of that class logit over samples
    from the same true class, then normalizes the row to sum to one.
    """
    model.eval()
    rows = []
    for class_idx in range(n_classes):
        class_indices = np.flatnonzero(y == class_idx)[:max_samples_per_class]
        if len(class_indices) == 0:
            rows.append(np.zeros(X.shape[1], dtype=np.float64))
            continue
        Xt = torch.tensor(
            X[class_indices],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        logits = model(Xt)
        model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()
        rows.append(Xt.grad.detach().abs().mean(dim=0).cpu().numpy().copy())
    return _normalize_rows(np.vstack(rows))


def class_weighted_bottleneck_attributions(bottleneck_attributions, classifier_weights):
    """
    Collapse bottleneck-unit input patterns into one nonnegative pattern per class.

    Only positive bottleneck-to-classifier weights are used, because those are
    the unit directions that increase the corresponding class logit.
    """
    positive_weights = np.maximum(classifier_weights, 0.0)
    class_patterns = positive_weights @ bottleneck_attributions
    return _normalize_rows(class_patterns)


def save_bottleneck_visualizations(
    model,
    X,
    y,
    spectra_df,
    feature_names,
    class_names,
    device,
    viz_prefix,
):
    """Save bottleneck activations, class means, classifier weights, and heatmaps."""
    if not viz_prefix or model.bottleneck is None:
        return {}

    prefix = Path(viz_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bottleneck_names = [f"bottleneck_{i}" for i in range(model.bottleneck_dim)]

    activations = extract_bottleneck_activations(model, X, device)
    activations_df = pd.DataFrame(activations, columns=bottleneck_names)
    activations_df.insert(0, "tumor_type", spectra_df["tumor_type"].to_numpy())
    activations_df.insert(0, "sample_id", spectra_df["sample_id"].to_numpy())
    activations_path = prefix.with_name(prefix.name + "_bottleneck_activations.csv")
    activations_df.to_csv(activations_path, index=False)

    means_df = activations_df.groupby("tumor_type", as_index=True)[bottleneck_names].mean()
    means_df = means_df.reindex(class_names)
    means_path = prefix.with_name(prefix.name + "_bottleneck_mean_by_tumor.csv")
    means_df.to_csv(means_path)
    means_plot_path = prefix.with_name(prefix.name + "_bottleneck_mean_by_tumor.svg")
    save_heatmap_svg(
        means_df.to_numpy(),
        row_labels=list(means_df.index),
        col_labels=bottleneck_names,
        out_path=means_plot_path,
        title="Mean Bottleneck Activation By Tumor Type",
    )

    classifier_weights = model.classifier.weight.detach().cpu().numpy()
    classifier_df = pd.DataFrame(
        classifier_weights,
        index=class_names,
        columns=bottleneck_names,
    )
    classifier_path = prefix.with_name(prefix.name + "_bottleneck_classifier_weights.csv")
    classifier_df.to_csv(classifier_path)
    classifier_plot_path = prefix.with_name(prefix.name + "_bottleneck_classifier_weights.svg")
    save_heatmap_svg(
        classifier_df.to_numpy(),
        row_labels=class_names,
        col_labels=bottleneck_names,
        out_path=classifier_plot_path,
        title="Classifier Weights From Bottleneck Units",
    )

    attributions = bottleneck_input_attributions(model, X, device)
    attribution_df = pd.DataFrame(
        attributions,
        index=bottleneck_names,
        columns=feature_names,
    )
    attribution_path = prefix.with_name(prefix.name + "_bottleneck_input_attributions.csv")
    attribution_df.to_csv(attribution_path)
    attribution_plot_path = prefix.with_name(prefix.name + "_bottleneck_input_attributions.svg")
    save_heatmap_svg(
        attribution_df.to_numpy(),
        row_labels=bottleneck_names,
        col_labels=feature_names,
        out_path=attribution_plot_path,
        title="Bottleneck Input Attribution Patterns",
    )

    outputs = {
        "bottleneck_activations": str(activations_path),
        "bottleneck_mean_by_tumor": str(means_path),
        "bottleneck_mean_by_tumor_plot": str(means_plot_path),
        "bottleneck_classifier_weights": str(classifier_path),
        "bottleneck_classifier_weights_plot": str(classifier_plot_path),
        "bottleneck_input_attributions": str(attribution_path),
        "bottleneck_input_attributions_plot": str(attribution_plot_path),
    }

    class_weighted = class_weighted_bottleneck_attributions(
        attributions,
        classifier_weights,
    )
    class_weighted_df = pd.DataFrame(
        class_weighted,
        index=class_names,
        columns=feature_names,
    )
    class_weighted_path = prefix.with_name(
        prefix.name + "_class_weighted_bottleneck_input_attributions.csv")
    class_weighted_df.to_csv(class_weighted_path)
    class_weighted_plot_path = prefix.with_name(
        prefix.name + "_class_weighted_bottleneck_input_attributions.svg")
    save_heatmap_svg(
        class_weighted_df.to_numpy(),
        row_labels=class_names,
        col_labels=feature_names,
        out_path=class_weighted_plot_path,
        title="Class-Weighted Bottleneck Input Attribution Patterns",
    )

    class_logit = class_logit_input_attributions(
        model,
        X,
        y,
        n_classes=len(class_names),
        device=device,
    )
    class_logit_df = pd.DataFrame(
        class_logit,
        index=class_names,
        columns=feature_names,
    )
    class_logit_path = prefix.with_name(prefix.name + "_class_logit_input_attributions.csv")
    class_logit_df.to_csv(class_logit_path)
    class_logit_plot_path = prefix.with_name(prefix.name + "_class_logit_input_attributions.svg")
    save_heatmap_svg(
        class_logit_df.to_numpy(),
        row_labels=class_names,
        col_labels=feature_names,
        out_path=class_logit_plot_path,
        title="Class Logit Input Attribution Patterns",
    )

    by_class = bottleneck_input_attributions_by_class(model, X, y, device)
    by_class_records = []
    by_class_rows = []
    by_class_labels = []
    for class_idx, class_name in enumerate(class_names):
        for unit_idx, bottleneck_name in enumerate(bottleneck_names):
            row = by_class[class_idx, unit_idx]
            by_class_records.append({
                "tumor_type": class_name,
                "bottleneck_unit": bottleneck_name,
                **{feature_name: row[col_idx] for col_idx, feature_name in enumerate(feature_names)},
            })
            by_class_rows.append(row)
            by_class_labels.append(f"{class_name}:{bottleneck_name}")
    by_class_df = pd.DataFrame.from_records(by_class_records)
    by_class_path = prefix.with_name(prefix.name + "_bottleneck_input_attributions_by_class.csv")
    by_class_df.to_csv(by_class_path, index=False)
    by_class_plot_path = prefix.with_name(prefix.name + "_bottleneck_input_attributions_by_class.svg")
    save_heatmap_svg(
        np.vstack(by_class_rows),
        row_labels=by_class_labels,
        col_labels=feature_names,
        out_path=by_class_plot_path,
        title="Bottleneck Input Attribution Patterns By Tumor Type",
    )

    outputs.update({
        "class_weighted_bottleneck_input_attributions": str(class_weighted_path),
        "class_weighted_bottleneck_input_attributions_plot": str(class_weighted_plot_path),
        "class_logit_input_attributions": str(class_logit_path),
        "class_logit_input_attributions_plot": str(class_logit_plot_path),
        "bottleneck_input_attributions_by_class": str(by_class_path),
        "bottleneck_input_attributions_by_class_plot": str(by_class_plot_path),
    })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-data", default=DEFAULT_PROCESSED_DATA,
                        help="Processed spectra CSV with sample_id, tumor_type, and 96 count columns.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional tumor labels to keep, e.g. BRCA SKCM LUAD.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Validation fraction of the full dataset.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of hidden Linear/BatchNorm/ReLU/Dropout blocks.")
    parser.add_argument("--bottleneck-dim", type=int, default=None,
                        help="Optional bottleneck layer width before classification.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalization", choices=["batch", "layer", "none"], default="batch",
                        help="Hidden-layer normalization type.")
    parser.add_argument("--feature-standardize", action="store_true",
                        help="Z-score model features using training-set mean and std.")
    parser.add_argument("--exclude-sbs96", action="store_true",
                        help="Remove the 96 normalized SBS spectrum features.")
    parser.add_argument("--include-sbs96-log-counts", action="store_true",
                        help="Append log1p raw count for each SBS96 channel.")
    parser.add_argument("--include-mutation-burden", action="store_true",
                        help="Append log1p(total SNV count) as an additional feature.")
    parser.add_argument("--include-driver-genes", action="store_true",
                        help="Append protein-altering driver gene mutation flags from MAF files.")
    parser.add_argument("--driver-feature-cache", default="data/processed/driver_gene_flags.csv",
                        help="CSV cache for per-sample driver gene mutation flags.")
    parser.add_argument("--driver-gene-file", default=None,
                        help="Optional TSV/CSV with driver genes in a Symbol column.")
    parser.add_argument("--data-dir", default="data/tcga_mafs",
                        help="Raw cached MAF directory used to build driver flags.")
    parser.add_argument("--force-driver-rebuild", action="store_true",
                        help="Reparse MAFs and overwrite --driver-feature-cache.")
    parser.add_argument("--include-maf-summary-features", action="store_true",
                        help="Append MAF-derived consequence/type/impact log-count features.")
    parser.add_argument("--maf-summary-feature-cache",
                        default="data/processed/maf_summary_features.csv",
                        help="CSV cache for per-sample MAF summary features.")
    parser.add_argument("--force-maf-summary-rebuild", action="store_true",
                        help="Reparse MAFs and overwrite --maf-summary-feature-cache.")
    parser.add_argument("--include-top-mutated-genes", action="store_true",
                        help="Append flags for the most frequently mutated genes in the MAF set.")
    parser.add_argument("--top-gene-feature-cache",
                        default="data/processed/top_mutated_gene_flags.csv",
                        help="CSV cache for top mutated gene flags.")
    parser.add_argument("--top-gene-count", type=int, default=1000,
                        help="Number of top mutated genes to keep when building top-gene features.")
    parser.add_argument("--top-gene-min-samples", type=int, default=5,
                        help="Minimum mutated samples required for a top-gene feature.")
    parser.add_argument("--force-top-gene-rebuild", action="store_true",
                        help="Reparse MAFs and overwrite --top-gene-feature-cache.")
    parser.add_argument("--include-cosmic-exposures", action="store_true",
                        help="Append NNLS-fitted COSMIC SBS exposure features.")
    parser.add_argument("--cosmic-path",
                        default="data/COSMIC_Human_SBS-96_GRCh38_v3.6.csv",
                        help="COSMIC SBS signature matrix for exposure features.")
    parser.add_argument("--cosmic-exposure-cache",
                        default="data/processed/cosmic_exposure_features.csv",
                        help="CSV cache for per-sample COSMIC exposure features.")
    parser.add_argument("--force-cosmic-exposure-rebuild", action="store_true",
                        help="Recompute and overwrite --cosmic-exposure-cache.")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable inverse-frequency class weighting in cross entropy.")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Sample training batches with inverse-frequency class weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many epochs without monitor improvement. 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum monitor improvement required to reset early stopping.")
    parser.add_argument("--monitor", choices=["val_acc", "val_loss"], default="val_acc",
                        help="Validation metric used for best checkpoint and early stopping.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--l1-lambda", type=float, default=0.0,
                        help="L1 regularization strength. 0 disables L1.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Cross-entropy label smoothing in [0, 1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--out", default="outputs/basic_nn_results.json")
    parser.add_argument("--model-out", default="outputs/basic_nn_model.pt")
    parser.add_argument("--plot-out", default="outputs/basic_nn_accuracy.svg")
    parser.add_argument("--viz-prefix", default="outputs/basic_nn",
                        help="Prefix for bottleneck visualization artifacts.")
    args = parser.parse_args()

    spectra_df = load_spectra_counts(args.processed_data)
    if args.labels is not None:
        spectra_df = spectra_df[spectra_df["tumor_type"].isin(set(args.labels))].copy()
        if spectra_df.empty:
            raise ValueError(f"No samples found for labels: {args.labels}")

    driver_features_df = None
    driver_genes = (
        load_driver_genes(args.driver_gene_file)
        if args.driver_gene_file else DEFAULT_DRIVER_GENES
    )
    if args.include_driver_genes:
        driver_features_df = load_or_build_driver_gene_features(
            cache_path=args.driver_feature_cache,
            data_dir=args.data_dir,
            driver_genes=driver_genes,
            force=args.force_driver_rebuild,
        )
    maf_summary_df = None
    if args.include_maf_summary_features:
        maf_summary_df = load_or_build_maf_summary_features(
            cache_path=args.maf_summary_feature_cache,
            data_dir=args.data_dir,
            force=args.force_maf_summary_rebuild,
        )
    top_gene_features_df = None
    if args.include_top_mutated_genes:
        top_gene_features_df = load_or_build_top_mutated_gene_features(
            cache_path=args.top_gene_feature_cache,
            data_dir=args.data_dir,
            max_genes=args.top_gene_count,
            min_samples=args.top_gene_min_samples,
            force=args.force_top_gene_rebuild,
        )
    cosmic_exposures_df = None
    if args.include_cosmic_exposures:
        cosmic_exposures_df = load_or_build_cosmic_exposure_features(
            spectra_df=spectra_df,
            cosmic_path=args.cosmic_path,
            cache_path=args.cosmic_exposure_cache,
            force=args.force_cosmic_exposure_rebuild,
        )

    X, y, class_names, feature_names = build_features(
        spectra_df,
        include_sbs96=not args.exclude_sbs96,
        include_sbs96_log_counts=args.include_sbs96_log_counts,
        include_mutation_burden=args.include_mutation_burden,
        driver_features_df=driver_features_df,
        maf_summary_df=maf_summary_df,
        top_gene_features_df=top_gene_features_df,
        cosmic_exposures_df=cosmic_exposures_df,
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=args.test_size + args.val_size,
        stratify=y,
        random_state=args.seed,
    )
    val_fraction_of_tmp = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        train_size=val_fraction_of_tmp,
        stratify=y_tmp,
        random_state=args.seed,
    )
    standardizer = None
    if args.feature_standardize:
        feature_mean, feature_scale = fit_standardizer(X_train)
        X_train = apply_standardizer(X_train, feature_mean, feature_scale)
        X_val = apply_standardizer(X_val, feature_mean, feature_scale)
        X_test = apply_standardizer(X_test, feature_mean, feature_scale)
        X = apply_standardizer(X, feature_mean, feature_scale)
        standardizer = {
            "mean": feature_mean.reshape(-1).tolist(),
            "scale": feature_scale.reshape(-1).tolist(),
        }

    model, history, device, best_epoch, best_score = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        args,
        n_classes=len(class_names),
    )
    test_loss, test_acc, test_preds = evaluate(
        model, X_test, y_test, get_device(), label_smoothing=args.label_smoothing)
    metrics = classification_metrics(y_test, test_preds, class_names=class_names)
    metrics.update(top_k_accuracy(model, X_test, y_test, get_device(), k_values=(2, 3)))
    visualization_outputs = save_bottleneck_visualizations(
        model=model,
        X=X,
        y=y,
        spectra_df=spectra_df,
        feature_names=feature_names,
        class_names=class_names,
        device=get_device(),
        viz_prefix=args.viz_prefix,
    )

    label_counts = spectra_df["tumor_type"].value_counts().sort_index().to_dict()
    config = {
        "processed_data": args.processed_data,
        "labels": class_names,
        "sample_counts_by_label": {k: int(v) for k, v in label_counts.items()},
        "n_samples": int(len(spectra_df)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "test_size": args.test_size,
        "val_size": args.val_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "bottleneck_dim": args.bottleneck_dim,
        "dropout": args.dropout,
        "normalization": args.normalization,
        "feature_standardize": args.feature_standardize,
        "include_sbs96": not args.exclude_sbs96,
        "include_sbs96_log_counts": args.include_sbs96_log_counts,
        "include_mutation_burden": args.include_mutation_burden,
        "include_driver_genes": args.include_driver_genes,
        "include_maf_summary_features": args.include_maf_summary_features,
        "include_top_mutated_genes": args.include_top_mutated_genes,
        "include_cosmic_exposures": args.include_cosmic_exposures,
        "driver_feature_cache": args.driver_feature_cache if args.include_driver_genes else None,
        "driver_gene_file": args.driver_gene_file if args.include_driver_genes else None,
        "driver_genes": driver_genes if args.include_driver_genes else None,
        "maf_summary_feature_cache": (
            args.maf_summary_feature_cache if args.include_maf_summary_features else None
        ),
        "top_gene_feature_cache": (
            args.top_gene_feature_cache if args.include_top_mutated_genes else None
        ),
        "top_gene_count": args.top_gene_count if args.include_top_mutated_genes else None,
        "top_gene_min_samples": args.top_gene_min_samples if args.include_top_mutated_genes else None,
        "cosmic_path": args.cosmic_path if args.include_cosmic_exposures else None,
        "cosmic_exposure_cache": (
            args.cosmic_exposure_cache if args.include_cosmic_exposures else None
        ),
        "class_weighted_cross_entropy": not args.no_class_weights,
        "balanced_sampler": args.balanced_sampler,
        "class_weights": (
            compute_class_weights(y_train, len(class_names)).tolist()
            if not args.no_class_weights else None
        ),
        "best_epoch": int(best_epoch),
        "best_monitor": args.monitor,
        "best_monitor_score": float(best_score),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "n_features": int(X.shape[1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "l1_lambda": args.l1_lambda,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "device": device,
        "viz_prefix": args.viz_prefix,
    }
    results = {
        "model": "basic_spectra_nn",
        "config": config,
        "feature_names": feature_names,
        "class_names": class_names,
        "metrics": {**metrics, "test_loss": test_loss, "test_accuracy": test_acc},
        "history": history,
        "visualization_outputs": visualization_outputs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    save_accuracy_plot(history, args.plot_out)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "feature_names": feature_names,
            "config": config,
            "standardizer": standardizer,
        },
        model_path,
    )

    print("\nBasic neural network baseline")
    print(f"  processed data: {args.processed_data}")
    print(f"  train/val/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"  classes: {class_names}")
    print(f"  test accuracy: {metrics['accuracy']:.3f}")
    print(f"  test top-2 accuracy: {metrics['top_2_accuracy']:.3f}")
    print(f"  test top-3 accuracy: {metrics['top_3_accuracy']:.3f}")
    print(f"  test weighted_f1: {metrics['weighted_f1']:.3f}")
    print(f"  best epoch: {best_epoch} ({args.monitor} score {best_score:.4f})")
    print(f"  results saved to {out_path}")
    print(f"  model saved to {model_path}")
    if args.plot_out:
        print(f"  accuracy plot saved to {args.plot_out}")
    if visualization_outputs:
        print("  bottleneck visualizations:")
        for name, path in visualization_outputs.items():
            print(f"    {name}: {path}")


if __name__ == "__main__":
    main()
