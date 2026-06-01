"""Run a hyperparameter sweep for the basic spectra neural network."""

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import classification_metrics
from data_processing_helpers import DEFAULT_PROCESSED_DATA, load_spectra_counts
from neural_network_baseline.train_nn import (
    BasicSpectraNN,
    build_features,
    compute_class_weights,
    get_device,
    l1_penalty,
)
from maf_features import (
    DEFAULT_DRIVER_GENES,
    load_driver_genes,
    load_or_build_driver_gene_features,
)


DEFAULT_OUT_DIR = "outputs/nn_sweep"


def parse_bottleneck(value):
    if value in {"none", "None", "null", "0"}:
        return None
    return int(value)


def evaluate_split(model, X, y, device, label_smoothing=0.0):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.long, device=device)
        logits = model(Xt)
        loss = F.cross_entropy(logits, yt, label_smoothing=label_smoothing).item()
        pred = logits.argmax(dim=1).cpu().numpy()
        max_k = min(3, logits.shape[1])
        topk = logits.topk(max_k, dim=1).indices.cpu().numpy()
    return {
        "loss": float(loss),
        "accuracy": float((pred == y).mean()),
        "top_2_accuracy": float(np.any(topk[:, :min(2, max_k)] == y[:, None], axis=1).mean()),
        "top_3_accuracy": float(np.any(topk[:, :min(3, max_k)] == y[:, None], axis=1).mean()),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pred": pred,
    }


def train_trial(config, splits, n_classes, device):
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    model = BasicSpectraNN(
        n_features=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        n_classes=n_classes,
        dropout=config.dropout,
        num_layers=config.num_layers,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    class_weights = None
    if config.class_weighted:
        class_weights = torch.tensor(
            compute_class_weights(y_train, n_classes),
            dtype=torch.float32,
            device=device,
        )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    sampler = None
    shuffle = True
    if config.balanced_sampler:
        class_counts = np.bincount(y_train, minlength=n_classes)
        class_counts = np.maximum(class_counts, 1)
        sample_weights = 1.0 / class_counts[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=None if sampler is not None else torch.Generator().manual_seed(config.seed),
    )

    best_state = None
    best_epoch = -1
    best_val_objective = -1.0
    best_val = None
    stale_epochs = 0
    history = []

    for epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(xb),
                yb,
                weight=class_weights,
                label_smoothing=config.label_smoothing,
            )
            if config.l1_lambda > 0:
                loss = loss + config.l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val = evaluate_split(model, X_val, y_val, device, config.label_smoothing)
        val_objective = 0.5 * (val["accuracy"] + val["weighted_f1"])
        history.append({
            "epoch": epoch,
            "train_batch_loss": float(np.mean(losses)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_weighted_f1": val["weighted_f1"],
            "val_macro_f1": val["macro_f1"],
            "val_objective": val_objective,
        })

        if val_objective > best_val_objective + config.min_delta:
            best_val_objective = val_objective
            best_epoch = epoch
            best_val = {k: v for k, v in val.items() if k != "pred"}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_split(model, X_test, y_test, device, config.label_smoothing)
    train = evaluate_split(model, X_train, y_train, device, config.label_smoothing)
    return {
        "model": model,
        "history": history,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_val_objective": best_val_objective,
        "train": {k: v for k, v in train.items() if k != "pred"},
        "test": {k: v for k, v in test.items() if k != "pred"},
        "test_pred": test["pred"],
    }


def build_grid(args):
    grid = list(itertools.product(
        args.hidden_dim,
        args.num_layers,
        [parse_bottleneck(x) for x in args.bottleneck_dim],
        args.dropout,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.label_smoothing,
        args.l1_lambda,
        args.include_mutation_burden,
        args.class_weighted,
        args.balanced_sampler,
    ))
    configs = []
    for values in grid:
        configs.append({
            "hidden_dim": values[0],
            "num_layers": values[1],
            "bottleneck_dim": values[2],
            "dropout": values[3],
            "lr": values[4],
            "weight_decay": values[5],
            "batch_size": values[6],
            "label_smoothing": values[7],
            "l1_lambda": values[8],
            "include_mutation_burden": values[9],
            "class_weighted": values[10],
            "balanced_sampler": values[11],
        })

    baseline = {
        "hidden_dim": 128,
        "num_layers": 2,
        "bottleneck_dim": 8,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "label_smoothing": 0.0,
        "l1_lambda": 0.0,
        "include_mutation_burden": True,
        "class_weighted": True,
        "balanced_sampler": False,
    }
    configs.insert(0, baseline)

    seen = set()
    deduped = []
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(config)

    rng = random.Random(args.seed)
    baseline_config = deduped[0]
    rest = deduped[1:]
    rng.shuffle(rest)
    deduped = [baseline_config] + rest
    if args.max_configs is not None:
        deduped = deduped[:args.max_configs]
    return deduped


def make_splits(
    spectra_df,
    include_mutation_burden,
    labels,
    test_size,
    val_size,
    seed,
    driver_features_df=None,
):
    work_df = spectra_df
    if labels is not None:
        work_df = work_df[work_df["tumor_type"].isin(set(labels))].copy()
        if work_df.empty:
            raise ValueError(f"No samples found for labels: {labels}")
    X, y, class_names, feature_names = build_features(
        work_df,
        include_mutation_burden=include_mutation_burden,
        driver_features_df=driver_features_df,
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=test_size + val_size,
        stratify=y,
        random_state=seed,
    )
    val_fraction = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        train_size=val_fraction,
        stratify=y_tmp,
        random_state=seed,
    )
    return (X_train, y_train, X_val, y_val, X_test, y_test), class_names, feature_names, work_df


def trial_config_dict(config):
    return {
        key: getattr(config, key)
        for key in [
            "hidden_dim",
            "num_layers",
            "bottleneck_dim",
            "dropout",
            "lr",
            "weight_decay",
            "batch_size",
            "label_smoothing",
            "l1_lambda",
            "include_mutation_burden",
            "class_weighted",
            "balanced_sampler",
            "epochs",
            "patience",
            "seed",
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-data", default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-configs", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--num-layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--bottleneck-dim", nargs="+", default=["none", "8", "16", "32", "64"])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3, 5e-4, 3e-4])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--label-smoothing", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--l1-lambda", type=float, nargs="+", default=[0.0, 1e-6])
    parser.add_argument("--include-mutation-burden", type=int, nargs="+", default=[1])
    parser.add_argument("--class-weighted", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--balanced-sampler", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--include-driver-genes", action="store_true")
    parser.add_argument("--driver-feature-cache", default="data/processed/driver_gene_flags.csv")
    parser.add_argument("--driver-gene-file", default=None)
    parser.add_argument("--data-dir", default="data/tcga_mafs")
    parser.add_argument("--force-driver-rebuild", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    spectra_df = load_spectra_counts(args.processed_data)
    driver_features_df = None
    if args.include_driver_genes:
        driver_genes = (
            load_driver_genes(args.driver_gene_file)
            if args.driver_gene_file else DEFAULT_DRIVER_GENES
        )
        driver_features_df = load_or_build_driver_gene_features(
            cache_path=args.driver_feature_cache,
            data_dir=args.data_dir,
            driver_genes=driver_genes,
            force=args.force_driver_rebuild,
        )
    configs = build_grid(args)

    print(f"Running {len(configs)} NN sweep trials on {device}.", flush=True)
    print(f"Results directory: {out_dir}", flush=True)

    split_cache = {}
    records = []
    best = None
    started = time.time()

    for trial_idx, config_dict in enumerate(configs, start=1):
        include_burden = bool(config_dict["include_mutation_burden"])
        if include_burden not in split_cache:
            split_cache[include_burden] = make_splits(
                spectra_df,
                include_burden,
                args.labels,
                args.test_size,
                args.val_size,
                args.seed,
                driver_features_df=driver_features_df,
            )
        splits, class_names, feature_names, work_df = split_cache[include_burden]
        config = SimpleNamespace(
            **config_dict,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed + trial_idx - 1,
        )

        trial_started = time.time()
        result = train_trial(config, splits, len(class_names), device)
        elapsed = time.time() - trial_started
        row = {
            "trial": trial_idx,
            **trial_config_dict(config),
            "best_epoch": result["best_epoch"],
            "best_val_objective": result["best_val_objective"],
            "val_accuracy": result["best_val"]["accuracy"],
            "val_weighted_f1": result["best_val"]["weighted_f1"],
            "val_macro_f1": result["best_val"]["macro_f1"],
            "val_loss": result["best_val"]["loss"],
            "test_accuracy": result["test"]["accuracy"],
            "test_top_2_accuracy": result["test"]["top_2_accuracy"],
            "test_top_3_accuracy": result["test"]["top_3_accuracy"],
            "test_weighted_f1": result["test"]["weighted_f1"],
            "test_macro_f1": result["test"]["macro_f1"],
            "test_loss": result["test"]["loss"],
            "train_accuracy": result["train"]["accuracy"],
            "train_weighted_f1": result["train"]["weighted_f1"],
            "elapsed_sec": elapsed,
        }
        records.append(row)
        pd.DataFrame(records).to_csv(out_dir / "sweep_results.csv", index=False)

        is_best = (
            best is None
            or row["best_val_objective"] > best["row"]["best_val_objective"]
        )
        if is_best:
            best = {
                "row": row,
                "state": result["best_state"],
                "class_names": class_names,
                "feature_names": feature_names,
                "config": trial_config_dict(config),
                "test_pred": result["test_pred"],
                "y_test": splits[-1],
            }
            torch.save(
                {
                    "model_state_dict": best["state"],
                    "class_names": class_names,
                    "feature_names": feature_names,
                    "config": {
                        **best["config"],
                        "processed_data": args.processed_data,
                        "labels": class_names,
                        "n_features": len(feature_names),
                        "test_size": args.test_size,
                        "val_size": args.val_size,
                        "selection_metric": "0.5 * (val_accuracy + val_weighted_f1)",
                        "include_driver_genes": args.include_driver_genes,
                        "driver_feature_cache": args.driver_feature_cache if args.include_driver_genes else None,
                        "driver_gene_file": args.driver_gene_file if args.include_driver_genes else None,
                    },
                    "sweep_row": row,
                },
                out_dir / "best_model.pt",
            )

        print(
            f"[{trial_idx:03d}/{len(configs):03d}] "
            f"val_obj={row['best_val_objective']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_wf1={row['val_weighted_f1']:.4f} "
            f"test_acc={row['test_accuracy']:.4f} test_top2={row['test_top_2_accuracy']:.4f} "
            f"test_wf1={row['test_weighted_f1']:.4f} "
            f"best={is_best} elapsed={elapsed:.1f}s "
            f"cfg={config_dict}",
            flush=True,
        )

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(
        ["best_val_objective", "val_weighted_f1", "val_accuracy"],
        ascending=False,
    )
    results_df.to_csv(out_dir / "sweep_results_ranked.csv", index=False)

    best_row = best["row"]
    metrics = classification_metrics(best["y_test"], best["test_pred"], best["class_names"])
    summary = {
        "selection_metric": "0.5 * (val_accuracy + val_weighted_f1)",
        "n_trials": len(configs),
        "elapsed_sec": time.time() - started,
        "best_trial": int(best_row["trial"]),
        "best_config": best["config"],
        "best_row": best_row,
        "best_test_metrics": metrics,
        "outputs": {
            "results": str(out_dir / "sweep_results.csv"),
            "ranked_results": str(out_dir / "sweep_results_ranked.csv"),
            "best_model": str(out_dir / "best_model.pt"),
        },
    }
    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 10 by validation objective:")
    print(results_df.head(10)[[
        "trial",
        "best_val_objective",
        "val_accuracy",
        "val_weighted_f1",
        "test_accuracy",
        "test_weighted_f1",
        "hidden_dim",
        "num_layers",
        "bottleneck_dim",
        "dropout",
        "lr",
        "weight_decay",
        "batch_size",
        "label_smoothing",
        "class_weighted",
        "balanced_sampler",
    ]].to_string(index=False))
    print(f"\nBest model saved to {out_dir / 'best_model.pt'}")
    print(f"Sweep summary saved to {out_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
