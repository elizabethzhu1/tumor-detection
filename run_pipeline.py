"""
End-to-end MVP pipeline.

Steps:
  1) Load data (synthetic or real MAFs).
  2) Split into train/val/test (70/15/15, stratified).
  3) Train baselines: logistic regression on raw spectra, logistic regression
     on COSMIC exposures, XGBoost on raw spectra.
  4) Train bottleneck NN.
  5) Evaluate all models on test set.
  6) Compute Hungarian-matched COSMIC alignment for NN bottleneck rows,
     compared against a random-vector null distribution.
  7) Save outputs (metrics JSON, learned signatures plot).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional XGBoost — skip if not installed.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from mutation_categories import CATEGORIES
from maf_processing import generate_synthetic_data, counts_to_probs, parse_maf_file
from tcga_gdc import DEFAULT_TCGA_PROJECTS, download_tcga_mafs, parse_project_specs
from cosmic import load_cosmic_signatures, fit_exposures
from model import train_model, predict
from evaluation import (classification_metrics, hungarian_match_to_cosmic,
                         null_alignment_score)


def load_data(synthetic=True, maf_paths=None, n_per_class=300, seed=0):
    """
    Returns:
      X_probs: (n, 96) probability spectra
      y: array of integer labels
      class_names: list, label index -> tumor type string
      sample_ids: list of sample ids
      spectra_df: DataFrame with sample_id, tumor_type, and 96 count columns
    """
    if synthetic:
        df = generate_synthetic_data(n_per_class=n_per_class, seed=seed)
    else:
        if not maf_paths:
            raise ValueError("Real-data mode requires --maf-paths or --download-tcga-mafs.")
        dfs = []
        for tumor, paths in maf_paths.items():
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            tumor_dfs = []
            for path in paths:
                print(f"  parsing {path} as {tumor}...")
                tumor_dfs.append(parse_maf_file(path, tumor))
            tumor_df = pd.concat(tumor_dfs, ignore_index=True)
            tumor_df = (
                tumor_df.groupby(["sample_id", "tumor_type"], as_index=False)[CATEGORIES]
                .sum()
            )
            dfs.append(tumor_df)
        df = pd.concat(dfs, ignore_index=True)

    counts = df[CATEGORIES].values.astype(np.float64)
    probs = counts_to_probs(counts).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["tumor_type"].values)
    return probs, y, list(le.classes_), df["sample_id"].tolist(), df


def stratified_three_split(X, y, train_frac=0.70, val_frac=0.15, seed=0):
    """70/15/15 stratified split. Returns (X_tr, X_va, X_te, y_tr, y_va, y_te)."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, train_size=train_frac, stratify=y, random_state=seed)
    val_size = val_frac / (1 - train_frac)  # fraction of remaining
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, train_size=val_size, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def run_baselines(X_tr, y_tr, X_va, y_va, X_te, y_te, signatures, class_names):
    """Train and evaluate baseline classifiers."""
    results = {}

    # Baseline A: logistic regression on raw 96-d spectra.
    print("Training LR on raw 96-d spectra...")
    lr_raw = LogisticRegression(max_iter=5000, C=1.0)
    lr_raw.fit(X_tr, y_tr)
    results["lr_raw"] = classification_metrics(
        y_te, lr_raw.predict(X_te), class_names=class_names)

    # Baseline B: logistic regression on COSMIC exposures.
    # This is the baseline your TA specifically asked for.
    print("Fitting COSMIC exposures, then LR on top...")
    expos_tr = fit_exposures(X_tr, signatures)
    expos_va = fit_exposures(X_va, signatures)
    expos_te = fit_exposures(X_te, signatures)
    lr_cosmic = LogisticRegression(max_iter=5000, C=1.0)
    lr_cosmic.fit(expos_tr, y_tr)
    results["lr_cosmic_exposures"] = classification_metrics(
        y_te, lr_cosmic.predict(expos_te), class_names=class_names)

    # Baseline C: XGBoost.
    if HAS_XGB:
        print("Training XGBoost on raw spectra...")
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            objective="multi:softprob", verbosity=0)
        xgb.fit(X_tr, y_tr)
        results["xgboost_raw"] = classification_metrics(
            y_te, xgb.predict(X_te), class_names=class_names)

    return results, (expos_tr, expos_va, expos_te)


def run_bottleneck_sweep(X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes,
                          K_values, signatures, sig_names, class_names,
                          seed=0, epochs=300):
    """Train the bottleneck model for several K and report metrics + alignment."""
    results = {}
    null_cache = {}

    for K in K_values:
        print(f"\nTraining bottleneck NN with K={K}...")
        # Reproducibility within the sweep.
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, _ = train_model(X_tr, y_tr, X_va, y_va,
                                K=K, n_classes=n_classes,
                                epochs=epochs, lr=1e-2, verbose=False)
        preds_te, _ = predict(model, X_te)
        cls_metrics = classification_metrics(y_te, preds_te, class_names=class_names)

        learned = model.signature_weights().detach().cpu().numpy()  # (K, 96)
        # Normalize rows to probabilities for fair cosine comparison.
        learned_normed = learned / (learned.sum(axis=1, keepdims=True) + 1e-12)
        align = hungarian_match_to_cosmic(learned_normed, signatures, sig_names)

        if K not in null_cache:
            null_cache[K] = null_alignment_score(
                K=K, cosmic_signatures=signatures, cosmic_names=sig_names,
                n_trials=100, seed=seed)

        results[K] = {
            "test_accuracy": cls_metrics["accuracy"],
            "test_macro_f1": cls_metrics["macro_f1"],
            "per_class_f1": cls_metrics["per_class_f1"],
            "confusion_matrix": cls_metrics["confusion_matrix"],
            "mean_matched_cosmic_similarity": align["mean_matched_similarity"],
            "matched_signatures": [m["cosmic_name"] for m in align["matches"]],
            "matched_similarities": [m["cosine_sim"] for m in align["matches"]],
            "null_mean": null_cache[K]["null_mean"],
            "null_p95": null_cache[K]["null_p95"],
            "alignment_above_null_p95": (
                align["mean_matched_similarity"] > null_cache[K]["null_p95"]),
            "learned_weights": learned_normed.tolist(),
        }
        print(f"  K={K}: test_acc={cls_metrics['accuracy']:.3f}, "
              f"cosmic_align={align['mean_matched_similarity']:.3f} "
              f"(null p95={null_cache[K]['null_p95']:.3f})")

    return results


def parse_maf_path_args(maf_path_args):
    """
    Parse --maf-paths entries.

    Each entry should be LABEL=path. Repeating a label appends another MAF for
    the same tumor type.
    """
    if not maf_path_args:
        return None
    out = {}
    for item in maf_path_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --maf-paths entry {item!r}. Use LABEL=/path/to/file.maf.gz."
            )
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(
                f"Invalid --maf-paths entry {item!r}. Use LABEL=/path/to/file.maf.gz."
            )
        out.setdefault(label, []).append(path)
    return out


def save_spectra_counts(spectra_df, out_path):
    """Save per-sample 96-dimensional count spectra."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "tumor_type"] + CATEGORIES
    spectra_df.loc[:, columns].to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data (default for MVP).")
    parser.add_argument("--no-synthetic", dest="synthetic", action="store_false",
                        help="Use real MAF files instead of synthetic data.")
    parser.add_argument("--download-tcga-mafs", action="store_true",
                        help="Download public TCGA MAFs from GDC before loading data.")
    parser.add_argument("--tcga-projects", nargs="*",
                        default=[f"{p}:{label}" for p, label in DEFAULT_TCGA_PROJECTS.items()],
                        help=("TCGA projects to download as PROJECT:LABEL. "
                              "Default: TCGA-BRCA:BRCA TCGA-ESCA:ESCA "
                              "TCGA-LAML:LAML TCGA-SKCM:SKCM."))
    parser.add_argument("--data-dir", default="data/tcga_mafs",
                        help="Directory for downloaded TCGA MAFs.")
    parser.add_argument("--force-download", action="store_true",
                        help="Redownload TCGA MAFs even when cached files exist.")
    parser.add_argument("--max-files-per-project", type=int, default=None,
                        help="Optional cap on downloaded MAF files per TCGA project.")
    parser.add_argument("--download-retries", type=int, default=3,
                        help="Retries for transient GDC file download failures.")
    parser.add_argument("--skip-failed-downloads", action="store_true",
                        help="Continue when one GDC MAF download fails after retries.")
    parser.add_argument("--maf-paths", nargs="*",
                        help="Real MAF inputs as LABEL=/path/to/file.maf.gz.")
    parser.add_argument("--cosmic-path", default=None,
                        help="Path to COSMIC SBS signatures TSV.")
    parser.add_argument("--spectra-out", default="outputs/spectra_counts.csv",
                        help="Where to save per-sample 96-d count spectra.")
    parser.add_argument("--out", default="outputs/results.json")
    parser.add_argument("--n-per-class", type=int, default=300,
                        help="Synthetic samples per tumor type.")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[4, 6, 8, 12, 16],
                        help="Bottleneck widths to train.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs for each bottleneck NN.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.download_tcga_mafs or args.maf_paths:
        args.synthetic = False

    print("=" * 60)
    print("CosmicNet MVP pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    maf_paths = None
    if not args.synthetic:
        maf_paths = parse_maf_path_args(args.maf_paths)
        if args.download_tcga_mafs:
            projects = parse_project_specs(args.tcga_projects)
            downloaded = download_tcga_mafs(
                projects=projects,
                output_dir=args.data_dir,
                force=args.force_download,
                max_files_per_project=args.max_files_per_project,
                retries=args.download_retries,
                skip_failed_downloads=args.skip_failed_downloads,
            )
            if maf_paths:
                for label, paths in downloaded.items():
                    maf_paths.setdefault(label, []).extend(paths)
            else:
                maf_paths = downloaded

    X, y, class_names, sample_ids, spectra_df = load_data(
        synthetic=args.synthetic,
        maf_paths=maf_paths,
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    save_spectra_counts(spectra_df, args.spectra_out)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(class_names)} classes: {class_names}")
    print(f"  96-d count spectra saved to {args.spectra_out}")

    print("\n[2/5] Splitting 70/15/15 stratified...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_three_split(X, y, seed=args.seed)
    print(f"  train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    print("\n[3/5] Loading COSMIC signatures...")
    signatures, sig_names = load_cosmic_signatures(args.cosmic_path)
    print(f"  {len(sig_names)} signatures loaded.")

    print("\n[4/5] Training baselines...")
    baseline_results, _ = run_baselines(
        X_tr, y_tr, X_va, y_va, X_te, y_te, signatures, class_names)
    for name, m in baseline_results.items():
        print(f"  {name}: accuracy={m['accuracy']:.3f}, macro_f1={m['macro_f1']:.3f}")

    print("\n[5/5] Training bottleneck NN across K values...")
    K_values = args.K_values
    nn_results = run_bottleneck_sweep(
        X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes=len(class_names),
        K_values=K_values, signatures=signatures, sig_names=sig_names,
        class_names=class_names, seed=args.seed, epochs=args.epochs)

    out = {
        "config": {"synthetic": args.synthetic, "seed": args.seed,
                   "class_names": class_names, "K_values": K_values,
                   "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
                   "n_test": int(len(X_te)),
                   "spectra_out": args.spectra_out,
                   "maf_paths": maf_paths},
        "baselines": baseline_results,
        "bottleneck_nn": nn_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Summary table.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 60)
    for name, m in baseline_results.items():
        print(f"{name:<35} {m['accuracy']:>10.3f} {m['macro_f1']:>10.3f}")
    for K, m in nn_results.items():
        label = f"bottleneck_NN K={K}"
        print(f"{label:<35} {m['test_accuracy']:>10.3f} {m['test_macro_f1']:>10.3f}")
    print("\nCOSMIC alignment (mean matched cosine similarity vs null p95):")
    for K, m in nn_results.items():
        flag = "***" if m["alignment_above_null_p95"] else "   "
        print(f"  K={K:>2}: {m['mean_matched_cosmic_similarity']:.3f} "
              f"vs {m['null_p95']:.3f} {flag}")


if __name__ == "__main__":
    main()
