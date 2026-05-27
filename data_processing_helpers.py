"""Shared loading and caching for processed MAF spectra."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from maf_processing import counts_to_probs, parse_maf_file
from mutation_categories import CATEGORIES


DEFAULT_PROCESSED_DATA = "data/processed/spectra_counts.csv"


def discover_maf_paths(data_dir, labels=None, max_files_per_label=None):
    """Return {label: [maf paths]} from data/tcga_mafs/<label>/*.maf.gz."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing MAF data directory: {data_dir}")

    if labels is None:
        label_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    else:
        label_dirs = [data_dir / label for label in labels]

    maf_paths = {}
    for label_dir in label_dirs:
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")
        paths = sorted(label_dir.glob("*.maf.gz"))
        if max_files_per_label is not None:
            paths = paths[:max_files_per_label]
        if paths:
            maf_paths[label_dir.name] = paths

    if len(maf_paths) < 2:
        raise ValueError("Need at least two tumor labels with MAF files.")
    return maf_paths


def load_maf_spectra(maf_paths, verbose=False):
    """Parse MAF files and return one per-sample count row per tumor label."""
    all_label_dfs = []
    for label, paths in maf_paths.items():
        print(f"\nParsing {len(paths)} MAF files for {label}...", flush=True)
        label_dfs = []
        for i, path in enumerate(paths, start=1):
            if verbose or i == 1 or i == len(paths) or i % 50 == 0:
                print(f"  [{i}/{len(paths)}] {path}", flush=True)
            parsed = parse_maf_file(path, label)
            if not parsed.empty:
                label_dfs.append(parsed)
        if not label_dfs:
            raise ValueError(f"No usable SNV spectra parsed for {label}.")

        label_df = pd.concat(label_dfs, ignore_index=True)
        label_df = (
            label_df.groupby(["sample_id", "tumor_type"], as_index=False)[CATEGORIES]
            .sum()
        )
        all_label_dfs.append(label_df)

    spectra_df = pd.concat(all_label_dfs, ignore_index=True)
    counts = spectra_df[CATEGORIES].to_numpy(dtype=np.float64)
    keep = counts.sum(axis=1) > 0
    return spectra_df.loc[keep].reset_index(drop=True)


def save_spectra_counts(spectra_df, out_path):
    """Write per-sample 96-channel mutation counts."""
    if out_path is None:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "tumor_type"] + CATEGORIES
    spectra_df.loc[:, columns].to_csv(out_path, index=False)


def load_spectra_counts(path):
    """Load a processed spectra table and validate the required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed spectra file does not exist: {path}")

    spectra_df = pd.read_csv(path)
    required = ["sample_id", "tumor_type"] + CATEGORIES
    missing = [col for col in required if col not in spectra_df.columns]
    if missing:
        raise ValueError(f"Processed spectra file is missing columns: {missing}")
    return spectra_df.loc[:, required].copy()


def load_or_build_spectra(
    processed_data=DEFAULT_PROCESSED_DATA,
    data_dir="data/tcga_mafs",
    labels=None,
    max_files_per_label=None,
    force_reprocess=False,
    verbose=False,
):
    """
    Load processed spectra if present; otherwise parse MAFs once and cache them.

    max_files_per_label only affects cache creation. If a processed file already
    exists, labels are filtered from that file and no raw MAFs are reparsed.
    """
    processed_path = Path(processed_data)
    if processed_path.exists() and not force_reprocess:
        spectra_df = load_spectra_counts(processed_path)
        source = "processed_cache"
    else:
        maf_paths = discover_maf_paths(
            data_dir,
            labels=labels,
            max_files_per_label=max_files_per_label,
        )
        spectra_df = load_maf_spectra(maf_paths, verbose=verbose)
        save_spectra_counts(spectra_df, processed_path)
        source = "raw_mafs"

    if labels is not None:
        label_set = set(labels)
        spectra_df = spectra_df[spectra_df["tumor_type"].isin(label_set)].copy()
        if spectra_df.empty:
            raise ValueError(f"No samples for requested labels: {sorted(label_set)}")

    return spectra_df.reset_index(drop=True), source


def encode_spectra(spectra_df):
    """Convert count spectra to normalized features and integer labels."""
    counts = spectra_df[CATEGORIES].to_numpy(dtype=np.float64)
    X = counts_to_probs(counts).astype(np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(spectra_df["tumor_type"].to_numpy())
    return X, y, list(label_encoder.classes_)


def stratified_split(X, y, test_size=0.15, seed=0):
    """Create a stratified split, keeping tiny capped runs valid."""
    n_classes = len(np.unique(y))
    n_samples = len(y)

    if isinstance(test_size, float):
        test_count = max(math.ceil(n_samples * test_size), n_classes)
        if n_samples - test_count < n_classes:
            raise ValueError(
                "Not enough samples for a stratified train/test split. "
                "Increase the processed dataset size or use fewer labels."
            )
        split_test_size = test_count
    else:
        split_test_size = test_size

    return train_test_split(
        X,
        y,
        test_size=split_test_size,
        stratify=y,
        random_state=seed,
    )


def load_baseline_dataset(args):
    """Load processed spectra and return train/test arrays for baselines."""
    spectra_df, data_source = load_or_build_spectra(
        processed_data=args.processed_data,
        data_dir=args.data_dir,
        labels=args.labels,
        max_files_per_label=args.max_files_per_label,
        force_reprocess=args.reprocess_data,
        verbose=args.verbose,
    )
    save_spectra_counts(spectra_df, args.spectra_out)

    X, y, class_names = encode_spectra(spectra_df)
    X_train, X_test, y_train, y_test = stratified_split(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
    )

    label_counts = spectra_df["tumor_type"].value_counts().sort_index().to_dict()
    config = {
        "data_source": data_source,
        "processed_data": args.processed_data,
        "data_dir": args.data_dir,
        "labels": class_names,
        "sample_counts_by_label": {k: int(v) for k, v in label_counts.items()},
        "n_samples": int(len(spectra_df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": args.test_size,
        "seed": args.seed,
        "spectra_out": args.spectra_out,
    }
    return X_train, X_test, y_train, y_test, class_names, config


def add_data_args(parser, default_spectra_out=None):
    """Add common processed-data CLI arguments to a parser."""
    parser.add_argument("--processed-data", default=DEFAULT_PROCESSED_DATA,
                        help="Processed spectra CSV shared by all models.")
    parser.add_argument("--reprocess-data", action="store_true",
                        help="Reparse raw MAFs and overwrite --processed-data.")
    parser.add_argument("--data-dir", default="data/tcga_mafs",
                        help="Raw cached MAF directory used when reprocessing.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional tumor labels, e.g. BRCA SKCM LUAD.")
    parser.add_argument("--max-files-per-label", type=int, default=None,
                        help="Optional cap used only when building processed data.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spectra-out", default=default_spectra_out,
                        help="Optional extra copy of the processed spectra.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every parsed MAF path when reprocessing.")
