"""
Load COSMIC SBS reference signatures and fit per-sample exposures.

Reference: COSMIC v3.4 GRCh38 SBS signatures
  https://cancer.sanger.ac.uk/signatures/downloads/
  File: COSMIC_v3.4_SBS_GRCh38.txt

The downloaded TSV has one row per mutation type ('A[C>A]A' style) and one
column per signature ('SBS1', 'SBS2', ...), matching our CATEGORIES ordering.

For development without the file, _synthetic_cosmic() returns placeholder
signatures so the rest of the pipeline can run.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from mutation_categories import CATEGORIES

COSMIC_URL = (
    "https://cancer.sanger.ac.uk/signatures/documents/2124/COSMIC_v3.4_SBS_GRCh38.txt"
)
DEFAULT_PATH = Path(__file__).parent / "data" / "cosmic" / "COSMIC_v3.4_SBS_GRCh38.txt"


def download_cosmic_signatures(dest: Path = DEFAULT_PATH) -> Path:
    """
    Download the COSMIC v3.4 GRCh38 SBS signatures file if not already present.

    Returns the path to the local file.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading COSMIC SBS signatures to {dest} ...")
    urllib.request.urlretrieve(COSMIC_URL, dest)
    print("Download complete.")
    return dest


def load_cosmic_signatures(path=None):
    """
    Load COSMIC SBS signatures as a (96, n_signatures) matrix.

    Args:
        path: path to the COSMIC SBS TSV file. If None, uses the default
              data/cosmic/COSMIC_v3.4_SBS_GRCh38.txt path, downloading it
              automatically if missing.

    Returns:
        matrix: np.ndarray of shape (96, n_signatures), columns are probability
                distributions over the 96 categories (each sums to ~1)
        signature_names: list of strings like ['SBS1', 'SBS2', ...]
    """
    if path is None:
        path = download_cosmic_signatures()

    df = pd.read_csv(path, sep="\t")

    # Identify and set the mutation-type index column.
    for candidate in ("Type", "MutationType", "SubType"):
        if candidate in df.columns:
            df = df.set_index(candidate)
            break
    else:
        df = df.set_index(df.columns[0])

    # Reorder rows to our canonical 96-category ordering.
    df = df.reindex(CATEGORIES)
    missing = df.index[df.isna().any(axis=1)].tolist()
    if missing:
        raise ValueError(
            f"{len(missing)} categories missing after reindexing COSMIC file. "
            f"First few: {missing[:5]}. Check that the file uses 'A[C>A]A' notation."
        )

    return df.values.astype(float), list(df.columns)


def _synthetic_cosmic(n_signatures=20, seed=42):
    """Generate placeholder signatures for development when the file is absent."""
    rng = np.random.default_rng(seed)
    mat = rng.dirichlet(alpha=np.ones(96) * 0.3, size=n_signatures).T  # (96, n_sig)
    names = [f"SBS_synth_{i+1}" for i in range(n_signatures)]
    return mat, names


def decompose_sample(mutation_counts, signatures):
    """
    Decompose a single sample's 96-dimensional mutation count (or frequency)
    vector into COSMIC signature exposures using non-negative least squares.

    Args:
        mutation_counts: array-like of length 96
        signatures: (96, n_signatures) np.ndarray

    Returns:
        exposures: np.ndarray of length n_signatures, non-negative, sums to 1.
                   Represents the fractional contribution of each signature.
    """
    counts = np.asarray(mutation_counts, dtype=float)
    x, _ = nnls(signatures, counts)
    total = x.sum()
    return x / total if total > 0 else x


def fit_exposures(spectra, signatures):
    """
    Fit nonnegative signature exposures for a batch of samples using NNLS.

    Args:
        spectra: (n_samples, 96) count or probability matrix
        signatures: (96, n_signatures) signature matrix

    Returns:
        exposures: (n_samples, n_signatures), non-negative, rows sum to 1
    """
    spectra = np.asarray(spectra, dtype=float)
    n_samples = spectra.shape[0]
    n_sig = signatures.shape[1]
    exposures = np.zeros((n_samples, n_sig))
    for i in range(n_samples):
        exposures[i] = decompose_sample(spectra[i], signatures)
    return exposures


def load_reference_dataframe(path=None):
    """
    Return the COSMIC signatures as a tidy DataFrame.

    Rows: 96 mutation categories (index = 'A[C>A]A' strings)
    Columns: signature names (SBS1, SBS2, ...)
    Values: probabilities (each column sums to ~1)
    """
    mat, names = load_cosmic_signatures(path)
    return pd.DataFrame(mat, index=CATEGORIES, columns=names)


if __name__ == "__main__":
    # ── Load real signatures ──────────────────────────────────────────────────
    ref_df = load_reference_dataframe()
    print(f"COSMIC reference matrix: {ref_df.shape[0]} categories × {ref_df.shape[1]} signatures")
    print(f"Column sums (should all be ≈1): {ref_df.sum().describe()}")
    print(f"First 5 categories:\n{ref_df.iloc[:5, :4]}\n")

    mat = ref_df.values  # (96, n_sig)

    # ── Test decompose_sample on a known mixture ──────────────────────────────
    # Construct a synthetic spectrum as 70% SBS1 + 30% SBS5, then recover it.
    true_weights = np.zeros(ref_df.shape[1])
    sbs1_idx = list(ref_df.columns).index("SBS1")
    sbs5_idx = list(ref_df.columns).index("SBS5")
    true_weights[sbs1_idx] = 0.7
    true_weights[sbs5_idx] = 0.3

    synthetic_spectrum = mat @ true_weights  # (96,)
    recovered = decompose_sample(synthetic_spectrum, mat)

    print("Decomposition test (70% SBS1 + 30% SBS5):")
    print(f"  Recovered SBS1: {recovered[sbs1_idx]:.4f}  (expected 0.70)")
    print(f"  Recovered SBS5: {recovered[sbs5_idx]:.4f}  (expected 0.30)")
    print(f"  Exposures sum to: {recovered.sum():.6f}")
    assert abs(recovered[sbs1_idx] - 0.7) < 1e-6, "SBS1 recovery failed"
    assert abs(recovered[sbs5_idx] - 0.3) < 1e-6, "SBS5 recovery failed"
    print("  Exact recovery confirmed.\n")

    # ── Test fit_exposures on a batch from real spectra_counts ────────────────
    spectra_path = Path(__file__).parent / "outputs" / "spectra_counts.csv"
    if spectra_path.exists():
        spectra_df = pd.read_csv(spectra_path)
        counts = spectra_df[CATEGORIES].values[:5]
        expos = fit_exposures(counts, mat)
        top_sigs = [ref_df.columns[i] for i in expos.argmax(axis=1)]
        print(f"Batch fit on 5 real TCGA samples:")
        print(f"  Exposures shape: {expos.shape}")
        print(f"  Row sums: {expos.sum(axis=1)}")
        print(f"  Top signature per sample: {top_sigs}")

    # ── Save reference matrix ─────────────────────────────────────────────────
    out_path = Path(__file__).parent / "outputs" / "cosmic_reference.csv"
    ref_df.to_csv(out_path)
    print(f"\nReference matrix saved to {out_path}")
