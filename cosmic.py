"""
Load COSMIC SBS reference signatures and fit per-sample exposures.

The COSMIC catalogue is downloadable from:
  https://cancer.sanger.ac.uk/signatures/sbs/
Use the GRCh38 SBS v3.x file. It comes as a TSV with one row per mutation type
('A[C>A]A' style) and one column per signature ('SBS1', 'SBS2', ...).

For development without the file, we synthesize a dictionary of toy signatures
that matches the same shape and ordering.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from mutation_categories import CATEGORIES


def load_cosmic_signatures(path=None):
    """
    Load COSMIC SBS signatures as a (96, n_signatures) matrix.

    Returns (matrix, signature_names).
    If path is None, returns synthetic placeholder signatures so the rest of
    the pipeline can run.
    """
    if path is None:
        return _synthetic_cosmic()

    df = pd.read_csv(path, sep="\t")
    # COSMIC files use a "Type" or "Substitution Type"/"Trinucleotide" column.
    # We expect categories in 'A[C>A]A' format. Normalize if needed.
    if "Type" in df.columns:
        df = df.set_index("Type")
    elif "MutationType" in df.columns:
        df = df.set_index("MutationType")
    else:
        df = df.set_index(df.columns[0])

    # Reorder rows to our canonical ordering.
    df = df.reindex(CATEGORIES)
    if df.isna().any().any():
        raise ValueError("Some categories missing after reindexing; check format.")
    return df.values, list(df.columns)


def _synthetic_cosmic(n_signatures=20, seed=42):
    """Generate placeholder signatures for development."""
    rng = np.random.default_rng(seed)
    mat = rng.dirichlet(alpha=np.ones(96) * 0.3, size=n_signatures).T  # (96, n_sig)
    names = [f"SBS_synth_{i+1}" for i in range(n_signatures)]
    return mat, names


def fit_exposures(spectra, signatures):
    """
    Fit nonnegative exposures of signatures for each sample using NNLS.

    Args:
        spectra: (n_samples, 96) probability or count matrix
        signatures: (96, n_signatures) signature matrix
    Returns:
        exposures: (n_samples, n_signatures), nonnegative, rows sum to 1
    """
    n_samples = spectra.shape[0]
    n_sig = signatures.shape[1]
    exposures = np.zeros((n_samples, n_sig))
    for i in range(n_samples):
        x, _ = nnls(signatures, spectra[i])
        exposures[i] = x
    # Normalize rows so each sample's exposures sum to 1 (interpretable as a
    # mixture). Avoid divide-by-zero for degenerate samples.
    row_sums = exposures.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return exposures / row_sums


if __name__ == "__main__":
    sigs, names = load_cosmic_signatures()
    print(f"Loaded {len(names)} signatures, matrix shape {sigs.shape}")
    print(f"Each signature sums to ~1: {sigs.sum(axis=0)[:3]}")

    # Fit on dummy spectra.
    rng = np.random.default_rng(0)
    fake_spectra = rng.dirichlet(np.ones(96) * 0.5, size=10)
    expos = fit_exposures(fake_spectra, sigs)
    print(f"Exposures shape: {expos.shape}, row sums: {expos.sum(axis=1)[:3]}")