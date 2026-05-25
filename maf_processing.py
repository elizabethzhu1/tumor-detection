"""
Parse MAF files into per-sample 96-channel mutation spectra.

Two paths:

1) parse_maf_file(path): reads a real GDC MAF file (TSV with '#' comments).
   Requires that sequence context is available. GDC MAFs include a CONTEXT
   column containing the reference allele plus five flanking bases on each side;
   this parser extracts the centered trinucleotide needed for 96-channel SBS
   spectra.

2) generate_synthetic_data(): produces fake 96-d spectra drawn from
   tumor-type-specific COSMIC-like signatures, so we can build and test
   the entire ML pipeline before the real data is downloaded.

For TCGA, MAF files for each project (SKCM, LUAD, BRCA, UCEC, COAD) are available
from the GDC Data Portal:
    https://portal.gdc.cancer.gov/
You typically want the 'Aliquot Ensemble Somatic Mutation' MAFs, one row per
mutation per sample, with Tumor_Sample_Barcode identifying the sample.
"""

import gzip
import io

import numpy as np
import pandas as pd

from mutation_categories import BASES, CATEGORIES, classify_mutation


def _open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {str(c).lower(): c for c in df.columns}
    for col in candidates:
        found = lower_map.get(col.lower())
        if found is not None:
            return found
    return None


def _trinuc_from_context(context, ref):
    """
    Convert GDC CONTEXT to a 3-base context centered on the reference allele.

    GDC MAF CONTEXT is documented as the reference allele plus five flanking
    bases. Older or non-GDC MAFs may already contain a 3-base context.
    """
    if pd.isna(context) or pd.isna(ref):
        return None
    seq = str(context).strip().upper()
    ref = str(ref).strip().upper()
    if len(ref) != 1 or ref not in BASES:
        return None
    if len(seq) == 3:
        return seq
    if len(seq) >= 3:
        mid = len(seq) // 2
        if len(seq) % 2 == 1 and seq[mid] == ref:
            return seq[mid - 1:mid + 2]
    return None


def parse_maf_file(path, tumor_label):
    """
    Read a MAF and return a DataFrame: one row per sample, 96 columns of counts
    plus a 'tumor_type' label.

    Assumes the MAF has columns: Tumor_Sample_Barcode, Reference_Allele,
    Tumor_Seq_Allele2, Variant_Type, and either CONTEXT (preferred) or
    chromosome+position which would require a reference lookup.
    """
    with _open_maybe_gz(path) as fh:
        # Skip MAF comment lines starting with '#'.
        lines = [ln for ln in fh if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)

    sample_col = _find_column(df, ["Tumor_Sample_Barcode"])
    ref_col = _find_column(df, ["Reference_Allele"])
    alt_col = _find_column(df, ["Tumor_Seq_Allele2", "Allele"])
    variant_type_col = _find_column(df, ["Variant_Type"])
    context_col = _find_column(df, ["CONTEXT"])
    required = {
        "Tumor_Sample_Barcode": sample_col,
        "Reference_Allele": ref_col,
        "Tumor_Seq_Allele2": alt_col,
        "Variant_Type": variant_type_col,
        "CONTEXT": context_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"MAF {path} lacks required columns: {missing}. GDC masked somatic "
            "mutation MAFs should include these columns."
        )

    # Keep only single-nucleotide variants and one-base alleles.
    df = df[df[variant_type_col].astype(str).str.upper().isin(["SNP", "SNV"])].copy()
    df[ref_col] = df[ref_col].astype(str).str.upper()
    df[alt_col] = df[alt_col].astype(str).str.upper()
    df = df[df[ref_col].isin(BASES) & df[alt_col].isin(BASES)]

    # Classify each row.
    cat_idx = [
        classify_mutation(
            r[ref_col],
            r[alt_col],
            _trinuc_from_context(r[context_col], r[ref_col]),
        )
        for _, r in df.iterrows()
    ]
    df["cat_idx"] = cat_idx
    df = df.dropna(subset=["cat_idx"])
    df["cat_idx"] = df["cat_idx"].astype(int)

    # Aggregate to per-sample 96-d count vectors.
    samples = sorted(df[sample_col].unique())
    counts = np.zeros((len(samples), 96), dtype=np.int64)
    sample_to_row = {s: i for i, s in enumerate(samples)}
    for _, r in df.iterrows():
        counts[sample_to_row[r[sample_col]], r["cat_idx"]] += 1

    out = pd.DataFrame(counts, columns=CATEGORIES)
    out.insert(0, "sample_id", samples)
    out["tumor_type"] = tumor_label
    return out


def counts_to_probs(counts_matrix, eps=0.0):
    """Row-normalize a (n_samples, 96) count matrix to probabilities."""
    row_sums = counts_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return (counts_matrix + eps) / (row_sums + eps * 96)


# ---------- Synthetic-data fallback ----------

def _make_signature(peaks, n_categories=96, noise=0.02, seed=None):
    """Make a fake 96-d signature with mass at given peak indices."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0, noise, size=n_categories)
    for idx, weight in peaks.items():
        sig[idx] += weight
    return sig / sig.sum()


def generate_synthetic_data(n_per_class=200, mutations_per_sample=(50, 5000), seed=0):
    """
    Make fake mutation count data for 5 tumor types, each driven by a different
    mixture of toy 'signatures' loosely inspired by COSMIC SBS dominant peaks.

    Returns a DataFrame in the same shape as parse_maf_file.
    """
    rng = np.random.default_rng(seed)

    # Toy "signatures": each tumor type gets a dominant signature plus shared
    # background. The peak indices are arbitrary but distinct per signature.
    sig_uv = _make_signature({CATEGORIES.index("C[C>T]C"): 0.4,
                              CATEGORIES.index("T[C>T]C"): 0.3,
                              CATEGORIES.index("C[C>T]T"): 0.2}, seed=1)  # SKCM-like
    sig_smoking = _make_signature({CATEGORIES.index("C[C>A]A"): 0.25,
                                   CATEGORIES.index("C[C>A]C"): 0.20,
                                   CATEGORIES.index("T[C>A]A"): 0.20,
                                   CATEGORIES.index("T[C>A]T"): 0.15}, seed=5)  # LUAD-like
    sig_apobec = _make_signature({CATEGORIES.index("T[C>T]A"): 0.35,
                                  CATEGORIES.index("T[C>G]A"): 0.3,
                                  CATEGORIES.index("T[C>T]T"): 0.15}, seed=2)  # BRCA
    sig_hrd = _make_signature({CATEGORIES.index("T[T>A]A"): 0.20,
                               CATEGORIES.index("T[T>G]T"): 0.18,
                               CATEGORIES.index("C[T>A]T"): 0.16}, seed=6)  # BRCA HRD-like
    sig_age = _make_signature({CATEGORIES.index("A[C>T]G"): 0.15,
                               CATEGORIES.index("C[C>T]G"): 0.18,
                               CATEGORIES.index("G[C>T]G"): 0.15,
                               CATEGORIES.index("T[C>T]G"): 0.15}, seed=3)  # SBS1/5-like aging
    sig_pole = _make_signature({CATEGORIES.index("T[C>A]T"): 0.30,
                                CATEGORIES.index("T[C>G]T"): 0.25,
                                CATEGORIES.index("A[C>A]A"): 0.15}, seed=4)  # UCEC/COAD POLE-like
    sig_mmr = _make_signature({CATEGORIES.index("A[C>T]A"): 0.20,
                               CATEGORIES.index("C[C>T]A"): 0.18,
                               CATEGORIES.index("G[C>T]A"): 0.16}, seed=7)  # UCEC/COAD MMR-like

    tumor_specs = {
        "SKCM": {"primary": sig_uv,      "secondary": sig_age, "p_mix": 0.85},
        "LUAD": {"primary": sig_smoking, "secondary": sig_age, "p_mix": 0.75},
        "BRCA": {"primary": sig_apobec,  "secondary": 0.5 * sig_hrd + 0.5 * sig_age, "p_mix": 0.60},
        "UCEC": {"primary": sig_pole,    "secondary": sig_mmr, "p_mix": 0.55},
        "COAD": {"primary": sig_mmr,     "secondary": sig_pole, "p_mix": 0.65},
    }

    rows = []
    for tumor, spec in tumor_specs.items():
        for i in range(n_per_class):
            n_mut = int(rng.integers(*mutations_per_sample))
            # Per-sample mixing weight has some variability.
            w = np.clip(rng.normal(spec["p_mix"], 0.08), 0.3, 0.99)
            true_dist = w * spec["primary"] + (1 - w) * spec["secondary"]
            true_dist = true_dist / true_dist.sum()
            counts = rng.multinomial(n_mut, true_dist)
            rows.append({
                "sample_id": f"{tumor}_{i:04d}",
                "tumor_type": tumor,
                **{c: counts[j] for j, c in enumerate(CATEGORIES)},
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_data(n_per_class=50)
    print(f"Generated {len(df)} samples.")
    print(df.groupby("tumor_type").size())
    counts = df[CATEGORIES].values
    print(f"Counts matrix shape: {counts.shape}")
    print(f"Mean mutations per sample: {counts.sum(axis=1).mean():.1f}")
    probs = counts_to_probs(counts)
    print(f"Row sums after normalization (should be 1): {probs.sum(axis=1)[:3]}")
