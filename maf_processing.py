"""
Parse real MAF files into per-sample 96-channel mutation spectra.

parse_maf_file(path) reads a GDC MAF file (TSV with '#' comments). It requires
that sequence context is available. GDC MAFs include a CONTEXT column containing
the reference allele plus five flanking bases on each side; this parser extracts
the centered trinucleotide needed for 96-channel SBS spectra.

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


if __name__ == "__main__":
    print(f"Configured {len(CATEGORIES)} SBS categories.")
