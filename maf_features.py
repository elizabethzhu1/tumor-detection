"""Feature extraction from cached TCGA MAF files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DRIVER_GENES = [
    "TP53",
    "KRAS",
    "APC",
    "BRAF",
    "EGFR",
    "PTEN",
    "PIK3CA",
    "IDH1",
    "IDH2",
    "FLT3",
    "NPM1",
    "DNMT3A",
    "RUNX1",
    "NRAS",
    "NF1",
    "CDKN2A",
    "STK11",
    "KEAP1",
    "ARID1A",
    "CTNNB1",
    "FBXW7",
    "GATA3",
    "CDH1",
    "MAP3K1",
    "POLE",
    "SMAD4",
    "NOTCH1",
    "ERBB2",
]

PROTEIN_ALTERING_CLASSIFICATIONS = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Splice_Region",
    "Translation_Start_Site",
}

SUMMARY_VARIANT_CLASSIFICATIONS = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Splice_Region",
    "Translation_Start_Site",
    "Silent",
    "Intron",
    "IGR",
    "RNA",
    "3'UTR",
    "5'UTR",
    "3'Flank",
    "5'Flank",
]

SUMMARY_VARIANT_TYPES = ["SNP", "DNP", "TNP", "ONP", "INS", "DEL"]
SUMMARY_IMPACTS = ["HIGH", "MODERATE", "LOW", "MODIFIER"]


def driver_feature_names(driver_genes=DEFAULT_DRIVER_GENES):
    return [f"{gene}_mutated" for gene in driver_genes]


def load_driver_genes(path):
    """Load driver gene symbols from a TSV/CSV with a Symbol column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Driver gene file does not exist: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    if "Symbol" in df.columns:
        symbols = df["Symbol"]
    elif "Hugo_Symbol" in df.columns:
        symbols = df["Hugo_Symbol"]
    else:
        symbols = df.iloc[:, 0]
    genes = []
    seen = set()
    for symbol in symbols.dropna().astype(str):
        gene = symbol.strip().strip('"').upper()
        if gene and gene not in seen:
            seen.add(gene)
            genes.append(gene)
    if not genes:
        raise ValueError(f"No driver genes found in {path}")
    return genes


def discover_maf_files(data_dir):
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*/*.maf.gz")) + sorted(data_dir.glob("*/*.maf"))
    if not paths:
        raise FileNotFoundError(f"No MAF files found under {data_dir}")
    return paths


def _read_driver_columns(path):
    try:
        return pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
            usecols=["Tumor_Sample_Barcode", "Hugo_Symbol", "Variant_Classification"],
        )
    except ValueError:
        return pd.DataFrame(columns=[
            "Tumor_Sample_Barcode",
            "Hugo_Symbol",
            "Variant_Classification",
        ])


def _read_summary_columns(path):
    usecols = [
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Variant_Classification",
        "Variant_Type",
        "IMPACT",
    ]
    try:
        return pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
            usecols=lambda column: column in set(usecols),
        )
    except ValueError:
        return pd.DataFrame(columns=usecols)


def build_driver_gene_features(
    data_dir="data/tcga_mafs",
    driver_genes=DEFAULT_DRIVER_GENES,
    protein_altering_only=True,
    verbose=False,
):
    """
    Return one row per sample with binary mutation flags for selected genes.
    """
    paths = discover_maf_files(data_dir)
    driver_set = set(driver_genes)
    records = []
    all_samples = set()

    for idx, path in enumerate(paths, start=1):
        if verbose and (idx == 1 or idx == len(paths) or idx % 100 == 0):
            print(f"[{idx}/{len(paths)}] {path}", flush=True)
        df = _read_driver_columns(path)
        if df.empty:
            continue
        df = df.dropna(subset=["Tumor_Sample_Barcode"])
        all_samples.update(df["Tumor_Sample_Barcode"].astype(str).unique())
        df = df[df["Hugo_Symbol"].astype(str).isin(driver_set)].copy()
        if protein_altering_only:
            df = df[df["Variant_Classification"].astype(str).isin(
                PROTEIN_ALTERING_CLASSIFICATIONS
            )]
        if df.empty:
            continue
        df["sample_id"] = df["Tumor_Sample_Barcode"].astype(str)
        df["gene"] = df["Hugo_Symbol"].astype(str)
        records.append(df[["sample_id", "gene"]])

    columns = ["sample_id"] + driver_feature_names(driver_genes)
    feature_columns = columns[1:]
    if all_samples:
        sample_ids = sorted(all_samples)
        out = pd.DataFrame(0, index=sample_ids, columns=feature_columns, dtype=int)
        out.index.name = "sample_id"
        out = out.reset_index()
    else:
        out = pd.DataFrame(columns=columns)

    if records:
        hits = pd.concat(records, ignore_index=True).drop_duplicates()
        hits["value"] = 1
        wide = hits.pivot_table(
            index="sample_id",
            columns="gene",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        wide = wide.rename(columns={gene: f"{gene}_mutated" for gene in wide.columns})
        out = out.set_index("sample_id")
        out.loc[wide.index, wide.columns] = wide
        out = out.reset_index()

    return out.loc[:, columns]


def load_or_build_driver_gene_features(
    cache_path="data/processed/driver_gene_flags.csv",
    data_dir="data/tcga_mafs",
    driver_genes=DEFAULT_DRIVER_GENES,
    force=False,
    verbose=False,
):
    cache_path = Path(cache_path)
    expected_columns = ["sample_id"] + driver_feature_names(driver_genes)
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path)
        missing = [column for column in expected_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Driver feature cache is missing columns: {missing}")
        return df.loc[:, expected_columns].copy()

    df = build_driver_gene_features(
        data_dir=data_dir,
        driver_genes=driver_genes,
        verbose=verbose,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def maf_summary_feature_names():
    names = [
        "log1p_maf_total_rows",
        "log1p_maf_unique_genes",
        "log1p_maf_protein_altering_rows",
        "log1p_maf_indel_rows",
    ]
    names.extend(f"log1p_variant_classification_{value}" for value in SUMMARY_VARIANT_CLASSIFICATIONS)
    names.extend(f"log1p_variant_type_{value}" for value in SUMMARY_VARIANT_TYPES)
    names.extend(f"log1p_impact_{value}" for value in SUMMARY_IMPACTS)
    return names


def build_maf_summary_features(data_dir="data/tcga_mafs", verbose=False):
    """Return per-sample log-count features from broad MAF annotations."""
    paths = discover_maf_files(data_dir)
    feature_columns = maf_summary_feature_names()
    records = []

    for idx, path in enumerate(paths, start=1):
        if verbose and (idx == 1 or idx == len(paths) or idx % 100 == 0):
            print(f"[{idx}/{len(paths)}] {path}", flush=True)
        df = _read_summary_columns(path)
        if df.empty or "Tumor_Sample_Barcode" not in df.columns:
            continue
        df = df.dropna(subset=["Tumor_Sample_Barcode"])
        if df.empty:
            continue
        df["sample_id"] = df["Tumor_Sample_Barcode"].astype(str)
        for sample_id, sample_df in df.groupby("sample_id", sort=False):
            variant_classification = sample_df.get(
                "Variant_Classification",
                pd.Series(dtype=str),
            ).astype(str)
            variant_type = sample_df.get("Variant_Type", pd.Series(dtype=str)).astype(str)
            impact = sample_df.get("IMPACT", pd.Series(dtype=str)).astype(str)
            genes = sample_df.get("Hugo_Symbol", pd.Series(dtype=str)).dropna().astype(str)
            record = {
                "sample_id": sample_id,
                "log1p_maf_total_rows": len(sample_df),
                "log1p_maf_unique_genes": genes[genes != ""].nunique(),
                "log1p_maf_protein_altering_rows": variant_classification.isin(
                    PROTEIN_ALTERING_CLASSIFICATIONS
                ).sum(),
                "log1p_maf_indel_rows": variant_type.isin({"INS", "DEL"}).sum(),
            }
            class_counts = variant_classification.value_counts()
            type_counts = variant_type.value_counts()
            impact_counts = impact.value_counts()
            for value in SUMMARY_VARIANT_CLASSIFICATIONS:
                record[f"log1p_variant_classification_{value}"] = class_counts.get(value, 0)
            for value in SUMMARY_VARIANT_TYPES:
                record[f"log1p_variant_type_{value}"] = type_counts.get(value, 0)
            for value in SUMMARY_IMPACTS:
                record[f"log1p_impact_{value}"] = impact_counts.get(value, 0)
            records.append(record)

    if not records:
        return pd.DataFrame(columns=["sample_id"] + feature_columns)

    out = pd.DataFrame.from_records(records)
    out = out.groupby("sample_id", as_index=False)[feature_columns].sum()
    out[feature_columns] = out[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    out[feature_columns] = np.log1p(out[feature_columns])
    return out.loc[:, ["sample_id"] + feature_columns]


def load_or_build_maf_summary_features(
    cache_path="data/processed/maf_summary_features.csv",
    data_dir="data/tcga_mafs",
    force=False,
    verbose=False,
):
    cache_path = Path(cache_path)
    expected_columns = ["sample_id"] + maf_summary_feature_names()
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path)
        missing = [column for column in expected_columns if column not in df.columns]
        if missing:
            raise ValueError(f"MAF summary feature cache is missing columns: {missing}")
        return df.loc[:, expected_columns].copy()

    df = build_maf_summary_features(data_dir=data_dir, verbose=verbose)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _top_gene_feature_column(gene):
    clean = str(gene).strip().upper()
    return f"topgene_{clean}_mutated"


def build_top_mutated_gene_features(
    data_dir="data/tcga_mafs",
    max_genes=1000,
    min_samples=5,
    protein_altering_only=True,
    verbose=False,
):
    """Return binary flags for the most frequently mutated genes across MAFs."""
    paths = discover_maf_files(data_dir)
    records = []
    all_samples = set()

    for idx, path in enumerate(paths, start=1):
        if verbose and (idx == 1 or idx == len(paths) or idx % 100 == 0):
            print(f"[{idx}/{len(paths)}] {path}", flush=True)
        df = _read_driver_columns(path)
        if df.empty:
            continue
        df = df.dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])
        if df.empty:
            continue
        all_samples.update(df["Tumor_Sample_Barcode"].astype(str).unique())
        if protein_altering_only:
            df = df[df["Variant_Classification"].astype(str).isin(
                PROTEIN_ALTERING_CLASSIFICATIONS
            )]
        if df.empty:
            continue
        df["sample_id"] = df["Tumor_Sample_Barcode"].astype(str)
        df["gene"] = df["Hugo_Symbol"].astype(str).str.strip().str.upper()
        df = df[df["gene"] != ""]
        if df.empty:
            continue
        records.append(df[["sample_id", "gene"]].drop_duplicates())

    if not all_samples:
        return pd.DataFrame(columns=["sample_id"])
    if not records:
        return pd.DataFrame({"sample_id": sorted(all_samples)})

    hits = pd.concat(records, ignore_index=True).drop_duplicates()
    gene_counts = hits.groupby("gene")["sample_id"].nunique().sort_values(ascending=False)
    selected_genes = gene_counts[gene_counts >= min_samples].head(max_genes).index.tolist()
    columns = ["sample_id"] + [_top_gene_feature_column(gene) for gene in selected_genes]
    out = pd.DataFrame(0, index=sorted(all_samples), columns=columns[1:], dtype=np.int8)
    out.index.name = "sample_id"
    if selected_genes:
        selected_hits = hits[hits["gene"].isin(selected_genes)].copy()
        selected_hits["feature"] = selected_hits["gene"].map(_top_gene_feature_column)
        selected_hits["value"] = 1
        wide = selected_hits.pivot_table(
            index="sample_id",
            columns="feature",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        out.loc[wide.index, wide.columns] = wide.astype(np.int8)
    out = out.reset_index()
    return out.loc[:, columns]


def load_or_build_top_mutated_gene_features(
    cache_path="data/processed/top_mutated_gene_flags.csv",
    data_dir="data/tcga_mafs",
    max_genes=1000,
    min_samples=5,
    force=False,
    verbose=False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    df = build_top_mutated_gene_features(
        data_dir=data_dir,
        max_genes=max_genes,
        min_samples=min_samples,
        verbose=verbose,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/tcga_mafs")
    parser.add_argument("--out", default="data/processed/driver_gene_flags.csv")
    parser.add_argument("--driver-gene-file", default=None,
                        help="Optional TSV/CSV with driver genes in a Symbol column.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    driver_genes = (
        load_driver_genes(args.driver_gene_file)
        if args.driver_gene_file else DEFAULT_DRIVER_GENES
    )

    features = load_or_build_driver_gene_features(
        cache_path=args.out,
        data_dir=args.data_dir,
        driver_genes=driver_genes,
        force=args.force,
        verbose=args.verbose,
    )
    print(f"Wrote {features.shape[0]} samples x {features.shape[1] - 1} driver flags to {args.out}")
