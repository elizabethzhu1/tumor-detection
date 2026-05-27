"""Feature extraction from cached TCGA MAF files."""

import argparse
from pathlib import Path

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
