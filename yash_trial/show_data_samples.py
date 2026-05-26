"""Inspect downloaded TCGA MAF data and display two sample rows per tumor type."""

from pathlib import Path
import sys
import gzip
import io
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from maf_processing import parse_maf_file, _find_column, _open_maybe_gz


def _read_raw_maf(path: Path, n_rows: int = 2):
    with _open_maybe_gz(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    return df.head(n_rows)


def show_two_samples_per_project(data_dir: Path):
    project_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not project_dirs:
        raise SystemExit(f"No project directories found in {data_dir}")

    print(f"Data directory: {data_dir}")
    print(f"Found {len(project_dirs)} tumor types:\n")

    for project in project_dirs:
        maf_files = sorted(project.glob("*.maf*"))
        if not maf_files:
            print(f"Skipping {project.name}: no MAF files found")
            continue

        sample_path = maf_files[0]
        print(f"=== Tumor type: {project.name} ===")
        print(f"Using file: {sample_path.name}")

        raw_df = _read_raw_maf(sample_path, n_rows=2)
        sample_col = _find_column(raw_df, ["Tumor_Sample_Barcode"])
        ref_col = _find_column(raw_df, ["Reference_Allele"])
        alt_col = _find_column(raw_df, ["Tumor_Seq_Allele2", "Allele"])
        context_col = _find_column(raw_df, ["CONTEXT"])
        variant_type_col = _find_column(raw_df, ["Variant_Type"])

        display_cols = [c for c in [sample_col, variant_type_col, ref_col, alt_col, context_col] if c]
        print("Raw MAF sample rows:")
        print(raw_df.loc[:, display_cols].to_string(index=False))
        print()

        df = parse_maf_file(sample_path, project.name)
        if df.empty:
            print("Parsed DataFrame is empty, skipping.\n")
            continue

        print(f"Parsed {len(df)} samples for {project.name}")
        print(df.loc[:, ["sample_id", "tumor_type"]].head(2).to_string(index=False))
        print()

    print("Done.")


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "tcga_mafs"
    show_two_samples_per_project(data_dir)


if __name__ == "__main__":
    main()
