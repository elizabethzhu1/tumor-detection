"""Build the shared processed spectra cache from cached TCGA MAF files."""

import argparse

from data_processing_helpers import (
    DEFAULT_PROCESSED_DATA,
    discover_maf_paths,
    load_maf_spectra,
    save_spectra_counts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/tcga_mafs",
                        help="Raw cached MAF directory.")
    parser.add_argument("--out", default=DEFAULT_PROCESSED_DATA,
                        help="Processed spectra CSV to write.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional tumor labels, e.g. BRCA SKCM LUAD.")
    parser.add_argument("--max-files-per-label", type=int, default=None,
                        help="Optional cap for quick test runs.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every parsed MAF path.")
    args = parser.parse_args()

    maf_paths = discover_maf_paths(
        args.data_dir,
        labels=args.labels,
        max_files_per_label=args.max_files_per_label,
    )
    spectra_df = load_maf_spectra(maf_paths, verbose=args.verbose)
    save_spectra_counts(spectra_df, args.out)

    counts = spectra_df["tumor_type"].value_counts().sort_index()
    print("\nProcessed spectra cache")
    print(f"  samples: {len(spectra_df)}")
    print(f"  labels: {counts.to_dict()}")
    print(f"  saved to: {args.out}")


if __name__ == "__main__":
    main()
