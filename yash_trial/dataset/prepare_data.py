"""
Prepare somatic mutation counts from raw MAF files and cache them to a CSV.
"""

from pathlib import Path
import gzip
import io
import pandas as pd
import numpy as np

# Constants
BASES = ["A", "C", "G", "T"]
PURINES = ["A", "G"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
SUBSTITUTIONS = [
    ("C", "A"), ("C", "G"), ("C", "T"),
    ("T", "A"), ("T", "C"), ("T", "G"),
]

def build_category_list():
    """Return the 96 categories as strings like 'A[C>A]A' in canonical order."""
    cats = []
    for ref, alt in SUBSTITUTIONS:
        for five_prime in BASES:
            for three_prime in BASES:
                cats.append(f"{five_prime}[{ref}>{alt}]{three_prime}")
    return cats

CATEGORIES = build_category_list()
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
assert len(CATEGORIES) == 96

def _open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def _trinuc_from_context(context, ref):
    """Convert GDC CONTEXT to a 3-base context centered on the reference allele."""
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

def reverse_complement(seq):
    return "".join(COMPLEMENT[b] for b in reversed(seq))

def classify_mutation(ref, alt, trinuc_context):
    """Map a (ref, alt, 3-base context) to one of the 96 category indices."""
    if trinuc_context is None:
        return None
    if len(trinuc_context) != 3 or "N" in trinuc_context:
        return None
    if ref not in BASES or alt not in BASES or ref == alt:
        return None
    if trinuc_context[1] != ref:
        return None

    # Fold to pyrimidine reference
    if ref in PURINES:
        ref = COMPLEMENT[ref]
        alt = COMPLEMENT[alt]
        trinuc_context = reverse_complement(trinuc_context)

    key = f"{trinuc_context[0]}[{ref}>{alt}]{trinuc_context[2]}"
    return CATEGORY_TO_IDX.get(key)

def load_maf_data(data_dir: Path):
    """Scan and parse MAF files from tumor type subdirectories."""
    project_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    all_samples = []

    print(f"Scanning data directory: {data_dir}")
    for project in project_dirs:
        maf_files = sorted(project.glob("*.maf*"))
        if not maf_files:
            continue
        print(f"Loading tumor type: {project.name} ({len(maf_files)} files)...")

        for path in maf_files:
            with _open_maybe_gz(path) as fh:
                lines = [ln for ln in fh if not ln.startswith("#")]
            if not lines:
                continue

            # Read header to find column mappings
            header = lines[0].split("\t")
            columns = [c.strip() for c in header]

            candidates_map = {
                "Tumor_Sample_Barcode": ["Tumor_Sample_Barcode"],
                "Reference_Allele": ["Reference_Allele"],
                "Tumor_Seq_Allele2": ["Tumor_Seq_Allele2", "Allele"],
                "Variant_Type": ["Variant_Type"],
                "CONTEXT": ["CONTEXT"]
            }

            usecols = []
            col_rename = {}
            for std_name, candidates in candidates_map.items():
                found = None
                for cand in candidates:
                    for c in columns:
                        if c.lower() == cand.lower():
                            found = c
                            break
                    if found:
                        break
                if found:
                    usecols.append(found)
                    col_rename[found] = std_name

            df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", usecols=usecols, low_memory=False)
            df = df.rename(columns=col_rename)

            required = ["Tumor_Sample_Barcode", "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Type", "CONTEXT"]
            missing = [r for r in required if r not in df.columns]
            if missing:
                print(f"Warning: File {path.name} lacks required columns {missing}. Skipping.")
                continue

            # Filter for SNPs/SNVs
            df = df[df["Variant_Type"].astype(str).str.upper().isin(["SNP", "SNV"])].copy()
            df["Reference_Allele"] = df["Reference_Allele"].astype(str).str.upper()
            df["Tumor_Seq_Allele2"] = df["Tumor_Seq_Allele2"].astype(str).str.upper()
            df = df[df["Reference_Allele"].isin(BASES) & df["Tumor_Seq_Allele2"].isin(BASES)]

            if df.empty:
                continue

            # Classify mutations
            def classify_row(row):
                trinuc = _trinuc_from_context(row["CONTEXT"], row["Reference_Allele"])
                return classify_mutation(row["Reference_Allele"], row["Tumor_Seq_Allele2"], trinuc)

            df["cat_idx"] = df.apply(classify_row, axis=1)
            df = df.dropna(subset=["cat_idx"])
            df["cat_idx"] = df["cat_idx"].astype(int)

            if df.empty:
                continue

            # Aggregate per sample in this file
            for sample_id, group in df.groupby("Tumor_Sample_Barcode"):
                counts = np.zeros(96, dtype=np.int64)
                for cat_idx in group["cat_idx"]:
                    counts[cat_idx] += 1

                all_samples.append({
                    "sample_id": sample_id,
                    "tumor_type": project.name,
                    "counts_96": counts
                })

    if not all_samples:
        raise ValueError(f"No valid sample data parsed from {data_dir}")

    # Build DataFrame
    rows = []
    for sample in all_samples:
        row_dict = {
            "sample_id": sample["sample_id"],
            "tumor_type": sample["tumor_type"],
        }
        for i, cat in enumerate(CATEGORIES):
            row_dict[cat] = sample["counts_96"][i]
        rows.append(row_dict)

    return pd.DataFrame(rows)

def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "tcga_mafs"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = results_dir / "somatic_mutations.csv"

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print("Loading and parsing somatic mutation data from raw MAF files...")
    df = load_maf_data(data_dir)
    
    print(f"Aggregated {len(df)} samples across {df['tumor_type'].nunique()} classes.")
    print(df.groupby('tumor_type').size())

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully saved aggregated counts to: {output_csv}")

if __name__ == "__main__":
    main()
