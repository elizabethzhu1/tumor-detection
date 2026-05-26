"""Plot distribution of 6 basic mutation types per tumor type."""

from pathlib import Path
import sys
import gzip
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from maf_processing import _find_column, _open_maybe_gz, BASES


def get_basic_mutation_type(ref, alt):
    """Extract basic mutation type (e.g., C>A) ignoring trinucleotide context."""
    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()
    if len(ref) == 1 and len(alt) == 1 and ref in BASES and alt in BASES and ref != alt:
        return f"{ref}>{alt}"
    return None


def count_mutation_types(maf_path: Path):
    """Parse a MAF file and count occurrences of each 6 basic mutation types."""
    with _open_maybe_gz(maf_path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)

    ref_col = _find_column(df, ["Reference_Allele"])
    alt_col = _find_column(df, ["Tumor_Seq_Allele2", "Allele"])
    variant_type_col = _find_column(df, ["Variant_Type"])

    if not all([ref_col, alt_col, variant_type_col]):
        return {}

    # Filter for SNPs/SNVs
    df = df[df[variant_type_col].astype(str).str.upper().isin(["SNP", "SNV"])].copy()

    mutation_counts = {}
    for _, row in df.iterrows():
        mut_type = get_basic_mutation_type(row[ref_col], row[alt_col])
        if mut_type:
            mutation_counts[mut_type] = mutation_counts.get(mut_type, 0) + 1

    return mutation_counts


def plot_mutation_distribution_combined(data_dir: Path, output_dir: Path, max_files_per_project: int = 3):
    """Plot mutation type distribution for all tumor types in a single grouped bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    project_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    all_mutation_types = {"C>A", "C>G", "C>T", "T>A", "T>C", "T>G"}
    
    # Collect data for all projects
    plot_data = {}
    
    for project in project_dirs:
        maf_files = sorted(project.glob("*.maf*"))[:max_files_per_project]
        if not maf_files:
            print(f"Skipping {project.name}: no MAF files found")
            continue

        print(f"Processing {project.name} ({len(maf_files)} files)...")

        # Aggregate counts from all MAF files for this tumor type
        total_counts = {mut: 0 for mut in all_mutation_types}
        for maf_path in maf_files:
            print(f"  Parsing {maf_path.name}...")
            counts = count_mutation_types(maf_path)
            for mut, count in counts.items():
                if mut in total_counts:
                    total_counts[mut] += count

        total = sum(total_counts.values())
        if total == 0:
            print(f"  No mutations found for {project.name}")
            continue

        # Convert to percentages
        percentages = {mut: (count / total * 100) for mut, count in total_counts.items()}
        plot_data[project.name] = {"percentages": percentages, "total": total, "counts": total_counts}
        print(f"  Counts: {total_counts}")
    
    if not plot_data:
        raise SystemExit("No valid projects found to plot.")
    
    # Create single grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    mutations = sorted(all_mutation_types)
    tumor_types = sorted(plot_data.keys())
    n_tumors = len(tumor_types)
    n_mutations = len(mutations)
    
    # Bar width and positions
    bar_width = 0.13
    x = np.arange(n_mutations)
    
    # Colors for each tumor type
    colors = plt.cm.Set3(np.linspace(0, 1, n_tumors))
    
    # Plot grouped bars
    for i, tumor_type in enumerate(tumor_types):
        percentages = plot_data[tumor_type]["percentages"]
        values = [percentages[mut] for mut in mutations]
        offset = (i - n_tumors / 2) * bar_width + bar_width / 2
        ax.bar(x + offset, values, bar_width, label=tumor_type, color=colors[i], edgecolor="black", linewidth=0.8)
    
    # Formatting
    ax.set_xlabel("Mutation Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percent Distribution (%)", fontsize=12, fontweight="bold")
    ax.set_title("Mutation Type Distribution Across All Tumor Types", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mutations, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "all_mutation_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nCombined plot saved to: {output_path}")



def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "tcga_mafs"
    output_dir = Path(__file__).resolve().parent.parent / "plots"
    
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    plot_mutation_distribution_combined(data_dir, output_dir, max_files_per_project=3)
    print(f"All plots saved to {output_dir}")



if __name__ == "__main__":
    main()
