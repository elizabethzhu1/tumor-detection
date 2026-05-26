"""Plot distribution of 96 trinucleotide mutation types per tumor type."""

from pathlib import Path
import sys
import gzip
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from maf_processing import _find_column, _open_maybe_gz, CATEGORIES, classify_mutation, _trinuc_from_context, BASES


def count_trinucleotide_mutations(maf_path: Path):
    """Parse a MAF file and count occurrences of each 96 trinucleotide mutation types."""
    with _open_maybe_gz(maf_path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)

    sample_col = _find_column(df, ["Tumor_Sample_Barcode"])
    ref_col = _find_column(df, ["Reference_Allele"])
    alt_col = _find_column(df, ["Tumor_Seq_Allele2", "Allele"])
    variant_type_col = _find_column(df, ["Variant_Type"])
    context_col = _find_column(df, ["CONTEXT"])

    if not all([ref_col, alt_col, variant_type_col, context_col]):
        return {}

    # Filter for SNPs/SNVs
    df = df[df[variant_type_col].astype(str).str.upper().isin(["SNP", "SNV"])].copy()
    df[ref_col] = df[ref_col].astype(str).str.upper()
    df[alt_col] = df[alt_col].astype(str).str.upper()
    df = df[df[ref_col].isin(BASES) & df[alt_col].isin(BASES)]

    mutation_counts = {cat: 0 for cat in CATEGORIES}
    for _, row in df.iterrows():
        trinuc = _trinuc_from_context(row[context_col], row[ref_col])
        cat_idx = classify_mutation(row[ref_col], row[alt_col], trinuc)
        if cat_idx is not None and not pd.isna(cat_idx):
            cat_idx = int(cat_idx)
            if 0 <= cat_idx < len(CATEGORIES):
                mutation_counts[CATEGORIES[cat_idx]] += 1

    return mutation_counts


def plot_trinucleotide_distribution_combined(data_dir: Path, output_dir: Path, max_files_per_project: int = 3):
    """Plot trinucleotide mutation distribution for all tumor types in a single grouped bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    project_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    
    # Collect data for all projects
    plot_data = {}
    
    for project in project_dirs:
        maf_files = sorted(project.glob("*.maf*"))[:max_files_per_project]
        if not maf_files:
            print(f"Skipping {project.name}: no MAF files found")
            continue

        print(f"Processing {project.name} ({len(maf_files)} files)...")

        # Aggregate counts from all MAF files for this tumor type
        total_counts = {cat: 0 for cat in CATEGORIES}
        for maf_path in maf_files:
            print(f"  Parsing {maf_path.name}...")
            counts = count_trinucleotide_mutations(maf_path)
            for cat, count in counts.items():
                total_counts[cat] += count

        total = sum(total_counts.values())
        if total == 0:
            print(f"  No mutations found for {project.name}")
            continue

        # Convert to percentages
        percentages = {cat: (count / total * 100) for cat, count in total_counts.items()}
        plot_data[project.name] = {"percentages": percentages, "total": total, "counts": total_counts}
        print(f"  Total mutations: {total}")
    
    if not plot_data:
        raise SystemExit("No valid projects found to plot.")
    
    # Create individual plots first (COSMIC standard)
    print("\nGenerating individual tumor-type plots...")
    for tumor_type, data in plot_data.items():
        _plot_individual_trinucleotide(tumor_type, data, output_dir)
    
    # Create single grouped bar chart
    print("Generating combined plot...")
    _plot_combined_trinucleotide(plot_data, output_dir)


def _plot_individual_trinucleotide(tumor_type: str, data: dict, output_dir: Path):
    """Plot trinucleotide distribution for a single tumor type in COSMIC style."""
    percentages = data["percentages"]
    total = data["total"]
    
    # Organize by mutation type groups
    mutation_types = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    colors_cosmic = {
        "C>A": "#3050A0",
        "C>G": "#000000",
        "C>T": "#E62020",
        "T>A": "#CCCCCC",
        "T>C": "#A0D055",
        "T>G": "#F5A623",
    }
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    x_pos = 0
    x_labels = []
    x_ticks = []
    colors_list = []
    values_list = []
    
    for mut_type in mutation_types:
        # Get all trinucleotides for this mutation type
        cats_for_type = [cat for cat in CATEGORIES if mut_type in cat]
        
        for cat in cats_for_type:
            val = percentages.get(cat, 0)
            values_list.append(val)
            colors_list.append(colors_cosmic[mut_type])
            x_labels.append(cat)
            x_ticks.append(x_pos)
            x_pos += 1
        
        # Add separator between mutation types
        x_pos += 0.5
    
    ax.bar(x_ticks, values_list, color=colors_list, edgecolor="black", linewidth=0.5, width=0.8)
    ax.set_ylabel("Percent Distribution (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trinucleotide Context", fontsize=12, fontweight="bold")
    ax.set_title(f"{tumor_type}: Trinucleotide Mutation Distribution (n={total:,} mutations)", 
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=90)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"{tumor_type}_trinucleotide_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path.name}")


def _plot_combined_trinucleotide(plot_data: dict, output_dir: Path):
    """Plot all tumor types in a single grouped bar chart."""
    # Create single grouped bar chart
    fig, ax = plt.subplots(figsize=(20, 7))
    
    trinucleotides = CATEGORIES
    tumor_types = sorted(plot_data.keys())
    n_tumors = len(tumor_types)
    n_trinuc = len(trinucleotides)
    
    # Bar width and positions
    bar_width = 0.11
    x = np.arange(n_trinuc)
    
    # Colors for each tumor type
    colors = plt.cm.Set3(np.linspace(0, 1, n_tumors))
    
    # Plot grouped bars
    for i, tumor_type in enumerate(tumor_types):
        percentages = plot_data[tumor_type]["percentages"]
        values = [percentages[cat] for cat in trinucleotides]
        offset = (i - n_tumors / 2) * bar_width + bar_width / 2
        ax.bar(x + offset, values, bar_width, label=tumor_type, color=colors[i], edgecolor="black", linewidth=0.5)
    
    # Formatting
    ax.set_xlabel("Trinucleotide Mutation Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percent Distribution (%)", fontsize=12, fontweight="bold")
    ax.set_title("Trinucleotide Mutation Distribution Across All Tumor Types", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(trinucleotides, fontsize=8, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "all_trinucleotide_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path.name}")


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "tcga_mafs"
    output_dir = Path(__file__).resolve().parent.parent / "plots"
    
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    plot_trinucleotide_distribution_combined(data_dir, output_dir, max_files_per_project=3)
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
