"""
Logistic Regression Classifier for Tumor Type Prediction
using 6 basic mutation types as features.
"""

from pathlib import Path
import sys
import gzip
import io
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Constants
BASES = ["A", "C", "G", "T"]
PURINES = ["A", "G"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
SUBSTITUTIONS = [
    ("C", "A"), ("C", "G"), ("C", "T"),
    ("T", "A"), ("T", "C"), ("T", "G"),
]
MUTATION_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
MUTATION_TYPE_TO_IDX = {m: i for i, m in enumerate(MUTATION_TYPES)}

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

# Map 96 category indices to 6 basic mutation type indices
CAT_IDX_TO_MUT_IDX = np.array([
    MUTATION_TYPE_TO_IDX[cat.split('[')[1].split(']')[0]]
    for cat in CATEGORIES
])

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
        # Sum 96 counts into 6 basic mutation counts
        counts_6 = np.zeros(6, dtype=np.int64)
        for cat_idx, count in enumerate(sample["counts_96"]):
            counts_6[CAT_IDX_TO_MUT_IDX[cat_idx]] += count

        row_dict = {
            "sample_id": sample["sample_id"],
            "tumor_type": sample["tumor_type"],
        }
        for i, m in enumerate(MUTATION_TYPES):
            row_dict[m] = counts_6[i]
        rows.append(row_dict)

    return pd.DataFrame(rows)

def counts_to_probs(counts_matrix, eps=0.0):
    """Row-normalize a count matrix to probabilities."""
    row_sums = counts_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return (counts_matrix + eps) / (row_sums + eps * counts_matrix.shape[1])

def plot_confusion_matrix(cm, class_names, save_path):
    """Save a clean heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_coefficients_heatmap(coefs, feature_names, class_names, save_path, annotate=True):
    """Save a heatmap of the logistic regression coefficients."""
    width = max(8, len(feature_names) * 0.18)
    fig, ax = plt.subplots(figsize=(width, 5))
    
    vmax = np.abs(coefs).max()
    vmin = -vmax
    im = ax.imshow(coefs, interpolation='nearest', cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label='Coefficient Value')
    
    ax.set(xticks=np.arange(len(feature_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=feature_names, yticklabels=class_names,
           title='Logistic Regression Coefficients',
           ylabel='Tumor Class',
           xlabel='Feature')
    
    rotation = 45 if len(feature_names) < 20 else 90
    fontsize = 10 if len(feature_names) < 20 else 6
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right" if rotation == 45 else "center",
             rotation_mode="anchor", fontsize=fontsize)
    
    if annotate and len(feature_names) < 20:
        thresh = vmax / 2.
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                val = coefs[i, j]
                color = "white" if np.abs(val) > thresh else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color=color)
                        
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "tcga_mafs"
    results_dir = Path(__file__).resolve().parent / "results" / "mutations"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print("Loading somatic mutation data...")
    df = load_maf_data(data_dir)
    print(f"Loaded data: {len(df)} samples across {df['tumor_type'].nunique()} classes.")
    print(df.groupby('tumor_type').size())

    # Prepare features and labels
    X_counts = df[MUTATION_TYPES].values.astype(np.float64)
    X_probs = counts_to_probs(X_counts)
    
    le = LabelEncoder()
    y = le.fit_transform(df["tumor_type"].values)
    class_names = list(le.classes_)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_probs, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTraining Logistic Regression with Cross-Validation...")
    clf = LogisticRegressionCV(cv=5, max_iter=10000, multi_class='multinomial', random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_report = classification_report(y_train, y_pred_train, target_names=class_names, output_dict=True, zero_division=0)
    test_report = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

    # Save metrics
    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_macro_f1": train_report["macro_f1"],
        "test_accuracy": test_report["accuracy"],
        "test_macro_f1": test_report["macro_f1"],
        "test_report": test_report,
        "confusion_matrix": cm.tolist()
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {results_dir / 'metrics.json'}")

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, results_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix plot to: {results_dir / 'confusion_matrix.png'}")

    # Plot and save coefficients
    plot_coefficients_heatmap(clf.coef_, MUTATION_TYPES, class_names, results_dir / "coefficients.png", annotate=True)
    print(f"Saved coefficients plot to: {results_dir / 'coefficients.png'}")

if __name__ == "__main__":
    main()
