"""
Logistic Regression Classifier for Tumor Type Prediction
using 96 trinucleotide mutation types as features.
Loads cached data from results/somatic_mutations.csv.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Constants
BASES = ["A", "C", "G", "T"]
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
assert len(CATEGORIES) == 96

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
    csv_path = Path(__file__).resolve().parent.parent / "results" / "somatic_mutations.csv"
    results_dir = Path(__file__).resolve().parent.parent / "results" / "trinucleotides"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"Cached data CSV not found at: {csv_path}\nPlease run 'python yash_trial/prepare_data.py' first.")

    print(f"Loading cached somatic mutation data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples across {df['tumor_type'].nunique()} classes.")
    print(df.groupby('tumor_type').size())

    # Prepare features and labels
    X_counts = df[CATEGORIES].values.astype(np.float64)
    X_probs = counts_to_probs(X_counts)
    
    le = LabelEncoder()
    y = le.fit_transform(df["tumor_type"].values)
    class_names = list(le.classes_)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_probs, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTraining Logistic Regression with Cross-Validation...")
    clf = LogisticRegressionCV(cv=5, max_iter=10000, random_state=42)
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
        "train_macro_f1": train_report["macro avg"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_macro_f1": test_report["macro avg"]["f1-score"],
        "test_report": test_report,
        "confusion_matrix": cm.tolist()
    }
    with open(results_dir / "lr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {results_dir / 'lr_metrics.json'}")

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, results_dir / "lr_confusion_matrix.png")
    print(f"Saved confusion matrix plot to: {results_dir / 'lr_confusion_matrix.png'}")

    # Plot and save coefficients (annotate=False because 96 features are too many for numeric labels)
    plot_coefficients_heatmap(clf.coef_, CATEGORIES, class_names, results_dir / "lr_coefficients.png", annotate=False)
    print(f"Saved coefficients plot to: {results_dir / 'lr_coefficients.png'}")

if __name__ == "__main__":
    main()
