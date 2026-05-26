"""
Neural Network Classifier for Tumor Type Prediction
using 6 basic mutation types as features.
Loads cached data from results/somatic_mutations.csv.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Constants
BASES = ["A", "C", "G", "T"]
SUBSTITUTIONS = [
    ("C", "A"), ("C", "G"), ("C", "T"),
    ("T", "A"), ("T", "C"), ("T", "G"),
]
MUTATION_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]

# Seed everything for deterministic output
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# Detect GPU / CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def plot_loss(train_losses, save_path):
    """Plot the training loss progression."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Neural Network Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Deeper, High-Capacity MLP Architecture with Batch Normalization
class HighCapacityMLP(nn.Module):
    def __init__(self, input_dim=6, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    csv_path = Path(__file__).resolve().parent.parent / "results" / "somatic_mutations.csv"
    results_dir = Path(__file__).resolve().parent.parent / "results" / "mutations"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"Cached data CSV not found at: {csv_path}\nPlease run 'python yash_trial/dataset/prepare_data.py' first.")

    print(f"Loading cached somatic mutation data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} samples across {df_raw['tumor_type'].nunique()} classes.")

    # Sum 96 trinucleotide counts into 6 basic mutation types
    df = pd.DataFrame()
    df["sample_id"] = df_raw["sample_id"]
    df["tumor_type"] = df_raw["tumor_type"]
    for m in MUTATION_TYPES:
        df[m] = 0
    for cat in CATEGORIES:
        mut_type = cat.split('[')[1].split(']')[0]
        df[mut_type] += df_raw[cat]

    # Prepare features and labels
    X_counts = df[MUTATION_TYPES].values.astype(np.float32)
    X_probs = counts_to_probs(X_counts)
    
    le = LabelEncoder()
    y = le.fit_transform(df["tumor_type"].values)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_probs, y, test_size=0.2, stratify=y, random_state=42
    )

    # Calculate class weights for training set
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # Datasets and Loaders (shuffle=True only for training)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Instantiate model, optimizer, loss
    model = HighCapacityMLP(input_dim=6, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # Training loop
    epochs = 200
    train_losses = []
    print(f"\nTraining Neural Network on device: {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Loss: {epoch_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits_train = model(X_train_t)
        logits_test = model(X_test_t)
        
        y_pred_train = torch.argmax(logits_train, dim=1).cpu().numpy()
        y_pred_test = torch.argmax(logits_test, dim=1).cpu().numpy()

    # Move target values back to CPU for evaluation
    y_train_cpu = y_train_t.cpu().numpy()
    y_test_cpu = y_test_t.cpu().numpy()

    train_report = classification_report(y_train_cpu, y_pred_train, target_names=class_names, output_dict=True, zero_division=0)
    test_report = classification_report(y_test_cpu, y_pred_test, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_cpu, y_pred_test)

    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test_cpu, y_pred_test, target_names=class_names, zero_division=0))

    # Save metrics
    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_macro_f1": train_report["macro avg"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_macro_f1": test_report["macro avg"]["f1-score"],
        "test_report": test_report,
        "confusion_matrix": cm.tolist()
    }
    with open(results_dir / "nn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {results_dir / 'nn_metrics.json'}")

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, results_dir / "nn_confusion_matrix.png")
    print(f"Saved confusion matrix plot to: {results_dir / 'nn_confusion_matrix.png'}")

    # Plot and save training loss
    plot_loss(train_losses, results_dir / "nn_loss.png")
    print(f"Saved training loss plot to: {results_dir / 'nn_loss.png'}")

if __name__ == "__main__":
    main()
