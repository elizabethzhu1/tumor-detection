"""
Neural Network Classifier for Tumor Type Prediction with Interpretable Signatures
using 96 trinucleotide mutation types as features.
Loads cached data from results/somatic_mutations.csv.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Seed everything for deterministic output
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# Insert root to import cosmic
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from cosmic import load_cosmic_signatures

# Constants
BASES = ["A", "C", "G", "T"]
SUBSTITUTIONS = [
    ("C", "A"), ("C", "G"), ("C", "T"),
    ("T", "A"), ("T", "C"), ("T", "G"),
]

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

def plot_losses(losses_dict, save_path):
    """Plot the training loss trajectories (classification vs reconstruction)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses_dict['total'], label='Total Loss', color='black', linewidth=1.5)
    ax.plot(losses_dict['class'], label='Classification Loss', color='#1f77b4', linestyle='--')
    ax.plot(losses_dict['recon'], label='Weighted Reconstruction Loss (x5)', color='#2ca02c', linestyle=':')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Joint Network Loss Trajectory', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_signature_importance_heatmap(weights, row_names, col_names, save_path):
    """Plot a heatmap showing signature influence on each tumor type."""
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(weights, cmap='RdBu_r', aspect='auto')
    ax.figure.colorbar(im, ax=ax, label='Log-odds Influence (Gradient * Exposure)')
    
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=10, fontweight='bold')
    
    ax.set_title('Mutational Signature Influence on Tumor Classification', fontsize=12, fontweight='bold')
    ax.set_xlabel('Reference-Matched Signatures', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tumor Type', fontsize=11, fontweight='bold')
    
    # Text annotations
    for i in range(len(row_names)):
        for j in range(len(col_names)):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", 
                    color="white" if abs(weights[i, j]) > np.max(abs(weights)) * 0.5 else "black",
                    fontsize=7)
            
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# High-Capacity Signature Bottleneck Neural Network
class DeepSignatureBottleneckNet(nn.Module):
    def __init__(self, input_dim=96, bottleneck_dim=30, num_classes=7):
        super().__init__()
        # Encoder: high-capacity mapping from 96 features to K bottleneck nodes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim)
        )
        
        # Decoder: K signatures represented by weights of shape (96, K)
        self.W = nn.Parameter(torch.randn(input_dim, bottleneck_dim) * 0.1)
        
        # Classifier: Multi-Layer Perceptron on exposures for high classification performance
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # 1. Exposures s (sum-to-1 and non-negative)
        s = F.softmax(self.encoder(x), dim=1)
        
        # 2. Signature matrix W (non-negative and column-normalized)
        W_non_neg = F.softplus(self.W)
        W_norm = W_non_neg / (W_non_neg.sum(dim=0, keepdim=True) + 1e-8)
        
        # 3. Reconstruction: x_hat = s * W_norm^T
        x_reconstructed = torch.matmul(s, W_norm.t())
        
        # 4. Classification: y_pred
        y_pred = self.classifier(s)
        
        return y_pred, x_reconstructed, s

def compute_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def main():
    csv_path = Path(__file__).resolve().parent.parent / "results" / "somatic_mutations.csv"
    results_dir = Path(__file__).resolve().parent.parent / "results" / "trinucleotides"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"Cached data CSV not found at: {csv_path}\nPlease run 'python yash_trial/dataset/prepare_data.py' first.")

    print(f"Loading cached somatic mutation data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples across {df['tumor_type'].nunique()} classes.")

    # Prepare features and labels
    X_counts = df[CATEGORIES].values.astype(np.float32)
    X_probs = counts_to_probs(X_counts)
    
    le = LabelEncoder()
    y = le.fit_transform(df["tumor_type"].values)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_probs, y, test_size=0.2, stratify=y, random_state=42
    )

    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    # Tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load reference signatures
    ref_sigs, ref_names = load_cosmic_signatures() # outputs shape (96, n_sig)
    print(f"Loaded {ref_sigs.shape[1]} reference signatures.")

    # Model and training parameters
    K = 30
    model = DeepSignatureBottleneckNet(input_dim=96, bottleneck_dim=K, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights_t)

    # Training loop
    epochs = 250
    recon_weight = 5.0
    history = {'total': [], 'class': [], 'recon': []}
    
    print(f"\nTraining Deep Signature-Bottleneck Neural Network on device: {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_total = 0.0
        epoch_class = 0.0
        epoch_recon = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred, x_recon, s = model(batch_x)
            
            # Classification Loss
            class_loss = criterion_class(y_pred, batch_y)
            
            # Class-weighted Reconstruction Loss
            batch_weights = class_weights_t[batch_y].unsqueeze(1)
            recon_errs = (x_recon - batch_x) ** 2
            recon_loss = torch.mean(recon_errs * batch_weights)
            
            # Total Loss
            total_loss = class_loss + recon_weight * recon_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_total += total_loss.item() * batch_x.size(0)
            epoch_class += class_loss.item() * batch_x.size(0)
            epoch_recon += recon_loss.item() * batch_x.size(0)
            
        n_samples = len(train_dataset)
        history['total'].append(epoch_total / n_samples)
        history['class'].append(epoch_class / n_samples)
        history['recon'].append((epoch_recon / n_samples) * recon_weight)
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Total Loss: {history['total'][-1]:.4f} | Class Loss: {history['class'][-1]:.4f} | Recon Loss: {history['recon'][-1]:.4f}")

    # Evaluate
    model.eval()
    
    # 1. Forward pass to get predictions and signature exposures
    with torch.no_grad():
        logits_train, _, _ = model(X_train_t)
        logits_test, _, exposures_test = model(X_test_t)
        
        y_pred_train = torch.argmax(logits_train, dim=1).cpu().numpy()
        y_pred_test = torch.argmax(logits_test, dim=1).cpu().numpy()
        
        # Get final learned signatures W_normalized
        W_non_neg = F.softplus(model.W)
        W_norm = (W_non_neg / (W_non_neg.sum(dim=0, keepdim=True) + 1e-8)).cpu().numpy() # shape (96, K)

    y_train_cpu = y_train_t.cpu().numpy()
    y_test_cpu = y_test_t.cpu().numpy()

    # 2. Compute Interpretability via Feature Attribution (Gradient * Input)
    # Enable gradients to run backprop on the exposures vector
    exposures_test_t = exposures_test.detach().clone().requires_grad_(True)
    classifier_logits_test = model.classifier(exposures_test_t)
    
    attributions = np.zeros((num_classes, K))
    for c in range(num_classes):
        if exposures_test_t.grad is not None:
            exposures_test_t.grad = None
            
        # Backprop through the classifier only (Logits with respect to Exposures s)
        logit_c = classifier_logits_test[:, c].sum()
        logit_c.backward(retain_graph=True)
        
        # Gradient * Input attribution
        grad_c = exposures_test_t.grad.cpu().numpy() # shape (n_test, K)
        s_c = exposures_test_t.detach().cpu().numpy() # shape (n_test, K)
        
        # We compute the average attribution of exposures for patients belonging to class c
        class_mask = (y_test_cpu == c)
        if np.sum(class_mask) > 0:
            attributions[c] = np.mean(grad_c[class_mask] * s_c[class_mask], axis=0)
        else:
            attributions[c] = np.mean(grad_c * s_c, axis=0)

    # 3. Map learned signatures to COSMIC reference signatures
    matched_ref_indices = []
    matched_ref_names = []
    similarities = []
    
    print("\n--- Signature Alignment with COSMIC Reference ---")
    for j in range(K):
        learned_sig = W_norm[:, j]
        best_sim = -1.0
        best_idx = -1
        
        for r in range(ref_sigs.shape[1]):
            ref_sig = ref_sigs[:, r]
            sim = compute_cosine_similarity(learned_sig, ref_sig)
            if sim > best_sim:
                best_sim = sim
                best_idx = r
                
        matched_ref_indices.append(best_idx)
        matched_ref_names.append(ref_names[best_idx])
        similarities.append(best_sim)
        print(f"Learned Sig {j+1:02d} -> Best Match: {ref_names[best_idx]} (cos sim: {best_sim:.3f})")

    # Rename duplicates to differentiate them in plots
    unique_col_names = []
    name_counts = {}
    for name in matched_ref_names:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    seen = {}
    for name in matched_ref_names:
        if name_counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            unique_col_names.append(f"{name}_(Node_{seen[name]})")
        else:
            unique_col_names.append(name)

    # Save signature matching and importance statistics
    signature_data = {
        "num_signatures": K,
        "learned_signatures": W_norm.tolist(),
        "reference_alignment": [
            {"learned_idx": j, "matched_ref": matched_ref_names[j], "similarity": similarities[j]}
            for j in range(K)
        ],
        "classifier_attributions": attributions.tolist(),
        "class_names": class_names,
        "signature_names": unique_col_names
    }
    with open(results_dir / "nn_signature_importance.json", "w") as f:
        json.dump(signature_data, f, indent=4)
    print(f"Saved signature importance JSON to: {results_dir / 'nn_signature_importance.json'}")

    # Plot signature importance heatmap (attributions)
    plot_signature_importance_heatmap(attributions, class_names, unique_col_names, results_dir / "nn_signature_importance.png")
    print(f"Saved signature importance heatmap plot to: {results_dir / 'nn_signature_importance.png'}")

    # Standard metrics
    train_report = classification_report(y_train_cpu, y_pred_train, target_names=class_names, output_dict=True, zero_division=0)
    test_report = classification_report(y_test_cpu, y_pred_test, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_cpu, y_pred_test)

    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test_cpu, y_pred_test, target_names=class_names, zero_division=0))

    # Save standard metrics
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

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, results_dir / "nn_confusion_matrix.png")
    print(f"Saved confusion matrix plot to: {results_dir / 'nn_confusion_matrix.png'}")

    # Plot training losses
    plot_losses(history, results_dir / "nn_loss.png")
    print(f"Saved training loss plot to: {results_dir / 'nn_loss.png'}")

if __name__ == "__main__":
    main()
