"""
Evaluation utilities:
  - Classification metrics (accuracy, macro-F1, per-class F1, confusion matrix)
  - Hungarian-matched cosine similarity between learned bottleneck rows and
    COSMIC signatures
  - Null distribution for the COSMIC alignment score
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)


def classification_metrics(y_true, y_pred, class_names=None):
    """Return a dict with the main classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names,
                                         output_dict=True, zero_division=0),
    }


def cosine_similarity_matrix(A, B):
    """
    A: (n_a, d), B: (n_b, d). Returns (n_a, n_b) cosine similarity matrix.
    """
    a_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def hungarian_match_to_cosmic(learned_weights, cosmic_signatures, cosmic_names):
    """
    Match each of the K learned rows to a distinct COSMIC signature that
    maximizes total cosine similarity (Hungarian algorithm).

    Args:
        learned_weights: (K, 96), the first-layer weight matrix
        cosmic_signatures: (96, n_sig)
        cosmic_names: list of n_sig signature names
    Returns:
        list of dicts: [{learned_idx, cosmic_idx, cosmic_name, cosine_sim}, ...]
        plus 'mean_matched_similarity'.
    """
    learned = learned_weights  # (K, 96)
    cosmic = cosmic_signatures.T  # (n_sig, 96)
    sim = cosine_similarity_matrix(learned, cosmic)  # (K, n_sig)

    # Hungarian maximizes by negating cost.
    row_ind, col_ind = linear_sum_assignment(-sim)
    matches = [
        {
            "learned_idx": int(r),
            "cosmic_idx": int(c),
            "cosmic_name": cosmic_names[c],
            "cosine_sim": float(sim[r, c]),
        }
        for r, c in zip(row_ind, col_ind)
    ]
    return {
        "matches": matches,
        "mean_matched_similarity": float(np.mean([m["cosine_sim"] for m in matches])),
        "full_similarity_matrix": sim,
    }


def null_alignment_score(K, cosmic_signatures, cosmic_names, n_trials=200,
                         alpha_dirichlet=0.5, seed=0):
    """
    Estimate the expected mean matched similarity if learned weights were random
    nonneg vectors with similar sparsity to COSMIC signatures. Lets us report
    whether the model's alignment exceeds chance.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_trials):
        random_w = rng.dirichlet(alpha=np.ones(96) * alpha_dirichlet, size=K)
        out = hungarian_match_to_cosmic(random_w, cosmic_signatures, cosmic_names)
        scores.append(out["mean_matched_similarity"])
    scores = np.array(scores)
    return {
        "null_mean": float(scores.mean()),
        "null_std": float(scores.std()),
        "null_p95": float(np.percentile(scores, 95)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    learned = rng.dirichlet(np.ones(96) * 0.5, size=8)
    cosmic = rng.dirichlet(np.ones(96) * 0.3, size=20).T
    names = [f"SBS{i+1}" for i in range(20)]
    res = hungarian_match_to_cosmic(learned, cosmic, names)
    print(f"Mean matched cosine similarity: {res['mean_matched_similarity']:.3f}")
    for m in res["matches"][:3]:
        print(f"  learned[{m['learned_idx']}] <-> {m['cosmic_name']}: {m['cosine_sim']:.3f}")

    null = null_alignment_score(K=8, cosmic_signatures=cosmic,
                                 cosmic_names=names, n_trials=50)
    print(f"Null mean: {null['null_mean']:.3f} +/- {null['null_std']:.3f}")