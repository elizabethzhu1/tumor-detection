# Basic Neural Network Baseline

Train a small PyTorch MLP to predict `tumor_type` from the 96 SBS count columns
in `data/processed/spectra_counts.csv`.

```bash
.venv/bin/python neural_network_baseline/train_nn.py
```

Quick smoke run:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --labels BRCA SKCM \
  --epochs 5
```

Outputs:

- `outputs/basic_nn_results.json`
- `outputs/basic_nn_model.pt`
- `outputs/basic_nn_accuracy.svg`

The training summary prints accuracy and weighted F1. The JSON metrics also
include macro F1, per-class F1, and the confusion matrix.

The script row-normalizes each sample's 96 raw mutation counts into a
probability spectrum before training.

The MLP uses configurable `Linear -> BatchNorm1d -> ReLU -> Dropout` hidden
blocks. Control depth with:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --num-layers 4 \
  --hidden-dim 256
```

To add a compact bottleneck layer before the classifier:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --bottleneck-dim 16
```

This bottleneck is useful for constraining capacity, but unlike `model.py` it is
not a COSMIC-aligned 96-bin signature layer.

When `--bottleneck-dim` is set, the script also writes bottleneck visualization
artifacts using `--viz-prefix`:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --bottleneck-dim 16 \
  --viz-prefix outputs/basic_nn
```

This saves per-sample bottleneck activations, mean activation by tumor type,
classifier weights from bottleneck units to tumor classes, and approximate
input-attribution patterns for each bottleneck unit.

Training uses inverse-frequency class-weighted cross entropy by default, which
helps smaller tumor classes contribute more to the loss. To disable it:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --no-class-weights
```

Regularization options:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --weight-decay 0.001 \
  --l1-lambda 0.000001 \
  --label-smoothing 0.05
```

`--weight-decay` uses AdamW L2-style regularization, `--l1-lambda` adds an L1
penalty to model parameters, and `--label-smoothing` makes the cross-entropy
target less overconfident.

To append mutation burden as a 97th feature:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --include-mutation-burden
```

This adds `log1p_total_snv_count`, computed from the row sum of the 96 raw count
columns before normalization.

To append protein-altering driver gene mutation flags parsed from cached MAFs:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --include-mutation-burden \
  --include-driver-genes
```

This builds or reuses `data/processed/driver_gene_flags.csv` and appends binary
features such as `TP53_mutated`, `KRAS_mutated`, and `APC_mutated`.

To choose a different plot path:

```bash
.venv/bin/python neural_network_baseline/train_nn.py \
  --plot-out outputs/my_accuracy_curve.svg
```

## Hyperparameter Sweep

Run a broad randomized grid sweep:

```bash
.venv/bin/python neural_network_baseline/sweep_nn.py
```

By default this evaluates up to 96 configurations with early stopping, ranks
models by `0.5 * (val_accuracy + val_weighted_f1)`, and writes artifacts under
`outputs/nn_sweep/`.
