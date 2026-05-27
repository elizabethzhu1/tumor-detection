# CosmicNet MVP

Tumor-type classification from 96-channel trinucleotide mutational spectra,
with a bottleneck neural network and COSMIC signature alignment evaluation.

## What's in here

```
mutation_categories.py   # 96-category definitions + mutation classification
maf_processing.py        # GDC MAF parsing
tcga_gdc.py              # TCGA MAF discovery/download through the GDC API
cosmic.py                # COSMIC signature loading + NNLS exposure fitting
processed_data.py        # Shared processed spectra cache loading/building
process_data.py          # One-time raw MAF -> processed spectra command
model.py                 # Bottleneck NN (Linear + softplus weights + softmax)
evaluation.py            # Metrics + Hungarian-matched COSMIC alignment + null
run_pipeline.py          # End-to-end orchestration
neural_network_baseline/ # Basic MLP classifier on processed spectra counts
data/
  processed/
    spectra_counts.csv    # Shared per-sample 96-d SNV count spectra
outputs/
  results.json            # Metrics and learned weights from a completed run
```

## Process Data Once

Build the shared processed spectra table from cached MAF files:

```bash
python process_data.py
```

This writes `data/processed/spectra_counts.csv`. The pipeline and standalone
baselines read that file by default, so they do not reparse the raw `.maf.gz`
files on every run.

## Run The Pipeline

```bash
python run_pipeline.py \
  --cosmic-path data/COSMIC_Human_SBS-96_GRCh38_v3.6.csv
```

By default, this uses `data/processed/spectra_counts.csv`. If that file is
missing, it builds it once from cached MAF files under `data/tcga_mafs/`.

This trains and evaluates:
- Logistic regression on raw 96-d spectra
- Logistic regression on COSMIC exposures
- XGBoost on raw spectra
- Bottleneck NN for K in {4, 6, 8, 12, 16}

And reports per-model accuracy/F1 plus Hungarian-matched COSMIC alignment
against a random-vector null distribution.

## Basic Neural Network Baseline

```bash
python neural_network_baseline/train_nn.py
```

This trains a small MLP to predict `tumor_type` from the processed 96-channel
spectra counts.

## Run on TCGA MAFs

The pipeline can download public TCGA Masked Somatic Mutation MAFs from GDC,
extract SNVs, build per-sample 96-dimensional SBS spectra, and train the
models:

```bash
python run_pipeline.py \
  --download-tcga-mafs \
  --cosmic-path data/COSMIC_Human_SBS-96_GRCh38_v3.6.csv
```

By default this downloads:
- `TCGA-SKCM` as `SKCM`
- `TCGA-LUAD` as `LUAD`
- `TCGA-BRCA` as `BRCA`
- `TCGA-UCEC` as `UCEC`
- `TCGA-COAD` as `COAD`

These defaults target tumor types with strong expected mutational signatures:
SKCM/UV (`SBS7`), LUAD/smoking (`SBS4`), BRCA/APOBEC-HRD-aging
(`SBS2/13/3/1/5`), UCEC/POLE-MMR (`SBS10/6`), and COAD/MMR-POLE.

Downloaded MAFs are cached under `data/tcga_mafs/`. The shared 96-dimensional
count matrix is written to `data/processed/spectra_counts.csv`; metrics and
learned bottleneck weights are written to `outputs/results.json`.

To use already downloaded MAFs instead of querying GDC:

```bash
python run_pipeline.py \
  --cosmic-path data/COSMIC_Human_SBS-96_GRCh38_v3.6.csv \
  --maf-paths BRCA=data/tcga_mafs/BRCA/example.maf.gz SKCM=data/tcga_mafs/SKCM/example.maf.gz
```

Repeat the same label to combine multiple MAFs for one tumor type.

Useful options:
- `--tcga-projects TCGA-LUAD:LUAD TCGA-LUSC:LUSC` changes the downloaded cohorts.
- `--force-download` redownloads cached MAFs.
- `--processed-data path/to/spectra_counts.csv` chooses the shared processed table.
- `--reprocess-data` reparses cached MAFs and overwrites `--processed-data`.
- `--max-files-per-project 20` limits cached/downloaded MAFs while building processed data.
- `--download-retries 5` retries transient GDC server errors.
- `--skip-failed-downloads` continues if one GDC file keeps returning an error.
- `--K-values 4 8 16` controls the bottleneck widths.
- `--epochs 500` controls neural-network training length.
- `--spectra-out path/to/counts.csv` writes an optional extra copy of the 96-d counts.

## COSMIC signatures

For real COSMIC exposure baselines and alignment, download the GRCh38 SBS v3.x
TSV from https://cancer.sanger.ac.uk/signatures/sbs/ and pass it with
`--cosmic-path`. A real COSMIC signature file is required.

## Current State

- **No reference-genome lookup.** Uses the GDC MAF `CONTEXT` column.
- **No hyperparameter search.** Just sensible defaults.
- **No batch training.** Full-batch Adam, fine for ~10^3 samples × 96 features.
- **No early stopping plot.** Best val checkpoint is saved internally.
- **No visualization.** Plot the learned signatures by reading
  `results.json["bottleneck_nn"][K]["learned_weights"]` as 96-bar plots.
# tumor-detection
