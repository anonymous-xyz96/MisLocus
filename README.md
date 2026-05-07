# MisLocus — companion repo

Reproduce the protein-localization representation benchmark from the published
**MisLocus single-cell crop dataset**. This repo ships the code needed
*downstream* of feature preprocessing — `features.parquet` files come
pre-computed in the dataset bundle, ready for classification and benchmarking.

## What you get

The dataset bundle (download separately, see below) ships into `data/`:

- **Per-rep features** — `data/interim/{cellprofiler,cytoself,subcell_portable_*,vit}/{batch}/features.parquet`.
- **Crop manifest** — `data/interim/crop_manifest/{batch}/manifest.parquet` (one row per cell).
- **QC'd single-cell crops** (optional) — `data/interim/single_cell_crops/{batch}/{allele}/*.npy` (128×128 uint16, 4 channels: DNA / GFP / AGP / Mito).

Reference annotations (HPA gene-localization + ClinVar / dbNSFP / pLDDT
allele collection) live in `annotations/` and are tracked in this repo
(~9.7 MB total). They're consumed by `10_benchmark_clinvar.py` and
`10b_benchmark_hpa.py` and don't need to be downloaded.

See [`docs/dataset_bundle.md`](docs/dataset_bundle.md) for the full layout
and the download script's CLI options.

## Quickstart

The five recipes you'll actually use (run `just --list` for the full set):

```bash
# 0. Install pixi: https://pixi.sh/install
just install

# 1. Browseable sample (~1.2 GB, no GPU needed) — 8 alleles × 2 batches
#    of single-cell crops, plus a sample manifest.
just download-sample
just inspect-sample

# 2. One batch of CellProfiler features (~3 GB), then classify on it.
just download-batch         # default: cellprofiler / 2025_01_27_Batch_13
just classify-batch         # XGBoost AUROC + copairs PA mAP

# 3. Or pull everything (every rep × every batch + crop tarballs).
just download-all
```

`just download-batch` and `just classify-batch` both accept `BATCH=…` and
`REP=…` overrides (`just classify-batch BATCH=2025_01_28_Batch_14
REP=cytoself`). `just classify-batch` runs `09_classify.py --gpu`
(GPU-required) followed by `09c_classify_PA.py` (CPU); the recipe sets the
two env vars (`CONDA_OVERRIDE_CUDA=12.0`, `CUDA_VERSION=12.0`) the
lab-server's pixi env needs to keep XGBoost on GPU.

If you're going to run the SubCell extractors (`08a` / `08c`), also fetch
the encoder weights:

```bash
just download-subcell       # ~2 GB, CZI's public S3 bucket
```

## Pipeline

```
data/interim/{rep}/{batch}/features.parquet      ← shipped in the bundle
            │
   ┌────────┴────────┐
   ▼                 ▼
09_classify.py    09c_classify_PA.py
(XGBoost AUROC)   (copairs mAP + p95 hit-call)
            │
            ▼
data/processed/classification{,_PA}/{rep}/{batch}/
            │
   ┌────────┴────────┐
   ▼                 ▼
10_benchmark_clinvar.py    10b_benchmark_hpa.py
            │
            ▼
11_summarize_across_reps.py
            │
            ▼
data/processed/benchmark/clinvar/full_dataset/summary_across_reps/
```

## Pixi environments

| Env       | Used for                                                 |
|-----------|----------------------------------------------------------|
| `default` | PA mAP (`09c`), benchmarks (`10` / `10b` / `11`). CPU only. |
| `gpu`     | XGBoost classification (`09 --gpu`). Inherits `default` + adds the CUDA-12 system requirement so conda-forge resolves to GPU-enabled xgboost. |

The `cytoself` / `subcell` / `vit` envs only matter if you re-extract or
retrain a representation from raw crops; they're not needed for the
classification + benchmark workflow.

## Adding a new representation

The benchmark is a contract — anything that writes
`data/interim/<myrep>/{batch}/features.parquet` with the right schema
plugs in without touching `09 / 09c / 10 / 10b / 11`. See
[`docs/dataset_bundle.md#adding-a-new-representation`](docs/dataset_bundle.md#adding-a-new-representation)
for the integration recipe.
