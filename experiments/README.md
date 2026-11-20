# Baseline experiment artifacts

This directory contains reproducibility metadata and aggregate results for the
HDFS full-dataset baseline experiments.

## Multi-seed evaluation

The five-seed evaluation uses seeds **11, 22, 33, 44, 55**. Each run uses:

- 575,061 labeled blocks and 45 features
- 60% training, 20% validation, 20% test split
- 345,036 training blocks, 115,012 validation blocks, 115,013 test blocks
- Metrics reported from each run's held-out test split

The aggregate table is at `summary/multi_seed_results.csv`. It contains the
mean and sample standard deviation (ddof=1) for precision, recall, F1-score,
and test inference time per block.

Regenerate the table locally from the original per-seed JSON files:

```powershell
python scripts/summarize_multiseed.py
```

The script expects the original run files at
`experiments/baseline_seed_<seed>/metrics.json`.

## Artifact policy

- Keep compact metrics JSON files and aggregate tables under version control.
- Keep large prediction CSVs and serialized model files in a separate experiment
  archive or managed artifact storage unless there is a specific reason to
  version them.
- The initial `baseline` run (seed 42) and `threshold_tuning` run are separate
  experiments; they are not included in the five-seed aggregate.
