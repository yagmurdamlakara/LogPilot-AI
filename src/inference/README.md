# Inference bundle and smoke test

The deployment bundle is generated locally from the completed Seed 11 train-only
experiment. Model binaries and parser state are intentionally not committed.

## 1. Export the bundle

Run from the repository root in the activated project virtual environment:

```powershell
git pull origin main
python -m scripts.export_inference_bundle
```

The exporter performs one pass over the raw HDFS log to reconstruct the Drain3
state from **training-block lines only**. It does not retrain Logistic Regression.
It verifies that the rebuilt Drain3 cluster IDs and templates exactly match the
historical Seed 11 template metadata. If they do not match, export stops.

Default local output:

```text
model_artifacts/seed_11/
  baseline_models.joblib
  drain3_state.bin
  drain3_templates.json
  drain3_train_only.ini
  metadata.json
```

Do not commit this directory: it contains generated model and parser artifacts.

## 2. Run the small-log smoke test

```powershell
python -m scripts.smoke_test_inference
```

By default this uses `data/samples/HDFS_2k.log`. It loads the saved parser state,
matches messages without changing the Drain3 clusters, builds block event-count
vectors in the exact saved feature order, and scores them with the Seed 11
Logistic Regression model and its validation-selected threshold.

## 3. Run tests

```powershell
pytest tests/test_inference.py
```

The smoke test is a functional integration check, not a new performance
evaluation. The published five-seed metrics remain the evaluation results.

## Important model boundary

The exported model is the existing Seed 11 Logistic Regression classifier.
Its test F1 from the completed train-only experiment is recorded in
`metadata.json`. The parser state is reconstructed from the same training
block split and committed Drain3 configuration, then checked against the
saved cluster/template metadata. No test labels are used by the inference
pipeline.
