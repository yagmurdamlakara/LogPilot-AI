# LogPilot AI

AI-powered HDFS log anomaly detection and analysis assistant.

## Project scope

LogPilot AI groups HDFS log lines by block ID, parses log messages into event templates with Drain3, creates block-level feature vectors, detects anomalous blocks with classical machine-learning models, and uses an LLM to explain selected anomalous blocks.

The HDFS ground-truth labels are block-level (normal/anomalous). They do not identify the root cause; generated explanations must therefore be presented as possible causes supported by log evidence.

## Repository structure

```text
data/
  raw/          # Downloaded source logs (not committed)
  processed/    # Generated datasets (not committed)
  samples/      # Small local samples (not committed)
src/
  preprocessing/
  feature_extraction/
  models/
  llm/
  app/
tests/
experiments/
reports/
```

## Local setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal session only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Verify the environment:

```powershell
python -c "import numpy, pandas, sklearn, drain3, streamlit; print('Environment OK')"
pytest
```

## Development principles

- Work at the HDFS block level, not individual log-line level.
- Split train/test data by block ID to prevent leakage.
- Fit preprocessing transformations on training data only.
- Report precision, recall, and F1 for the anomalous class; do not rely on accuracy alone.
- Send only selected anomalous block evidence to the LLM, never the entire log file.
- Do not commit raw datasets, generated datasets, credentials, or virtual environments.
