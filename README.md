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

## Full HDFS_v1 preprocessing (memory-bounded)

Keep the raw HDFS_v1 files under `data/raw/HDFS/`. Do not commit the downloaded
dataset. The full-log pipeline keeps one Drain3 miner alive for the complete
file and stores block/event counts in a temporary SQLite database instead of
holding all parsed log lines in RAM.

Run from the repository root in the activated virtual environment:

```powershell
python -m src.preprocessing.full_dataset data/raw/HDFS/HDFS.log
```

Default outputs:

- `data/processed/hdfs_full_block_features.csv`: one row per block and one
  count column per Drain3 event cluster.
- `data/processed/hdfs_full_event_templates.csv`: event IDs and their latest
  mined templates.
- `data/processed/.hdfs_feature_work.sqlite3`: temporary work database,
  removed after successful export.

Then merge the official block-level ground-truth labels:

```powershell
python -m src.preprocessing.label_merger data/processed/hdfs_full_block_features.csv data/raw/HDFS/preprocessed/anomaly_label.csv --output data/processed/hdfs_full_labeled.csv
```

The parser reports processed lines, lines containing block IDs, unique blocks,
unique event clusters, and the total feature count. It prints progress every
500,000 non-empty lines by default. Adjust `--batch-size` or
`--progress-every` if needed.

The Loghub-provided `Event_occurrence_matrix.csv` is a reference artifact,
not the output of our Drain3 parser. Its E1-E29 identifiers must not be assumed
to match Drain3 cluster IDs. Use it for a separate reference comparison.

## Run the Streamlit application

After exporting the local Seed 11 inference bundle:

```powershell
python -m scripts.export_inference_bundle
streamlit run src/app/streamlit_app.py
```

Open the local Streamlit URL shown in the terminal, upload an HDFS `.log` or
`.txt` file, and select **Logları analiz et**. The interface displays block-level
anomaly scores, lets you download the anomaly list as CSV, and shows up to 100
raw log lines for a selected anomalous block. The **LLM açıklama** tab can generate
a structured explanation for one selected anomalous block using the local Ollama
service (default model: `qwen3:4b`). It displays possible causes, selected source
log lines, recommended checks, limitations, and a conservative confidence level.
Only that block's bounded raw evidence is sent to the LLM; the classifier prediction
and anomaly score remain unchanged. Start Ollama separately before requesting an
explanation. If Ollama is unavailable, anomaly detection and the rest of the
dashboard remain usable.

The model bundle under `model_artifacts/seed_11/` is generated locally and is
not committed to Git. The app uses the frozen train-only Drain3 parser and the
Seed 11 Logistic Regression classifier. LLM-based root-cause explanations are
not part of this first UI increment; detected anomalies are not, by themselves,
proof of a root cause.
