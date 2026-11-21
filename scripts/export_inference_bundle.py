"""Export a deployable train-only Drain3 + Logistic Regression bundle.

This intentionally rebuilds the Seed 11 parser from training-block messages because
the historical experiment saved cluster metadata but not a Drain3 state snapshot.
It does not retrain the classifier. It performs one raw-log pass to reconstruct the
parser state using the same split and committed Drain3 configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.model_selection import train_test_split

from src.preprocessing.log_parser import HDFSLogParser

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/hdfs_full_labeled.csv"
DEFAULT_LOG = ROOT / "data/raw/HDFS/HDFS.log"
DEFAULT_EXPERIMENT = ROOT / "experiments/train_only_drain3/seed_11"
DEFAULT_CONFIG = ROOT / "configs/drain3_train_only.ini"
DEFAULT_OUTPUT = ROOT / "model_artifacts/seed_11"
UNMATCHED_EVENT = "__UNMATCHED__"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_block_ids(frame: pd.DataFrame, seed: int) -> set[str]:
    train_val, test = train_test_split(
        frame.index, test_size=0.2, random_state=seed, stratify=frame["label"]
    )
    train, _validation = train_test_split(
        train_val,
        test_size=0.2 / 0.8,
        random_state=seed,
        stratify=frame.loc[train_val, "label"],
    )
    return set(frame.loc[train, "block_id"].astype(str))


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=DEFAULT_DATA)
    cli.add_argument("--raw-log", type=Path, default=DEFAULT_LOG)
    cli.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    cli.add_argument("--drain3-config", type=Path, default=DEFAULT_CONFIG)
    cli.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--seed", type=int, default=11)
    args = cli.parse_args()

    required = [
        args.data, args.raw_log, args.drain3_config,
        args.experiment_dir / "baseline_models.joblib",
        args.experiment_dir / "metrics.json",
        args.experiment_dir / "drain3_templates.json",
        args.experiment_dir / "experiment_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required local experiment files are missing:\n" + "\n".join(missing))

    frame = pd.read_csv(args.data, dtype={"block_id": "string"})
    frame = frame[["block_id", "label"]].dropna().copy()
    frame["block_id"] = frame["block_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["block_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate block IDs.")

    experiment = json.loads((args.experiment_dir / "experiment_report.json").read_text(encoding="utf-8"))
    if int(experiment["seed"]) != args.seed:
        raise ValueError("Experiment report seed does not match requested seed.")

    templates = json.loads((args.experiment_dir / "drain3_templates.json").read_text(encoding="utf-8"))
    feature_columns = sorted([str(item["event_id"]) for item in templates] + [UNMATCHED_EVENT])
    models = joblib.load(args.experiment_dir / "baseline_models.joblib")
    if "logistic_regression" not in models:
        raise ValueError("Logistic Regression model is missing.")
    classifier = models["logistic_regression"]
    expected_features = int(classifier.n_features_in_)
    if len(feature_columns) != expected_features:
        raise ValueError(
            f"Feature schema mismatch: templates imply {len(feature_columns)} features, "
            f"but classifier expects {expected_features}."
        )
    metrics = json.loads((args.experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["models"]["logistic_regression"]["threshold"])

    config = TemplateMinerConfig()
    config.load(str(args.drain3_config))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "drain3_state.bin"
    state_path.unlink(missing_ok=True)
    miner = TemplateMiner(persistence_handler=FilePersistence(str(state_path)), config=config)

    train_ids = training_block_ids(frame, args.seed)
    scanned = 0
    used = 0
    print(f"Rebuilding Drain3 state for seed {args.seed}; only training-block lines are added.", flush=True)
    with args.raw_log.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            message = raw_line.strip()
            if not message:
                continue
            scanned += 1
            if HDFSLogParser.extract_block_id(message) in train_ids:
                miner.add_log_message(message)
                used += 1
            if scanned % 500_000 == 0:
                print(f"Scanned {scanned:,}; training lines used {used:,}; clusters {len(miner.drain.clusters)}", flush=True)

    miner.save_state("deployment bundle export")
    rebuilt_templates = {
        str(cluster.cluster_id): cluster.get_template()
        for cluster in miner.drain.clusters
    }
    saved_templates = {
        str(item["event_id"]): str(item["template"])
        for item in templates
    }
    if rebuilt_templates != saved_templates:
        raise RuntimeError(
            "Rebuilt Drain3 cluster IDs/templates do not exactly match the "
            "historical experiment metadata. Do not deploy this bundle."
        )

    shutil.copy2(args.experiment_dir / "baseline_models.joblib", args.output_dir / "baseline_models.joblib")
    shutil.copy2(args.experiment_dir / "drain3_templates.json", args.output_dir / "drain3_templates.json")
    shutil.copy2(args.drain3_config, args.output_dir / "drain3_train_only.ini")
    metadata = {
        "bundle_name": "logpilot_hdfs_train_only_seed_11",
        "seed": args.seed,
        "primary_model": "logistic_regression",
        "threshold": threshold,
        "positive_class": 1,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "unmatched_event": UNMATCHED_EVENT,
        "parser_fit_scope": "training block IDs only",
        "inference_method": "TemplateMiner.match; no cluster updates",
        "drain3_config_path": "drain3_train_only.ini",
        "drain3_config_sha256": sha256(args.drain3_config),
        "drain3_state_sha256": sha256(state_path),
        "classifier_sha256": sha256(args.output_dir / "baseline_models.joblib"),
        "dataset_rows": int(len(frame)),
        "parser_fit_raw_lines_scanned": scanned,
        "parser_fit_training_block_lines": used,
        "drain3_cluster_count": len(miner.drain.clusters),
        "historical_test_f1": float(metrics["models"]["logistic_regression"]["f1"]),
        "note": (
            "The classifier is the already-trained historical Seed 11 model. "
            "The Drain3 state is reconstructed from training-block lines using the "
            "committed config; the classifier is not retrained."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Bundle exported to: {args.output_dir}")
    print(f"Parser clusters: {metadata['drain3_cluster_count']} | features: {metadata['feature_count']}")
    print(f"Logistic Regression threshold: {threshold:.10f}")
    print("Classifier was loaded from the completed experiment; no model training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
