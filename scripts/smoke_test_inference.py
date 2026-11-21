"""Run a small-file smoke test against a locally exported inference bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.predictor import LogPilotPredictor

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--log", type=Path, default=ROOT / "data/samples/HDFS_2k.log")
    cli.add_argument("--artifacts", type=Path, default=ROOT / "model_artifacts/seed_11")
    args = cli.parse_args()
    if not args.log.is_file():
        raise FileNotFoundError(f"Sample log not found: {args.log}")

    predictor = LogPilotPredictor(args.artifacts)
    with args.log.open("r", encoding="utf-8", errors="replace") as source:
        predictions, stats = predictor.predict_lines(source)

    print("Inference smoke test")
    print(f"Log lines read: {stats['lines_seen']:,}")
    print(f"Lines with block IDs: {stats['lines_with_block_ids']:,}")
    print(f"Blocks scored: {stats['blocks_with_features']:,}")
    print(f"Unmatched lines: {stats['unmatched_lines']:,}")
    print(f"Predicted anomalies: {int(predictions['prediction'].sum()) if not predictions.empty else 0:,}")
    if not predictions.empty:
        print("\nTop 10 blocks by anomaly score:")
        print(predictions.head(10).to_string(index=False))
    print("\nPASS: bundle loaded and inference completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
