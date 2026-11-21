"""Load a packaged LogPilot model and score HDFS blocks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from src.inference.feature_builder import FrozenBlockFeatureBuilder


class LogPilotPredictor:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        metadata_path = self.artifact_dir / "metadata.json"
        model_path = self.artifact_dir / "baseline_models.joblib"
        state_path = self.artifact_dir / "drain3_state.bin"
        config_path = self.artifact_dir / "drain3_train_only.ini"
        for path in (metadata_path, model_path, state_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required inference artifact not found: {path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = TemplateMinerConfig()
        config.load(str(config_path))
        self.miner = TemplateMiner(
            persistence_handler=FilePersistence(str(state_path)),
            config=config,
        )
        if len(self.miner.drain.clusters) != int(self.metadata["drain3_cluster_count"]):
            raise ValueError("Loaded Drain3 state cluster count does not match metadata.")

        models = joblib.load(model_path)
        self.model = models[self.metadata["primary_model"]]
        self.feature_columns = list(self.metadata["feature_columns"])
        if int(self.model.n_features_in_) != len(self.feature_columns):
            raise ValueError("Model input dimension does not match the feature schema.")
        self.threshold = float(self.metadata["threshold"])

    def predict_lines(self, lines: Iterable[str]) -> tuple[pd.DataFrame, dict[str, int]]:
        builder = FrozenBlockFeatureBuilder(self.miner, self.feature_columns)
        builder.add_lines(lines)
        features = builder.to_frame()
        if features.empty:
            return pd.DataFrame(columns=[
                "block_id", "anomaly_score", "prediction", "prediction_label"
            ]), builder.stats()

        matrix = features.reindex(columns=["block_id", *self.feature_columns])
        scores = self.model.predict_proba(matrix[self.feature_columns])[:, 1]
        predictions = (scores >= self.threshold).astype(int)
        result = pd.DataFrame({
            "block_id": matrix["block_id"].astype(str),
            "anomaly_score": scores,
            "prediction": predictions,
            "prediction_label": ["Anomaly" if value else "Normal" for value in predictions],
        })
        result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        return result, builder.stats()

    @staticmethod
    def extract_block_evidence(
        log_path: str | Path,
        block_ids: set[str],
        *,
        max_lines_per_block: int = 100,
    ) -> dict[str, list[str]]:
        """Second-pass, bounded evidence extraction for selected anomalous blocks."""
        if max_lines_per_block < 1:
            raise ValueError("max_lines_per_block must be positive.")
        evidence = {block_id: [] for block_id in block_ids}
        remaining = set(block_ids)
        with Path(log_path).open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                message = raw_line.strip()
                block_id = __import__(
                    "src.preprocessing.log_parser", fromlist=["HDFSLogParser"]
                ).HDFSLogParser.extract_block_id(message)
                if block_id not in remaining:
                    continue
                evidence[block_id].append(message)
                if len(evidence[block_id]) >= max_lines_per_block:
                    remaining.remove(block_id)
                if not remaining:
                    break
        return evidence
