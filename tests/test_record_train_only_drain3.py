from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import record_train_only_drain3 as recorder


def make_inputs(root: Path) -> None:
    (root / "experiments/train_only_drain3").mkdir(parents=True)
    (root / "reports/model_evaluation").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    seeds = []
    audit_seeds = []
    for seed in recorder.EXPECTED_SEEDS:
        seed_dir = root / "experiments/train_only_drain3" / f"seed_{seed}"
        seed_dir.mkdir()
        for name in recorder.REQUIRED_ARTIFACTS:
            (seed_dir / name).write_bytes(f"{seed}:{name}".encode())
        seeds.append(
            {
                "seed": seed,
                "train_blocks": 1,
                "validation_blocks": 1,
                "test_blocks": 1,
                "parser_fit": {"trained_clusters": 2},
                "frozen_feature_build": {
                    "matched_log_lines": 10,
                    "unmatched_log_lines": 1,
                },
                "unmatched_fraction": 0.1,
                "metrics": {"example": {"f1": 0.9}},
            }
        )
        audit_seeds.append({"seed": seed, "status": "PASS"})
    (root / "experiments/train_only_drain3/summary.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    (root / "reports/model_evaluation/train_only_drain3_split_audit.json").write_text(
        json.dumps({"status": "PASS", "rows": 5, "seeds": audit_seeds}),
        encoding="utf-8",
    )
    (root / "configs/drain3_train_only.ini").write_text("[DRAIN]\r\nsim_th=0.4\r\n", encoding="utf-8")
    (root / "data/processed/hdfs_full_labeled.csv").write_bytes(b"block_id,label\r\na,0\r\n")


def point_recorder_at(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recorder, "ROOT", root)
    monkeypatch.setattr(recorder, "SUMMARY", root / "experiments/train_only_drain3/summary.json")
    monkeypatch.setattr(
        recorder,
        "AUDIT",
        root / "reports/model_evaluation/train_only_drain3_split_audit.json",
    )
    monkeypatch.setattr(recorder, "CONFIG", root / "configs/drain3_train_only.ini")
    monkeypatch.setattr(recorder, "DATASET", root / "data/processed/hdfs_full_labeled.csv")
    monkeypatch.setattr(
        recorder, "OUTPUT", root / "reports/model_evaluation/train_only_drain3_manifest.json"
    )
    monkeypatch.setattr(recorder, "command_output", lambda *args, **kwargs: "test-value")
    monkeypatch.setattr(recorder, "package_versions", lambda: {"drain3": "test"})


def test_file_record_streams_hash_and_size_in_same_read(tmp_path: Path, monkeypatch) -> None:
    content = b"0123456789abcdef" * 32
    path = tmp_path / "payload.bin"
    path.write_bytes(content)
    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "CHUNK_SIZE", 13)

    record = recorder.file_record(path)

    assert record == {
        "path": "payload.bin",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_build_manifest_records_dataset_and_five_artifacts_per_seed(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)

    manifest = recorder.build_manifest()

    assert manifest["schema_version"] == 2
    dataset = tmp_path / "data/processed/hdfs_full_labeled.csv"
    assert manifest["dataset"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert [item["seed"] for item in manifest["seed_results"]] == [11, 22, 33, 44, 55]
    for result in manifest["seed_results"]:
        assert len(result["artifacts"]) == 5
        assert {Path(item["path"]).name for item in result["artifacts"]} == set(
            recorder.REQUIRED_ARTIFACTS
        )
        assert all(item["size_bytes"] > 0 and len(item["sha256"]) == 64 for item in result["artifacts"])
    assert manifest["drain3_configuration"]["historical_run"]["ini_loaded"] is False
    assert manifest["drain3_configuration"]["historical_run"]["effective_defaults"]["similarity_threshold"] == 0.4
    defaults = manifest["drain3_configuration"]["historical_run"]["effective_defaults"]
    assert defaults["profiling_report_sec"] == 60
    assert defaults["mask_prefix"] == "<"
    assert defaults["mask_suffix"] == ">"
    assert manifest["drain3_configuration"]["future_run"]["path"] == "configs/drain3_train_only.ini"
    assert "historical experiment run time" in manifest["limitations"][2]
    assert "manifest_sha256" not in json.dumps(manifest)


def test_missing_artifact_fails_explicitly(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    missing = tmp_path / "experiments/train_only_drain3/seed_33/drain3_templates.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="drain3_templates.json"):
        recorder.build_manifest()


def test_unreadable_artifact_fails_explicitly(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    target = tmp_path / "experiments/train_only_drain3/seed_11/metrics.json"
    original_open = Path.open

    def fail_target_open(path, *args, **kwargs):
        if path == target:
            raise PermissionError("test read denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_target_open)
    with pytest.raises(OSError, match="Could not read required file"):
        recorder.build_manifest()


def update_seed_list(path: Path, seeds: list[int]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["seeds"]
    by_seed = {item["seed"]: item for item in entries}
    payload["seeds"] = [by_seed.get(seed, {"seed": seed, "status": "PASS"}) for seed in seeds]
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("source", ["summary", "audit"])
@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ([11, 22, 33, 44, 44], "duplicate"),
        ([11, 22, 33, 44], "missing"),
        ([11, 22, 33, 44, 66], "unexpected"),
    ],
)
def test_seed_lists_reject_duplicates_missing_and_unexpected(
    tmp_path, monkeypatch, source, seeds, message
) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    path = recorder.SUMMARY if source == "summary" else recorder.AUDIT
    update_seed_list(path, seeds)

    with pytest.raises(ValueError, match=message):
        recorder.build_manifest()


@pytest.mark.parametrize("source", ["summary", "audit"])
@pytest.mark.parametrize("invalid_seed", [11.0, "11", True])
def test_seed_lists_require_exact_integer_type(tmp_path, monkeypatch, source, invalid_seed) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    path = recorder.SUMMARY if source == "summary" else recorder.AUDIT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seeds"][0]["seed"] = invalid_seed
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integer seed"):
        recorder.build_manifest()


def test_seed_output_order_is_deterministic(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    for path in (recorder.SUMMARY, recorder.AUDIT):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seeds"].reverse()
        path.write_text(json.dumps(payload), encoding="utf-8")

    assert [item["seed"] for item in recorder.build_manifest()["seed_results"]] == [11, 22, 33, 44, 55]


def test_malformed_json_fails_explicitly(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    recorder.SUMMARY.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="valid JSON"):
        recorder.build_manifest()


@pytest.mark.parametrize("source", ["summary", "audit"])
def test_json_top_level_must_be_object(tmp_path, monkeypatch, source) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    path = recorder.SUMMARY if source == "summary" else recorder.AUDIT
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        recorder.build_manifest()


@pytest.mark.parametrize(
    ("source", "mutate", "message"),
    [
        ("summary", lambda value: value.pop("seeds"), "seeds list"),
        ("summary", lambda value: value.update(seeds={}), "seeds list"),
        ("audit", lambda value: value.pop("rows"), "rows must be an integer"),
        ("audit", lambda value: value.update(rows="5"), "rows must be an integer"),
        ("audit", lambda value: value.update(status=1), "status must be a string"),
        (
            "summary_seed",
            lambda value: value["seeds"][0].pop("parser_fit"),
            "parser_fit.trained_clusters",
        ),
        (
            "summary_seed",
            lambda value: value["seeds"][0].update(metrics=[]),
            "metrics must be an object",
        ),
    ],
)
def test_required_json_fields_and_types_are_validated(tmp_path, monkeypatch, source, mutate, message) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    if source == "summary_seed":
        path = recorder.SUMMARY
    else:
        path = recorder.SUMMARY if source == "summary" else recorder.AUDIT
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        recorder.build_manifest()


def test_existing_manifest_is_never_overwritten(tmp_path, monkeypatch) -> None:
    point_recorder_at(tmp_path, monkeypatch)
    existing = tmp_path / "reports/model_evaluation/existing.json"
    existing.parent.mkdir(parents=True)
    original = b"keep this manifest exactly\r\n"
    existing.write_bytes(original)
    monkeypatch.setattr(recorder, "build_manifest", lambda: pytest.fail("must stop before building"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        recorder.main(["--output", str(existing)])

    assert existing.read_bytes() == original


def test_empty_working_tree_status_is_recordable(monkeypatch) -> None:
    monkeypatch.setattr(recorder.subprocess, "check_output", lambda *args, **kwargs: "\r\n")

    assert recorder.command_output("git", "status", "--short", allow_empty=True) == ""


@pytest.mark.parametrize("command", ["branch", "head", "status"])
def test_git_metadata_command_failure_stops_manifest(tmp_path, monkeypatch, command) -> None:
    make_inputs(tmp_path)
    real_command_output = recorder.command_output
    point_recorder_at(tmp_path, monkeypatch)
    monkeypatch.setattr(recorder, "command_output", real_command_output)
    original_check_output = recorder.subprocess.check_output

    def fail_selected(args, **kwargs):
        if (command == "branch" and args[-1] == "--show-current") or (
            command == "head" and args[-1] == "HEAD"
        ) or (command == "status" and "status" in args):
            raise recorder.subprocess.CalledProcessError(1, args)
        return original_check_output(args, **kwargs)

    monkeypatch.setattr(recorder.subprocess, "check_output", fail_selected)
    with pytest.raises(RuntimeError, match="Could not capture source-control metadata"):
        recorder.build_manifest()


@pytest.mark.parametrize("command", ["branch", "head"])
def test_required_git_metadata_empty_output_stops_manifest(tmp_path, monkeypatch, command) -> None:
    make_inputs(tmp_path)
    real_command_output = recorder.command_output
    point_recorder_at(tmp_path, monkeypatch)
    monkeypatch.setattr(recorder, "command_output", real_command_output)

    def empty_selected(args, **kwargs):
        if (command == "branch" and args[-1] == "--show-current") or (
            command == "head" and args[-1] == "HEAD"
        ):
            return "\r\n"
        return "value\r\n"

    monkeypatch.setattr(recorder.subprocess, "check_output", empty_selected)
    with pytest.raises(RuntimeError, match="empty output"):
        recorder.build_manifest()


def test_relative_output_path_is_rooted_at_repository(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)

    assert recorder.main(["--output", "generated/manifest.json"]) == 0
    assert (tmp_path / "generated/manifest.json").is_file()
    assert not recorder.OUTPUT.exists()


@pytest.mark.parametrize("output", ["../escape.json"])
def test_output_outside_repository_is_rejected(tmp_path, monkeypatch, output) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_inputs(root)
    point_recorder_at(root, monkeypatch)
    monkeypatch.setattr(recorder, "build_manifest", lambda: pytest.fail("must reject output first"))

    with pytest.raises(ValueError, match="inside repository root"):
        recorder.main(["--output", output])
    assert not (tmp_path / "escape.json").exists()


def test_read_error_after_open_is_reported(tmp_path, monkeypatch) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"long enough for multiple reads")
    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "CHUNK_SIZE", 4)
    original_open = Path.open

    class FailingReader:
        def __init__(self, source):
            self.source = source
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def read(self, size):
            self.read_count += 1
            if self.read_count == 2:
                raise OSError("simulated mid-stream failure")
            return self.source.read(size)

    def fail_during_read(candidate, *args, **kwargs):
        opened = original_open(candidate, *args, **kwargs)
        return FailingReader(opened) if candidate == path else opened

    monkeypatch.setattr(Path, "open", fail_during_read)
    with pytest.raises(OSError, match="Could not read required file"):
        recorder.file_record(path)


def test_concurrent_manifest_creation_does_not_overwrite(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    output = tmp_path / "reports/model_evaluation/race.json"
    original = Path.open
    competitor_bytes = b"created by another process"

    def create_competitor_before_exclusive_open(candidate, mode="r", *args, **kwargs):
        if candidate == output and mode == "x":
            candidate.write_bytes(competitor_bytes)
        return original(candidate, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", create_competitor_before_exclusive_open)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        recorder.main(["--output", str(output)])
    assert output.read_bytes() == competitor_bytes


def test_main_writes_only_requested_temporary_output(tmp_path, monkeypatch) -> None:
    make_inputs(tmp_path)
    point_recorder_at(tmp_path, monkeypatch)
    output = tmp_path / "new-output/manifest.json"

    assert recorder.main(["--output", str(output)]) == 0
    raw_output = output.read_bytes()
    assert b"\r" not in raw_output
    manifest = json.loads(raw_output.decode("utf-8"))

    assert manifest["schema_version"] == 2
    assert set(manifest) == {
        "schema_version",
        "experiment",
        "recorded_at_utc",
        "source_control",
        "environment",
        "dataset",
        "split_audit",
        "drain3_configuration",
        "experiment_settings",
        "seed_results",
        "limitations",
    }
    assert set(manifest["dataset"]) == {
        "path",
        "size_bytes",
        "sha256",
        "rows",
        "modified_time",
        "identity_scope",
    }
    assert set(manifest["source_control"]) == {
        "repository",
        "branch",
        "commit",
        "working_tree_status",
    }
    assert set(manifest["environment"]) == {
        "python",
        "python_executable",
        "platform",
        "machine",
        "packages",
    }
    assert set(manifest["split_audit"]) == {
        "status",
        "seed_count",
        "all_seed_statuses",
        "report_path",
    }
    assert set(manifest["drain3_configuration"]) == {"historical_run", "future_run"}
    assert set(manifest["drain3_configuration"]["historical_run"]) == {
        "drain3_version",
        "config_source",
        "ini_loaded",
        "effective_defaults",
        "note",
    }
    assert set(manifest["drain3_configuration"]["future_run"]) == {
        "path",
        "size_bytes",
        "sha256",
        "text",
    }
    assert set(manifest["experiment_settings"]) == {
        "seeds",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "split_strategy",
        "parser_fit_scope",
        "validation_test_parser_behavior",
        "unmatched_message_handling",
        "standard_deviation",
    }
    assert [result["seed"] for result in manifest["seed_results"]] == [11, 22, 33, 44, 55]
    assert all(
        set(result)
        == {
            "seed",
            "train_blocks",
            "validation_blocks",
            "test_blocks",
            "trained_drain_clusters",
            "matched_log_lines",
            "unmatched_log_lines",
            "unmatched_fraction",
            "metrics",
            "artifacts",
        }
        for result in manifest["seed_results"]
    )
    assert all(
        {Path(artifact["path"]).name for artifact in result["artifacts"]}
        == set(recorder.REQUIRED_ARTIFACTS)
        for result in manifest["seed_results"]
    )
    assert all(
        set(artifact) == {"path", "size_bytes", "sha256"}
        for result in manifest["seed_results"]
        for artifact in result["artifacts"]
    )
    assert not recorder.OUTPUT.exists()
