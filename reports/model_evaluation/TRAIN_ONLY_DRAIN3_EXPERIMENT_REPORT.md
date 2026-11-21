# Train-Only Drain3 Experiment Report

## 1. Executive summary

This experiment evaluates HDFS block anomaly detection with an **inductive, train-only Drain3 parser**. Unlike the earlier full-corpus parsing workflow, Drain3 is fitted only on raw log lines belonging to the training blocks. The resulting parser is then frozen: feature construction for all labelled blocks uses `TemplateMiner.match()`, which does not add or update clusters. Messages that do not match a learned cluster are assigned to a dedicated `__UNMATCHED__` event feature.

The experiment was completed for five random seeds (11, 22, 33, 44, 55). The split audit passed for all five seeds. Across the five runs, Logistic Regression achieved a mean test F1-score of **0.990725 ± 0.006098**, while Isolation Forest achieved **0.684958 ± 0.004165**. Values are mean ± sample standard deviation (`ddof=1`).

## 2. Dataset

| Property | Value |
|---|---:|
| Dataset | `data/processed/hdfs_full_labeled.csv` |
| Labelled blocks | 575,061 |
| Dataset file size at manifest capture | 67,505,299 bytes |
| Features in the original baseline dataset | 45 |
| Normal blocks (label 0) | 558,223 |
| Anomaly blocks (label 1) | 16,838 |

The dataset contains one row per block ID. The dataset audit previously reported no null or duplicate block IDs, no invalid labels, no missing/non-numeric feature values, and no suspicious feature names.

## 3. Experimental methodology

### 3.1 Block-level data split

Each seed uses a random, stratified block-level split:

| Subset | Fraction | Blocks |
|---|---:|---:|
| Training | 60% | 345,036 |
| Validation | 20% | 115,012 |
| Test | 20% | 115,013 |
| **Total** | **100%** | **575,061** |

The split is stratified by the binary label. Seeds 11, 22, 33, 44, and 55 produce separate partitions. The split is random rather than chronological.

### 3.2 Train-only Drain3 fitting

For each seed:

1. The block IDs belonging to the training partition are reconstructed.
2. The raw HDFS log is scanned, and Drain3 receives only log messages whose block IDs belong to the training partition.
3. The learned Drain3 parser is frozen.
4. The raw log is scanned again. Messages belonging to labelled blocks are mapped using `TemplateMiner.match()`; this matching pass does not create or update clusters.
5. Unmatched messages are assigned to the `__UNMATCHED__` event bucket.
6. Event counts are aggregated by block ID to form the block-level feature matrix.
7. The existing baseline training/evaluation routine is applied using the same seed and split fractions.

The parser is therefore not fitted on validation or test log messages. The validation and test messages are only matched against the parser learned from training-block messages.

### 3.3 Models and evaluation

The experiment retains the existing baseline model implementations:

- **Isolation Forest**
- **Logistic Regression**

The existing baseline routine uses the validation partition for Logistic Regression threshold selection and evaluates the selected threshold on the held-out test partition. Test predictions and metrics are saved separately for each seed.

## 4. Drain3 parsing results

| Seed | Learned Drain3 clusters | Unmatched log messages |
|---:|---:|---:|
| 11 | 46 | 4 |
| 22 | 45 | 2 |
| 33 | 45 | 5 |
| 44 | 44 | 17 |
| 55 | 45 | 4 |
| **Total** | — | **32** |

Each seed's frozen feature-building pass scanned 11,175,629 raw log lines and found 11,175,629 labelled block log lines. Across the five runs, 32 messages were unmatched out of 55,878,145 labelled block log lines (approximately **0.000057%**). The unmatched rate is therefore very small; it is not exactly zero.

The event feature count includes the learned clusters and the dedicated unmatched bucket. For example, Seed 11 has 46 learned clusters and an additional `__UNMATCHED__` feature.

## 5. Test performance

### 5.1 Per-seed F1-score

| Seed | Isolation Forest F1 | Logistic Regression F1 |
|---:|---:|---:|
| 11 | 0.691285 | 0.995262 |
| 22 | 0.686511 | 0.991589 |
| 33 | 0.680821 | 0.996004 |
| 44 | 0.684275 | 0.980775 |
| 55 | 0.681898 | 0.989997 |

### 5.2 Five-seed aggregate metrics

All values below are mean ± sample standard deviation (`ddof=1`).

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Isolation Forest | 0.835355 ± 0.011007 | 0.580582 ± 0.008566 | 0.684958 ± 0.004165 |
| Logistic Regression | 0.982952 ± 0.012447 | 0.998694 ± 0.000881 | 0.990725 ± 0.006098 |

### 5.3 Interpretation

Logistic Regression provides substantially higher recall and F1-score than Isolation Forest on these test partitions. Its mean recall is close to 1.0, while its mean precision is also high. Isolation Forest has lower recall, meaning that it misses a larger share of anomalous blocks at the selected operating point.

The five-seed standard deviations indicate that the reported performance is reasonably consistent across the tested random partitions, although Logistic Regression shows a larger F1 variation than Isolation Forest in absolute terms. The per-seed results, rather than the mean alone, should be retained when discussing this variation.

## 6. Comparison with the earlier full-corpus Drain3 baseline

The earlier five-seed baseline parsed the full HDFS log corpus before the block-level model split. That parser therefore had exposure to messages from blocks later assigned to validation and test partitions. This is an unsupervised, transductive preprocessing setup; it is methodologically different from the train-only parser evaluated here.

| Model | Metric | Earlier full-corpus mean | Train-only mean | Difference (train-only − earlier) |
|---|---|---:|---:|---:|
| Isolation Forest | Precision | 0.814006 | 0.835355 | +0.021349 |
| Isolation Forest | Recall | 0.592043 | 0.580582 | −0.011461 |
| Isolation Forest | F1-score | 0.685290 | 0.684958 | −0.000332 |
| Logistic Regression | Precision | 0.979603 | 0.982952 | +0.003349 |
| Logistic Regression | Recall | 0.998931 | 0.998694 | −0.000237 |
| Logistic Regression | F1-score | 0.989133 | 0.990725 | +0.001592 |

The train-only experiment does **not** show a meaningful F1-score decrease relative to the earlier full-corpus baseline: the Isolation Forest mean F1 changes by approximately −0.000332, and Logistic Regression mean F1 changes by approximately +0.001592. This suggests that fitting Drain3 on the full corpus did not materially inflate the measured F1-score in these experiments.

This comparison is evidence about the specific parser-exposure concern tested here. It is not proof that every possible source of data leakage or dependence has been eliminated.

## 7. Split integrity audit

The dedicated train-only split audit returned **PASS for all five seeds**.

For each seed, the audit reconstructed the stratified block split and checked:

- Pairwise overlap of block IDs between training, validation, and test partitions: **zero**.
- Whether saved test prediction IDs exactly match the reconstructed test IDs: **yes**.
- Whether saved test prediction labels match the labelled dataset: **yes**.
- Whether the split sizes recorded in each experiment report match the reconstructed split: **yes**.

The audit verifies block-level partition integrity and consistency of the saved test predictions. It does not test temporal or host-level independence and does not prove the absence of every possible leakage path.

## 8. Reproducibility and software environment

The local environment manifest recorded the following versions:

| Component | Version |
|---|---:|
| Python | 3.13.13 |
| Drain3 | 0.9.11 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.5 |
| joblib | 1.6.0 |
| SciPy | 1.18.1 |

The earlier manifest record, `reports/model_evaluation/train_only_drain3_manifest.json`, captured this repository state when that manifest was created:

- Repository: `yagmurdamlakara/LogPilot-AI`
- Branch: `main`
- Commit: `f06baaf48060acf255ac9f011dcb0ea46acf3c9e`

This commit belongs to the earlier manifest record and is retained as its creation-time Git snapshot. The current schema v2 reproducibility manifest, `reports/model_evaluation/train_only_drain3_manifest_v2.json`, separately records commit `ece190ac7175bd22921a502d08231e609dc063b7`. Its commit and working-tree status describe the Git state captured when the schema v2 manifest was created. Both records are snapshots of their respective manifest creation times and may therefore differ.

### Drain3 configuration provenance

An important reproducibility distinction must be preserved: the completed five-seed historical experiment instantiated `TemplateMiner()` without a `drain3.ini` in the working directory. Drain3 0.9.11 therefore used its `TemplateMinerConfig` defaults.

The committed file `configs/drain3_train_only.ini` was added **after** that historical run. It explicitly records the intended default settings for future runs; it was not loaded by the completed experiment. Its SHA-256 at manifest capture was:

```
d3807c585bacaf2b62a95f4445c17296827dc307090393bb95104ad476ff38bf
```

Future experiments should use the committed INI explicitly and record its hash, so the configuration used by the run is unambiguous.

## 9. Limitations and interpretation boundaries

1. **Random rather than chronological split:** the current evaluation does not establish performance on a future time period.
2. **No host-level independence test:** blocks are separated between partitions, but this audit does not establish that related blocks from the same host or other correlated sources cannot occur across partitions.
3. **Audit scope:** the audit verifies block-ID separation, prediction IDs/labels, and split sizes. It is not a universal leakage detector.
4. **Historical configuration provenance:** the INI file documents settings for future runs and was not used in the completed five-seed experiment.
5. **Dataset identity in the earlier manifest:** the earlier manifest file records the dataset path, row count, file size, and modification time, but does not contain a cryptographic hash of the dataset contents.

The schema v2 manifest generator records a streaming SHA-256 and byte size for the
dataset and for each seed's metrics, test predictions, serialized models, Drain3
templates, and experiment report. These hashes identify the bytes read when a
schema v2 manifest is created; they do **not** prove that the same bytes were
present when the historical experiment ran. The generator refuses to overwrite
an existing output; use a new `--output` path when retaining an earlier manifest.

## 10. Conclusion

The train-only Drain3 experiment provides a stronger inductive evaluation than fitting the parser on the full log corpus before splitting blocks. Drain3 learned 44–46 clusters per seed, and only 32 messages were unmatched across the five frozen feature-building passes. The block-level split audit passed for all five seeds.

Logistic Regression achieved the strongest results, with a five-seed mean F1-score of **0.990725 ± 0.006098**. Isolation Forest achieved **0.684958 ± 0.004165**. Compared with the earlier full-corpus parser baseline, the F1 results are effectively unchanged. This supports the conclusion that full-corpus unsupervised parsing did not materially improve the measured F1 in this experiment, while the limitations around temporal and host-level independence remain open for future evaluation.

## 11. Local experiment artifacts

The completed local experiment produced, for each seed:

- `metrics.json`
- `test_predictions.csv`
- `baseline_models.joblib`
- `drain3_templates.json`
- `experiment_report.json`

The local summary is `experiments/train_only_drain3/summary.json`, and the split
audit is `reports/model_evaluation/train_only_drain3_split_audit.json`. The file
`reports/model_evaluation/train_only_drain3_manifest.json` is the earlier
manifest record. The current schema v2 reproducibility manifest is
`reports/model_evaluation/train_only_drain3_manifest_v2.json`.

Schema v2 records the dataset SHA-256 and file size, plus SHA-256 and byte size
for the 25 experiment artifacts across the five seeds. These hashes identify
the files as read when the manifest was created; they do not prove the files had
the same identity when the historical experiment ran.

Configuration provenance is separate: the completed historical run used
Drain3 0.9.11 `TemplateMinerConfig` defaults with no INI loaded, while
`configs/drain3_train_only.ini` records configuration for future runs. The INI
must not be described as the configuration consumed by the historical run.

Large serialized model files and prediction CSVs should remain outside normal source control unless there is a specific artifact-management requirement. The compact report and reproducibility metadata are suitable for version control.
