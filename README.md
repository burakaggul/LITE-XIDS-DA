# LITE-XIDS-DA

Reproducible leakage-aware cross-dataset intrusion-detection evaluation for IoT and IIoT traffic tables.

This repository implements an evaluation framework, not a claim of a universally transferable or deployment-ready detector. In particular, the inherited 12 features are explicitly documented as proxies with known packet/window/flow mismatches. Distinct native records that collide after mapping are retained and grouped rather than silently deleted.

## Setup

Use Python 3.13 for the recorded desktop environment. Install `requirements.txt` in an isolated environment. No private traffic, raw dataset files, peer-review reports, or editorial correspondence are redistributed.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s revision/code -p 'test_*.py' -v
python revision/code/download_data.py
python revision/code/run_all_verified.py
```

Run from the repository root. The main experiment uses seeds 11, 23, 42, 67, and 89, and up to 30,000 rows per downloaded pool. Native, proxy-12, and proxy-8 within-dataset comparisons use identical source partitions. All six ordered transfer directions are evaluated across CIC-IoT-2023, TON_IoT, and mirrored CICIoMT2024 packet-window CSV tables. Original capture train/test files are pooled for the declared resampling study, not presented as official-split scores.

Candidate selection takes source-calibration fields only and rejects external metrics. Models are fixed before target evaluation labels enter metric calculation. Source-only FPR constraints and target alarm budgets are distinct policies, not target-FPR guarantees. Raw margins avoid sigmoid score saturation. Boundary ties are handled with reproducible, label-independent hashes; the adaptation workload matches the requested budget in expectation, not necessarily on a future stream.

The native control includes allowed numeric variables and source-fitted low-cardinality categorical encoding (at most 30 observed categories); unseen categories never fit on held-out data. The legacy `deduplicate_semantic_features` helper in the inherited mapping utilities is **not called** by this protocol. Record identity and grouping are implemented in `data_revision.py` and `protocol.py`. Original identifiers participate in native duplicate checks but not model inputs.

## Data provenance and scope

- CIC-IoT-2023: `lacg030175/CIC-IoT-2023-neto-subsample`, already a class-conditioned mirror. Its mirror train/test files are pooled for this declared resampling protocol; this is not official-split evaluation.
- TON_IoT: `codymlewis/TON_IoT_network`.
- CICIoMT2024: version 1 of `outstandard09/ciciomt2024` on Kaggle, retaining the published Wi-Fi/MQTT capture folders and packet-window feature schema. Labels are inherited from the published benign/attack capture filenames, not independently reverified per row. A CICFlowMeter mirror with NeedManualLabel entries was rejected. Capture/group and extraction-window limitations remain explicit.
- UL-ECE-MQTT-DDoS-H-IoT2025 and UL-ECE-UDP-DDoS-H-IoT2025: exact Zenodo record 15305814. Used for label-rule auditing only: the published preprocessing constructs outcomes from frequency-based predictors. Rule recovery is not independently established attack-detection performance.

Official descriptions: https://www.unb.ca/cic/datasets/iotdataset-2023.html ; https://research.unsw.edu.au/projects/toniot-datasets ; https://www.unb.ca/cic/datasets/iomt-dataset-2024.html ; https://zenodo.org/records/15305814 . Upstream dataset licenses and access terms apply. The download provenance and SHA256 manifest identify the exact local files.

The downloader pins both Hugging Face revisions and verifies all source checksums. CICIoMT uses the version-1 archive hash. A hash mismatch is an error, not permission to silently substitute newer data.

## Outputs

`revision/results/` contains per-run numeric metrics, class counts, feature mapping warnings, collision/conflict audits, fixed-model ablations, score quantiles, source-selected results, compression costs, and external SHAP rankings. Five-seed means and sample SD describe resampling variability conditional on the downloaded pools, not independent deployment sites. Paired tests are exploratory and Holm-adjusted; no state-of-the-art superiority is asserted across incompatible published protocols.

The tabular TCN uses a declared feature order. It is not presented as a reproduction of a chronological TCN detection/mitigation pipeline. TON source-device and CICIoMT2024 capture grouping are attempted only with real metadata. Missing or unestimable splits are reported without fabricated chronology. A matched ten-window-descriptor CIC/IoMT control excludes unavailable duration and non-harmonized IAT; it does not establish identical extraction-window settings.

The extra capture-disjoint final test reserves one benign capture and 20% of attack captures; calibration is predictor-grouped within the remaining training captures, not a third independent capture pool. Baseline inference profiles are auxiliary seed-42 timings, not five-seed training uncertainty. `postcheck_controls.py --phase extras` can rerun these additional controls separately. The historical execution record discloses the native-only corrections and matching proxy/label/record fingerprints. It distinguishes the original run hashes from current public source hashes.

## Actual device measurement

The `revision/code/edge_benchmark.py` script measures a teacher and compact students on the machine where it is executed. Generate the author-controlled artifacts by running the compression stage, then pass `--artifacts revision/results --device-label "ACTUAL MODEL AND RAM"`. Actual Raspberry Pi/Jetson measurements must be acquired separately; desktop CPU timings are not edge-device evidence. Use a supported 64-bit OS and record the device label, power mode, thread count, temperature, and complete generated JSON. Five timing trials of one fixed artifact are not five independently trained models. The default `first` input policy matches the desktop warm-cache microbenchmark; `cycle` is a separate traffic-vector sensitivity condition. Never run untrusted pickle/model files. Only load author-generated artifacts whose hashes you have checked.

## Experimental Common-Flow Follow-Up

`revision/flow_v2/` contains a separately versioned follow-up, not a replacement
for the v1.0.0 manuscript results. Its fixed cohort comprises 19 mirrored
CIC-IoT-2023 captures, all 25 captures in the selected TON mirror, and 23 original
Zeek-schema label tables. Both sources are processed with NFStream 6.6.0 using
the same 12 physical feature definitions. Original timestamps and complete packet
bytes are preserved through chronology normalization; two structurally incomplete
source tails are quarantined without fabricating bytes.

The common-flow experiment has **not passed its predeclared annotation-quality
gate**. Label coverage is 93.6348%, below the required 95%. An additional download
of all 18 original author-hosted `GroundTruth_Network` CSV files did not resolve
the unsupported flows under the unchanged tuple/time and unanimous-label rules.
Of 41,621 unresolved sampled flows, 32,972 have no matching tuple in any of the
41 annotation tables at any timestamp. Common training, common final evaluation,
and common LR/RF/XAI controls therefore remain unexecuted. Missing events are not
assigned benign labels and low-coverage final captures are not silently dropped.

The separate historical diagnostic completes five-seed source-only cost-sensitive
teacher/student experiments, 80 matched-cost component controls, matched-backend
float/INT8 CPU measurements and external probability-scale SHAP/PDP analyses.
These are diagnostic results on legacy proxy features, not new raw-flow evidence
or proof that generalization is solved. The cost intervention does not reliably
reduce external false alarms. INT8 reduces serialized size but is slower in the
recorded matched desktop benchmark. No Raspberry Pi or Jetson execution is claimed.

See `revision/flow_v2/README.txt` for the two-runtime workflow and its stopped gate.
The `evidence/` directory contains aggregate numerical outputs, provenance hashes
and execution audits only. It excludes raw packets, original CSV annotations,
model pickles, reviewer correspondence and private working reports. New runs
create their own code/data freezes; author execution snapshots are not silently
reused as fresh model choices. The unit tests cover both rejection of failed
annotation gates and preservation of original packet/event timing semantics.

Windows ML additions are in `revision/flow_v2/requirements-windows.txt`; Linux
extraction additions are in `requirements-linux-extraction.txt`. The existing
root requirements still describe the original revision environment.

## Author

Burak Aggul, Istanbul Topkapi University. Contact: burakaggul@topkapi.edu.tr.
