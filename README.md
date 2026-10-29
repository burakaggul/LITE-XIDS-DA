# LITE-XIDS-DA

Reproducible leakage-aware cross-dataset intrusion-detection evaluation for IoT and IIoT traffic tables.

This repository implements an evaluation framework, not a claim of a universally transferable or deployment-ready detector. In particular, the inherited 12 features are explicitly documented as proxies with known packet/window/flow mismatches. Distinct native records that collide after mapping are retained and grouped rather than silently deleted.

## Setup

Use Python 3.13 for the recorded desktop environment. Install `requirements.txt` in an isolated environment. No private traffic, raw dataset files, peer-review reports, or editorial correspondence are redistributed.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s revision/code -p 'test_*.py' -v
python revision/code/download_data.py
python revision/code/run_revision.py --stage all
python revision/code/run_mapping_controls.py
python revision/code/run_sensitivity.py
python revision/code/run_sensitivity.py --statistics-only
python revision/code/run_external_xai.py
```

Run from the repository root. The main experiment uses seeds 11, 23, 42, 67, and 89, and up to 30,000 rows per downloaded pool. Native, proxy-12, and proxy-8 within-dataset comparisons use identical source partitions. All six ordered transfer directions are evaluated across CIC-IoT-2023, TON_IoT, and a six-capture CICIoMT2024 flow-mirror subset.

Candidate selection takes source-calibration fields only and rejects external metrics. Models are fixed before target evaluation labels enter metric calculation. Source-only FPR constraints and target alarm budgets are distinct policies, not target-FPR guarantees. Raw margins avoid sigmoid score saturation. Boundary ties are handled with reproducible, label-independent hashes; the adaptation workload matches the requested budget in expectation, not necessarily on a future stream.

## Data provenance and scope

- CIC-IoT-2023: `lacg030175/CIC-IoT-2023-neto-subsample`, already a class-conditioned mirror. Its mirror train/test files are pooled for this declared resampling protocol; this is not official-split evaluation.
- TON_IoT: `codymlewis/TON_IoT_network`.
- CICIoMT2024: six named capture CSVs from `somnath0100/CICIoMT2024Small`. These are CICFlowMeter-derived tables, not the official release's packet-window feature tables. The subset does not preserve full benchmark attack coverage or prevalence.
- UL-ECE-MQTT-DDoS-H-IoT2025 and UL-ECE-UDP-DDoS-H-IoT2025: exact Zenodo record 15305814. Used for label-rule auditing only: the published preprocessing constructs outcomes from frequency-based predictors. Rule recovery is not independently established attack-detection performance.

Official descriptions: https://www.unb.ca/cic/datasets/iotdataset-2023.html ; https://research.unsw.edu.au/projects/toniot-datasets ; https://www.unb.ca/cic/datasets/iomt-dataset-2024.html ; https://zenodo.org/records/15305814 . Upstream dataset licenses and access terms apply. The download provenance and SHA256 manifest identify the exact local files.

## Outputs

`revision/results/` contains per-run numeric metrics, class counts, feature mapping warnings, collision/conflict audits, fixed-model ablations, score quantiles, source-selected results, compression costs, and external SHAP rankings. Five-seed means and sample SD describe resampling variability conditional on the downloaded pools, not independent deployment sites. Paired tests are exploratory and Holm-adjusted; no state-of-the-art superiority is asserted across incompatible published protocols.

The tabular TCN uses a declared feature order. It is not presented as a reproduction of a chronological TCN detection/mitigation pipeline. Device grouping is evaluated only where metadata exist. Missing timestamps are not replaced with fabricated chronology; unestimable temporal splits are reported.

## Actual device measurement

The `edge_benchmark.py` script measures a teacher and compact students on the machine where it is executed. Actual Raspberry Pi/Jetson measurements must be acquired separately; desktop CPU timings are not edge-device evidence. Use a supported 64-bit OS and record the device label, power mode, thread count, temperature, and complete generated JSON. Never run untrusted pickle/model files. Only load the accompanying author-generated artifacts whose hashes you have checked.

## Author

Burak Aggul, Istanbul Topkapi University. Contact: burakaggul@topkapi.edu.tr.
