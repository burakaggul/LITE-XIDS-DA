COMMON-FLOW REVISION EXPERIMENT

Status is recorded in results/implementation_checks.json and the actual stage
logs. config.json is an immutable pre-execution snapshot, not a live status file.
The original submitted manuscript and historical revision outputs are retained.

DATA AND PROVENANCE
The fixed cohort contains 19 complete mirrored CIC-IoT-2023 objects, 25 mirrored
TON_IoT capture objects and 23 original-schema TON label tables (21.32 GiB).
These are bounded mirrored cohorts, not the complete original benchmark releases
or natural-prevalence traffic. Repository revisions and object checksums are
pinned in manifests/capture_plan.json. Object integrity does not independently
authenticate the mirrors against official source checksums or guarantee that
every object ends on a structurally complete packet record. No raw data are
redistributed. Upstream access terms and dataset citations still apply.

Wireshark reordercap normalizes frame order according to original timestamps.
Pinned source files remain unchanged. Complete packet payloads, exact recorded
timestamps, lengths and counts are checked using order-independent SHA256-based
multiset fingerprints. Incomplete raw tails are preserved in the original and
explicitly quarantined/reported; missing bytes are never padded or fabricated.
Sorting does not independently establish synchronization of acquisition clocks.

Both sources use NFStream 6.6.0, IP-layer byte accounting, 30s idle/120s active
timeouts, no tunnel decoding, no L7 dissection, and bidirectional statistics.
The 12 model features have one common physical definition. Endpoints, ports,
timestamps, capture IDs and label metadata are never model inputs.

CIC labels inherit the published capture category, not independent per-packet
verification. TON labels require the same orientation-invariant tuple, unanimous
overlapping binary labels and full time support from their interval union, with
a fixed 1s timestamp tolerance. Unsupported/conflicting flows remain unknown.
At least 95 percent overall TON label coverage is required before primary
full-traffic training; the separately registered conditional diagnostic below
does not override that failed gate.
The initial 23-table join failed this gate (93.6348 percent); that failed data
snapshot is retained in archive/pre_official_annotations. The official public
UNSW SharePoint folder was subsequently accessible through the browser and its
18 GroundTruth_Network CSV files were downloaded as the original folder ZIP.
official_events.py independently pins and indexes that snapshot. Event rows have
timestamps but no durations: their intervals remain start=end=ts. Only explicit
normal types are benign; missing events are never presumed benign. The same
one-second tolerance, exact tuple and unanimous full-union support rules apply.
The source amendment precedes any common model performance; capture roles,
coverage requirements, training code and hyperparameters are unchanged.

Capture roles are frozen before scores. Different stored native identities are
retained even when the 12 feature values coincide. Cross-role overlap exclusions
use native endpoints, timing, directional counters and common statistics, with
orientation normalized. They are not proof of raw packet-byte identity. Capture
disjointness does not prove acquisition-session independence.

LEARNING AND EVALUATION
Five seeds: 11, 23, 42, 67, 89. Hash-priority capture samples retain at most 30000
flows per capture. Source train/calibration samples are capped at 30000 each.
Final partitions retain all eligible sampled flows. Sampling is not prevalence
representative; identical observational fingerprints share hash priority.

All imputation, signed-log scaling, class weights, early stopping, costs and
thresholds use source train/calibration only. Source calibration is reused for
early stopping and operating-point selection; its FPR constraints are empirical,
not population or external-FPR guarantees. All five costs (1,2,5,10,20) and both
source FPR limits (0.01,0.05) are declared before common final evaluation.

Controls are XGBoost teacher, capacity-matched supervised/distilled 64/32 MLP,
derived dynamic INT8 student, logistic regression and random forest. INT8 uses
row-wise scripted inference so changing outer batch size cannot change activation
quantization. Matched-cost component controls and matched-backend float/INT8
timings prevent cost-selection and execution-backend differences being presented
as isolated distillation or quantization effects.

Historical A/C and common-flow B/D contrasts are paired within their own cohorts.
They are not a capture-matched causal feature-extractor comparison. Historical
target results have previously been examined and are diagnostic. Five-seed
intervals describe training/sampling variability conditional on fixed cohorts,
not independent real-world deployment sites. All prespecified results are shown;
lower FPR with collapsed attack recall is not successful intrusion detection.

Post-freeze XAI uses probability-scale permutation SHAP, source training
backgrounds, fixed within/external query records, five-seed/domain rank stability,
full known-label subtype errors, and descriptive PDP. Correlated/derived features
permit off-manifold perturbations. Explanations are not causal attack evidence.

DESKTOP AND DEVICE SCOPE
CPU measurements state whether they include preprocessing or model inference
only. The conditional-v3 matched profiles are model-only. Packet capture,
flow construction and the 30/120s aggregation delays
are excluded. Process RSS includes Python/frameworks and cached artifacts; it is
not isolated model memory. Desktop timings are not Raspberry Pi/Jetson evidence.
Actual school-device tests are deferred and must not be described as completed.

RUNNING
Keep this directory at revision/flow_v2 in the repository. Windows ML runtime
and the separate Linux extraction runtime are recorded in runtime_environment.
Use Linux NFStream and Wireshark reordercap; no Windows capture-driver changes
are required for offline processing. The author used unprivileged extraction of
Ubuntu's wireshark-common, libwiretap14t64 and libwsutil15t64 packages into a
user-local directory. The usual installed reordercap executable also works.

Start complete downloads, label indexing, source auditing and Linux extraction
in separate terminals, then run the gated pipeline:

  python revision/flow_v2/plan_and_download.py --kind all --workers 4
  python revision/flow_v2/label_flows.py --stage index --watch
  python revision/flow_v2/audit_sources.py --stage sources --watch
  python revision/flow_v2/extract_flows.py --watch             [Linux runtime]
  python revision/flow_v2/run_common_pipeline.py
  python revision/flow_v2/matched_cpu.py
  python revision/flow_v2/summarize.py

For the explicit official-annotation amendment, first import the folder ZIP:

  python revision/flow_v2/official_events.py --archive PATH_TO_OFFICIAL_ZIP
  python revision/flow_v2/run_common_pipeline.py --official-events

The data-freeze stage pins all six prepared partitions and label provenance
before common training; subsequent verification rejects modified snapshots.
Do not describe the primary common-flow experiment as completed unless its label
gate, full result audit, LR/RF controls, XAI and component/timing controls passed.

The optional Linux prefetch_chronology.py stages byte-identical local copies to
avoid WSL shared-mount per-packet I/O overhead. It uses the same normalization
algorithm and never overwrites another extractor's completed proof.

For the separate historical diagnostic, run experiment.py --track legacy
--stage train, then --stage evaluate and audit_results.py --track legacy.
The author's earlier batch-calibration records and code are archived. The v2.1
source-only INT8 recalibration amendment predates that diagnostic's final scores.
Do not treat archived exploratory outputs as corrected common-flow results.
ablate_common.py --track legacy, matched_cpu.py --track legacy, and
explain_legacy.py provide separately named posthoc diagnostic analyses of the
frozen historical models. They are not raw-flow or independent new test evidence.

CONDITIONAL RAW DIAGNOSTIC V3
The original full-traffic study remains stopped. Versioned IPv6/ICMP key
normalization recovers 437 retained labels. Packet-label witnesses independently
reconstruct native packet counts, bytes, timing, flags and UID; 6,645 of 6,742
empty-time-gap candidates pass. Ninety-seven rejected candidates remain unknown.
There are three quarantined incomplete tails, including injection_normal4; its
re-extracted flow values are unchanged after complete-record normalization.
Coverage is 94.71785699 percent, with 34,539 unknown outcomes. The unchanged
95-percent primary gate remains failed.

The separate labeled_raw_v3 study was registered before common-model scores.
It estimates conditional performance among traceably supported sampled flows,
not all traffic. All source/seed models freeze before final evaluation. Unknown
labels never enter training or selection. Missing-label identification bounds
permit every possible class assignment, including nonrandom missingness, and
must accompany conditional scores. These bounds are not confidence intervals.
Original interval-only provenance strings in the frozen snapshot predate packet
witnesses; documentation_receipt.json explicitly clarifies the 6,645 witnessed
flows without modifying frozen data or model artifacts.

After the documented downloads, original indexes, official-event import and
Linux extraction, run the versioned schema and packet inspection stages:

  python revision/flow_v2/label_flows.py --stage prepare --official-events
    [Expected exit 3 when the unchanged full-traffic coverage gate fails.]
  python revision/flow_v2/label_schema_v3.py
  python revision/flow_v2/label_flows.py --stage prepare --official-events --label-schema v3
    [Expected exit 3; this prepares the versioned unresolved cohort.]
  python revision/flow_v2/packet_label_witness.py --prepare
  python revision/flow_v2/resume_packet_witnesses_v3.py   [Linux runtime]
  python revision/flow_v2/run_labeled_revision_v3.py      [Windows ML runtime]

Inspect each script's --help and configured paths. The author's WSL helper
uses a user-local environment and mirror/cache paths; replace these for a new
installation rather than treating the author's machine layout as portable.
run_labeled_revision_v3.py accepts only the documented code-3 failed primary
coverage gate before registering the conditional study; other failures stop it.
Do not reuse author evidence snapshots as a new run's data/model freeze.

The completed study has 480 primary policy rows, 80 matched-cost component
rows and 560 missing-outcome-bound rows, independently audited. Five-seed
source-limit-0.01 KD external recall is 0.0659 (CIC-to-TON) and 0.1133
(TON-to-CIC); these are not successful transfer results. Smaller INT8 artifacts
do not establish faster inference. Actual-device measurements remain absent.

Only load author-generated model artifacts after checking their frozen hashes.
Do not publish raw traffic, original label tables, peer-review correspondence,
private working reports or school-device credentials with the code release.
