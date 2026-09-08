# M4 first-innovation formal-test clean10 V1 protocol

Frozen on 2026-09-08 before any M4 formal-test sample or formal result was
accessed in this task. Base implementation commit:
`f5bc5ab74a5d095ff2f26a8e35b914057731cda4`.
Branch: `m4_osseous_formal_test_v1_protocol`.

```text
protocol_status=FROZEN_BEFORE_FIRST_M4_FORMAL_TEST
FORMAL_TEST_ACCESSED_AT_FREEZE=false
SELECTED_LAMBDA_OSS=0.5
TEST_GRID_CASE_COUNT=750
CHECKPOINT_BINDING=PENDING_SERVER_CPU_AUDIT
FORMAL_TEST_ACCESSED=false
GPU_USED=false
```

This is a protocol and implementation freeze, not an executed experiment or a
claim that M4 improves on M3. Its purpose is a fair paired evaluation of the
fixed first-innovation configuration against the frozen M3 defect baseline.
No formal test, training, inference on real samples, or checkpoint modification
was performed. Existing M3 results were neither read nor modified.

## Source audit and binding

The new protocol is
`experiments/geotransformer.pointct.baseline_v1/protocols/m4_osseous_formal_test_clean10_v1.json`.
Its canonical SHA-256 is
`e33c0c4721e1dc303b9f5211b3f7ca2854f15e7933ae6ebd7a14303cd6c67f22`.
The adjacent `.sha256` sidecar uses sorted, compact, ASCII-escaped JSON with
`allow_nan=False`, UTF-8 encoding, excluding only top-level `protocol_sha256`.

| Frozen source | SHA-256 |
| --- | --- |
| `m3_6b_5fold_clean10_v2` | `34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c` |
| `m3_defect_eval_clean10_v2` | `cfd519b923be3b623cffec8c8f5830cb5b1160859461180eddeb7134a0adb6b0` |
| `m3_metric_diagnostic_clean10_v2` | `5f40c322433c2274d356213969f9041d2931ff0e058efb0c541bcf5e1a9624bc` |
| `m4_osseous_strength_selection_clean10_v1` | `a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38` |
| `m4_osseous_strength_selection_clean10_v2_amendment` | `fe59f307eca6355d835fad5a9a5d18f8b7f24b940747700149b7a5668413e13c` |

All five source JSON hashes are recomputed using their original excluded hash
field, then compared with embedded trust anchors. The new protocol also binds
23 unchanged implementation source files, covering manifests, perturbations,
M3 metrics, M4 inference and priors, registration, matching, encoders, config,
and dataset implementation. Source hashes normalize CRLF to LF for cross-platform
Git checkouts; checkpoint hashes always use exact binary file bytes.

Audited implementation interfaces include
`defect_evaluation.build_defect_test_manifest`,
`perturbation.derive_perturbation_seed`,
`evaluate_m3._build_models_from_checkpoint`,
`evaluate_m4_osseous_prior.run_m4_osseous_prior_inference`,
`evaluation.evaluate_registration_result`,
`evaluation.correspondence_inlier_metrics`, and
`m3_metric_diagnostic.compute_tre_diagnostics`.

The unittest suite checks every generated identity, seed, case key and
perturbation bound against the existing M3 manifest generator without creating
a dataset. The protocol's `manifest_sha256` additionally fixes the complete
ordered in-memory manifest definition.

## Fixed model and selection meaning

| Component | Fixed setting |
| --- | --- |
| Hard defect constraint | ON |
| Soft defect modulation | ON; sigma 60 mm; strength 2.0 |
| Continuous Osseous Support Prior | ON; center 300 HU; tau 100 HU; radius 20 mm |
| `lambda_oss` | 0.5 |

M4-2D V1 remains `FAILED_CLEANLY`, with no selected lambda. V2 remains `PASS`,
with selection basis `POST_HOC_VALIDATION_ONLY` and selected lambda 0.5.
This protocol does not recast V2 as a successful preregistered V1 selection.
Neither lambda nor any other model, ranking, threshold, or comparison rule may
be adjusted after observing formal-test outcomes.

The existing Hard, Soft, and Osseous mathematical definitions and the solver's
minimum correspondence rule are unchanged. Runtime uses the existing M4
inference implementation with the fixed parameters and checks its returned
active flags and parameter diagnostics.

## Exact inherited formal grid

| Fold | Frozen test patients | Cases |
| --- | --- | ---: |
| Fold1 | Pat12, Pat5 | 150 |
| Fold2 | Pat7, Pat4 | 150 |
| Fold3 | Pat11, Pat3 | 150 |
| Fold4 | Pat8, Pat9 | 150 |
| Fold5 | Pat1, Pat2 | 150 |

The JSON freezes all three partitions for every fold, directly from the clean10
source. There are ten unique patients; Pat6 is excluded and Pat10 remains a
failed-manifest patient. Each patient has the five source defect conditions:

1. `defect_001_left_maxilla_cheek_small`
2. `defect_001_left_maxilla_cheek_medium`
3. `defect_001_left_maxilla_cheek_large`
4. `defect_001_right_maxilla_cheek_small`
5. `defect_001_right_maxilla_cheek_medium`

For each patient/defect, the test perturbations are mild (5 degrees / 5 mm),
moderate (10 degrees / 10 mm), and hard (20 degrees / 20 mm), each with five
variants indexed 0 through 4. Thus there are 15 cases per defect instance,
75 per patient, 150 per fold, and 750 overall. Counts are derived and validated
against the frozen sources, not taken from the earlier validation grid.

The identity is `(fold_id, subject_id, defect_id, severity, variant_id,
perturbation_seed)`. Seed generation uses `m3_6b_seed_v1`, the clean10 protocol
hash, root seed 20260815, `purpose=test`, `epoch=None`, subject, severity and
variant. It deliberately excludes defect ID, exactly as M3 does. Runtime uses
the original perturbation sampler and effective Point Cloud -> CT transform
composition. Coordinates remain LPS, in millimetres.

## Metrics and source differences

Registration Recall retains the original inclusive rule:
`solver_success AND RRE <= 5 degrees AND legacy RTE <= 10 mm`.
Its denominator includes every frozen case. Correspondence inliers use distance
`<= 15 mm`; mutual-filter minimum confidence remains `null`.

Each case records success/status, correspondence count, inlier count/ratio,
Recall hit, RRE, legacy RTE, centroid TRE, and point TRE mean/median/RMSE/linear
p95/max. Inference runtime retains the M3 scope: model forward, matching,
mutual filtering, and weighted registration. Runtime excludes dataset loading
and metric calculation. Legacy RTE is origin-sensitive; it is preserved for
historical comparison and Recall, not replaced by TRE.

TRE is calculated by the unchanged M3 diagnostic function using the same
`inference.point_physical` reference and effective ground truth as the existing
M3 diagnostic evaluator. No new accuracy definition is introduced.

Source audit distinctions are explicitly retained:

- M3's `required_enable_m4_defect_mapping=false` describes the baseline model.
  M4 intentionally enables the frozen Hard/Soft/Osseous configuration while
  preserving the M3 test grid and all evaluation thresholds.
- The source M3 formal evaluator converts an invalid solver-declared success
  transform to `invalid_registration_output`. The separate TRE diagnostic
  entrypoint would instead raise on the disagreement with solver-declared
  success. M4 formal evaluation preserves the original **formal** failure
  status and emits null error diagnostics; it does not repair the transform.
- Training checkpoint metadata contains historical development/smoke status
  strings. Binding preserves those strings exactly as historical training
  provenance. The present protocol independently freezes the formal settings.
- Formal test has 750 cases; selection validation has 600 total cases across
  four lambdas. These grids and their evidential roles are not interchangeable.

Solver-declared failures and invalid-registration failures have null RRE, RTE
and TRE fields and false Recall. They remain in case denominators. A successful
solver with fewer than three correspondences is rejected as a contract breach;
the actual solver minimum has not changed.

## Pre-execution checkpoint binding

All five exact paths are fixed to:

```text
checkpoints/m4_osseous_strength_selection_v1/lambda_0p50/fold1/checkpoints/best_val_loss.pt
checkpoints/m4_osseous_strength_selection_v1/lambda_0p50/fold2/checkpoints/best_val_loss.pt
checkpoints/m4_osseous_strength_selection_v1/lambda_0p50/fold3/checkpoints/best_val_loss.pt
checkpoints/m4_osseous_strength_selection_v1/lambda_0p50/fold4/checkpoints/best_val_loss.pt
checkpoints/m4_osseous_strength_selection_v1/lambda_0p50/fold5/checkpoints/best_val_loss.pt
```

These five files are absent locally. No checkpoint SHA, best epoch, or objective
has been invented, and no placeholder binding manifest has been emitted.

On the server, the independent CPU binding step reads these five checkpoints
and their sibling fold `train.jsonl` validation-training records. It checks
the 20-epoch budget, strict-less validation minimum and earliest tied epoch,
full training config, optimizer learning rate/decay, fold, train/val identities,
defect-instance provenance, and fixed M4 parameters. It hashes each original
checkpoint buffer and uses `torch.load(..., map_location='cpu', weights_only=True)`
on that same buffer. There is no unsafe deserialization fallback.

The resulting no-clobber manifest is:
`experiments/geotransformer.pointct.baseline_v1/protocols/m4_osseous_formal_test_clean10_v1_checkpoint_binding.json`.
It includes all five SHA-256 values, exact paths, epochs/objectives, full config,
config hashes, train/val identities and training-log hashes, plus a canonical
self-hash and formal protocol hash. Generating it does not authorize execution:
the exact manifest and formal protocol must be present in **Git HEAD**, not
merely untracked or staged. Commit the binding before any formal sample access.

Only after verifying the committed binding and re-auditing all five current
checkpoint hashes/configs does the evaluator instantiate the current fold's
dataset. Its explicit selector includes only that fold's two test patients and
five defects. There is no checkpoint override, lambda override, training mode,
or automatic checkpoint replacement.

## Fixed outputs and paired analysis

Future outputs use `checkpoints/m4_osseous_formal_test_clean10_v1/`:

- `fold1/` through `fold5/`: `formal_test_cases.jsonl` and
  `formal_test_summary.json`.
- Root: `formal_test_summary.json` and `m3_vs_m4_paired_comparison.json`.

An exclusive fold-directory reservation prevents concurrent or partial-run
overwrite. Individual files use flushed temporary files and atomic no-clobber
hard-link publication. An existing fold or destination is a hard error; the
runner never silently resumes, overwrites, or deletes formal artifacts.

The independent comparison requires explicit future M3 roots. It reads only
`Fold1` through `Fold5` / `cases.jsonl` for legacy metrics and the corresponding
`diagnostic_cases.jsonl` files for TRE. It does not search directories for
results. Each file must match its frozen fold; both M3 unions and the M4 union
must match the exact 750 identities. Original/diagnostic M3 source protocol
hashes, success, status, Recall and counts are checked, and continuous
cross-checks use the frozen M3 diagnostic tolerances (absolute 1e-6, relative
1e-7). M4 rows are checked against the formal and checkpoint-binding hashes.

Report patient-level, fold-level and overall descriptive statistics, plus
patient macro summaries. Success and Recall deltas use all exact case pairs.
Error deltas are **M4 minus M3** on both-solver-success pairs only. Report common
success, M3-only success, M4-only success and neither-success counts separately.
Do not impute errors for failures. Every patient remains visible, including
zero-success patients and those with no common-success pair (null metrics,
count zero). Means, medians, linear p95 and metric counts are fixed now.

## Entrypoints and current verification

Commands permitted and used in this task:

```console
python -B -m unittest discover -s tests/pointct -p test_m4_osseous_formal_protocol.py -v
python -B experiments/geotransformer.pointct.baseline_v1/m4_osseous_formal_protocol.py --contract-audit
```

The portable runner and Bash wrapper have no default execution mode.
`--contract-audit` imports no runtime, dataset or PyTorch. Checkpoint binding
imports PyTorch only for CPU deserialization and imports no dataset or training
entrypoint. Genuine execution is isolated behind explicit `--evaluate`.

Future server steps, **not executed in this task**:

```console
python -B experiments/geotransformer.pointct.baseline_v1/bind_m4_osseous_formal_checkpoints.py --checkpoint-bind
```

Review and commit the generated binding manifest, then CPU-verify it:

```console
python -B experiments/geotransformer.pointct.baseline_v1/bind_m4_osseous_formal_checkpoints.py --verify-checkpoints
```

Formal execution remains pending separate authorization and a valid committed
binding. Its future explicit command is:

```console
bash run_m4_osseous_formal_test_clean10_v1.sh --evaluate --data-root <formal-data-root> --device cuda
```

After future completion, the independent CPU comparison command is:

```console
python -B experiments/geotransformer.pointct.baseline_v1/aggregate_m4_osseous_formal_test.py --compare --m3-legacy-root <frozen-M3-root> --m3-diagnostic-root <frozen-M3-diagnostic-root>
```

Current validation: **53/53 unittest PASS; CPU contract audit PASS**.
The machine-readable record is
`M4_OSSEOUS_FORMAL_TEST_CLEAN10_V1_CPU_AUDIT.json` and is explicitly marked
`artifact_scope=protocol_building_cpu_audit`, not a formal result.

The final suite and contract audit ran under a Python file-access audit hook
that rejected every repository `checkpoints/` and `local_data/` file open.
There were zero such opens, zero real dataset instantiations, and PyTorch was
not imported. Runtime tests used synthetic NumPy arrays or mocked datasets,
checkpoints and output directories in temporary storage. Tests cover all 34
requested contracts, including SHA/config/fold failures, exact M3 manifests,
no-clobber races, committed binding checks, patient stability and paired
missingness. Passing these tests is implementation evidence, not evidence of
formal registration performance.
