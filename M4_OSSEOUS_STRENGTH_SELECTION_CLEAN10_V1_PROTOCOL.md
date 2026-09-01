# M4-2D Osseous Strength Train/Validation-only Selection V1

Protocol status: **FROZEN BEFORE GPU SELECTION**  
Expected branch: `m4_osseous_strength_selection_v1`  
Base commit: `d6a10cf092089426307b5f93766b8c5a5a2636f9`  
Canonical protocol SHA-256: `a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38`

The machine-readable authority is
`experiments/geotransformer.pointct.baseline_v1/protocols/m4_osseous_strength_selection_clean10_v1.json`.
The SHA-256 is computed from canonical JSON after removing only the top-level
`protocol_sha256` field. No lambda result may change this document or the JSON.

## Frozen purpose and firewall

This protocol selects `lambda_oss` using train/validation data only. Test
samples, test evaluation, test Recall/RRE/RTE/TRE, and any test result root are
forbidden until a lambda has been frozen in a later stage. The existing
training checkpoint may retain upstream test subject IDs as unread split
provenance, but neither the runner's validation producer nor the aggregator may
load or consume them. Validation artifacts containing a result key whose token
is `test`, or a value claiming test scope, fail closed.

No formal test entry point may be invoked. In particular, this stage forbids
`evaluate_m3.py --execute-test`, `evaluate_m3_defect.py --execute-test`, and
the test-manifest diagnostic execution path.

## Frozen clean10 folds

`Pat6` is excluded from all formal statistics.

| Fold | Train | Validation | Sealed test (never executed here) |
| --- | --- | --- | --- |
| Fold1 | Pat11, Pat3, Pat8, Pat9, Pat1, Pat2 | Pat7, Pat4 | Pat12, Pat5 |
| Fold2 | Pat12, Pat5, Pat8, Pat9, Pat1, Pat2 | Pat11, Pat3 | Pat7, Pat4 |
| Fold3 | Pat12, Pat5, Pat7, Pat4, Pat1, Pat2 | Pat8, Pat9 | Pat11, Pat3 |
| Fold4 | Pat12, Pat5, Pat7, Pat4, Pat11, Pat3 | Pat1, Pat2 | Pat8, Pat9 |
| Fold5 | Pat7, Pat4, Pat11, Pat3, Pat8, Pat9 | Pat12, Pat5 | Pat1, Pat2 |

The source clean10 protocol is `m3_6b_5fold_clean10_v2`, canonical SHA-256
`34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c`.

## Frozen candidate grid

The exact ordered grid is:

```text
lambda_oss = [0.25, 0.50, 1.00, 2.00]
```

No value may be added, removed, or altered after this protocol is hashed.
`1.00` is a development-only mechanism check, not a default winner.

## Frozen model and training configuration

Every fold and lambda uses the same full selection budget and random
conditions. Only `lambda_oss` changes.

| Setting | Frozen value |
| --- | --- |
| Epochs | 20 (full budget) |
| Global seed | 20260815 |
| Perturbation seed scheme/root | `m3_6b_seed_v1` / 20260815 |
| Optimizer | AdamW, LR 0.0003, weight decay 0.0001 |
| Scheduler | none; constant LR |
| Batch / precision | 1 / fp32 |
| Matcher | temperature 0.10, Sinkhorn 20, alpha init 1.0 |
| M4 hard | enabled and active |
| M4 soft | enabled; sigma 60.0 mm; strength 2.0 |
| Osseous score | center 300 HU; tau 100 HU; radius 20 mm |

The 60 mm soft scale equals three frozen 20 mm coarse-support cells. It is
pre-registered here because the repository had no authoritative formal soft
value and a 4 mm test fixture would be nearly inactive at 20 mm token spacing.
The strength 2.0 value is the common osseous integration fixture. Both values
are now fixed before any GPU selection result.

The frozen formulas remain:

```text
R_oss[i,j] = R_soft[i,j] * o_j
S_final[i,j] = S_soft[i,j] - lambda_oss * (1 - R_oss[i,j])
```

## Frozen checkpoint rule

For each fold/lambda, the selected epoch minimizes the existing mean
validation training loss over the frozen validation defect/severity cases.
Updates use strict `<`; an exact equal value retains the earlier epoch. The
checkpoint basename is `best_val_loss.pt`. The completion record must contain
`best_epoch`, `best_val_objective`, absolute checkpoint path, and the actual
checkpoint SHA-256. No test metric may select an epoch.

## Frozen validation unit and metrics

Each validation patient contributes five defect conditions under mild,
moderate, and hard validation perturbations, with `variant_id=0`: 15 cases per
patient, 30 cases per fold, and 150 cases per lambda. These cases are repeated
measurements, not independent patients.

Solver-failure TRE/RRE values remain null and failures stay in descriptive
denominators. A patient's selection metric is the median over that patient's
solver-success cases. A patient with zero successful cases makes that lambda
incomplete and forbids selection.

For every lambda, exactly ten patient rows are required. The selection metrics
are:

1. Primary: median of the ten patient-level median centroid TRE values.
2. Secondary: median of the ten patient-level median point-mean TRE values.
3. Tertiary: median of the ten patient-level median RRE values.

Legacy parameter RTE and registration Recall are descriptive only. Each lambda
also reports per-patient rows, per-fold summaries, mean, median, linear p95,
and best/worst patient.

## Frozen deterministic tie algorithm

Starting with all four complete candidates:

1. Keep candidates no more than 0.25 mm above the minimum primary value.
2. Among survivors, keep candidates no more than 0.25 mm above the minimum
   secondary value.
3. Among survivors, keep candidates no more than 0.25 degrees above the
   minimum tertiary value.
4. Select the smallest remaining `lambda_oss`.

The anchored-to-minimum stages avoid a non-transitive pairwise-tie ambiguity.
`SELECTED_LAMBDA_OSS` may be emitted only after all four lambdas and all five
folds pass completeness and firewall checks.

## Frozen output isolation

The root is `checkpoints/m4_osseous_strength_selection_v1/`, with
`lambda_0p25`, `lambda_0p50`, `lambda_1p00`, and `lambda_2p00`, then `fold1`
through `fold5`. Each fold has isolated checkpoints, training JSONL/log,
validation artifacts, checkpoint record, and a validated completion marker.
A non-empty incomplete directory fails closed; only a fully verified marker
may be skipped. M3 and earlier M4 outputs are never overwritten.

This local protocol implementation performs no training, uses no GPU or
AutoDL, and performs no commit or push.
