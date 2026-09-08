# M4-2D V1 official validation-only result

Recorded on 2026-09-08 from the original AutoDL validation artifacts copied
unchanged into `checkpoints/m4_osseous_strength_selection_v1/` by the user.

```text
M4_2D_V1_SELECTION_STATUS=FAILED_CLEANLY
M4_2D_V1_SELECTED_LAMBDA_OSS=None
TRAIN_VALIDATION_COMBINATIONS_COMPLETE=20/20
CPU_VERIFICATION=20/20 PASS
AGGREGATION=PASS
selection_eligible=false
selection_reason=one_or_more_zero_success_patients
FORMAL_TEST_ACCESSED=false
```

The 20/20 training/validation completion and prior 20/20 CPU verification are
the completed-run facts supplied with this amendment request. This local task
independently validated all 20 original validation JSONL files (30 cases each),
all 600 frozen case identities, and recomputed the V1 summary with the unchanged
V1 aggregator. It did not reopen training logs, checkpoints, or completion
markers, and it did not rerun training or model inference.

V1 aggregation succeeded as a valid calculation of an ineligible grid. Its
global eligibility condition forbids selection when any candidate contains a
zero-success patient. A computational PASS does not imply selection eligibility.

The original summary omits `SELECTED_LAMBDA_OSS` on ineligibility; its meaning is
no selected lambda (`None` / JSON `null`). This record makes that meaning explicit
without adding a key to, or overwriting, the V1 summary.

## Solver outcomes

Each lambda has 150 validation cases, comprising ten unique patients with the
original 15 cases per patient. Failed cases remain in every rate denominator.

| lambda_oss | Success | Failure | Success rate | Zero-success validation patients |
| --- | ---: | ---: | ---: | --- |
| 0.25 | 128/150 | 22/150 | 85.333333% | Fold4 / Pat2 (0/15) |
| 0.50 | 132/150 | 18/150 | 88.000000% | None |
| 1.00 | 109/150 | 41/150 | 72.666667% | Fold4 / Pat1 (0/15); Fold4 / Pat2 (0/15) |
| 2.00 | 7/150 | 143/150 | 4.666667% | Fold1 / Pat7 (0/15); Fold1 / Pat4 (0/15); Fold2 / Pat11 (0/15); Fold2 / Pat3 (0/15); Fold3 / Pat8 (0/15); Fold3 / Pat9 (0/15); Fold4 / Pat1 (0/15); Fold4 / Pat2 (0/15) |

All 224 failed cases report `insufficient_correspondences`. The solver definition
previously confirmed in the task is: on entering weighted registration with
`num_correspondences < 3`, return `success=False` and
`failure_reason='insufficient_correspondences'`. This amendment does not change
that condition or any Hard / Soft / Continuous Osseous Support Prior definition.

The larger-lambda degradation is consistent with stronger osseous modulation
reducing the surviving correspondences. These validation artifacts do not prove
a unique causal mechanism and do not establish an evaluator bug.

## Frozen provenance

| Item | SHA-256 |
| --- | --- |
| V1 protocol canonical JSON, excluding `protocol_sha256` | `a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38` |
| V1 protocol original file bytes | `5e8a987ffa3d5d0abcb544388c659f7c6a37bf5dc2b3f23608d4043929d39902` |
| V1 protocol SHA sidecar original bytes | `4153385b5ae84d0f5fce341e799fdcbec71212aae04b69789dcb87ceacfb960d` |
| V1 frozen Markdown original bytes | `5e5d3dfbb9689a1a639548babc178a30f44dd8f76e617864cbd0d0a188d5304d` |
| Original `validation_selection_summary.json` bytes | `dc8ef14428481bea6dc2addaf6b555b90f999b023252dcc0bb75b33063762b42` |

The V1 protocol version is `m4_osseous_strength_selection_clean10_v1`.
The protocol JSON, its sidecar, the frozen Markdown, and the original summary
retain their original bytes. Each original validation JSONL byte hash is bound
in the separate V2 amendment protocol and repeated in its output summary.

The source summary agrees with local CPU recomputation in structure, types,
identities, counts, statuses, and selection outcome. One descriptive float,
lambda 0.5 centroid TRE mean over patient medians, is
`7.954210859143404` in the original summary and `7.954210859143403` in the local
recomputation. Verification permits `rel_tol=1e-12, abs_tol=1e-12` for finite
floating-point values only. SHA binding remains exact over the original bytes.

## Relationship to V2

The separate V2 amendment was established after observing V1 validation
failures and before formal test access. Its selection is explicitly
`POST_HOC_VALIDATION_ONLY`. It does not convert this failed V1 result into a
successful V1 preregistered selection and does not provide formal test evidence.
