# M4-2D clean10 V2 validation-only post-hoc amendment

Date: 2026-09-08. Branch: `m4_osseous_strength_selection_v2_amendment`.
Source code base before this amendment: `ed2caf7`.

```text
protocol_status=POST_HOC_VALIDATION_ONLY_AMENDMENT_FROZEN_BEFORE_FORMAL_TEST
M4_2D_V1_SELECTION_STATUS=FAILED_CLEANLY
M4_2D_V1_SELECTED_LAMBDA_OSS=None
M4_2D_V2_AMENDMENT_STATUS=PASS
M4_2D_V2_SELECTION_BASIS=POST_HOC_VALIDATION_ONLY
M4_2D_V2_SELECTED_LAMBDA_OSS=0.5
FORMAL_TEST_ACCESSED=false
```

This amendment changes candidate eligibility after seeing the completed V1
validation results. It was frozen before any formal test access in this task.
The selected lambda is a post-hoc validation-only choice, not a V1 preregistered
selection. No model training, checkpoint modification, GPU computation, dataset
loading, or formal test evaluation was performed for this amendment.

## Binding and reproducibility

Machine-readable protocol:
`experiments/geotransformer.pointct.baseline_v1/protocols/m4_osseous_strength_selection_clean10_v2_amendment.json`.
The adjacent `.sha256` file binds canonical JSON after removing only the
top-level `protocol_sha256` field. Serialization uses sorted keys, compact
separators, ASCII escaping, no NaN/Inf, and UTF-8 encoding.

| Binding | Value |
| --- | --- |
| amendment version | `m4_osseous_strength_selection_clean10_v2_amendment` |
| amendment canonical SHA-256 | `fe59f307eca6355d835fad5a9a5d18f8b7f24b940747700149b7a5668413e13c` |
| source V1 protocol version | `m4_osseous_strength_selection_clean10_v1` |
| source V1 protocol canonical SHA-256 | `a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38` |
| source V1 summary original-byte SHA-256 | `dc8ef14428481bea6dc2addaf6b555b90f999b023252dcc0bb75b33063762b42` |

All 20 validation artifact original-byte hashes are frozen in the protocol.
The standalone V2 executable embeds the V2 and V1 summary trust anchors and
uses V1's unchanged protocol trust anchor and identity/patient-collapse code.
Hashing and parsing consume the same byte buffer for each input.

V1 summary comparison requires exact fields, types, counts, identities, states,
and list order. Finite floats alone allow `rel_tol=1e-12, abs_tol=1e-12` for
cross-runtime rounding; the one observed last-bit difference is recorded in
`M4_OSSEOUS_STRENGTH_SELECTION_V1_RESULT.md`. No original artifact is rewritten.

## Frozen validation units

The lambda grid is exactly `[0.25, 0.5, 1.0, 2.0]`. Each lambda uses all five
folds, with 30 cases per fold and 150 cases per lambda. Each patient retains the
same 15 V1 cases: five defect conditions times three perturbation severities,
with V1's original variant and deterministic seed identities.

| Validation fold | Patients |
| --- | --- |
| Fold1 | Pat7, Pat4 |
| Fold2 | Pat11, Pat3 |
| Fold3 | Pat8, Pat9 |
| Fold4 | Pat1, Pat2 |
| Fold5 | Pat12, Pat5 |

These are ten unique clean10 patients. Pat6 is forbidden. Duplicates, missing
cases, unexpected patients, unexpected fields, invalid seeds, wrong scopes,
and nonfinite values invalidate the entire aggregation. A malformed or missing
artifact is not treated as a candidate that can simply be dropped.

## Stage 0 and inherited ranking

For each candidate, recompute each patient's `solver_success_count` from its
15 original cases. If any of the ten patients has zero successes, set
`candidate_complete=false` and exclude that lambda from the deployable
candidate set. If every patient has at least one success, set
`candidate_complete=true`.

After this gate:

1. Zero complete candidates: fail closed with eligibility false and selection
   null. The CLI returns a nonzero exit status.
2. One complete candidate: select it directly, without applying metric ranking.
3. Multiple complete candidates: retain the original V1 sequence below, applied
   only to the surviving complete candidates.

The original ranking sequence is:

1. Median across the ten patient-level median centroid TRE values: retain values
   at most the current stage minimum plus 0.25 mm.
2. Among survivors, median across the ten patient-level median point-mean TRE
   values: retain values at most this stage minimum plus 0.25 mm.
3. Among survivors, median across the ten patient-level median RRE values:
   retain values at most this stage minimum plus 0.25 degrees.
4. Select the smallest lambda among remaining candidates.

Each patient median uses solver-success cases only; failures remain in the
15-case success-rate denominator. Threshold comparisons include equality and
are anchored to each stage's minimum, not pairwise chained ties. The protocol
retains the full V1 selection rule as provenance and explicitly replaces only
its global eligibility policy for V2. Metrics, priors, and the solver minimum
correspondence condition are unchanged.

## Validation-only outcome

| lambda_oss | Success | Success rate | Complete | Zero-success patients |
| --- | ---: | ---: | --- | --- |
| 0.25 | 128/150 | 85.333333% | false | Fold4 / Pat2 |
| 0.50 | 132/150 | 88.000000% | true | None |
| 1.00 | 109/150 | 72.666667% | false | Fold4 / Pat1; Fold4 / Pat2 |
| 2.00 | 7/150 | 4.666667% | false | Fold1 / Pat7; Fold1 / Pat4; Fold2 / Pat11; Fold2 / Pat3; Fold3 / Pat8; Fold3 / Pat9; Fold4 / Pat1; Fold4 / Pat2 |

Every listed zero-success patient has 0/15 successful cases. Stage 0 survivors
are `[0.5]`, so the selection reason is `single_complete_candidate`. Passing
this minimum gate does not establish uniformly reliable registration: at 0.5,
Fold4 / Pat1 has 9/15 successes and Fold4 / Pat2 has 3/15 successes.

All observed failures are `insufficient_correspondences`. The larger-lambda
degradation is consistent with stronger osseous modulation leaving fewer
surviving correspondences. It is not proof of a unique causal mechanism or an
evaluator bug. No formal test performance claim follows from this amendment.

## File access and independent output

The V2 executable has no result-root, dataset, checkpoint, protocol-path,
output-path, or test-path argument. It reads only the exact 20 paths of the form
`checkpoints/m4_osseous_strength_selection_v1/lambda_*/fold*/validation/validation_cases.jsonl`,
plus the bound V1 summary and V1/V2 protocol metadata and sidecars. It uses no
glob expansion. Symlink/reparse path components and paths outside this allowlist
are rejected before reading. Training logs, `train.jsonl`, checkpoints, test
datasets, and other evaluation results are never opened.

The independent output is
`checkpoints/m4_osseous_strength_selection_v1/validation_selection_v2_amendment_summary.json`.
It records the input hashes, all patient success/failure counts and rates, the
candidate-completeness trace, and the selection trace, while preserving:

```text
V1_SELECTION_ELIGIBLE=false
V1_SELECTED_LAMBDA_OSS=null
V2_AMENDMENT_SELECTION_ELIGIBLE=true
V2_SELECTED_LAMBDA_OSS=0.5
```

An existing output is rejected before reading result inputs. A flushed temporary
file is published with an atomic hard link that cannot replace an existing
destination, including a concurrent writer's output. Unsupported publication
fails closed. `--verify` only recomputes and compares this fixed V2 output; it
never rewrites the output or the V1 summary.

## CPU verification

Run from the repository root, with Python standard library only:

```console
python -B -m unittest discover -s tests/pointct -p test_m4_osseous_strength_selection_v2_amendment.py -v
python -B experiments/geotransformer.pointct.baseline_v1/aggregate_m4_osseous_strength_selection_v2_amendment.py --contract-audit
python -B experiments/geotransformer.pointct.baseline_v1/aggregate_m4_osseous_strength_selection_v2_amendment.py --aggregate
python -B experiments/geotransformer.pointct.baseline_v1/aggregate_m4_osseous_strength_selection_v2_amendment.py --verify
```

The `--aggregate` command is one-shot; once the output exists, use `--verify`.
The validation inputs must be provisioned at the exact local paths before
running the integration unittest; missing inputs fail, with no synthetic fallback.

Validation completed in this task: 36/36 unittest PASS; CPU contract audit PASS
(20 lambda/fold pairs, 600 expected identities); validation-only aggregation
PASS selecting 0.5; fixed-output read-only recomputation PASS; repeated real
aggregation correctly refused overwrite. V1 protocol, sidecar, frozen Markdown,
and original summary byte hashes remained unchanged. The generated V2 summary
byte SHA-256 is `cef85af30a69b5f8b265de6f643d103d6d335bcefd92f5132ebbafa56c1fb4f2`.
The suite covers all 20 requested behaviors, including the
real V1 grid, all four candidate states, no/multiple complete candidates, all
three original ranking stages, deterministic ties, invalid identities/scopes/
fields/numbers/hashes, restricted file reads, and output overwrite races.

Only the two V2 Python files, two new Markdown records, V2 protocol/sidecar, and
V2 generated summary belong to this amendment commit. Original V1 validation
inputs remain local input artifacts with their exact source-byte hash bindings.
