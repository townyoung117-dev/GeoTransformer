# M4-2B2 Continuous CT-derived Osseous Support CPU Validation V2 Report

## Provenance and frozen protocol

- Branch: `m4_continuous_osseous_prior_validation_v2`
- HEAD: `d3c294cf55bd4b6ad48a982f52b0ffba7293de6a`
- Unique legal base HEAD exact match: **PASS**
- Data root: `D:\Medical_AR_Facial_Data\outputs`
- Formal coverage: **50/50** (`10 patients x 5 defects`)
- Score execution: CPU only

The protocol, score family, parameter grid, and gates were frozen before the
formal50 score run. No parameter or gate was changed after reading results.

The exact continuous evidence is

`e(HU; c, tau) = sigmoid((HU - c) / tau)`.

The exact token score is

`s_j(c, tau, r) = mean_{x in N_r(X_j)} e(HU(x); c, tau)`,

where `N_r(X_j)` contains only valid in-volume voxel centres with physical-mm
Euclidean distance `<= r` from support location `X_j`. Spacing, origin, and
direction participate in the image-index-to-physical mapping.

The frozen parameter family is:

- centre: `c = 300 HU`;
- tau: `75, 100, 150 HU`;
- radius: `15, 20, 25 mm`;
- primary: `c = 300 HU, tau = 100 HU, radius = 20 mm`.

The unchanged V1 gates are: non-degeneracy range `>= 0.05`, standard deviation
`>= 0.01`, endpoint fraction `<= 0.98`, unique values `>= 3`; cross-defect
patient median Spearman rho `>= 0.70` and Top-20% Jaccard `>= 0.50`; tau
sensitivity rho `>= 0.85`; radius sensitivity rho `>= 0.70`; M4-2A diagnostic
median absolute rho `< 0.95`.

## Files

All implementation files are independent V2 additions. No V1, M3, M4-1, or
M4-2A file was overwritten.

- `M4_CONTINUOUS_OSSEOUS_PRIOR_VALIDATION_V2_PROTOCOL.md`
- `experiments/geotransformer.pointct.baseline_v1/m4_continuous_osseous_prior.py`
- `experiments/geotransformer.pointct.baseline_v1/validate_m4_continuous_osseous_prior.py`
- `tests/pointct/test_m4_continuous_osseous_prior.py`
- `experiments/geotransformer.pointct.baseline_v1/results/m4_continuous_osseous_prior_validation_v2.json`
- `M4_CONTINUOUS_OSSEOUS_PRIOR_VALIDATION_V2_REPORT.md`

## Formal50 numeric and non-degeneracy results

- Numeric contract: **50/50 PASS**
- Non-empty and exact token count: **50/50 PASS**
- Finite, no NaN/Inf, strictly bounded score: **50/50 PASS**
- Non-degeneracy: **50/50 PASS**
- Support-token count across cases: min `1391`, median `1685`, max `2082`

The following table summarizes each primary-score statistic across the 50
per-case statistic values. Columns are the minimum, median, and maximum across
cases.

| Statistic | Minimum | Median | Maximum |
|---|---:|---:|---:|
| min | 0.000001427 | 0.000004697 | 0.000012379 |
| mean | 0.036881 | 0.049981 | 0.072930 |
| max | 0.392591 | 0.479612 | 0.597965 |
| std | 0.058424 | 0.069594 | 0.077606 |
| P10 | 0.000562 | 0.003183 | 0.007456 |
| P50 | 0.007231 | 0.024869 | 0.043612 |
| P90 | 0.104810 | 0.136999 | 0.182474 |

## Patient-level validation and V1 comparison

All reported rho values below are the frozen V1-style patient medians. The V1
column is the recomputed frozen hard-threshold sensitivity family at
`200/300/400 HU`; its failed-patient set exactly reproduced the frozen V1
history.

| Patient | Cross rho | Top-20 Jaccard | V2 tau rho | Radius rho | abs(V2 vs M4-2A rho) | V1 threshold rho | V1 failure repaired |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Pat1 | 0.999798 | 0.992857 | 0.963156 | 0.954427 | 0.181195 | 0.870458 | n/a |
| Pat2 | 0.999931 | 0.980952 | 0.992770 | 0.966549 | 0.436255 | 0.931840 | n/a |
| Pat3 | 0.999886 | 0.994460 | 0.973877 | 0.957080 | 0.301771 | 0.805637 | **YES** |
| Pat4 | 0.999917 | 0.954930 | 0.972440 | 0.969476 | 0.205967 | 0.879159 | n/a |
| Pat5 | 0.999909 | 0.994898 | 0.985595 | 0.959879 | 0.392840 | 0.860842 | n/a |
| Pat7 | 0.999926 | 0.964072 | 0.970764 | 0.962338 | 0.270419 | 0.822435 | **YES** |
| Pat8 | 0.999630 | 0.984207 | 0.977487 | 0.965081 | 0.252907 | 0.839419 | **YES** |
| Pat9 | 0.999742 | 0.981538 | 0.960505 | 0.948501 | 0.171890 | 0.877938 | n/a |
| Pat11 | 0.999870 | 0.986622 | 0.964459 | 0.962492 | 0.157369 | 0.865030 | n/a |
| Pat12 | 0.999853 | 0.955844 | 0.974096 | 0.965782 | 0.272029 | 0.816528 | **YES** |

Gate counts:

- Cross-defect stability: **10/10 PASS**
- Tau sensitivity: **10/10 PASS**
- Radius sensitivity: **10/10 PASS**
- M4-2A independence: **10/10 PASS**

The three declared tau comparisons were all retained. Across their ten
patient-level medians:

| Tau pair (HU) | Minimum | Median | Maximum |
|---|---:|---:|---:|
| 75 vs 100 | 0.968691 | 0.979203 | 0.996596 |
| 75 vs 150 | 0.871383 | 0.910143 | 0.982185 |
| 100 vs 150 | 0.960505 | 0.973159 | 0.992770 |

The continuous family therefore removed the observed V1 hard-threshold
failure mode for all four previously failing patients: Pat3, Pat7, Pat8, and
Pat12. This conclusion is limited to tau stability of continuous CT evidence;
it is not based on registration performance.

## Leakage, tests, and regression

Leakage audit: **PASS**.

- Both score APIs have exactly these parameters: `ct_volume`, `ct_spacing`,
  `ct_origin`, `ct_direction`, `support_locations_mm`, `center_hu`, `tau_hu`,
  `radius_mm`.
- Forbidden signature intersection: empty.
- Forbidden AST identifier intersection: empty.
- Helper imports: `math`, `typing`, `numpy` only.
- Defect mask and M4-2A reliability are not read during score generation.
- The M4 annotation is opened only after both CT-only V2 and historical V1
  score grids are complete, for diagnostic independence comparison only.

New unittest: **23/23 PASS** using `unittest`.

Related regression: **69 tests run, 0 failures**. Of these, 36 passed and 33
were skipped by existing no-PyTorch skip contracts. No PyTorch installation,
training, matching run, or registration evaluation was performed.

Whitespace audit: `git diff --check` and explicit no-index checks for all new
files: **PASS**.

## Decision and interpretation

`CONTINUOUS_OSSEOUS_PRIOR_VALIDATION = PASS`

`ANATOMY_PRIOR_IMPLEMENTATION_READY = CONDITIONAL`

The result supports only the interpretation that CT-derived continuous
osseous support evidence is a stable, input-only CT-side reliability candidate
that is independent of the defect-distance diagnostic under the frozen gates.
It does not establish nasal, orbital, frontal, or any other structure-level
anatomical localization. The conditional status remains because independent
structure-level anatomical verification has not been performed.

Recommended next step: freeze this validation evidence, then design a separate
pre-registered structure-level anatomical verification stage before assigning
anatomical semantics or integrating the candidate into matching/training.

- `GPU_USED = NO`
- `TRAINING_PERFORMED = NO`
- `AUTODL_USED = NO`
- `CHECKPOINT_GENERATED = NO`
- `COMMIT = NO`
- `PUSH = NO`
