# M4-2B2 Continuous CT-derived Osseous Support CPU Validation V2

## Pre-formal protocol freeze

This protocol was frozen on branch
`m4_continuous_osseous_prior_validation_v2` at
`d3c294cf55bd4b6ad48a982f52b0ffba7293de6a`, before any V2 formal score was
computed.

The authoritative data root is:

`D:\Medical_AR_Facial_Data\outputs`

The pre-run coverage audit found exactly 50 ready cases from the declared
clean10 cohort (10 patients times 5 defects). The formal defective CT audit
found HU extrema from -3024 HU to 3071 HU, both signed int16 and signed int32
storage, one observed spacing `(1, 1, 1)` mm, one observed identity direction,
and three observed image shapes. The implementation nevertheless retains the
general physical-image contract for anisotropic spacing, nonzero origin, and
orthonormal direction.

## Frozen evidence and score family

For finite CT intensity `HU`, centre `c`, and positive smoothness `tau`, the
continuous evidence is

`e(HU; c, tau) = sigmoid((HU - c) / tau)`.

For CT support token `j` at physical location `X_j`, the support score is

`s_j(c, tau, r) = mean_{x in N_r(X_j)} e(HU(x); c, tau)`,

where `N_r(X_j)` contains only valid in-volume voxel centres whose Euclidean
physical-mm distance from `X_j` is at most `r`. Image spacing, origin, and
direction are part of the index-to-physical mapping. A fixed voxel-index
radius is not permitted.

The frozen parameter grid is:

- centre: `c = 300 HU` only;
- smoothness: `tau = 75, 100, 150 HU`;
- physical radius: `r = 15, 20, 25 mm`;
- primary configuration: `c = 300 HU, tau = 100 HU, r = 20 mm`.

No centre, smoothness, or radius may be added or selected after formal results
are read. Registration performance is not part of this validation.

## Frozen gates

The V1 non-degeneracy gate remains mathematically applicable to a continuous
bounded score and is retained unchanged, before formal execution:

- score range at least `0.05`;
- score standard deviation at least `0.01`;
- fraction at `<= 0.01` or `>= 0.99` at most `0.98`;
- at least `3` unique score values.

The remaining frozen patient-level gates are:

- cross-defect median Spearman rho at least `0.70`;
- cross-defect median Top-20% Jaccard at least `0.50`;
- tau-sensitivity median Spearman rho at least `0.85`;
- radius-sensitivity median Spearman rho at least `0.70`;
- median absolute Spearman correlation with the diagnostic-only frozen M4-2A
  reliability strictly below `0.95`.

The final decision is `PASS` only if coverage, all 50 numeric contracts, all 50
non-degeneracy gates, all 10 patient gates in every declared category, and the
score-helper leakage audit pass. There is no conditional validation pass.

Even after a pass, the permitted interpretation is limited to stable,
input-only CT-derived continuous osseous support evidence. No nasal, orbital,
frontal, or other structure-level anatomical localization is claimed without
separate anatomical verification.
