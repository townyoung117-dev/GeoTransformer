# M4-2B3 Independent Structure-level Anatomical Verification Protocol

## Status and scope

This protocol was frozen before review of any sampled anatomy labels. It creates
an independent, blinded, patient-level human verification of the frozen V2
continuous osseous score. It does not train a model, run matching or
registration, infer anatomy automatically, or change the score parameters.

- Branch: `m4_anatomy_verification_v1`
- Unique legal base: `ed4b39f91bbc7b931edf2988b007024eaab44b3b`
- Frozen evidence wording: **CT-derived continuous osseous support evidence**
- Human review required: **YES**
- Status before completed external labels:
  `STRUCTURE_LEVEL_ANATOMY_VERIFICATION = PENDING_HUMAN_REVIEW`

No HU value, physical coordinate threshold, score, defect annotation, patient
identity, registration result, or atlas surrogate may be used to assign a true
anatomical category.

## Independent units and fixed case

The independent units are the ten clean10 patients:

`Pat1`, `Pat2`, `Pat3`, `Pat4`, `Pat5`, `Pat7`, `Pat8`, `Pat9`, `Pat11`,
and `Pat12`.

Exactly one case is used per patient. The canonical defect is frozen as
`defect_001_left_maxilla_cheek_medium`. The choice cannot be changed using the
current score, anatomy appearance, or registration performance.

## Frozen score and token construction

The existing read-only V2 helper is reused with exactly:

- `center_hu = 300 HU`
- `tau_hu = 100 HU`
- `radius_mm = 20 mm`

External CT support and 20 mm support tokens use the already frozen PointCT
preprocessing definitions. Only the defective CT volume and its image header
are opened. No ground truth, defect mask, registration metric, training output,
or matching output is an input.

## Blinded sampling and identity separation

The fixed random seed is `20260830`, using NumPy `PCG64` with deterministic
SHA-256-derived per-purpose sub-seeds.

For every patient:

1. HIGH comprises the ten highest V2-score tokens. Ties are resolved by
   ascending frozen support linear ID.
2. CONTROL comprises ten tokens drawn uniformly without replacement from all
   non-HIGH tokens using the patient-specific CONTROL sub-seed.
3. The selected 20 tokens are shuffled using a separate patient-specific POINT
   sub-seed.

The ten cases are shuffled with the CASE sub-seed. Anonymous identifiers have
the form `CASE01_POINT001`. The reviewer-facing files contain no patient ID,
defect ID, HIGH/CONTROL group, score, success/failure result, registration
result, or M4-2A reliability. The private `hidden_key.json` is written outside
the reviewer packet directory. The reviewer ZIP contains only the reviewer
directory.

## Fixed review material

Each point receives axial, coronal, and sagittal PNG views centered on its
physical LPS location. All views use:

- fixed `160 mm x 160 mm` physical context;
- fixed `1 mm` output pixel spacing;
- fixed CT bone window level `500 HU`, width `2000 HU`;
- a fixed red crosshair at the current point;
- no score, group, patient, or defect annotation.

Display orientation is frozen as:

- axial: image left-to-right `R -> L`, top-to-bottom `P -> A`;
- coronal: image left-to-right `R -> L`, top-to-bottom `S -> I`;
- sagittal: image left-to-right `A -> P`, top-to-bottom `S -> I`.

## Frozen human categories

Exactly one primary category must be selected for every point:

1. `NASAL_BRIDGE`
2. `ORBITAL_RIM`
3. `FRONTAL_BONE`
4. `MAXILLA_ZYGOMATIC`
5. `TEETH_DENTOALVEOLAR`
6. `MANDIBLE`
7. `OTHER_CRANIAL_BONE`
8. `NON_BONE_OR_UNCERTAIN`

Confidence is an integer from 1 to 5. Comments are optional. The target set is
frozen as `NASAL_BRIDGE`, `ORBITAL_RIM`, and `FRONTAL_BONE`. The distractor set
is frozen as `MAXILLA_ZYGOMATIC`, `TEETH_DENTOALVEOLAR`, and `MANDIBLE`.

At least one reviewer with craniofacial/CT anatomy competence is recommended.
The software never fills anatomy labels. If a genuinely independent second
reviewer is available, the first supplied form remains the preregistered primary
decision source; the second is scored separately and Cohen's kappa is reported
as additional evidence only.

## Frozen patient-level metrics and engineering gate

For every patient:

- `target_fraction_high = target HIGH labels / 10`
- `target_fraction_control = target CONTROL labels / 10`
- `target_enrichment = target_fraction_high - target_fraction_control`
- `distractor_high_fraction = distractor HIGH labels / 10`

`STRUCTURE_LEVEL_ANATOMY_VERIFICATION = PASS` requires all five clauses:

1. at least 8/10 patients have
   `target_fraction_high > target_fraction_control`;
2. patient-level median `target_fraction_high >= 0.50`;
3. patient-level median `target_enrichment >= 0.20`;
4. patient-level median `distractor_high_fraction <= 0.30`;
5. all ten patients have complete labels for all 20 points.

A complete review that misses any clause is `FAIL`. An intact 200-row form with
one or more blank category/confidence pairs remains `PENDING_HUMAN_REVIEW` and
cannot produce PASS. Missing, extra, or duplicate point rows, invalid
categories, and invalid confidence values fail closed as malformed input.

This gate is exploratory engineering evidence, not a significance test. It may
not be relaxed after labels are observed.
