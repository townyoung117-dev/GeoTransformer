# M4-2B3 Independent Structure-level Anatomical Verification Report

## Outcome

- `PACKET_GENERATION = PASS`
- `STRUCTURE_LEVEL_ANATOMY_VERIFICATION = PENDING_HUMAN_REVIEW`
- `HUMAN_REVIEW_REQUIRED = YES`
- `GPU_USED = NO`
- `TRAINING_PERFORMED = NO`
- `COMMIT = NO`
- `PUSH = NO`

This stage prepared and verified the blinded review workflow. It did not assign
an anatomical category to any real point and does not establish a nasal bridge,
orbital rim, frontal bone, or stable anatomical prior. The supported wording
remains **CT-derived continuous osseous support evidence**.

## Frozen execution identity

1. Branch: `m4_anatomy_verification_v1`
2. HEAD: `ed4b39f91bbc7b931edf2988b007024eaab44b3b`
3. Data root: `D:\Medical_AR_Facial_Data\outputs`
4. Canonical defect: `defect_001_left_maxilla_cheek_medium`
5. Independent patient count: 10 clean10 patients
6. Blinded point count: 200, exactly 20 per patient

The ten patients are Pat1, Pat2, Pat3, Pat4, Pat5, Pat7, Pat8, Pat9, Pat11,
and Pat12. Synthetic defect variants were not treated as independent patients.

## Sampling and anonymization

7. HIGH rule: use the frozen V2 helper at `center_hu=300`, `tau_hu=100`, and
   `radius_mm=20`; take the ten largest scores per patient, with ascending
   support linear ID as the deterministic tie-break.
8. CONTROL rule: uniformly sample ten tokens without replacement from all
   non-HIGH tokens using the patient-specific CONTROL sub-seed.
9. Random seed: `20260830`; RNG is NumPy PCG64 with SHA-256-derived independent
   CASE, CONTROL, and POINT sub-seeds.
10. Anonymous mapping: the cases are seeded-randomized, each selected 20-point
    set is independently seeded-randomized, and reviewer IDs use
    `CASE##_POINT###`. Patient, defect, group, coordinate, and score are retained
    only in `private/hidden_key.json`. The key is outside the reviewer directory
    and absent from the reviewer ZIP.

The generated canonical support-token counts were checked patient by patient
against the frozen V2 formal50 result for this exact defect; all 10/10 counts
matched exactly.

## Human categories and gate

11. Frozen reviewer categories:
    `NASAL_BRIDGE`, `ORBITAL_RIM`, `FRONTAL_BONE`, `MAXILLA_ZYGOMATIC`,
    `TEETH_DENTOALVEOLAR`, `MANDIBLE`, `OTHER_CRANIAL_BONE`, and
    `NON_BONE_OR_UNCERTAIN`.
12. Target categories: `NASAL_BRIDGE`, `ORBITAL_RIM`, `FRONTAL_BONE`.
13. Distractor categories: `MAXILLA_ZYGOMATIC`, `TEETH_DENTOALVEOLAR`,
    `MANDIBLE`.
14. Frozen PASS clauses, all required:
    at least 8/10 patients have target HIGH fraction greater than CONTROL;
    median target HIGH fraction is at least 0.50; median target enrichment is
    at least 0.20; median distractor HIGH fraction is at most 0.30; and 10/10
    patients have all 20 human labels.

An intact form with blank labels remains `PENDING_HUMAN_REVIEW`. Missing,
extra, duplicate, invalid-category, or invalid-confidence input fails closed.
For two genuinely independent reviewers, the first remains the primary gate
source; the second is scored separately and Cohen's kappa is additional
evidence only.

## Generated files

15. Reviewer-facing output:

- `output/m4_anatomy_verification/reviewer_packet.zip`
- `output/m4_anatomy_verification/reviewer_packet/README.md`
- `output/m4_anatomy_verification/reviewer_packet/blinded_points.csv`
- `output/m4_anatomy_verification/reviewer_packet/reviewer_form.csv`
- `output/m4_anatomy_verification/reviewer_packet/packet_manifest.json`
- `output/m4_anatomy_verification/reviewer_packet/blinded_images/` containing
  600 PNGs: axial, coronal, and sagittal for each of 200 points

Private/non-reviewer output:

- `output/m4_anatomy_verification/private/hidden_key.json`
- `output/m4_anatomy_verification/generation_manifest.json`

The reviewer form has 200/200 blank anatomy labels and 200/200 blank confidence
values. The ZIP contains 604 files in total, including 600 PNGs and no private
key entry. All 600 PNGs passed CPU decode, fixed 161x161 RGB shape, and centered
red-crosshair checks. ZIP CRC passed and reviewer-facing text leaked none of the
frozen patient IDs, canonical defect ID, or HIGH/CONTROL identities.

The generated artifacts are under the repository's ignored `output/` tree so
medical review images and the private unblinding key are not staged by normal
source-control workflows.

## Verification

16. New unittest: **25/25 PASS** using `unittest`.
17. Full PointCT regression: **675 tests run, 0 failures, 439 passed, 236 skipped**.
    Skips are the existing no-PyTorch/GPU dependency contracts. Source
    compilation with `py_compile` also passed.
18. `git diff --check`: **PASS**, including explicit no-index whitespace checks
    for new untracked files.
19. `git status --short`: only the new protocol, report, preparation script,
    scoring script, and unittest are untracked; ignored packet artifacts are not
    shown. No frozen file is modified.

The explicitly prohibited training, matching, M3/M4 evaluation, hard/soft M4,
and frozen V2 helper files were checked against their pre-work SHA-256 values;
all remained unchanged.

## Final declarations

20. `PACKET_GENERATION = PASS`
21. `STRUCTURE_LEVEL_ANATOMY_VERIFICATION = PENDING_HUMAN_REVIEW`
22. `HUMAN_REVIEW_REQUIRED = YES`
23. `GPU_USED = NO`
24. `TRAINING_PERFORMED = NO`
25. `COMMIT = NO`
26. `PUSH = NO`
