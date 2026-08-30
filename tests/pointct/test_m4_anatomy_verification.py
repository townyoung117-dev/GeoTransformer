import csv
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import prepare_m4_anatomy_verification as prepare
import score_m4_anatomy_verification as scoring


def synthetic_cases():
    cases = {}
    for patient_number, subject_id in enumerate(prepare.CLEAN10_SUBJECT_IDS):
        count = 40
        cases[subject_id] = {
            'support_linear': np.arange(count, dtype=np.int64) + patient_number * 1000,
            'support_locations_mm': np.column_stack(
                (
                    np.arange(count, dtype=np.float64),
                    np.full(count, patient_number, dtype=np.float64),
                    np.arange(count, dtype=np.float64) * 0.5,
                )
            ),
            'score': np.linspace(0.01, 0.99, count, dtype=np.float64),
        }
    return cases


def synthetic_hidden_key():
    assignments = prepare.build_blinded_assignments(synthetic_cases())
    return {
        'schema_version': 1,
        'records': assignments['records'],
    }


def blank_rows(hidden_key):
    _, rows = prepare.reviewer_rows_from_hidden(hidden_key['records'])
    return rows


def completed_rows(hidden_key, *, passing=True):
    rows = blank_rows(hidden_key)
    row_by_point = {row['anonymous_point']: row for row in rows}
    counters = {
        (subject_id, group): 0
        for subject_id in prepare.CLEAN10_SUBJECT_IDS
        for group in ('HIGH', 'CONTROL')
    }
    positive_patients = set(prepare.CLEAN10_SUBJECT_IDS[:7]) if not passing else set(
        prepare.CLEAN10_SUBJECT_IDS
    )
    for record in hidden_key['records']:
        subject_id = record['subject_id']
        group = record['group']
        index = counters[(subject_id, group)]
        counters[(subject_id, group)] += 1
        if group == 'HIGH':
            target_count = 6 if subject_id in positive_patients else 2
            if index < target_count:
                category = 'NASAL_BRIDGE'
            elif index < target_count + 3:
                category = 'MAXILLA_ZYGOMATIC'
            else:
                category = 'OTHER_CRANIAL_BONE'
        else:
            category = 'ORBITAL_RIM' if index < 2 else 'OTHER_CRANIAL_BONE'
        row = row_by_point[record['anonymous_point']]
        row['anatomical_category'] = category
        row['reviewer_confidence'] = '4'
    return rows


class BlindedSamplingTest(unittest.TestCase):
    def setUp(self):
        self.cases = synthetic_cases()
        self.assignments = prepare.build_blinded_assignments(self.cases)

    def test_fixed_seed_reproducibility(self):
        repeated = prepare.build_blinded_assignments(self.cases)
        self.assertEqual(self.assignments, repeated)

    def test_exactly_ten_high_per_patient(self):
        for subject_id in prepare.CLEAN10_SUBJECT_IDS:
            patient = [
                row for row in self.assignments['records'] if row['subject_id'] == subject_id
            ]
            self.assertEqual(sum(row['group'] == 'HIGH' for row in patient), 10)

    def test_exactly_ten_control_per_patient(self):
        for subject_id in prepare.CLEAN10_SUBJECT_IDS:
            patient = [
                row for row in self.assignments['records'] if row['subject_id'] == subject_id
            ]
            self.assertEqual(sum(row['group'] == 'CONTROL' for row in patient), 10)

    def test_high_and_control_never_overlap(self):
        for subject_id in prepare.CLEAN10_SUBJECT_IDS:
            patient = [
                row for row in self.assignments['records'] if row['subject_id'] == subject_id
            ]
            high = {row['support_linear_id'] for row in patient if row['group'] == 'HIGH'}
            control = {
                row['support_linear_id'] for row in patient if row['group'] == 'CONTROL'
            }
            self.assertFalse(high & control)

    def test_exactly_ten_patients_and_two_hundred_points(self):
        records = self.assignments['records']
        self.assertEqual(len(set(row['subject_id'] for row in records)), 10)
        self.assertEqual(len(records), 200)

    def test_high_rule_uses_score_then_linear_id_tie_break(self):
        case = self.cases['Pat1'].copy()
        case['score'] = np.full(40, 0.5, dtype=np.float64)
        selected = prepare.sample_patient_tokens(case, 'Pat1')
        high = sorted(row['support_linear_id'] for row in selected if row['group'] == 'HIGH')
        self.assertEqual(high, list(range(10)))


class AnonymizationTest(unittest.TestCase):
    def setUp(self):
        self.hidden = synthetic_hidden_key()
        self.blinded, self.reviewer = prepare.reviewer_rows_from_hidden(
            self.hidden['records']
        )

    def test_anonymization_hides_patient_and_defect_ids(self):
        reviewer_text = json.dumps(
            {'blinded': self.blinded, 'reviewer': self.reviewer},
            sort_keys=True,
        )
        for subject_id in prepare.CLEAN10_SUBJECT_IDS:
            self.assertNotIn(subject_id, reviewer_text)
        self.assertNotIn(prepare.CANONICAL_DEFECT, reviewer_text)

    def test_reviewer_csv_has_no_score_or_group(self):
        self.assertEqual(tuple(self.reviewer[0]), prepare.REVIEWER_FIELDS)
        self.assertNotIn('score', self.reviewer[0])
        self.assertNotIn('group', self.reviewer[0])

    def test_hidden_key_recovers_every_group(self):
        by_point = {row['anonymous_point']: row['group'] for row in self.hidden['records']}
        self.assertEqual(len(by_point), 200)
        self.assertEqual(sum(value == 'HIGH' for value in by_point.values()), 100)
        self.assertEqual(sum(value == 'CONTROL' for value in by_point.values()), 100)

    def test_anonymous_case_order_is_seeded_not_patient_order(self):
        self.assertEqual(
            set(row['anonymous_case'] for row in self.hidden['records']),
            {f'CASE{index:02d}' for index in range(1, 11)},
        )
        case01_patient = next(
            row['subject_id']
            for row in self.hidden['records']
            if row['anonymous_case'] == 'CASE01'
        )
        self.assertEqual(case01_patient, prepare.build_blinded_assignments(synthetic_cases())['case_order'][0])


class ReviewerValidationTest(unittest.TestCase):
    def setUp(self):
        self.hidden = synthetic_hidden_key()

    def test_all_allowed_categories_validate(self):
        rows = blank_rows(self.hidden)
        for index, row in enumerate(rows):
            row['anatomical_category'] = prepare.ALLOWED_CATEGORIES[
                index % len(prepare.ALLOWED_CATEGORIES)
            ]
            row['reviewer_confidence'] = str(index % 5 + 1)
        validated = scoring.validate_reviewer_rows(rows, self.hidden)
        self.assertTrue(validated['is_complete'])

    def test_invalid_category_fails_closed(self):
        rows = completed_rows(self.hidden)
        rows[0]['anatomical_category'] = 'AUTOMATIC_NASAL_GUESS'
        with self.assertRaises(scoring.AnatomyScoringError):
            scoring.validate_reviewer_rows(rows, self.hidden)

    def test_missing_point_fails_closed(self):
        with self.assertRaises(scoring.AnatomyScoringError):
            scoring.validate_reviewer_rows(completed_rows(self.hidden)[:-1], self.hidden)

    def test_duplicate_point_fails_closed(self):
        rows = completed_rows(self.hidden)
        rows[-1] = dict(rows[0])
        with self.assertRaises(scoring.AnatomyScoringError):
            scoring.validate_reviewer_rows(rows, self.hidden)

    def test_incomplete_human_review_cannot_pass(self):
        report = scoring.score_reviewer_rows(blank_rows(self.hidden), self.hidden)
        self.assertEqual(report['status'], scoring.STATUS_PENDING)
        self.assertIsNone(report['gate'])
        self.assertNotEqual(report['status'], scoring.STATUS_PASS)

    def test_invalid_confidence_fails_closed(self):
        rows = completed_rows(self.hidden)
        rows[0]['reviewer_confidence'] = '5.0'
        with self.assertRaises(scoring.AnatomyScoringError):
            scoring.validate_reviewer_rows(rows, self.hidden)


class FrozenScoringTest(unittest.TestCase):
    def setUp(self):
        self.hidden = synthetic_hidden_key()

    def test_scoring_formulas_are_exact(self):
        report = scoring.score_reviewer_rows(completed_rows(self.hidden), self.hidden)
        first = report['patient_metrics'][0]
        self.assertAlmostEqual(first['target_fraction_high'], 0.6)
        self.assertAlmostEqual(first['target_fraction_control'], 0.2)
        self.assertAlmostEqual(first['target_enrichment'], 0.4)
        self.assertAlmostEqual(first['distractor_high_fraction'], 0.3)

    def test_pass_gate_toy_case(self):
        report = scoring.score_reviewer_rows(completed_rows(self.hidden), self.hidden)
        self.assertEqual(report['status'], scoring.STATUS_PASS)
        self.assertTrue(report['gate']['all_clauses_pass'])

    def test_fail_gate_toy_case(self):
        report = scoring.score_reviewer_rows(
            completed_rows(self.hidden, passing=False),
            self.hidden,
        )
        self.assertEqual(report['status'], scoring.STATUS_FAIL)
        self.assertEqual(report['gate']['positive_patient_count'], 7)
        self.assertFalse(
            report['gate']['clauses']['at_least_8_of_10_high_greater_than_control']
        )

    def test_identical_real_second_form_has_kappa_one_without_changing_gate(self):
        rows = completed_rows(self.hidden)
        report = scoring.score_verification(rows, self.hidden, [dict(row) for row in rows])
        self.assertEqual(report['inter_rater_agreement']['cohens_kappa'], 1.0)
        self.assertTrue(report['primary_gate_unchanged_by_agreement'])
        self.assertEqual(report['structure_level_anatomy_verification'], scoring.STATUS_PASS)


class IndependenceAndRenderingTest(unittest.TestCase):
    def test_formal_cli_has_no_gt_mask_or_registration_metric_input(self):
        prepare_destinations = {
            action.dest for action in prepare.build_arg_parser()._actions
        }
        score_destinations = {
            action.dest for action in scoring.build_arg_parser()._actions
        }
        forbidden = {
            'gt',
            'ground_truth',
            'defect_mask',
            'registration_metric',
            'registration_result',
        }
        self.assertFalse(prepare_destinations & forbidden)
        self.assertFalse(score_destinations & forbidden)
        self.assertEqual(
            tuple(inspect.signature(prepare.prepare_packet).parameters),
            ('data_root', 'output_root'),
        )

    def test_fixed_score_values_are_not_a_search_grid(self):
        self.assertEqual(prepare.CENTER_HU, 300.0)
        self.assertEqual(prepare.TAU_HU, 100.0)
        self.assertEqual(prepare.RADIUS_MM, 20.0)
        self.assertFalse(any(isinstance(value, (tuple, list)) for value in (
            prepare.CENTER_HU,
            prepare.TAU_HU,
            prepare.RADIUS_MM,
        )))

    def test_surface_extraction_reuses_frozen_foreground_threshold(self):
        source = inspect.getsource(prepare._extract_external_surface_sitk)
        self.assertIn('CT_FOREGROUND_HU', source)
        self.assertNotIn('> -900', source)

    def test_cpu_renderer_writes_three_channel_png_with_center_crosshair(self):
        try:
            import SimpleITK as sitk
        except ModuleNotFoundError:
            self.skipTest('SimpleITK is unavailable.')
        ct = {
            'ct_volume': np.full((9, 9, 9), 500, dtype=np.int16),
            'ct_spacing': np.ones(3, dtype=np.float64),
            'ct_origin': np.zeros(3, dtype=np.float64),
            'ct_direction': np.eye(3, dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'view.png'
            prepare.render_review_png(ct, (4.0, 4.0, 4.0), 'axial', path)
            array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        self.assertEqual(array.shape[-1], 3)
        center = array.shape[0] // 2
        self.assertEqual(tuple(array[center, center]), (255, 0, 0))

    def test_hidden_key_loader_round_trip(self):
        hidden = synthetic_hidden_key()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'hidden_key.json'
            path.write_text(json.dumps(hidden), encoding='utf-8')
            loaded = scoring.load_hidden_key(path)
        self.assertEqual(len(loaded['records']), 200)


if __name__ == '__main__':
    unittest.main()
