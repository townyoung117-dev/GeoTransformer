import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
EVALUATION_PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_7_eval_v1.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluation import (
    M3EvaluationContractError,
    aggregate_case_metrics,
    aggregate_runtime_metrics,
    correspondence_inlier_metrics,
    evaluate_registration_result,
    load_evaluation_protocol,
    rotation_error_degrees,
    translation_error_mm,
)


def _rotation_z(degrees):
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform(rotation=None, translation=None):
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value


class M3EvaluationMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_evaluation_protocol(EVALUATION_PROTOCOL_PATH)

    def test_rre_identity_is_zero(self):
        self.assertEqual(
            rotation_error_degrees(np.eye(3), np.eye(3)),
            0.0,
        )

    def test_rre_known_ninety_degree_rotation(self):
        self.assertAlmostEqual(
            rotation_error_degrees(_rotation_z(90.0), np.eye(3)),
            90.0,
            places=12,
        )

    def test_rte_is_euclidean_distance_in_mm(self):
        self.assertEqual(
            translation_error_mm([4.0, 6.0, 3.0], [1.0, 2.0, 3.0]),
            5.0,
        )

    def test_rre_acos_input_is_clamped(self):
        slightly_over_trace = np.eye(3, dtype=np.float64)
        slightly_over_trace[0, 0] += 1e-12
        self.assertEqual(
            rotation_error_degrees(slightly_over_trace, np.eye(3)),
            0.0,
        )

    def test_correspondence_inlier_ratio_uses_effective_point_to_ct_gt(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        transform = _transform(translation=[10.0, 0.0, 0.0])
        ct_points = np.asarray(
            [[10.0, 0.0, 0.0], [26.0, 0.0, 0.0], [27.1, 0.0, 0.0]],
            dtype=np.float32,
        )
        metrics = correspondence_inlier_metrics(
            points,
            ct_points,
            np.asarray([0, 1, 2]),
            np.asarray([0, 1, 2]),
            transform,
            15.0,
        )
        self.assertEqual(metrics['num_correspondences'], 3)
        self.assertEqual(metrics['num_inliers'], 2)
        self.assertAlmostEqual(metrics['inlier_ratio'], 2.0 / 3.0)

    def test_zero_correspondence_has_zero_inliers_and_ratio(self):
        metrics = correspondence_inlier_metrics(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.eye(4),
            15.0,
        )
        self.assertEqual(
            metrics,
            {'num_correspondences': 0, 'num_inliers': 0, 'inlier_ratio': 0.0},
        )

    def test_solver_failure_keeps_rre_and_rte_null(self):
        result = evaluate_registration_result(
            {
                'success': False,
                'failure_reason': 'insufficient_correspondences',
                'rotation': np.eye(3),
                'translation': np.zeros(3),
            },
            np.eye(4),
            5.0,
            10.0,
        )
        self.assertFalse(result['solver_success'])
        self.assertEqual(result['solver_status'], 'insufficient_correspondences')
        self.assertIsNone(result['rre_deg'])
        self.assertIsNone(result['rte_mm'])
        self.assertFalse(result['registration_recall_hit'])

    def test_exact_recall_threshold_boundary_is_a_hit(self):
        result = evaluate_registration_result(
            {
                'success': True,
                'failure_reason': None,
                'rotation': _rotation_z(5.0),
                'translation': np.asarray([10.0, 0.0, 0.0]),
            },
            np.eye(4),
            self.protocol['registration_rre_threshold_deg'],
            self.protocol['registration_rte_threshold_mm'],
        )
        self.assertTrue(result['solver_success'])
        self.assertTrue(result['registration_recall_hit'])
        self.assertAlmostEqual(result['rre_deg'], 5.0, places=10)
        self.assertAlmostEqual(result['rte_mm'], 10.0, places=10)

    def test_above_either_recall_threshold_is_not_a_hit(self):
        cases = (
            (_rotation_z(5.01), np.asarray([10.0, 0.0, 0.0])),
            (_rotation_z(5.0), np.asarray([10.01, 0.0, 0.0])),
        )
        for rotation, translation in cases:
            with self.subTest(translation=translation.tolist()):
                result = evaluate_registration_result(
                    {
                        'success': True,
                        'failure_reason': None,
                        'rotation': rotation,
                        'translation': translation,
                    },
                    np.eye(4),
                    5.0,
                    10.0,
                )
                self.assertTrue(result['solver_success'])
                self.assertFalse(result['registration_recall_hit'])

    def test_invalid_success_transform_fails_closed_without_fake_identity(self):
        result = evaluate_registration_result(
            {
                'success': True,
                'failure_reason': None,
                'rotation': np.zeros((3, 3)),
                'translation': np.zeros(3),
            },
            np.eye(4),
            5.0,
            10.0,
        )
        self.assertEqual(result['solver_status'], 'invalid_registration_output')
        self.assertFalse(result['solver_success'])
        self.assertIsNone(result['rre_deg'])
        self.assertIsNone(result['rte_mm'])

    def test_nonfinite_geometry_fails_closed(self):
        with self.assertRaisesRegex(M3EvaluationContractError, 'finite'):
            translation_error_mm([float('nan'), 0.0, 0.0], [0.0, 0.0, 0.0])


class M3EvaluationAggregationTest(unittest.TestCase):
    @staticmethod
    def _cases():
        return [
            {
                'subject_id': 'Pat1',
                'severity': 'mild',
                'solver_success': True,
                'rre_deg': 1.0,
                'rte_mm': 2.0,
                'registration_recall_hit': True,
                'inlier_ratio': 0.5,
                'inference_runtime_ms': 10.0,
            },
            {
                'subject_id': 'Pat1',
                'severity': 'hard',
                'solver_success': True,
                'rre_deg': 9.0,
                'rte_mm': 12.0,
                'registration_recall_hit': False,
                'inlier_ratio': 0.25,
                'inference_runtime_ms': 20.0,
            },
            {
                'subject_id': 'Pat2',
                'severity': 'hard',
                'solver_success': False,
                'rre_deg': None,
                'rte_mm': None,
                'registration_recall_hit': False,
                'inlier_ratio': 0.0,
                'inference_runtime_ms': 30.0,
            },
        ]

    def test_global_and_group_aggregation_keeps_failures_in_denominator(self):
        summary = aggregate_case_metrics(self._cases())
        self.assertEqual(summary['total_cases'], 3)
        self.assertEqual(summary['solver_success_cases'], 2)
        self.assertEqual(summary['solver_failure_cases'], 1)
        self.assertAlmostEqual(summary['solver_success_rate'], 2.0 / 3.0)
        self.assertAlmostEqual(summary['solver_failure_rate'], 1.0 / 3.0)
        self.assertAlmostEqual(summary['registration_recall'], 1.0 / 3.0)
        self.assertEqual(summary['rre_deg']['success_sample_count'], 2)
        self.assertEqual(summary['rre_deg']['mean'], 5.0)
        self.assertEqual(summary['rte_mm']['median'], 7.0)
        self.assertEqual(summary['inlier_ratio']['sample_count'], 3)
        self.assertEqual(set(summary['per_severity']), {'hard', 'mild'})
        self.assertEqual(set(summary['per_subject']), {'Pat1', 'Pat2'})
        self.assertEqual(summary['per_subject']['Pat2']['solver_failure_cases'], 1)

    def test_runtime_mean_median_and_p95_include_all_cases(self):
        runtime = aggregate_runtime_metrics(self._cases())
        self.assertEqual(runtime['sample_count'], 3)
        self.assertEqual(runtime['mean_ms'], 20.0)
        self.assertEqual(runtime['median_ms'], 20.0)
        self.assertEqual(runtime['p95_ms'], 29.0)

    def test_failed_case_with_numeric_error_is_rejected(self):
        cases = self._cases()
        cases[-1]['rre_deg'] = 0.0
        with self.assertRaisesRegex(M3EvaluationContractError, 'must keep'):
            aggregate_case_metrics(cases)


if __name__ == '__main__':
    unittest.main()
