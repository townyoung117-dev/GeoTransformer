import argparse
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
DIAGNOSTIC_PROTOCOL_PATH = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_diagnostic_v1.json'
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_m3_defect_metric_diagnostic as diagnostic_cli
from audit_m3_metric_diagnostic import run_audit
from defect_evaluation import (
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
)
from m3_metric_diagnostic import (
    DIAGNOSTIC_TRE_FIELDS,
    EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD,
    EXPECTED_TEST_CASE_COUNT,
    M3MetricDiagnosticContractError,
    aggregate_diagnostic_cases,
    build_pat6_forensic,
    compute_tre_diagnostics,
    cross_check_legacy_cases,
    legacy_parameter_rte_mm,
    load_diagnostic_protocol,
    make_rigid_transform,
    safe_correlation,
    shift_transform_origin,
    summarize_confidence,
    validate_case_identity_set,
    validate_diagnostic_protocol,
    validate_legacy_result_tree,
)
from training_protocol import FOLD_SUBJECTS, load_training_protocol


def _rotation_z(angle_deg):
    angle = np.deg2rad(angle_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _success_case(
    index=0,
    *,
    subject_id='Pat1',
    rre_deg=5.0,
    legacy_rte=20.0,
    centroid_tre=3.0,
    point_tre=4.0,
    inlier_ratio=0.5,
):
    return {
        'case_key': f'Fold1|{subject_id}|defect_a|hard|{index}',
        'fold_id': 'Fold1',
        'subject_id': subject_id,
        'defect_id': 'defect_a',
        'severity': 'hard',
        'variant_id': index,
        'perturbation_seed': 1000 + index,
        'solver_success': True,
        'solver_status': 'success',
        'registration_recall_hit': False,
        'rre_deg': rre_deg,
        'legacy_parameter_rte_mm': legacy_rte,
        'centroid_tre_mm': centroid_tre,
        'point_tre_mean_mm': point_tre,
        'point_tre_median_mm': point_tre,
        'point_tre_rmse_mm': point_tre + 1.0,
        'point_tre_p95_mm': point_tre + 2.0,
        'point_tre_max_mm': point_tre + 3.0,
        'inlier_ratio': inlier_ratio,
        'num_correspondences': 10,
        'num_inliers': 5,
        'confidence_mean': 0.7,
        'confidence_median': 0.7,
        'confidence_p05': 0.5,
        'confidence_p95': 0.9,
        'confidence_min': 0.4,
        'confidence_max': 0.95,
        'source_centroid_norm_mm': 800.0,
        'origin_rotation_lever_proxy_mm': 70.0,
    }


def _failure_case(index=1, *, subject_id='Pat2'):
    row = _success_case(index, subject_id=subject_id)
    row.update(
        {
            'solver_success': False,
            'solver_status': 'insufficient_correspondences',
            'registration_recall_hit': False,
            'rre_deg': None,
            'legacy_parameter_rte_mm': None,
            'inlier_ratio': 0.0,
            'num_correspondences': 0,
            'num_inliers': 0,
            'confidence_mean': None,
            'confidence_median': None,
            'confidence_p05': None,
            'confidence_p95': None,
            'confidence_min': None,
            'confidence_max': None,
            'origin_rotation_lever_proxy_mm': None,
        }
    )
    for field in DIAGNOSTIC_TRE_FIELDS:
        row[field] = None
    return row


def _formal_cases_by_fold():
    training = load_training_protocol(diagnostic_cli.DEFAULT_TRAINING_PROTOCOL)
    evaluation = load_defect_evaluation_protocol(
        diagnostic_cli.DEFAULT_EVALUATION_PROTOCOL
    )
    return {
        fold_id: build_defect_test_manifest(
            training,
            evaluation,
            fold_id,
        )['cases']
        for fold_id in FOLD_SUBJECTS
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
        encoding='utf-8',
    )


_ABSENT_ROOT = object()


def _write_legacy_tree(
    root,
    cases_by_fold,
    *,
    root_rows=_ABSENT_ROOT,
    per_fold_overrides=None,
):
    overrides = {} if per_fold_overrides is None else per_fold_overrides
    for fold_id in FOLD_SUBJECTS:
        rows = overrides.get(fold_id, cases_by_fold[fold_id])
        _write_jsonl(Path(root) / fold_id / 'cases.jsonl', rows)
    if root_rows is not _ABSENT_ROOT:
        _write_jsonl(Path(root) / 'cases.jsonl', root_rows)


class M3MetricMathTest(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray(
            [
                [1000.0, -800.0, 500.0],
                [1010.0, -800.0, 500.0],
                [1000.0, -790.0, 500.0],
                [1000.0, -800.0, 510.0],
            ],
            dtype=np.float64,
        )
        self.identity = np.eye(4, dtype=np.float64)

    def test_identical_transforms_make_every_tre_zero(self):
        result = compute_tre_diagnostics(
            self.points,
            self.identity,
            self.identity,
            solver_success=True,
        )
        for field in DIAGNOSTIC_TRE_FIELDS:
            self.assertEqual(result[field], 0.0)
        self.assertEqual(result['legacy_parameter_rte_mm'], 0.0)

    def test_pure_translation_mismatch_equals_translation_norm(self):
        predicted = make_rigid_transform(np.eye(3), [3.0, 4.0, 12.0])
        result = compute_tre_diagnostics(
            self.points,
            predicted,
            self.identity,
            solver_success=True,
        )
        for field in DIAGNOSTIC_TRE_FIELDS:
            self.assertAlmostEqual(result[field], 13.0, places=12)

    def test_rotation_mismatch_about_anatomy_has_nonzero_point_tre(self):
        centroid = np.mean(self.points, axis=0)
        rotation = _rotation_z(20.0)
        translation = centroid - rotation @ centroid
        predicted = make_rigid_transform(rotation, translation)
        result = compute_tre_diagnostics(
            self.points,
            predicted,
            self.identity,
            solver_success=True,
        )
        self.assertAlmostEqual(result['centroid_tre_mm'], 0.0, places=10)
        self.assertGreater(result['point_tre_mean_mm'], 0.0)
        self.assertGreater(result['legacy_parameter_rte_mm'], 100.0)

    def test_origin_shift_invariance_for_all_tre_statistics(self):
        predicted = make_rigid_transform(
            _rotation_z(9.0),
            [20.0, -5.0, 3.0],
        )
        ground_truth = make_rigid_transform(
            _rotation_z(-3.0),
            [-4.0, 2.0, 7.0],
        )
        original = compute_tre_diagnostics(
            self.points,
            predicted,
            ground_truth,
            solver_success=True,
        )
        shift = np.asarray([1000.0, -800.0, 500.0])
        shifted = compute_tre_diagnostics(
            self.points - shift,
            shift_transform_origin(predicted, shift),
            shift_transform_origin(ground_truth, shift),
            solver_success=True,
        )
        for field in DIAGNOSTIC_TRE_FIELDS:
            self.assertAlmostEqual(original[field], shifted[field], places=9)

    def test_legacy_parameter_rte_is_origin_sensitive_for_rotation_error(self):
        predicted = make_rigid_transform(_rotation_z(10.0), [0.0, 0.0, 0.0])
        ground_truth = self.identity
        before = legacy_parameter_rte_mm(
            predicted[:3, 3],
            ground_truth[:3, 3],
        )
        shift = np.asarray([1000.0, -800.0, 500.0])
        predicted_shifted = shift_transform_origin(predicted, shift)
        gt_shifted = shift_transform_origin(ground_truth, shift)
        after = legacy_parameter_rte_mm(
            predicted_shifted[:3, 3],
            gt_shifted[:3, 3],
        )
        self.assertEqual(before, 0.0)
        self.assertGreater(after, 100.0)

    def test_percentile_uses_numpy_linear_method(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        predicted = make_rigid_transform(_rotation_z(90.0), [0.0, 0.0, 0.0])
        result = compute_tre_diagnostics(
            points,
            predicted,
            self.identity,
            solver_success=True,
        )
        distances = np.sqrt(2.0) * np.asarray([0.0, 1.0, 2.0, 3.0])
        expected = float(np.percentile(distances, 95, method='linear'))
        self.assertAlmostEqual(result['point_tre_p95_mm'], expected, places=12)

    def test_rmse_is_sqrt_mean_squared_distance(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        predicted = make_rigid_transform(_rotation_z(90.0), [0.0, 0.0, 0.0])
        result = compute_tre_diagnostics(
            points,
            predicted,
            self.identity,
            solver_success=True,
        )
        distances = np.sqrt(2.0) * np.asarray([0.0, 1.0, 2.0, 3.0])
        expected = math.sqrt(float(np.mean(np.square(distances))))
        self.assertAlmostEqual(result['point_tre_rmse_mm'], expected, places=12)

    def test_solver_failure_keeps_all_tre_and_predicted_transform_null(self):
        result = compute_tre_diagnostics(
            self.points,
            None,
            self.identity,
            solver_success=False,
        )
        for field in DIAGNOSTIC_TRE_FIELDS:
            self.assertIsNone(result[field])
        self.assertIsNone(result['legacy_parameter_rte_mm'])
        self.assertIsNone(result['predicted_transform_4x4'])
        self.assertIsNone(result['origin_rotation_lever_proxy_mm'])
        self.assertEqual(len(result['source_centroid_mm']), 3)

    def test_invalid_so3_fails_closed(self):
        invalid = self.identity.copy()
        invalid[0, 0] = 2.0
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            r'SO\(3\)|repair',
        ):
            compute_tre_diagnostics(
                self.points,
                invalid,
                self.identity,
                solver_success=True,
            )

    def test_nan_and_inf_points_fail_closed(self):
        for invalid_value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(invalid_value=invalid_value):
                points = self.points.copy()
                points[0, 0] = invalid_value
                with self.assertRaisesRegex(
                    M3MetricDiagnosticContractError,
                    'finite',
                ):
                    compute_tre_diagnostics(
                        points,
                        self.identity,
                        self.identity,
                        solver_success=True,
                    )

    def test_empty_points_fail_closed(self):
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'N>0',
        ):
            compute_tre_diagnostics(
                np.empty((0, 3), dtype=np.float64),
                self.identity,
                self.identity,
                solver_success=True,
            )

    def test_confidence_summary_is_null_for_zero_correspondences(self):
        summary = summarize_confidence(np.empty((0,)), 0)
        self.assertTrue(all(value is None for value in summary.values()))


class M3MetricProtocolTest(unittest.TestCase):
    def test_protocol_hash_passes_and_tampering_fails(self):
        protocol = load_diagnostic_protocol(DIAGNOSTIC_PROTOCOL_PATH)
        self.assertEqual(
            protocol['diagnostic_protocol_version'],
            'm3_metric_diagnostic_v1',
        )
        tampered = copy.deepcopy(protocol)
        tampered['catastrophic_rotation_threshold_deg'] = 46.0
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'diagnostic_protocol_hash mismatch',
        ):
            validate_diagnostic_protocol(tampered)

    def test_cpu_audit_reports_all_required_contracts(self):
        result = run_audit()
        self.assertEqual(result['DIAGNOSTIC_PROTOCOL_AUDIT'], 'PASS')
        self.assertEqual(result['TRE_ORIGIN_SHIFT_INVARIANCE'], 'PASS')
        self.assertEqual(result['FROZEN_BASELINE_DIFF'], 'PASS')
        self.assertFalse(result['LOCAL_REAL_DATA_AUDIT_RUN'])


class M3MetricAggregationTest(unittest.TestCase):
    def test_null_failure_metrics_are_not_aggregated_as_zero(self):
        summary = aggregate_diagnostic_cases(
            [_success_case(0, centroid_tre=10.0), _failure_case(1)],
            expected_case_count=2,
            expected_pat6_case_count=0,
        )
        overall = summary['overall']
        self.assertEqual(overall['case_count'], 2)
        self.assertEqual(overall['solver_success_count'], 1)
        self.assertEqual(overall['solver_failure_count'], 1)
        self.assertEqual(overall['centroid_tre_mm']['sample_count'], 1)
        self.assertEqual(overall['centroid_tre_mm']['mean'], 10.0)
        self.assertEqual(overall['rre_deg']['sample_count'], 1)

    def test_zero_variance_correlation_returns_null(self):
        self.assertIsNone(safe_correlation([1.0, 1.0], [2.0, 3.0]))
        self.assertIsNone(safe_correlation([1.0, 2.0], [3.0, 3.0]))

    def test_pat6_catastrophic_count_and_worst_order_are_correct(self):
        rows = [
            _success_case(0, subject_id='Pat6', rre_deg=10.0),
            _success_case(1, subject_id='Pat6', rre_deg=45.0),
            _success_case(2, subject_id='Pat6', rre_deg=90.0),
        ]
        forensic = build_pat6_forensic(rows, expected_case_count=3)
        self.assertEqual(forensic['catastrophic_rotation_case_count'], 2)
        self.assertAlmostEqual(
            forensic['catastrophic_rotation_case_ratio'],
            2.0 / 3.0,
        )
        self.assertEqual(forensic['worst_10_cases'][0]['rre_deg'], 90.0)


class M3MetricIdentityAndLegacyTest(unittest.TestCase):
    def test_legacy_case_identity_duplicate_fails(self):
        first = _success_case(0)
        duplicate = copy.deepcopy(first)
        expected = [_success_case(0), _success_case(1)]
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'duplicate legacy case identity',
        ):
            validate_case_identity_set(
                [first, duplicate],
                expected,
                expected_count=2,
                label='legacy',
            )

    def test_exact_825_to_825_cross_check_contract(self):
        legacy_rows = []
        diagnostic_rows = []
        for index in range(EXPECTED_TEST_CASE_COUNT):
            diagnostic = _success_case(index)
            legacy = copy.deepcopy(diagnostic)
            legacy['rte_mm'] = legacy.pop('legacy_parameter_rte_mm')
            legacy_rows.append(legacy)
            diagnostic_rows.append(diagnostic)
        result = cross_check_legacy_cases(
            legacy_rows,
            diagnostic_rows,
            expected_count=EXPECTED_TEST_CASE_COUNT,
        )
        self.assertTrue(result['cross_check_pass'])
        self.assertEqual(result['case_count'], EXPECTED_TEST_CASE_COUNT)
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'case count mismatch',
        ):
            cross_check_legacy_cases(
                legacy_rows[:-1],
                diagnostic_rows,
                expected_count=EXPECTED_TEST_CASE_COUNT,
            )

    def test_legacy_continuous_mismatch_fails_with_first_case(self):
        diagnostic = _success_case(0)
        legacy = copy.deepcopy(diagnostic)
        legacy['rte_mm'] = legacy.pop('legacy_parameter_rte_mm') + 0.01
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'legacy mismatch at case',
        ):
            cross_check_legacy_cases([legacy], [diagnostic], expected_count=1)


class M3LegacyResultTreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases_by_fold = _formal_cases_by_fold()
        cls.ordered_union = [
            row
            for fold_id in FOLD_SUBJECTS
            for row in cls.cases_by_fold[fold_id]
        ]

    def _validate(self, root):
        return validate_legacy_result_tree(root, self.cases_by_fold)

    def test_complete_825_root_union_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(
                directory,
                self.cases_by_fold,
                root_rows=self.ordered_union,
            )
            result = self._validate(directory)
        self.assertEqual(
            result['legacy_root_cases_status'],
            'complete_union_verified',
        )
        self.assertEqual(result['legacy_root_cases_count'], 825)
        self.assertIsNone(result['legacy_root_cases_detected_fold'])

    def test_fold5_only_root_is_valid_single_fold_invocation_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(
                directory,
                self.cases_by_fold,
                root_rows=self.cases_by_fold['Fold5'],
            )
            result = self._validate(directory)
        self.assertEqual(
            result['legacy_root_cases_status'],
            'single_fold_invocation_artifact',
        )
        self.assertEqual(result['legacy_root_cases_count'], 150)
        self.assertEqual(result['legacy_root_cases_detected_fold'], 'Fold5')

    def test_fold1_only_root_is_valid_single_fold_invocation_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(
                directory,
                self.cases_by_fold,
                root_rows=self.cases_by_fold['Fold1'],
            )
            result = self._validate(directory)
        self.assertEqual(
            result['legacy_root_cases_status'],
            'single_fold_invocation_artifact',
        )
        self.assertEqual(result['legacy_root_cases_count'], 225)
        self.assertEqual(result['legacy_root_cases_detected_fold'], 'Fold1')

    def test_absent_root_passes_with_complete_per_fold_union(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(directory, self.cases_by_fold)
            result = self._validate(directory)
        self.assertEqual(result['legacy_root_cases_status'], 'absent')
        self.assertEqual(result['legacy_root_cases_count'], 0)
        self.assertIsNone(result['root'])
        self.assertIsNone(result['legacy_root_cases_detected_fold'])

    def test_root_149_151_or_mixed_identities_fails_closed(self):
        invalid_roots = {
            '149': self.cases_by_fold['Fold5'][:-1],
            '151': self.cases_by_fold['Fold5']
            + [self.cases_by_fold['Fold1'][0]],
            'mixed_150': self.cases_by_fold['Fold4'][:75]
            + self.cases_by_fold['Fold5'][:75],
        }
        for label, root_rows in invalid_roots.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                _write_legacy_tree(
                    directory,
                    self.cases_by_fold,
                    root_rows=root_rows,
                )
                with self.assertRaisesRegex(
                    M3MetricDiagnosticContractError,
                    'neither the complete 825-case|single-Fold',
                ):
                    self._validate(directory)

    def test_missing_one_case_from_any_per_fold_file_fails_closed(self):
        for fold_id in FOLD_SUBJECTS:
            with self.subTest(fold_id=fold_id), tempfile.TemporaryDirectory() as directory:
                _write_legacy_tree(
                    directory,
                    self.cases_by_fold,
                    per_fold_overrides={
                        fold_id: self.cases_by_fold[fold_id][:-1]
                    },
                )
                with self.assertRaisesRegex(
                    M3MetricDiagnosticContractError,
                    f'legacy {fold_id} case count mismatch',
                ):
                    self._validate(directory)

    def test_per_fold_union_duplicate_fails_closed(self):
        duplicate_fold = list(self.cases_by_fold['Fold3'])
        duplicate_fold[-1] = copy.deepcopy(duplicate_fold[0])
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(
                directory,
                self.cases_by_fold,
                per_fold_overrides={'Fold3': duplicate_fold},
            )
            with self.assertRaisesRegex(
                M3MetricDiagnosticContractError,
                'duplicate legacy Fold3 case identity',
            ):
                self._validate(directory)

    def test_complete_per_fold_union_is_authoritative_825_source(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(directory, self.cases_by_fold)
            result = self._validate(directory)
        self.assertEqual(
            result['per_fold_counts'],
            EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD,
        )
        self.assertEqual(result['per_fold_union_count'], 825)
        self.assertEqual(result['per_fold_union_identity_audit'], 'PASS')
        self.assertEqual(len(result['authoritative_union']), 825)
        self.assertEqual(result['authoritative_union'], self.ordered_union)


class M3MetricCLISafetyTest(unittest.TestCase):
    def test_no_explicit_mode_fails_before_execution(self):
        args = argparse.Namespace(audit_only=False, execute_diagnostic=False)
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'explicitly choose',
        ):
            diagnostic_cli._explicit_mode(args)

    def test_cli_default_device_is_cpu(self):
        parser = diagnostic_cli.build_argument_parser()
        args = parser.parse_args(
            [
                '--legacy-results-root',
                'legacy',
                '--output-root',
                'diagnostic',
                '--fold-id',
                'Fold1',
                '--audit-only',
            ]
        )
        self.assertEqual(args.device, 'cpu')

    def test_nonempty_output_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.mkdir()
            (output / 'existing.txt').write_text('do not overwrite', encoding='utf-8')
            with self.assertRaisesRegex(
                M3MetricDiagnosticContractError,
                'non-empty',
            ):
                diagnostic_cli._require_safe_empty_output_root(
                    output,
                    Path(directory) / 'legacy',
                )

    def test_audit_only_validates_825_legacy_cases_without_model_or_gpu(self):
        training = load_training_protocol(
            diagnostic_cli.DEFAULT_TRAINING_PROTOCOL
        )
        evaluation = load_defect_evaluation_protocol(
            diagnostic_cli.DEFAULT_EVALUATION_PROTOCOL
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root = root / 'legacy'
            fold5_rows = None
            for fold_id in FOLD_SUBJECTS:
                manifest = build_defect_test_manifest(
                    training,
                    evaluation,
                    fold_id,
                )
                rows = manifest['cases']
                if fold_id == 'Fold5':
                    fold5_rows = rows
                fold_dir = legacy_root / fold_id
                fold_dir.mkdir(parents=True)
                (fold_dir / 'cases.jsonl').write_text(
                    ''.join(
                        json.dumps(row, sort_keys=True) + '\n' for row in rows
                    ),
                    encoding='utf-8',
                )
            (legacy_root / 'cases.jsonl').write_text(
                ''.join(
                    json.dumps(row, sort_keys=True) + '\n' for row in fold5_rows
                ),
                encoding='utf-8',
            )
            args = argparse.Namespace(
                audit_only=True,
                execute_diagnostic=False,
                all_folds=True,
                fold_id=None,
                protocol_manifest=diagnostic_cli.DEFAULT_TRAINING_PROTOCOL,
                evaluation_protocol=diagnostic_cli.DEFAULT_EVALUATION_PROTOCOL,
                diagnostic_protocol=diagnostic_cli.DEFAULT_DIAGNOSTIC_PROTOCOL,
                legacy_results_root=legacy_root,
                output_root=root / 'output',
                checkpoint_root=None,
                json_log_root=None,
                data_root=Path('must-not-be-read'),
                device='cuda',
            )
            with mock.patch.object(
                diagnostic_cli,
                '_execute_fold',
                side_effect=AssertionError('model/GPU execution'),
            ):
                result = diagnostic_cli.run_evaluation(args)
            self.assertEqual(result['mode'], 'audit-only')
            self.assertEqual(result['global_expected_case_count'], 825)
            self.assertEqual(
                result['LEGACY_PER_FOLD_COUNTS'],
                EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD,
            )
            self.assertEqual(result['LEGACY_PER_FOLD_UNION_COUNT'], 825)
            self.assertEqual(
                result['LEGACY_PER_FOLD_UNION_IDENTITY_AUDIT'],
                'PASS',
            )
            self.assertEqual(
                result['legacy_root_cases_status'],
                'single_fold_invocation_artifact',
            )
            self.assertEqual(result['legacy_root_cases_count'], 150)
            self.assertEqual(result['legacy_root_cases_detected_fold'], 'Fold5')
            self.assertFalse(result['model_loaded'])
            self.assertFalse(result['gpu_used'])


if __name__ == '__main__':
    unittest.main()
