import ast
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import validate_m3_registration as validation


class ValidateM3RegistrationTest(unittest.TestCase):
    @staticmethod
    def _filter_output():
        return {
            'point_indices': np.asarray([0, 1, 2], dtype=np.int64),
            'ct_indices': np.asarray([2, 1, 0], dtype=np.int64),
            'confidence': np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
            'num_mutual_before_confidence': 3,
            'num_correspondences': 3,
        }

    @staticmethod
    def _registration_output(success=True, failure_reason=None):
        return {
            'success': success,
            'rotation': np.eye(3, dtype=np.float32) if success else None,
            'translation': np.zeros(3, dtype=np.float32) if success else None,
            'failure_reason': failure_reason,
            'num_correspondences': 3,
            'num_positive_weights': 3,
            'weight_sum': 2.1,
            'source_rank': 2,
            'target_rank': 2,
            'covariance_rank': 2,
            'det_rotation': 1.0 if success else None,
        }

    def _execute_tail(self, correspondence_filter, registration_estimator):
        return validation.execute_registration_tail(
            log_assignment=object(),
            point_valid_mask=object(),
            ct_valid_mask=object(),
            point_encoder_output={'Xp_phys_coarse': object()},
            ct_encoder_output={'Xv_phys_coarse': object()},
            correspondence_filter=correspondence_filter,
            registration_estimator=registration_estimator,
        )

    def test_validation_module_imports_without_real_data(self):
        self.assertTrue(hasattr(validation, 'validate_real_dataset'))
        self.assertTrue(hasattr(validation, 'validate_real_registration_case'))

    def test_manifest_subject_order_contract_is_reused(self):
        class DatasetContractOnly:
            records = [{'subject_id': 'Pat2'}, {'subject_id': 'Pat1'}]
            skipped_records = [{'subject_id': 'Pat10'}]

        ready_ids, skipped_ids = validation.extract_manifest_subject_ids(DatasetContractOnly())
        self.assertEqual(ready_ids, ['Pat2', 'Pat1'])
        self.assertEqual(skipped_ids, ['Pat10'])

    def test_validation_reuses_matching_filter_and_registration(self):
        source = (EXPERIMENT_DIR / 'validate_m3_registration.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.setdefault(node.module, set()).update(alias.name for alias in node.names)
        self.assertIn('matching_validation._load_runtime_components()', source)
        self.assertEqual(
            imported['matching_filter'],
            {'extract_dustbin_aware_mutual_correspondences'},
        )
        self.assertEqual(
            imported['registration'],
            {'estimate_weighted_point_to_ct_transform'},
        )
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn('PointCTMatcher', class_names)
        self.assertNotIn('PointCTLogSinkhorn', class_names)
        self.assertNotIn('_weighted_procrustes', function_names)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn('svd', called_attributes)
        self.assertNotIn('argmax', called_attributes)

    def test_physical_coordinates_and_confidence_are_forwarded_by_identity(self):
        point_physical = object()
        ct_physical = object()
        confidence = object()
        inputs = validation.assemble_registration_inputs(
            {'Xp_phys_coarse': point_physical, 'Xp_net_coarse': object()},
            {'Xv_phys_coarse': ct_physical},
            {
                'point_indices': object(),
                'ct_indices': object(),
                'confidence': confidence,
            },
        )
        self.assertIs(inputs['point_physical'], point_physical)
        self.assertIs(inputs['ct_physical'], ct_physical)
        self.assertIs(inputs['weights'], confidence)
        self.assertNotIn('Xp_net_coarse', inputs)

    def test_forbidden_supervision_is_not_an_input(self):
        forbidden = {
            'gt_transform',
            'gt_primary_ct_index',
            'gt_primary_valid',
            'gt_high_confidence',
        }
        for function in (
            validation.assemble_registration_inputs,
            validation.execute_registration_tail,
            validation.validate_real_registration_case,
        ):
            self.assertTrue(set(inspect.signature(function).parameters).isdisjoint(forbidden))
        source = (EXPERIMENT_DIR / 'validate_m3_registration.py').read_text(encoding='utf-8').lower()
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_registration_success_is_recorded_with_identity_diagnostic(self):
        tail = {
            'filter_output': self._filter_output(),
            'registration_output': self._registration_output(),
            'filtering_time': 0.01,
            'registration_time': 0.02,
        }
        fields = validation.build_registration_case_fields(tail)
        self.assertTrue(fields['registration_success'])
        self.assertIsNone(fields['failure_reason'])
        self.assertEqual(fields['rotation'], np.eye(3).tolist())
        self.assertEqual(fields['translation'], [0.0, 0.0, 0.0])
        self.assertEqual(fields['identity_rotation_deviation_deg'], 0.0)
        self.assertEqual(fields['identity_translation_norm_mm'], 0.0)

    def test_registration_mathematical_failure_is_a_completed_case(self):
        tail = {
            'filter_output': self._filter_output(),
            'registration_output': self._registration_output(
                success=False,
                failure_reason='insufficient_correspondences',
            ),
            'filtering_time': 0.01,
            'registration_time': 0.02,
        }

        def completed_case():
            return {'subject_id': 'Pat1', **validation.build_registration_case_fields(tail)}

        statistics_dict, pipeline_failure = validation.capture_pipeline_case(
            'Pat1',
            completed_case,
        )
        self.assertIsNone(pipeline_failure)
        self.assertFalse(statistics_dict['registration_success'])
        self.assertEqual(statistics_dict['failure_reason'], 'insufficient_correspondences')
        self.assertIsNone(statistics_dict['rotation'])
        self.assertIsNone(statistics_dict['translation'])

    def test_pipeline_exception_is_recorded_as_failure(self):
        def broken_case():
            raise ValueError('shape mismatch')

        statistics_dict, pipeline_failure = validation.capture_pipeline_case(
            'Pat3',
            broken_case,
        )
        self.assertIsNone(statistics_dict)
        self.assertEqual(
            pipeline_failure,
            {
                'subject_id': 'Pat3',
                'error_type': 'ValueError',
                'error_message': 'shape mismatch',
            },
        )

    def test_failed_registration_cannot_supply_identity_fallback(self):
        output = self._registration_output(
            success=False,
            failure_reason='insufficient_correspondences',
        )
        output['rotation'] = np.eye(3, dtype=np.float32)
        output['translation'] = np.zeros(3, dtype=np.float32)
        with self.assertRaisesRegex(
            validation.M3RegistrationSmokeValidationError,
            'must not provide a fallback transform',
        ):
            validation.validate_registration_output(output)

    def test_min_confidence_none_and_confidence_weights_are_exact(self):
        filter_output = self._filter_output()
        calls = {}

        def correspondence_filter(log_assignment, **kwargs):
            calls['log_assignment'] = log_assignment
            calls['filter_kwargs'] = kwargs
            return filter_output

        def registration_estimator(**kwargs):
            calls['registration_kwargs'] = kwargs
            return self._registration_output()

        tail = self._execute_tail(correspondence_filter, registration_estimator)
        self.assertIsNone(calls['filter_kwargs']['min_confidence'])
        self.assertIs(calls['registration_kwargs']['weights'], filter_output['confidence'])
        self.assertIs(tail['filter_output'], filter_output)

    def test_successful_rotation_must_be_in_so3(self):
        invalid = self._registration_output()
        invalid['rotation'] = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
        invalid['det_rotation'] = -1.0
        with self.assertRaisesRegex(
            validation.M3RegistrationSmokeValidationError,
            r'determinant \+1',
        ):
            validation.validate_registration_output(invalid)

    def test_identity_cannot_claim_success_with_insufficient_correspondences(self):
        invalid = self._registration_output()
        invalid['num_correspondences'] = 0
        invalid['num_positive_weights'] = 0
        invalid['source_rank'] = None
        invalid['target_rank'] = None
        invalid['covariance_rank'] = None
        with self.assertRaisesRegex(
            validation.M3RegistrationSmokeValidationError,
            'at least three correspondences',
        ):
            validation.validate_registration_output(invalid)

    def test_filter_and_registration_count_mismatch_is_pipeline_error(self):
        def correspondence_filter(*args, **kwargs):
            return self._filter_output()

        output = self._registration_output()
        output['num_correspondences'] = 4
        with self.assertRaisesRegex(
            validation.M3RegistrationSmokeValidationError,
            'counts are inconsistent',
        ):
            self._execute_tail(correspondence_filter, lambda **kwargs: output)

    def test_smoke_parameters_are_explicitly_not_frozen(self):
        self.assertEqual(validation.SMOKE_TEMPERATURE, 0.1)
        self.assertEqual(validation.SMOKE_SINKHORN_ITERATIONS, 20)
        self.assertEqual(validation.SMOKE_ALPHA_INIT, 1.0)
        self.assertIsNone(validation.SMOKE_MIN_CONFIDENCE)
        self.assertIn('SMOKE VALIDATION ONLY', validation.SMOKE_PARAMETER_NOTICE)
        self.assertIn('NOT FROZEN', validation.SMOKE_PARAMETER_NOTICE)
        self.assertIn('NOT FROZEN EVALUATION', validation.IDENTITY_DIAGNOSTIC_NOTICE)

    def test_summary_separates_pipeline_and_mathematical_failures(self):
        success = {
            'subject_id': 'Pat1',
            'Np': 5,
            'Nv': 6,
            'num_correspondences': 3,
            'registration_success': True,
            'failure_reason': None,
            'source_rank': 2,
            'target_rank': 2,
            'covariance_rank': 2,
            'det_rotation': 1.0,
            'matching_sinkhorn_time': 0.4,
            'filtering_time': 0.1,
            'registration_time': 0.2,
            'total_case_time': 1.0,
            'identity_rotation_deviation_deg': 2.0,
            'identity_translation_norm_mm': 4.0,
        }
        mathematical_failure = {
            **success,
            'subject_id': 'Pat2',
            'num_correspondences': 0,
            'registration_success': False,
            'failure_reason': 'insufficient_correspondences',
            'source_rank': None,
            'target_rank': None,
            'covariance_rank': None,
            'det_rotation': None,
            'identity_rotation_deviation_deg': None,
            'identity_translation_norm_mm': None,
        }
        pipeline_failures = [{'subject_id': 'Pat3'}]
        summary = validation.aggregate_registration_statistics(
            ['Pat1', 'Pat2', 'Pat3'],
            [success, mathematical_failure],
            pipeline_failures,
            total_runtime=3.0,
        )
        self.assertEqual(summary['ready_subjects'], 3)
        self.assertEqual(summary['completed_subjects'], 2)
        self.assertEqual(summary['pipeline_failed_subjects'], 1)
        self.assertEqual(summary['registration_success_count'], 1)
        self.assertEqual(summary['registration_failure_count'], 1)
        self.assertEqual(
            summary['failure_reason_histogram'],
            {'insufficient_correspondences': 1},
        )
        self.assertEqual(summary['total_correspondences'], 3)

    def test_source_stops_at_smoke_boundary_and_has_no_cuda_hardcode(self):
        source = (EXPERIMENT_DIR / 'validate_m3_registration.py').read_text(encoding='utf-8')
        lowered = source.lower()
        prohibited = {
            'training',
            'optimizer',
            'loss integration',
            'ransac',
            'icp',
            'localglobalregistration',
            'defect',
            'anatomical',
            'entropy',
            'fusion',
            'm4+',
            'cuda:0',
        }
        for token in prohibited:
            self.assertNotIn(token, lowered)
        tree = ast.parse(source)
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id.lower())
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr.lower())
        self.assertNotIn('backward', called_names)
        self.assertNotIn('step', called_names)


if __name__ == '__main__':
    unittest.main()
