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

import validate_m3_matching as validation


class ValidateM3MatchingTest(unittest.TestCase):
    @staticmethod
    def _ct_branch():
        return {
            'ct_context_features': np.ones((2, 1), dtype=np.float32),
            'ct_context_indices': np.zeros((2, 3), dtype=np.int32),
            'ct_context_spatial_shape': np.asarray([2, 2, 2], dtype=np.int64),
            'ct_support_indices_20mm': np.zeros((1, 3), dtype=np.int32),
            'ct_support_linear_20mm': np.zeros((1,), dtype=np.int64),
            'ct_support_spatial_shape_20mm': np.asarray([1, 1, 1], dtype=np.int64),
            'ct_support_phys_20mm': np.zeros((1, 3), dtype=np.float64),
            'physical_unit': 'mm',
            'coordinate_system': 'DICOM LPS',
            'ct_volume': object(),
        }

    def test_validation_module_imports_without_real_data(self):
        self.assertTrue(hasattr(validation, 'validate_real_dataset'))
        self.assertTrue(hasattr(validation, 'validate_real_case'))

    def test_validation_source_explicitly_adds_project_root_to_sys_path(self):
        source = (EXPERIMENT_DIR / 'validate_m3_matching.py').read_text(encoding='utf-8')
        self.assertIn('if str(PROJECT_ROOT) not in sys.path:', source)
        self.assertIn('sys.path.insert(0, str(PROJECT_ROOT))', source)

    def test_real_forward_helper_has_no_forbidden_input_parameters(self):
        parameters = set(inspect.signature(validation.validate_real_case).parameters)
        forbidden = {
            'gt_transform',
            'gt_primary_ct_index',
            'gt_primary_valid',
            'gt_high_confidence',
        }
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_matcher_input_assembly_contains_only_descriptors_and_structural_masks(self):
        q = object()
        k = object()
        point_mask = object()
        ct_mask = object()
        inputs = validation.assemble_matcher_inputs(
            {'Q': q, 'Xp_phys_coarse': object()},
            {'K': k, 'Xv_phys_coarse': object()},
            point_mask,
            ct_mask,
        )
        self.assertEqual(
            set(inputs),
            {'q', 'k', 'point_valid_mask', 'ct_valid_mask'},
        )
        self.assertIs(inputs['q'], q)
        self.assertIs(inputs['k'], k)
        self.assertIs(inputs['point_valid_mask'], point_mask)
        self.assertIs(inputs['ct_valid_mask'], ct_mask)

    def test_descriptor_and_matcher_shape_validation_fail_closed(self):
        with self.assertRaisesRegex(validation.M3MatchingValidationError, 'Q must have shape'):
            validation.validate_descriptor_shapes(
                np.ones((0, 256), dtype=np.float32),
                np.ones((3, 256), dtype=np.float32),
            )
        with self.assertRaisesRegex(validation.M3MatchingValidationError, 'K must have shape'):
            validation.validate_descriptor_shapes(
                np.ones((2, 256), dtype=np.float32),
                np.ones((3, 128), dtype=np.float32),
            )
        with self.assertRaisesRegex(validation.M3MatchingValidationError, 'S must have shape'):
            validation.validate_matching_shapes(
                np.ones((2, 2), dtype=np.float32),
                np.ones((3, 4), dtype=np.float32),
                2,
                3,
            )

    def test_marginal_qa_accepts_correct_small_transport(self):
        probability = np.asarray(
            [[0.5, 0.5], [0.5, 0.5]],
            dtype=np.float32,
        )
        statistics = validation.calculate_transport_qa(
            probability,
            np.asarray([True]),
            np.asarray([True]),
        )
        validation.validate_marginal_qa(statistics, tolerance=1e-5)
        self.assertEqual(statistics['max_marginal_residual'], 0.0)

    def test_marginal_qa_rejects_wrong_transport(self):
        probability = np.asarray(
            [[1.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        statistics = validation.calculate_transport_qa(
            probability,
            np.asarray([True]),
            np.asarray([True]),
        )
        with self.assertRaisesRegex(validation.M3MatchingValidationError, 'exceeds'):
            validation.validate_marginal_qa(statistics, tolerance=1e-5)

    def test_helper_combined_residual_does_not_overwrite_ordinary_row_semantics(self):
        probability = np.asarray(
            [[0.5, 0.5], [0.3, 0.3]],
            dtype=np.float32,
        )
        statistics = validation.calculate_transport_qa(
            probability,
            np.asarray([True]),
            np.asarray([True]),
        )
        ordinary_point_residual = statistics['max_abs_point_row_residual']
        ordinary_ct_residual = statistics['max_abs_ct_column_residual']
        helper_row_residual = max(
            ordinary_point_residual,
            statistics['point_dustbin_row_residual'],
        )
        helper_col_residual = max(
            ordinary_ct_residual,
            statistics['ct_dustbin_column_residual'],
        )
        validation.record_helper_marginal_residuals(
            statistics,
            helper_row_residual,
            helper_col_residual,
        )
        self.assertEqual(statistics['max_abs_point_row_residual'], ordinary_point_residual)
        self.assertEqual(statistics['max_abs_ct_column_residual'], ordinary_ct_residual)
        self.assertGreater(
            statistics['helper_max_abs_row_residual'],
            statistics['max_abs_point_row_residual'],
        )
        self.assertEqual(
            statistics['max_marginal_residual'],
            max(
                statistics['point_dustbin_row_residual'],
                statistics['ct_dustbin_column_residual'],
                statistics['total_mass_residual'],
                statistics['helper_max_abs_row_residual'],
                statistics['helper_max_abs_col_residual'],
            ),
        )

    def test_runtime_helper_preserves_tensor_object_and_values(self):
        tensor = np.arange(6, dtype=np.float32).reshape(2, 3)
        original = tensor.copy()
        synchronizations = []
        result, elapsed = validation.measure_smoke_stage(
            lambda: tensor,
            lambda: synchronizations.append('sync'),
        )
        self.assertIs(result, tensor)
        np.testing.assert_array_equal(tensor, original)
        self.assertEqual(synchronizations, ['sync', 'sync'])
        self.assertGreaterEqual(elapsed, 0.0)

    def test_source_stops_at_forward_smoke_boundary(self):
        source_path = EXPERIMENT_DIR / 'validate_m3_matching.py'
        source = source_path.read_text(encoding='utf-8')
        lowered = source.lower()
        prohibited = {
            'loss',
            'mutual',
            'correspondence',
            'svd',
            'procrustes',
            'registration',
            'training',
            'ransac',
            'geometrictransformer',
            'localglobalregistration',
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

    def test_validation_reuses_matching_without_reimplementing_it(self):
        source_path = EXPERIMENT_DIR / 'validate_m3_matching.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        imported_from_matching = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'matching':
                imported_from_matching.update(alias.name for alias in node.names)
        self.assertNotIn('PointCTMatcher', class_names)
        self.assertNotIn('compute_cross_modal_similarity', function_names)
        self.assertNotIn('PointCTLogSinkhorn', class_names)
        self.assertEqual(
            imported_from_matching,
            {'PointCTMatcher', 'compute_marginal_residual'},
        )

    def test_ct_device_input_assembly_excludes_complete_volume(self):
        branch = self._ct_branch()
        moved_fields = []

        def adapter(value, name):
            moved_fields.append(name)
            return value

        encoder_input = validation.assemble_ct_encoder_input(branch, adapter)
        self.assertNotIn('ct_volume', encoder_input)
        self.assertNotIn('ct_volume', moved_fields)
        self.assertEqual(
            set(moved_fields),
            {
                'ct_context_features',
                'ct_context_indices',
                'ct_support_indices_20mm',
                'ct_support_linear_20mm',
                'ct_support_phys_20mm',
            },
        )

    def test_manifest_subject_extraction_preserves_record_order(self):
        class DatasetContractOnly:
            records = [{'subject_id': 'Pat2'}, {'subject_id': 'Pat1'}]
            skipped_records = [{'subject_id': 'Pat10'}]

        ready_ids, skipped_ids = validation.extract_manifest_subject_ids(DatasetContractOnly())
        self.assertEqual(ready_ids, ['Pat2', 'Pat1'])
        self.assertEqual(skipped_ids, ['Pat10'])


if __name__ == '__main__':
    unittest.main()
