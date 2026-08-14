import ast
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

from audit_m3_contract import (
    M3ContractAuditError,
    audit_evaluation_perturbation_protocol,
    audit_subject_splits,
    calculate_matrix_memory,
    compute_distance_statistics,
    compute_primary_statistics,
    extract_subject_ids,
    find_evaluation_perturbation_manifests,
    is_identity_gt_transform,
)


class M3ContractAuditTest(unittest.TestCase):
    @staticmethod
    def _primary(indices, valid=None, high_confidence=None, num_ct_tokens=None):
        indices = np.asarray(indices, dtype=np.int64)
        if valid is None:
            valid = np.ones(indices.shape, dtype=bool)
        if high_confidence is None:
            high_confidence = np.ones(indices.shape, dtype=bool)
        if num_ct_tokens is None:
            num_ct_tokens = int(indices.max()) + 1 if indices.size else 1
        return compute_primary_statistics(
            indices,
            np.asarray(valid, dtype=bool),
            np.asarray(high_confidence, dtype=bool),
            num_ct_tokens=num_ct_tokens,
        )

    def test_unique_primary_ct_count(self):
        statistics = self._primary([4, 1, 4, 2, 1])
        self.assertEqual(statistics['unique_primary_ct_count'], 3)

    def test_no_collision_contract(self):
        statistics = self._primary([0, 1, 2, 3])
        self.assertEqual(statistics['multiplicity_hist'], {1: 4})
        self.assertEqual(statistics['collision_group_count'], 0)
        self.assertEqual(statistics['collision_point_count'], 0)
        self.assertEqual(statistics['excess_collision_count'], 0)
        self.assertEqual(statistics['collision_point_ratio'], 0.0)
        self.assertEqual(statistics['excess_collision_ratio'], 0.0)

    def test_simple_collision_contract(self):
        statistics = self._primary([0, 1, 1, 2])
        self.assertEqual(statistics['unique_primary_ct_count'], 3)
        self.assertEqual(statistics['collision_group_count'], 1)
        self.assertEqual(statistics['collision_point_count'], 2)
        self.assertEqual(statistics['excess_collision_count'], 1)
        self.assertEqual(statistics['multiplicity_hist'], {1: 2, 2: 1})
        self.assertEqual(statistics['multiplicity_max'], 2)

    def test_complex_collision_contract(self):
        statistics = self._primary([0, 0, 0, 1, 1, 2])
        self.assertEqual(statistics['unique_primary_ct_count'], 3)
        self.assertEqual(statistics['collision_group_count'], 2)
        self.assertEqual(statistics['collision_point_count'], 5)
        self.assertEqual(statistics['excess_collision_count'], 3)
        self.assertEqual(statistics['multiplicity_hist'], {1: 1, 2: 1, 3: 1})
        self.assertEqual(statistics['multiplicity_max'], 3)
        self.assertAlmostEqual(statistics['multiplicity_mean'], 2.0)

    def test_invalid_points_are_excluded_from_collision_statistics(self):
        statistics = self._primary(
            [0, 0, 0, 1],
            valid=[True, False, False, True],
            high_confidence=[True, False, False, True],
        )
        self.assertEqual(statistics['valid_point_count'], 2)
        self.assertEqual(statistics['invalid_point_count'], 2)
        self.assertEqual(statistics['unique_primary_ct_count'], 2)
        self.assertEqual(statistics['collision_group_count'], 0)
        self.assertEqual(statistics['collision_point_count'], 0)
        self.assertEqual(statistics['excess_collision_count'], 0)

    def test_high_confidence_flag_does_not_change_primary_collision_statistics(self):
        first = self._primary([0, 1, 1, 2], high_confidence=[True, True, True, True])
        second = self._primary([0, 1, 1, 2], high_confidence=[False, False, True, False])
        collision_keys = (
            'unique_primary_ct_count',
            'collision_group_count',
            'collision_point_count',
            'collision_point_ratio',
            'excess_collision_count',
            'excess_collision_ratio',
            'multiplicity_max',
            'multiplicity_mean',
            'multiplicity_hist',
        )
        self.assertNotEqual(first['high_confidence_count'], second['high_confidence_count'])
        for key in collision_keys:
            self.assertEqual(first[key], second[key])

    def test_matrix_memory_calculation_is_exact(self):
        statistics = calculate_matrix_memory(2, 3)
        self.assertEqual(statistics['matrix_elements'], 6)
        self.assertEqual(statistics['raw_similarity_float32_MiB'], 2 * 3 * 4 / 1024 ** 2)
        self.assertEqual(statistics['dustbin_matrix_float32_MiB'], 3 * 4 * 4 / 1024 ** 2)

    def test_identity_and_non_identity_rigid_transform_detection(self):
        identity = np.eye(4, dtype=np.float64)
        translated = np.eye(4, dtype=np.float64)
        translated[:3, 3] = [1.0, -2.0, 3.0]
        self.assertTrue(is_identity_gt_transform(identity, 'Point Cloud -> CT'))
        self.assertFalse(is_identity_gt_transform(translated, 'Point Cloud -> CT'))

    def test_wrong_gt_direction_fails_closed(self):
        with self.assertRaisesRegex(M3ContractAuditError, 'not inverted automatically'):
            is_identity_gt_transform(np.eye(4), 'CT -> Point Cloud')

    def test_empty_valid_point_contract_has_no_nan_or_division_by_zero(self):
        statistics = self._primary(
            [0, 1, 1],
            valid=[False, False, False],
            high_confidence=[False, False, False],
        )
        self.assertEqual(statistics['valid_point_count'], 0)
        self.assertEqual(statistics['invalid_point_count'], 3)
        self.assertEqual(statistics['unique_primary_ct_count'], 0)
        self.assertEqual(statistics['multiplicity_hist'], {})
        self.assertEqual(statistics['multiplicity_max'], 0)
        self.assertEqual(statistics['multiplicity_mean'], 0.0)
        self.assertEqual(statistics['collision_point_ratio'], 0.0)
        self.assertEqual(statistics['excess_collision_ratio'], 0.0)

    def test_distance_statistics_use_valid_physical_mm_only(self):
        statistics = compute_distance_statistics(
            np.asarray([5.0, 10.0, 15.0, 100.0]),
            np.asarray([True, True, True, False]),
        )
        self.assertEqual(statistics['distance_count'], 3)
        self.assertEqual(statistics['p50_mm'], 10.0)
        self.assertEqual(statistics['max_mm'], 15.0)
        self.assertAlmostEqual(statistics['coverage_le_5_mm'], 1.0 / 3.0)
        self.assertAlmostEqual(statistics['coverage_le_10_mm'], 2.0 / 3.0)
        self.assertEqual(statistics['coverage_le_15_mm'], 1.0)
        self.assertEqual(statistics['coverage_le_17.5_mm'], 1.0)
        self.assertEqual(statistics['coverage_le_20_mm'], 1.0)

    def test_split_audit_reports_absent_present_and_leakage(self):
        self.assertEqual(
            audit_subject_splits([{'subject_id': 'A'}, {'subject_id': 'B'}])['split_status'],
            'ABSENT',
        )
        present = audit_subject_splits(
            [
                {'subject_id': 'A', 'split': 'train'},
                {'subject_id': 'B', 'split': 'val'},
                {'subject_id': 'C', 'split': 'test'},
            ]
        )
        self.assertEqual(present, {'split_status': 'PRESENT', 'splits': ['test', 'train', 'val']})
        with self.assertRaisesRegex(M3ContractAuditError, 'leakage'):
            audit_subject_splits(
                [
                    {'subject_id': 'A', 'derived_sample_id': 'one', 'split': 'train'},
                    {'subject_id': 'A', 'derived_sample_id': 'two', 'split': 'test'},
                ]
            )

    def test_subject_id_extraction_uses_frozen_record_contract_and_preserves_order(self):
        class RecordContractOnly:
            records = [{'subject_id': 'Pat2'}, {'subject_id': 'Pat1'}]
            skipped_records = [{'subject_id': 'Pat10'}, {'subject_id': 'Pat9'}]

        ready_ids, skipped_ids = extract_subject_ids(RecordContractOnly())
        self.assertEqual(ready_ids, ['Pat2', 'Pat1'])
        self.assertEqual(skipped_ids, ['Pat10', 'Pat9'])

    def test_filename_matched_empty_manifest_is_only_an_unverified_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(find_evaluation_perturbation_manifests(root), [])
            manifest = root / 'pointct_evaluation_perturbation_manifest.json'
            manifest.write_text('{}', encoding='utf-8')
            self.assertEqual(find_evaluation_perturbation_manifests(root), [manifest.resolve()])
            audit = audit_evaluation_perturbation_protocol(root)
            self.assertEqual(audit['perturbation_manifest_status'], 'ABSENT')
            self.assertEqual(audit['unverified_candidates'], [manifest.resolve()])

    def test_missing_explicit_perturbation_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.json'
            with self.assertRaisesRegex(M3ContractAuditError, 'missing'):
                audit_evaluation_perturbation_protocol(
                    directory,
                    explicit_manifest=missing,
                )

    def test_source_prohibits_post_m3_1_algorithm_imports_and_calls(self):
        source_path = EXPERIMENT_DIR / 'audit_m3_contract.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        prohibited_symbols = {
            'learnablelogoptimaltransport',
            'optimaltransport',
            'weighted_procrustes',
            'svd',
            'ransac',
            'geometrictransformer',
            'localglobalregistration',
            'normalize',
        }
        imported_names = set()
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split('.')[-1].lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id.lower())
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr.lower())
        self.assertTrue(prohibited_symbols.isdisjoint(imported_names))
        self.assertTrue(prohibited_symbols.isdisjoint(called_names))
        self.assertNotIn('torch', imported_names)
        self.assertTrue(
            {
                'm2_point_collate_fn',
                'm2_ct_collate_fn',
                'build_coarse_gt_correspondence',
            }.issubset(called_names)
        )


if __name__ == '__main__':
    unittest.main()
