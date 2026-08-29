import ast
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch

    import m4_hard_constraint
    from gt_correspondence import build_coarse_gt_correspondence
    from m4_hard_constraint import (
        M4HardConstraintError,
        build_m4_mask_aware_gt,
        resolve_m4_matching_masks,
    )
else:
    torch = None


class M4HardConstraintSourceContractTest(unittest.TestCase):
    def test_independent_module_exports_frozen_public_api(self):
        source_path = EXPERIMENT_DIR / 'm4_hard_constraint.py'
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertIn('M4HardConstraintError', definitions)
        self.assertEqual(
            [argument.arg for argument in definitions['resolve_m4_matching_masks'].args.args],
            ['point_encoder_output', 'ct_encoder_output', 'device'],
        )
        self.assertEqual(
            [argument.arg for argument in definitions['build_m4_mask_aware_gt'].args.args],
            [
                'Xp_phys_coarse',
                'Xv_phys_coarse',
                'gt_transform',
                'gt_transform_direction',
                'primary_max_distance_mm',
                'high_confidence_distance_mm',
                'point_valid_mask',
                'ct_valid_mask',
            ],
        )

    def test_gt_wrapper_reuses_frozen_builder_without_changing_feature_descriptors(self):
        source = (EXPERIMENT_DIR / 'm4_hard_constraint.py').read_text(encoding='utf-8')
        self.assertIn('from gt_correspondence import build_coarse_gt_correspondence', source)
        self.assertIn('subset = build_coarse_gt_correspondence(', source)
        tree = ast.parse(source)
        resolver = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'resolve_m4_matching_masks'
        )
        assigned_names = {
            target.id
            for node in ast.walk(resolver)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertNotIn('Q', assigned_names)
        self.assertNotIn('K', assigned_names)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M4 hard-constraint tests.')
class M4MatchingMaskResolverTest(unittest.TestCase):
    @staticmethod
    def _outputs(point_count=3, ct_count=4):
        return (
            {'Q': torch.arange(point_count * 2, dtype=torch.float32).reshape(point_count, 2)},
            {'K': torch.arange(ct_count * 2, dtype=torch.float32).reshape(ct_count, 2)},
        )

    @staticmethod
    def _add_mapping(output, prefix, intact):
        intact = torch.as_tensor(intact, dtype=torch.bool)
        total = torch.full(intact.shape, 2, dtype=torch.int64)
        defect = torch.where(
            intact,
            torch.zeros_like(total),
            torch.ones_like(total),
        )
        output[f'{prefix}_intact_coarse'] = intact
        output[f'{prefix}_raw_total_count_coarse'] = total
        output[f'{prefix}_raw_defect_count_coarse'] = defect

    def _mapped_outputs(self, point_intact=(True, False, True), ct_intact=(False, True, True, False)):
        point_output, ct_output = self._outputs(len(point_intact), len(ct_intact))
        self._add_mapping(point_output, 'point', point_intact)
        self._add_mapping(ct_output, 'ct', ct_intact)
        return point_output, ct_output

    def test_mapping_off_is_frozen_all_true_fallback(self):
        point_output, ct_output = self._outputs()
        result = resolve_m4_matching_masks(point_output, ct_output, torch.device('cpu'))

        self.assertIs(result['enabled'], False)
        torch.testing.assert_close(result['point_valid_mask'], torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(result['ct_valid_mask'], torch.ones(4, dtype=torch.bool))
        self.assertEqual(result['point_total_count'], 3)
        self.assertEqual(result['point_intact_count'], 3)
        self.assertEqual(result['point_excluded_count'], 0)
        self.assertEqual(result['ct_total_count'], 4)
        self.assertEqual(result['ct_intact_count'], 4)
        self.assertEqual(result['ct_excluded_count'], 0)

    def test_mapping_on_returns_exact_encoder_masks_and_diagnostics(self):
        point_output, ct_output = self._mapped_outputs()
        q_before = point_output['Q'].clone()
        k_before = ct_output['K'].clone()

        result = resolve_m4_matching_masks(point_output, ct_output, 'cpu')

        self.assertIs(result['enabled'], True)
        self.assertIs(result['point_valid_mask'], point_output['point_intact_coarse'])
        self.assertIs(result['ct_valid_mask'], ct_output['ct_intact_coarse'])
        torch.testing.assert_close(result['point_valid_mask'], torch.tensor([True, False, True]))
        torch.testing.assert_close(result['ct_valid_mask'], torch.tensor([False, True, True, False]))
        self.assertEqual(
            (
                result['point_total_count'],
                result['point_intact_count'],
                result['point_excluded_count'],
            ),
            (3, 2, 1),
        )
        self.assertEqual(
            (
                result['ct_total_count'],
                result['ct_intact_count'],
                result['ct_excluded_count'],
            ),
            (4, 2, 2),
        )
        torch.testing.assert_close(point_output['Q'], q_before)
        torch.testing.assert_close(ct_output['K'], k_before)

    def test_point_only_mapping_fails_closed(self):
        point_output, ct_output = self._outputs()
        self._add_mapping(point_output, 'point', (True, False, True))
        with self.assertRaisesRegex(M4HardConstraintError, 'Point-only'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_ct_only_mapping_fails_closed(self):
        point_output, ct_output = self._outputs()
        self._add_mapping(ct_output, 'ct', (True, False, True, True))
        with self.assertRaisesRegex(M4HardConstraintError, 'CT-only'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_partial_mapping_fails_closed(self):
        point_output, ct_output = self._mapped_outputs()
        del point_output['point_raw_defect_count_coarse']
        with self.assertRaisesRegex(M4HardConstraintError, 'incomplete.*missing fields'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_zero_intact_point_fails_closed(self):
        point_output, ct_output = self._mapped_outputs(point_intact=(False, False))
        with self.assertRaisesRegex(M4HardConstraintError, 'at least one intact Point'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_zero_intact_ct_fails_closed(self):
        point_output, ct_output = self._mapped_outputs(ct_intact=(False, False))
        with self.assertRaisesRegex(M4HardConstraintError, 'at least one intact CT'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_mapping_shape_mismatch_fails_closed(self):
        point_output, ct_output = self._mapped_outputs()
        point_output['point_intact_coarse'] = point_output['point_intact_coarse'][:-1]
        with self.assertRaisesRegex(M4HardConstraintError, 'exact token shape'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_mapping_dtype_mismatch_fails_closed(self):
        point_output, ct_output = self._mapped_outputs()
        ct_output['ct_raw_total_count_coarse'] = ct_output[
            'ct_raw_total_count_coarse'
        ].to(torch.int32)
        with self.assertRaisesRegex(M4HardConstraintError, 'torch.int64'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_mapping_device_mismatch_fails_closed(self):
        point_output, ct_output = self._mapped_outputs()
        point_output['point_intact_coarse'] = torch.ones(
            3,
            dtype=torch.bool,
            device='meta',
        )
        with self.assertRaisesRegex(M4HardConstraintError, 'requested token device'):
            resolve_m4_matching_masks(point_output, ct_output, 'cpu')

    def test_count_and_intact_consistency_fail_closed(self):
        for field, replacement, message in (
            ('point_raw_total_count_coarse', torch.tensor([2, 0, 2]), 'positive'),
            ('point_raw_defect_count_coarse', torch.tensor([0, 3, 0]), 'lie in'),
            ('point_intact_coarse', torch.tensor([True, True, True]), 'must equal'),
        ):
            with self.subTest(field=field):
                point_output, ct_output = self._mapped_outputs()
                point_output[field] = replacement
                with self.assertRaisesRegex(M4HardConstraintError, message):
                    resolve_m4_matching_masks(point_output, ct_output, 'cpu')


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M4 mask-aware GT tests.')
class M4MaskAwareGTTest(unittest.TestCase):
    @staticmethod
    def _build(points, ct_support, point_mask, ct_mask):
        return build_m4_mask_aware_gt(
            Xp_phys_coarse=np.asarray(points, dtype=np.float64),
            Xv_phys_coarse=np.asarray(ct_support, dtype=np.float64),
            gt_transform=np.eye(4, dtype=np.float64),
            gt_transform_direction='Point Cloud -> CT',
            primary_max_distance_mm=17.5,
            high_confidence_distance_mm=15.0,
            point_valid_mask=point_mask,
            ct_valid_mask=ct_mask,
        )

    def test_defect_point_is_unsupervised_in_full_length_output(self):
        output = self._build(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            np.asarray([True, False, True]),
            np.asarray([True, True]),
        )

        self.assertEqual(output['gt_primary_ct_index'].shape, (3,))
        np.testing.assert_array_equal(output['gt_primary_ct_index'], [0, -1, 1])
        np.testing.assert_array_equal(output['gt_primary_valid'], [True, False, True])
        np.testing.assert_array_equal(output['gt_high_confidence'], [True, False, True])
        self.assertTrue(np.isinf(output['gt_primary_distance_mm'][1]))

    def test_nearest_defect_ct_is_replaced_by_nearest_intact_ct(self):
        output = self._build(
            [[0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            torch.tensor([True]),
            torch.tensor([False, True, True]),
        )

        self.assertEqual(int(output['gt_primary_ct_index'][0]), 1)
        self.assertAlmostEqual(float(output['gt_primary_distance_mm'][0]), 1.9)
        self.assertTrue(bool(output['gt_primary_valid'][0]))
        self.assertTrue(bool(np.asarray([False, True, True])[output['gt_primary_ct_index'][0]]))

    def test_wrapper_calls_frozen_builder_on_intact_subsets_and_remaps_targets(self):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return build_coarse_gt_correspondence(**kwargs)

        with mock.patch.object(m4_hard_constraint, 'build_coarse_gt_correspondence', spy):
            output = self._build(
                [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                np.asarray([True, False, True]),
                np.asarray([False, True, True]),
            )

        np.testing.assert_array_equal(
            captured['Xp_phys_coarse'],
            [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(
            captured['Xv_phys_coarse'],
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(output['gt_primary_ct_index'], [1, -1, 2])
        supervised_targets = output['gt_primary_ct_index'][output['gt_primary_valid']]
        self.assertTrue(np.all(np.asarray([False, True, True])[supervised_targets]))

    def test_mask_shape_dtype_and_nonempty_contracts_fail_closed(self):
        cases = (
            (np.asarray([1], dtype=np.int64), np.asarray([True]), 'bool dtype'),
            (np.asarray([True, False]), np.asarray([True]), 'exact token shape'),
            (np.asarray([False]), np.asarray([True]), 'at least one intact Point'),
            (np.asarray([True]), np.asarray([False]), 'at least one intact CT'),
        )
        for point_mask, ct_mask, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(M4HardConstraintError, message):
                    self._build(
                        [[0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0]],
                        point_mask,
                        ct_mask,
                    )


if __name__ == '__main__':
    unittest.main()
