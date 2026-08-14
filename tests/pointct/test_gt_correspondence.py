import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GT_HIGH_CONFIDENCE_DISTANCE_MM,
    GT_PRIMARY_MAX_DISTANCE_MM,
    make_cfg,
)
from dataset import build_ct_support_20mm, m2_point_collate_fn
from gt_correspondence import (
    GTCorrespondenceContractError,
    build_coarse_gt_correspondence,
)


class GTCorrespondenceTest(unittest.TestCase):
    def _build(self, points, ct_support, transform=None, direction='Point Cloud -> CT'):
        if transform is None:
            transform = np.eye(4, dtype=np.float64)
        return build_coarse_gt_correspondence(
            Xp_phys_coarse=np.asarray(points),
            Xv_phys_coarse=np.asarray(ct_support),
            gt_transform=transform,
            gt_transform_direction=direction,
            primary_max_distance_mm=GT_PRIMARY_MAX_DISTANCE_MM,
            high_confidence_distance_mm=GT_HIGH_CONFIDENCE_DISTANCE_MM,
        )

    def test_frozen_thresholds_use_independent_namespace(self):
        cfg = make_cfg()
        self.assertEqual(GT_PRIMARY_MAX_DISTANCE_MM, 17.5)
        self.assertEqual(GT_HIGH_CONFIDENCE_DISTANCE_MM, 15.0)
        self.assertEqual(cfg.gt_coarse.primary_max_distance_mm, 17.5)
        self.assertEqual(cfg.gt_coarse.high_confidence_distance_mm, 15.0)
        self.assertFalse(hasattr(cfg.point, 'primary_max_distance_mm'))
        self.assertFalse(hasattr(cfg.ct, 'primary_max_distance_mm'))

    def test_identity_transform_nearest_indices_and_distances(self):
        output = self._build(
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [20.0, 20.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 20.0, 0.0]],
        )
        np.testing.assert_allclose(
            output['Xp_gt_ct_phys'],
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [20.0, 20.0, 0.0]],
        )
        np.testing.assert_array_equal(output['gt_primary_ct_index'], [0, 1, 2])
        np.testing.assert_allclose(output['gt_primary_distance_mm'], [0.0, 1.0, 0.0])

    def test_nontrivial_point_to_ct_rigid_transform(self):
        rotation = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        translation = np.asarray([10.0, -2.0, 3.0], dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 4.0, 0.0]])
        expected = points @ rotation.T + translation
        ct_support = np.stack((expected[1], [100.0, 100.0, 100.0], expected[0]))

        output = self._build(points, ct_support, transform=transform)

        np.testing.assert_allclose(output['Xp_gt_ct_phys'], expected)
        np.testing.assert_array_equal(output['gt_primary_ct_index'], [2, 0])
        np.testing.assert_allclose(output['gt_primary_distance_mm'], [0.0, 0.0])

    def test_wrong_transform_direction_fails_closed_without_inversion(self):
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'must be exactly'):
            self._build(
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                direction='CT -> Point Cloud',
            )

    def test_nearest_one_returns_only_one_primary_index(self):
        output = self._build(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        )
        self.assertEqual(output['gt_primary_ct_index'].shape, (1,))
        self.assertEqual(int(output['gt_primary_ct_index'][0]), 2)
        self.assertAlmostEqual(float(output['gt_primary_distance_mm'][0]), 0.25)

    def test_exact_distance_tie_selects_smallest_ct_support_index(self):
        output = self._build(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        self.assertEqual(int(output['gt_primary_ct_index'][0]), 0)
        self.assertAlmostEqual(float(output['gt_primary_distance_mm'][0]), 1.0)

    def test_valid_and_high_confidence_threshold_boundaries(self):
        output = self._build(
            [[15.0, 0.0, 0.0], [17.5, 0.0, 0.0], [17.5001, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(output['gt_primary_valid'], [True, True, False])
        np.testing.assert_array_equal(output['gt_high_confidence'], [True, False, False])

    def test_empty_or_nonfinite_coordinates_fail_closed(self):
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'at least one'):
            self._build(np.empty((0, 3)), [[0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'at least one'):
            self._build([[0.0, 0.0, 0.0]], np.empty((0, 3)))
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'NaN or Inf'):
            self._build([[np.nan, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'NaN or Inf'):
            self._build([[0.0, 0.0, 0.0]], [[np.inf, 0.0, 0.0]])

    def test_transform_and_threshold_contracts_fail_closed(self):
        nonrigid = np.eye(4, dtype=np.float64)
        nonrigid[0, 0] = 2.0
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'orthonormal'):
            self._build([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], transform=nonrigid)
        with self.assertRaisesRegex(GTCorrespondenceContractError, 'must not exceed'):
            build_coarse_gt_correspondence(
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                np.eye(4),
                'Point Cloud -> CT',
                primary_max_distance_mm=10.0,
                high_confidence_distance_mm=11.0,
            )

    def test_output_is_sparse_primary_contract_not_pairwise_labels(self):
        output = self._build(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        self.assertEqual(
            set(output),
            {
                'Xp_phys_coarse',
                'Xp_gt_ct_phys',
                'Xv_phys_coarse',
                'gt_primary_ct_index',
                'gt_primary_distance_mm',
                'gt_primary_valid',
                'gt_high_confidence',
            },
        )
        for name in (
            'gt_primary_ct_index',
            'gt_primary_distance_mm',
            'gt_primary_valid',
            'gt_high_confidence',
        ):
            self.assertEqual(output[name].shape, (2,))
        self.assertEqual(output['gt_primary_ct_index'].dtype, np.int64)
        self.assertEqual(output['gt_primary_valid'].dtype, np.bool_)
        self.assertEqual(output['gt_high_confidence'].dtype, np.bool_)

    def test_production_source_is_physical_only_and_contains_no_forbidden_matching(self):
        source = (EXPERIMENT_DIR / 'gt_correspondence.py').read_text(encoding='utf-8')
        lower_source = source.lower()
        self.assertNotIn('xp_net_coarse', lower_source)
        self.assertNotIn('ct_support_indices_20mm', lower_source)
        self.assertNotIn('sparse indices', lower_source)
        for prohibited in (
            'sinkhorn',
            'procrustes',
            'ransac',
            'f.normalize',
            'cosine',
            'attention',
            'svd',
            'feature similarity',
        ):
            self.assertNotIn(prohibited, lower_source)

        parameters = tuple(inspect.signature(build_coarse_gt_correspondence).parameters)
        self.assertEqual(
            parameters,
            (
                'Xp_phys_coarse',
                'Xv_phys_coarse',
                'gt_transform',
                'gt_transform_direction',
                'primary_max_distance_mm',
                'high_confidence_distance_mm',
            ),
        )

    def test_gt_is_not_used_by_formal_point_or_ct_support_preprocessing(self):
        self.assertNotIn('gt_transform', inspect.getsource(m2_point_collate_fn))
        self.assertNotIn('gt_transform', inspect.getsource(build_ct_support_20mm))


if __name__ == '__main__':
    unittest.main()
