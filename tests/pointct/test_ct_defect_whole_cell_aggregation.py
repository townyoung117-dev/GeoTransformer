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

from dataset import (
    CTDefectAggregationContractError,
    aggregate_ct_defect_whole_cells,
    m2_ct_collate_fn,
)


class CTDefectWholeCellAggregationTest(unittest.TestCase):
    @staticmethod
    def aggregate(mask, spacing, support):
        return aggregate_ct_defect_whole_cells(
            mask,
            mask.shape,
            np.asarray(spacing, dtype=np.float64),
            np.asarray(support, dtype=np.int64),
        )

    def assert_aggregation(self, result, *, total, defect, intact):
        np.testing.assert_array_equal(
            result['ct_raw_total_count_coarse'], np.asarray(total, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            result['ct_raw_defect_count_coarse'], np.asarray(defect, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            result['ct_intact_coarse'], np.asarray(intact, dtype=np.bool_)
        )
        self.assertEqual(result['ct_raw_total_count_coarse'].dtype, np.int64)
        self.assertEqual(result['ct_raw_defect_count_coarse'].dtype, np.int64)
        self.assertEqual(result['ct_intact_coarse'].dtype, np.bool_)

    def test_simple_whole_cell_count_and_one_defect_voxel(self):
        mask = np.ones((2, 2, 4), dtype=np.bool_)
        mask[1, 1, 0] = False

        result = self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=[0])

        self.assert_aggregation(result, total=[8], defect=[1], intact=[False])

    def test_entire_cell_counts_background_and_foreground_voxels(self):
        volume = np.asarray(
            [[[-1000, 500], [-700, 1000]], [[-1000, -1000], [800, 1200]]],
            dtype=np.int16,
        )
        mask = np.ones(volume.shape, dtype=np.bool_)

        result = aggregate_ct_defect_whole_cells(
            mask,
            volume.shape,
            np.asarray([10.0, 10.0, 10.0]),
            np.asarray([0], dtype=np.int64),
        )

        self.assert_aggregation(result, total=[volume.size], defect=[0], intact=[True])

    def test_all_intact(self):
        mask = np.ones((2, 2, 2), dtype=np.uint8)
        result = self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=[0])
        self.assert_aggregation(result, total=[8], defect=[0], intact=[True])

    def test_all_defect(self):
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        result = self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=[0])
        self.assert_aggregation(result, total=[8], defect=[8], intact=[False])

    def test_multiple_authoritative_support_rows_use_ascending_order(self):
        mask = np.ones((1, 1, 6), dtype=np.bool_)
        mask[0, 0, 4] = False
        mask[0, 0, 2:4] = False

        result = self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=[0, 1, 2])

        self.assert_aggregation(
            result,
            total=[2, 2, 2],
            defect=[0, 2, 1],
            intact=[True, False, False],
        )

    def test_boundary_cell_counts_only_existing_voxels(self):
        mask = np.ones((1, 1, 5), dtype=np.bool_)
        mask[0, 0, 4] = False

        result = self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=[2])

        self.assert_aggregation(result, total=[1], defect=[1], intact=[False])

    def test_unsorted_support_fails_closed(self):
        mask = np.ones((1, 1, 6), dtype=np.bool_)
        for support in ([1, 0], [2, 0, 1], [0, 2, 1]):
            with self.subTest(support=support):
                with self.assertRaisesRegex(CTDefectAggregationContractError, 'strictly increasing'):
                    self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=support)

    def test_anisotropic_spacing_uses_xyz_spacing_for_zyx_array(self):
        mask = np.ones((3, 4, 5), dtype=np.bool_)
        mask[2, 3, 4] = False
        # spacing=[sx,sy,sz]=[6,11,9] makes cell [qz,qy,qx]=[0,1,1]
        # contain z=0..2, y=2..3, x=4: 3*2*1 raw voxels. Its key is 3.
        result = self.aggregate(mask, spacing=[6.0, 11.0, 9.0], support=[3])

        self.assert_aggregation(result, total=[6], defect=[1], intact=[False])

    def test_invalid_mask_fails_closed(self):
        valid_shape = (2, 2, 2)
        invalid_masks = (
            np.ones((2, 2), dtype=np.bool_),
            np.ones((2, 2, 3), dtype=np.bool_),
            np.full(valid_shape, 2, dtype=np.uint8),
            np.ones(valid_shape, dtype=np.float32),
        )
        for mask in invalid_masks:
            with self.subTest(shape=mask.shape, dtype=str(mask.dtype)):
                with self.assertRaises(CTDefectAggregationContractError):
                    aggregate_ct_defect_whole_cells(
                        mask,
                        valid_shape,
                        np.asarray([10.0, 10.0, 10.0]),
                        np.asarray([0], dtype=np.int64),
                    )

    def test_invalid_support_fails_closed(self):
        mask = np.ones((1, 1, 4), dtype=np.bool_)
        for support in (
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        ):
            with self.subTest(support=support.tolist()):
                with self.assertRaises(CTDefectAggregationContractError):
                    self.aggregate(mask, spacing=[10.0, 10.0, 10.0], support=support)

        with self.assertRaisesRegex(CTDefectAggregationContractError, 'no existing raw voxel'):
            self.aggregate(
                np.ones((1, 1, 2), dtype=np.bool_),
                spacing=[50.0, 10.0, 10.0],
                support=[1],
            )

    def test_support_requires_nonempty_1d_integer_input(self):
        mask = np.ones((1, 1, 2), dtype=np.bool_)
        invalid_supports = (
            np.asarray([], dtype=np.int64),
            np.asarray([[0]], dtype=np.int64),
            np.asarray([0.0], dtype=np.float64),
        )
        for support in invalid_supports:
            with self.subTest(shape=support.shape, dtype=str(support.dtype)):
                with self.assertRaises(CTDefectAggregationContractError):
                    aggregate_ct_defect_whole_cells(
                        mask,
                        mask.shape,
                        np.asarray([10.0, 10.0, 10.0]),
                        support,
                    )

    def test_membership_api_is_independent_of_origin_direction_and_intensity(self):
        parameters = inspect.signature(aggregate_ct_defect_whole_cells).parameters
        self.assertNotIn('ct_volume', parameters)
        self.assertNotIn('ct_origin', parameters)
        self.assertNotIn('ct_direction', parameters)
        self.assertNotIn('grid_mm', parameters)

    def test_b0_default_ct_preprocessing_does_not_call_c1_aggregation(self):
        signature = inspect.signature(m2_ct_collate_fn)
        self.assertIs(signature.parameters['enable_m4_defect_mapping'].default, False)
        collate_source = inspect.getsource(m2_ct_collate_fn)
        self.assertIn('if enable_m4_defect_mapping:', collate_source)
        self.assertIn('aggregate_ct_defect_whole_cells(', collate_source)
        ct_encoder_source = (EXPERIMENT_DIR / 'ct_encoder.py').read_text(encoding='utf-8')
        self.assertNotIn('aggregate_ct_defect_whole_cells', ct_encoder_source)


if __name__ == '__main__':
    unittest.main()
