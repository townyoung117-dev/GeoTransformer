import ast
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataset as experiment_dataset
from dataset import (
    CTDefectAggregationContractError,
    CTPreprocessingContractError,
    PointPreprocessingContractError,
    m2_ct_collate_fn,
    m2_point_collate_fn,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None
    TORCH_SKIP_REASON = 'PyTorch is not installed in this local environment.'
else:
    from config import make_cfg
    from ct_encoder import (
        CTEncoderContractError,
        recover_x20_support_rows,
        validate_ct_defect_mapping,
    )
    from point_encoder import PointEncoder
    from geotransformer.modules.ops.point_defect_hierarchy import PointDefectHierarchyError
    from training import assemble_ct_encoder_input, assemble_point_encoder_input

    TORCH_SKIP_REASON = ''

TorchModuleBase = nn.Module if nn is not None else object


POINT_MAPPING_KEYS = (
    'point_intact_coarse',
    'point_raw_total_count_coarse',
    'point_raw_defect_count_coarse',
)
CT_MAPPING_KEYS = (
    'ct_intact_coarse',
    'ct_raw_total_count_coarse',
    'ct_raw_defect_count_coarse',
)


def make_sample():
    point_xyz = np.asarray(
        [[float(index), float(index % 2), 0.0] for index in range(6)],
        dtype=np.float32,
    )
    point_mask = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.bool_)
    ct_volume = np.full((1, 1, 6), 500, dtype=np.int16)
    ct_mask = np.asarray([[[1, 1, 0, 1, 1, 1]]], dtype=np.bool_)
    return {
        'subject_id': 'SyntheticM4',
        'case_id': 'SyntheticM4',
        'derived_sample_id': 'SyntheticM4_defect_001',
        'defect_id': 'defect_001',
        'point_xyz_phys': point_xyz,
        'point_normal': np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (6, 1)),
        'point_count': point_xyz.shape[0],
        'point_defect_mask': point_mask,
        'ct_volume': ct_volume,
        'ct_spacing': np.asarray([10.0, 10.0, 10.0], dtype=np.float64),
        'ct_origin': np.zeros(3, dtype=np.float64),
        'ct_direction': np.eye(3, dtype=np.float64),
        'ct_shape': np.asarray(ct_volume.shape, dtype=np.int64),
        'ct_array_axis_order': ('z', 'y', 'x'),
        'ct_image_index_convention': ('x', 'y', 'z'),
        'ct_defect_mask': ct_mask,
        'gt_transform': np.eye(4, dtype=np.float64),
        'gt_transform_direction': 'Point Cloud -> CT',
        'physical_unit': 'mm',
        'coordinate_system': 'DICOM LPS physical coordinates',
    }


class B0StackCollate:
    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        data_dicts,
        num_stages,
        voxel_size,
        search_radius,
        neighbor_limits,
        precompute_data,
    ):
        self.calls += 1
        points = data_dicts[0]['points']
        features = data_dicts[0]['feats']
        stages = [points, points[::2], points[::4], points[::6]]
        return {
            'features': features,
            'points': stages,
            'lengths': [np.asarray([stage.shape[0]], dtype=np.int64) for stage in stages],
            'neighbors': [np.zeros((stage.shape[0], 1), dtype=np.int64) for stage in stages],
            'subsampling': [np.zeros((stage.shape[0], 1), dtype=np.int64) for stage in stages[1:]],
            'upsampling': [np.zeros((stage.shape[0], 1), dtype=np.int64) for stage in stages[:-1]],
            'batch_size': 1,
        }


class M4IntegrationSourceAndB0Tests(unittest.TestCase):
    def test_explicit_mapping_flags_default_false(self):
        self.assertIs(
            inspect.signature(m2_point_collate_fn).parameters['enable_m4_defect_mapping'].default,
            False,
        )
        self.assertIs(
            inspect.signature(m2_ct_collate_fn).parameters['enable_m4_defect_mapping'].default,
            False,
        )

    def test_default_and_provenance_precompute_are_separate_paths(self):
        source = (PROJECT_ROOT / 'geotransformer' / 'utils' / 'data.py').read_text(encoding='utf-8')
        module = ast.parse(source)
        functions = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
        }
        default_source = ast.get_source_segment(source, functions['precompute_data_stack_mode'])
        provenance_source = ast.get_source_segment(
            source,
            functions['precompute_data_stack_mode_with_parent'],
        )
        self.assertIn('grid_subsample(', default_source)
        self.assertNotIn('grid_subsample_with_parent(', default_source)
        self.assertIn('grid_subsample_with_parent(', provenance_source)
        self.assertNotIn('grid_subsample(', provenance_source)

    def test_b0_uses_old_point_stack_and_emits_no_mapping_artifacts(self):
        stack_collate = B0StackCollate()
        sample = make_sample()
        point_batch = m2_point_collate_fn([sample], stack_collate_fn=stack_collate)

        self.assertEqual(stack_collate.calls, 1)
        self.assertNotIn('parent_indices', point_batch['point'])
        self.assertNotIn('m4_defect_mapping_enabled', point_batch['point'])
        for key in POINT_MAPPING_KEYS:
            self.assertNotIn(key, point_batch['point'])

    def test_b0_ct_does_not_call_c1_and_emits_no_mapping_artifacts(self):
        sample = make_sample()
        with mock.patch.object(
            experiment_dataset,
            'aggregate_ct_defect_whole_cells',
            side_effect=AssertionError('C1 must not run in B0'),
        ):
            ct_batch = m2_ct_collate_fn([sample])

        self.assertNotIn('m4_defect_mapping_enabled', ct_batch['ct'])
        for key in CT_MAPPING_KEYS:
            self.assertNotIn(key, ct_batch['ct'])

    def test_non_bool_mapping_gate_fails_closed(self):
        with self.assertRaises(TypeError):
            m2_point_collate_fn(
                [make_sample()],
                stack_collate_fn=B0StackCollate(),
                enable_m4_defect_mapping=1,
            )
        with self.assertRaises(TypeError):
            m2_ct_collate_fn([make_sample()], enable_m4_defect_mapping='true')


class CTMappingIntegrationTests(unittest.TestCase):
    def test_ct_enabled_integration_uses_authoritative_support_rows(self):
        ct = m2_ct_collate_fn([make_sample()], enable_m4_defect_mapping=True)['ct']

        np.testing.assert_array_equal(ct['ct_support_linear_20mm'], np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(ct['ct_raw_total_count_coarse'], np.asarray([2, 2, 2]))
        np.testing.assert_array_equal(ct['ct_raw_defect_count_coarse'], np.asarray([0, 1, 0]))
        np.testing.assert_array_equal(ct['ct_intact_coarse'], np.asarray([True, False, True]))
        self.assertIs(ct['m4_defect_mapping_enabled'], True)
        self.assertEqual(ct['ct_intact_coarse'].dtype, np.bool_)
        self.assertEqual(ct['ct_raw_total_count_coarse'].dtype, np.int64)
        self.assertEqual(ct['ct_raw_defect_count_coarse'].dtype, np.int64)
        support_count = ct['ct_support_linear_20mm'].shape[0]
        self.assertEqual(ct['ct_support_indices_20mm'].shape[0], support_count)
        self.assertEqual(ct['ct_support_phys_20mm'].shape[0], support_count)

    def test_ct_support_order_mismatch_fails_closed(self):
        invalid_support = {
            'ct_support_indices_20mm': np.asarray([[0, 0, 1], [0, 0, 0]], dtype=np.int32),
            'ct_support_linear_20mm': np.asarray([1, 0], dtype=np.int64),
            'ct_support_spatial_shape_20mm': np.asarray([1, 1, 3], dtype=np.int64),
            'ct_support_phys_20mm': np.zeros((2, 3), dtype=np.float64),
        }
        with mock.patch.object(experiment_dataset, 'build_ct_support_20mm', return_value=invalid_support):
            with self.assertRaisesRegex(CTDefectAggregationContractError, 'strictly increasing'):
                m2_ct_collate_fn([make_sample()], enable_m4_defect_mapping=True)

    def test_ct_coarse_count_mismatch_fails_closed(self):
        invalid_mapping = {
            'ct_intact_coarse': np.ones(2, dtype=np.bool_),
            'ct_raw_total_count_coarse': np.ones(2, dtype=np.int64),
            'ct_raw_defect_count_coarse': np.zeros(2, dtype=np.int64),
        }
        with mock.patch.object(
            experiment_dataset,
            'aggregate_ct_defect_whole_cells',
            return_value=invalid_mapping,
        ):
            with self.assertRaisesRegex(CTPreprocessingContractError, 'shape'):
                m2_ct_collate_fn([make_sample()], enable_m4_defect_mapping=True)

    def test_ct_enabled_requires_raw_mask_and_explicit_defect_identity(self):
        sample = make_sample()
        del sample['ct_defect_mask']
        with self.assertRaisesRegex(CTPreprocessingContractError, 'ct_defect_mask'):
            m2_ct_collate_fn([sample], enable_m4_defect_mapping=True)

        sample = make_sample()
        del sample['defect_id']
        with self.assertRaisesRegex(CTPreprocessingContractError, 'defect_id'):
            m2_ct_collate_fn([sample], enable_m4_defect_mapping=True)


@unittest.skipUnless(torch is not None, TORCH_SKIP_REASON)
class PointMappingIntegrationRuntimeTests(unittest.TestCase):
    class ProvenanceStackCollate:
        def __init__(self, invalid_parent=False):
            self.invalid_parent = invalid_parent
            self.calls = 0

        def __call__(
            self,
            data_dicts,
            num_stages,
            voxel_size,
            search_radius,
            neighbor_limits,
            precompute_data,
        ):
            self.calls += 1
            raw = torch.from_numpy(data_dicts[0]['points'])
            features = torch.from_numpy(data_dicts[0]['feats'])
            parents = (
                torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
                torch.tensor([0, 0, 1], dtype=torch.int64),
                torch.tensor([0, 1], dtype=torch.int64),
            )
            points = [raw]
            lengths = [torch.tensor([raw.shape[0]], dtype=torch.int64)]
            for parent in parents:
                output_count = int(parent.max().item()) + 1
                output = torch.zeros((output_count, 3), dtype=raw.dtype)
                counts = torch.zeros(output_count, dtype=torch.int64)
                output.index_add_(0, parent, points[-1])
                counts.index_add_(0, parent, torch.ones(parent.shape[0], dtype=torch.int64))
                output = output / counts[:, None]
                points.append(output)
                lengths.append(torch.tensor([output_count], dtype=torch.int64))
            parents = list(parents)
            if self.invalid_parent:
                parents[0] = parents[0][:-1]
            return {
                'features': features,
                'points': points,
                'lengths': lengths,
                'parent_indices': parents,
                'neighbors': [torch.zeros((stage.shape[0], 1), dtype=torch.int64) for stage in points],
                'subsampling': [torch.zeros((stage.shape[0], 1), dtype=torch.int64) for stage in points[1:]],
                'upsampling': [torch.zeros((stage.shape[0], 1), dtype=torch.int64) for stage in points[:-1]],
                'batch_size': 1,
            }

    class SyntheticBackbone(TorchModuleBase):
        def forward(self, features, point_dict):
            counts = [point_dict['points'][index].shape[0] for index in (1, 2, 3)]
            return [
                torch.ones((counts[0], 256), dtype=features.dtype, device=features.device),
                torch.ones((counts[1], 512), dtype=features.dtype, device=features.device),
                torch.ones((counts[2], 1024), dtype=features.dtype, device=features.device),
            ]

    def test_point_enabled_same_pass_integration_and_encoder_row_identity(self):
        stack_collate = self.ProvenanceStackCollate()
        point = m2_point_collate_fn(
            [make_sample()],
            stack_collate_fn=stack_collate,
            enable_m4_defect_mapping=True,
        )['point']

        self.assertEqual(stack_collate.calls, 1)
        self.assertEqual(point['points'][3].shape[0], 2)
        raw_net = torch.from_numpy(make_sample()['point_xyz_phys']) * 0.001
        expected_coarse = torch.stack((raw_net[:4].mean(dim=0), raw_net[4:].mean(dim=0)))
        torch.testing.assert_close(point['points'][3], expected_coarse)
        torch.testing.assert_close(point['point_raw_total_count_coarse'], torch.tensor([4, 2]))
        torch.testing.assert_close(point['point_raw_defect_count_coarse'], torch.tensor([1, 0]))
        torch.testing.assert_close(point['point_intact_coarse'], torch.tensor([False, True]))

        encoder_input = assemble_point_encoder_input(point, torch.device('cpu'))
        output = PointEncoder(make_cfg(), backbone=self.SyntheticBackbone())(encoder_input)
        self.assertEqual(output['P_raw'].shape[0], point['points'][3].shape[0])
        self.assertEqual(output['Q'].shape[0], point['point_intact_coarse'].shape[0])
        self.assertEqual(output['Xp_phys_coarse'].shape[0], point['point_intact_coarse'].shape[0])
        torch.testing.assert_close(output['Xp_phys_coarse'], expected_coarse / 0.001)
        torch.testing.assert_close(output['point_intact_coarse'], torch.tensor([False, True]))

    def test_combined_sample_produces_six_mapping_outputs(self):
        sample = make_sample()
        point = m2_point_collate_fn(
            [sample],
            stack_collate_fn=self.ProvenanceStackCollate(),
            enable_m4_defect_mapping=True,
        )['point']
        ct = m2_ct_collate_fn([sample], enable_m4_defect_mapping=True)['ct']
        for key in POINT_MAPPING_KEYS:
            self.assertIn(key, point)
        for key in CT_MAPPING_KEYS:
            self.assertIn(key, ct)
        ct_input = assemble_ct_encoder_input(ct, torch.device('cpu'))
        validated = validate_ct_defect_mapping(
            ct_input,
            ct['ct_support_linear_20mm'].shape[0],
            torch.device('cpu'),
        )
        self.assertEqual(set(validated), set(CT_MAPPING_KEYS))

    def test_point_raw_mask_parent_and_coarse_mismatch_fail_closed(self):
        sample = make_sample()
        sample['point_defect_mask'] = sample['point_defect_mask'][:-1]
        with self.assertRaises(PointDefectHierarchyError):
            m2_point_collate_fn(
                [sample],
                stack_collate_fn=self.ProvenanceStackCollate(),
                enable_m4_defect_mapping=True,
            )

        with self.assertRaises(PointDefectHierarchyError):
            m2_point_collate_fn(
                [make_sample()],
                stack_collate_fn=self.ProvenanceStackCollate(invalid_parent=True),
                enable_m4_defect_mapping=True,
            )

        invalid_mapping = {
            'point_intact_coarse': torch.ones(1, dtype=torch.bool),
            'point_raw_total_count_coarse': torch.ones(1, dtype=torch.int64),
            'point_raw_defect_count_coarse': torch.zeros(1, dtype=torch.int64),
        }
        with mock.patch(
            'geotransformer.modules.ops.aggregate_point_defect_hierarchy',
            return_value=invalid_mapping,
        ):
            with self.assertRaisesRegex(PointPreprocessingContractError, 'shape'):
                m2_point_collate_fn(
                    [make_sample()],
                    stack_collate_fn=self.ProvenanceStackCollate(),
                    enable_m4_defect_mapping=True,
                )


@unittest.skipUnless(torch is not None, TORCH_SKIP_REASON)
class X20AlignmentRuntimeTests(unittest.TestCase):
    def test_recovery_returns_rows_in_authoritative_support_order(self):
        active = torch.tensor([[0, 0, 2], [0, 0, 0], [0, 0, 1]], dtype=torch.int32)
        support = torch.tensor([0, 1, 2], dtype=torch.int64)
        rows = recover_x20_support_rows(active, (1, 1, 3), support)
        torch.testing.assert_close(rows, torch.tensor([1, 2, 0], dtype=torch.int64))
        features = torch.tensor([[20.0], [0.0], [10.0]])
        torch.testing.assert_close(features[rows], torch.tensor([[0.0], [10.0], [20.0]]))

    def test_x20_out_of_bounds_duplicate_and_missing_support_fail_closed(self):
        support = torch.tensor([0, 1, 2], dtype=torch.int64)
        cases = (
            (torch.tensor([[0, 0, 3]], dtype=torch.int32), 'out-of-bounds'),
            (torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=torch.int32), 'unique'),
            (torch.tensor([[0, 0, 0], [0, 0, 2]], dtype=torch.int32), 'missing'),
        )
        for active, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(CTEncoderContractError, message):
                    recover_x20_support_rows(active, (1, 1, 3), support)


if __name__ == '__main__':
    unittest.main()
