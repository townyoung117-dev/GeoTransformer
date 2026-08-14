import ast
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import make_cfg
from dataset import PointPreprocessingContractError, m2_point_collate_fn

if torch is not None:
    from point_encoder import PointEncoder


class SyntheticStackCollate:
    def __init__(self, test_case):
        self.test_case = test_case

    def __call__(
        self,
        data_dicts,
        num_stages,
        voxel_size,
        search_radius,
        neighbor_limits,
        precompute_data,
    ):
        self.test_case.assertEqual(len(data_dicts), 1)
        self.test_case.assertEqual(num_stages, 4)
        self.test_case.assertEqual(voxel_size, 0.0025)
        self.test_case.assertEqual(search_radius, 0.00625)
        self.test_case.assertEqual(neighbor_limits, [177, 32, 33, 34])
        self.test_case.assertTrue(precompute_data)

        points = data_dicts[0]['points']
        features = data_dicts[0]['feats']
        points_list = [points, points[::2], points[::4], points[::8]]
        return {
            'features': features,
            'points': points_list,
            'lengths': [np.asarray([stage.shape[0]], dtype=np.int64) for stage in points_list],
            'neighbors': [np.zeros((stage.shape[0], 1), dtype=np.int64) for stage in points_list],
            'subsampling': [np.zeros((stage.shape[0], 1), dtype=np.int64) for stage in points_list[1:]],
            'upsampling': [np.zeros((stage.shape[0], 1), dtype=np.int64) for stage in points_list[:-1]],
            'batch_size': 1,
        }


if torch is not None:
    class SyntheticBackbone(nn.Module):
        def forward(self, features, data_dict):
            device = features.device
            dtype = features.dtype
            fine_count = data_dict['points'][1].shape[0]
            middle_count = data_dict['points'][2].shape[0]
            coarse_count = data_dict['points'][3].shape[0]
            fine = torch.full((fine_count, 256), 0.25, device=device, dtype=dtype)
            middle = torch.full((middle_count, 512), 0.5, device=device, dtype=dtype)
            coarse = torch.arange(
                coarse_count * 1024,
                device=device,
                dtype=dtype,
            ).reshape(coarse_count, 1024)
            return [fine, middle, coarse]


class PointEncoderTest(unittest.TestCase):
    def setUp(self):
        self.point_xyz_phys = np.asarray(
            [
                [100.0, -50.0, 25.0],
                [102.0, -48.0, 27.0],
                [104.0, -46.0, 29.0],
                [106.0, -44.0, 31.0],
                [108.0, -42.0, 33.0],
                [110.0, -40.0, 35.0],
                [112.0, -38.0, 37.0],
                [114.0, -36.0, 39.0],
                [116.0, -34.0, 41.0],
            ],
            dtype=np.float64,
        )
        self.point_normal = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (9, 1))
        self.ct_volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        self.ct_metadata = {'modality': 'CT', 'physical_unit': 'mm'}
        self.pair_metadata = {'relationship': 'synthetic Point-to-CT pair'}
        self.gt_transform = np.eye(4, dtype=np.float64)
        self.sample = {
            'subject_id': 'Synthetic01',
            'case_id': 'Synthetic01',
            'derived_sample_id': 'Synthetic01_original',
            'point_xyz_phys': self.point_xyz_phys,
            'point_normal': self.point_normal,
            'point_count': self.point_xyz_phys.shape[0],
            'ct_volume': self.ct_volume,
            'ct_spacing': np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
            'ct_origin': np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            'ct_direction': np.eye(3, dtype=np.float64),
            'ct_shape': np.asarray(self.ct_volume.shape, dtype=np.int64),
            'ct_array_axis_order': ('z', 'y', 'x'),
            'ct_image_index_convention': ('x', 'y', 'z'),
            'ct_metadata': self.ct_metadata,
            'pair_metadata': self.pair_metadata,
            'gt_transform': self.gt_transform,
            'gt_transform_direction': 'Point Cloud -> CT',
            'physical_unit': 'mm',
            'coordinate_system': 'synthetic physical coordinates',
        }

    def _preprocess(self):
        return m2_point_collate_fn(
            [self.sample],
            stack_collate_fn=SyntheticStackCollate(self),
        )

    def test_mm_to_m_scaling_does_not_overwrite_physical_coordinates(self):
        collated = self._preprocess()
        point = collated['point']
        self.assertNotIn('points', collated)
        self.assertNotIn('features', collated)
        self.assertIs(point['point_xyz_phys'], self.point_xyz_phys)
        np.testing.assert_array_equal(point['point_xyz_phys'], self.point_xyz_phys)
        self.assertEqual(point['point_xyz_net'].dtype, np.float32)
        np.testing.assert_allclose(
            point['point_xyz_net'],
            self.point_xyz_phys.astype(np.float32) * 0.001,
        )
        self.assertEqual(point['features'].shape, (9, 1))
        self.assertEqual(point['features'].dtype, np.float32)
        np.testing.assert_array_equal(point['features'], np.ones((9, 1), dtype=np.float32))

    def test_point_preprocessing_does_not_modify_ct_branch_or_m1_contract(self):
        collated = self._preprocess()
        self.assertIn('point', collated)
        self.assertIn('ct', collated)
        self.assertIs(collated['ct']['ct_volume'], self.ct_volume)
        self.assertIs(collated['ct']['ct_metadata'], self.ct_metadata)
        self.assertIs(collated['pair_metadata'], self.pair_metadata)
        self.assertIs(collated['point']['point_normal'], self.point_normal)
        self.assertIs(collated['gt_transform'], self.gt_transform)
        self.assertEqual(collated['gt_transform_direction'], 'Point Cloud -> CT')
        np.testing.assert_array_equal(collated['ct']['ct_volume'], self.ct_volume)

    def test_missing_physical_unit_fails_closed(self):
        sample = dict(self.sample)
        del sample['physical_unit']
        with self.assertRaisesRegex(PointPreprocessingContractError, 'explicit physical_unit'):
            m2_point_collate_fn(
                [sample],
                stack_collate_fn=SyntheticStackCollate(self),
            )

    @unittest.skipIf(torch is None, 'PyTorch is not installed in this local environment.')
    def test_encoder_output_contract_and_inverse_scaling(self):
        collated = self._preprocess()
        point = collated['point']
        for key in ('features',):
            point[key] = torch.from_numpy(point[key])
        for key in ('points', 'lengths', 'neighbors', 'subsampling', 'upsampling'):
            point[key] = [torch.from_numpy(value) for value in point[key]]
        cfg = make_cfg()
        encoder = PointEncoder(cfg, backbone=SyntheticBackbone())
        output = encoder(point)

        coarse_count = point['points'][3].shape[0]
        self.assertEqual(tuple(output['P_raw'].shape), (coarse_count, 1024))
        self.assertEqual(tuple(output['Q'].shape), (coarse_count, 256))
        self.assertEqual(tuple(output['Xp_net_coarse'].shape), (coarse_count, 3))
        self.assertEqual(tuple(output['Xp_phys_coarse'].shape), (coarse_count, 3))
        self.assertEqual(encoder.point_proj.in_features, 1024)
        self.assertEqual(encoder.point_proj.out_features, 256)

        torch.testing.assert_close(
            output['Xp_phys_coarse'],
            output['Xp_net_coarse'] / 0.001,
        )
        torch.testing.assert_close(
            output['Xp_phys_coarse'] * 0.001,
            output['Xp_net_coarse'],
        )
        for tensor in output.values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))

    @unittest.skipIf(torch is None, 'PyTorch is not installed in this local environment.')
    def test_physical_geometry_coordinate_is_not_the_network_coordinate(self):
        collated = self._preprocess()
        point = collated['point']
        point['features'] = torch.from_numpy(point['features'])
        for key in ('points', 'lengths', 'neighbors', 'subsampling', 'upsampling'):
            point[key] = [torch.from_numpy(value) for value in point[key]]
        output = PointEncoder(make_cfg(), backbone=SyntheticBackbone())(point)

        # Future GT geometry/SVD consumers must select this explicit physical
        # field. It is in mm and is neither aliased to nor numerically confused
        # with the KPConv coordinate in metres.
        geometry_coordinate = output['Xp_phys_coarse']
        self.assertNotEqual(geometry_coordinate.data_ptr(), output['Xp_net_coarse'].data_ptr())
        self.assertFalse(torch.allclose(geometry_coordinate, output['Xp_net_coarse']))
        expected_phys = torch.from_numpy(self.point_xyz_phys[::8].astype(np.float32))
        torch.testing.assert_close(geometry_coordinate, expected_phys)

    def test_frozen_point_configuration(self):
        point_cfg = make_cfg().point
        self.assertEqual(point_cfg.physical_unit, 'mm')
        self.assertEqual(point_cfg.point_network_scale_mm_to_m, 0.001)
        self.assertEqual(point_cfg.num_stages, 4)
        self.assertEqual(point_cfg.init_voxel_size_m, 0.0025)
        self.assertEqual(point_cfg.init_radius_m, 0.00625)
        self.assertEqual(point_cfg.init_sigma_m, 0.005)
        self.assertEqual(point_cfg.neighbor_limits, [177, 32, 33, 34])
        self.assertEqual(point_cfg.input_dim, 1)
        self.assertEqual(point_cfg.init_dim, 64)
        self.assertEqual(point_cfg.group_norm, 32)
        self.assertEqual(point_cfg.coarse_raw_dim, 1024)
        self.assertEqual(point_cfg.projected_dim, 256)

    def test_encoder_source_has_explicit_coarse_physical_contract(self):
        source_path = EXPERIMENT_DIR / 'point_encoder.py'
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        self.assertIn("point_dict['points'][3]", source)
        self.assertIn('Xp_phys_coarse = Xp_net_coarse / self.point_network_scale_mm_to_m', source)
        self.assertIn("'P_raw': P_raw", source)
        self.assertIn("'Q': Q", source)
        self.assertIn("'Xp_net_coarse': Xp_net_coarse", source)
        self.assertIn("'Xp_phys_coarse': Xp_phys_coarse", source)
        self.assertEqual(source.count('self.point_proj(P_raw)'), 1)
        self.assertNotIn('F.normalize', source)
        linear_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'Linear'
        ]
        self.assertEqual(len(linear_calls), 1)

        backbone_source = (EXPERIMENT_DIR / 'backbone.py').read_text(encoding='utf-8')
        self.assertIn('return [latent_s2, latent_s3, feats_s4]', backbone_source)

        dataset_source = (EXPERIMENT_DIR / 'dataset.py').read_text(encoding='utf-8')
        self.assertIn('pointct_collate_fn(samples)', dataset_source)
        self.assertNotIn('registration_collate_fn_stack_mode', dataset_source)


if __name__ == '__main__':
    unittest.main()
