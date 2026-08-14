import ast
import importlib.util
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

from config import make_cfg
from dataset import (
    CTPreprocessingContractError,
    build_ct_context_5mm,
    build_ct_support_20mm,
    extract_external_surface,
    find_boundary_connected_outside_air,
    m2_ct_collate_fn,
)


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
SPCONV_AVAILABLE = importlib.util.find_spec('spconv') is not None
if TORCH_AVAILABLE:
    import torch
else:
    torch = None

if TORCH_AVAILABLE and SPCONV_AVAILABLE:
    from ct_encoder import CTEncoder


class CTEncoderTest(unittest.TestCase):
    def setUp(self):
        self.point_xyz_phys = np.asarray(
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            dtype=np.float64,
        )
        self.point_normal = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.ct_volume = np.full((5, 5, 5), -1000, dtype=np.int16)
        self.ct_volume[2, 2, 2] = 500
        self.ct_metadata = {'modality': 'CT', 'physical_unit': 'mm'}
        self.pair_metadata = {'relationship': 'synthetic Point-to-CT pair'}
        self.gt_transform = np.eye(4, dtype=np.float64)
        self.sample = {
            'subject_id': 'SyntheticCT01',
            'case_id': 'SyntheticCT01',
            'derived_sample_id': 'SyntheticCT01_original',
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
            'coordinate_system': 'DICOM LPS physical coordinates',
        }

    def test_frozen_ct_configuration(self):
        ct_cfg = make_cfg().ct
        self.assertEqual(ct_cfg.physical_unit, 'mm')
        self.assertEqual(ct_cfg.foreground_hu, -500.0)
        self.assertEqual(ct_cfg.context_grid_mm, 5.0)
        self.assertEqual(ct_cfg.support_grid_mm, 20.0)
        self.assertEqual(ct_cfg.input_dim, 1)
        self.assertEqual(ct_cfg.stem_dim, 32)
        self.assertEqual(ct_cfg.mid_dim, 64)
        self.assertEqual(ct_cfg.coarse_raw_dim, 128)
        self.assertEqual(ct_cfg.projected_dim, 256)
        self.assertEqual(ct_cfg.hu_clip_min, -500.0)
        self.assertEqual(ct_cfg.hu_clip_max, 2000.0)

    def test_context_5mm_threshold_mean_clip_normalise_and_order(self):
        volume = np.full((2, 2, 7), -1000, dtype=np.int16)
        volume[0, 0, 0] = -500  # Strict threshold: this voxel is excluded.
        volume[0, 0, 1] = -400
        volume[0, 1, 4] = 600
        volume[1, 1, 5] = 3000
        output = build_ct_context_5mm(
            volume,
            [1.0, 1.0, 1.0],
            physical_unit='mm',
        )

        np.testing.assert_array_equal(output['ct_context_indices'], [[0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(output['ct_context_linear'], [0, 1])
        np.testing.assert_array_equal(output['ct_context_spatial_shape'], [1, 1, 2])
        self.assertEqual(output['ct_context_features'].shape, (2, 1))
        self.assertEqual(output['ct_context_features'].dtype, np.float32)
        np.testing.assert_allclose(
            output['ct_context_features'][:, 0],
            np.asarray([(100.0 + 500.0) / 2500.0, 1.0], dtype=np.float32),
        )

    def test_context_requires_explicit_mm_and_nonempty_foreground(self):
        with self.assertRaisesRegex(CTPreprocessingContractError, 'explicit physical_unit'):
            build_ct_context_5mm(self.ct_volume, [1.0, 1.0, 1.0], physical_unit=None)
        with self.assertRaisesRegex(CTPreprocessingContractError, 'physical_unit'):
            build_ct_context_5mm(self.ct_volume, [1.0, 1.0, 1.0], physical_unit='m')
        with self.assertRaisesRegex(CTPreprocessingContractError, 'no foreground'):
            build_ct_context_5mm(
                np.full((2, 2, 2), -500, dtype=np.int16),
                [1.0, 1.0, 1.0],
                physical_unit='mm',
            )

    def test_external_surface_uses_six_connected_outside_air(self):
        volume = np.zeros((5, 5, 5), dtype=np.int16)
        volume[0, 0, 0] = -1000  # Boundary-connected outside-air seed.
        volume[1, 1, 1] = -1000  # Only diagonally connected to that seed.
        foreground = volume > -500
        outside_air = find_boundary_connected_outside_air(foreground)
        surface = extract_external_surface(volume)

        self.assertTrue(outside_air[0, 0, 0])
        self.assertFalse(outside_air[1, 1, 1])
        self.assertFalse(surface[2, 1, 1])  # Adjacent only to the sealed cavity.
        self.assertFalse(bool(np.any(surface & ~foreground)))
        self.assertTrue(surface[0, 4, 4])  # Foreground on a volume face is included.

        boundary = np.zeros_like(surface)
        boundary[0, :, :] = boundary[-1, :, :] = True
        boundary[:, 0, :] = boundary[:, -1, :] = True
        boundary[:, :, 0] = boundary[:, :, -1] = True
        for z, y, x in np.argwhere(surface & ~boundary):
            neighbours = (
                outside_air[z - 1, y, x],
                outside_air[z + 1, y, x],
                outside_air[z, y - 1, x],
                outside_air[z, y + 1, x],
                outside_air[z, y, x - 1],
                outside_air[z, y, x + 1],
            )
            self.assertTrue(any(neighbours))

    def test_support_order_and_lps_physical_coordinates(self):
        surface = np.zeros((6, 2, 11), dtype=bool)
        surface[0, 0, 0] = True
        surface[1, 1, 1] = True  # Same 20 mm cell as [0,0,0].
        surface[0, 0, 10] = True
        surface[5, 0, 0] = True
        spacing = np.asarray([2.0, 3.0, 4.0])
        origin = np.asarray([10.0, -5.0, 100.0])
        direction = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        output = build_ct_support_20mm(
            surface,
            spacing,
            origin,
            direction,
            volume_shape_zyx=surface.shape,
            physical_unit='mm',
            coordinate_system='left-posterior-superior (LPS)',
        )

        np.testing.assert_array_equal(output['ct_support_spatial_shape_20mm'], [2, 1, 2])
        np.testing.assert_array_equal(output['ct_support_linear_20mm'], [0, 1, 2])
        np.testing.assert_array_equal(
            output['ct_support_indices_20mm'],
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
        )
        displacement_xyz = np.asarray(
            [[10.0, 10.0, 10.0], [30.0, 10.0, 10.0], [10.0, 10.0, 30.0]]
        )
        expected_phys = origin + displacement_xyz @ direction.T
        np.testing.assert_allclose(output['ct_support_phys_20mm'], expected_phys)
        self.assertFalse(
            np.array_equal(
                output['ct_support_phys_20mm'],
                output['ct_support_indices_20mm'].astype(np.float64),
            )
        )

    def test_ct_preprocessing_preserves_dual_branch_and_gt(self):
        collated = m2_ct_collate_fn([self.sample])
        self.assertIn('point', collated)
        self.assertIn('ct', collated)
        self.assertNotIn('points', collated)
        self.assertNotIn('features', collated)
        self.assertIs(collated['point']['point_xyz_phys'], self.point_xyz_phys)
        self.assertIs(collated['point']['point_normal'], self.point_normal)
        self.assertIs(collated['ct']['ct_volume'], self.ct_volume)
        self.assertIs(collated['ct']['ct_metadata'], self.ct_metadata)
        self.assertIs(collated['pair_metadata'], self.pair_metadata)
        self.assertIs(collated['gt_transform'], self.gt_transform)
        self.assertEqual(collated['gt_transform_direction'], 'Point Cloud -> CT')
        self.assertEqual(collated['physical_unit'], 'mm')
        self.assertIn('ct_context_features', collated['ct'])
        self.assertIn('ct_support_phys_20mm', collated['ct'])

    def test_ct_collate_rejects_missing_unit_or_wrong_axes(self):
        sample = dict(self.sample)
        del sample['physical_unit']
        with self.assertRaisesRegex(CTPreprocessingContractError, 'explicit physical_unit'):
            m2_ct_collate_fn([sample])

        sample = dict(self.sample)
        sample['ct_array_axis_order'] = ('x', 'y', 'z')
        with self.assertRaisesRegex(CTPreprocessingContractError, 'array_axis_order'):
            m2_ct_collate_fn([sample])

    def test_encoder_source_topology_projection_and_prohibitions(self):
        source = (EXPERIMENT_DIR / 'ct_encoder.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        call_names = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertEqual(call_names.count('SubMConv3d'), 3)
        self.assertEqual(call_names.count('SparseConv3d'), 2)
        self.assertEqual(call_names.count('BatchNorm1d'), 5)
        self.assertEqual(call_names.count('ReLU'), 5)
        self.assertEqual(call_names.count('Linear'), 1)
        self.assertEqual(source.count('self.ct_proj(V_raw)'), 1)
        self.assertIn("x20.indices[:, 1:]", source)
        self.assertIn('torch.searchsorted', source)

        prohibited = (
            'sinkhorn',
            'weighted svd',
            'procrustes',
            'ransac',
            'cross-modal attention',
            'f.normalize',
            'registration_collate_fn_stack_mode',
        )
        production_paths = (
            EXPERIMENT_DIR / 'config.py',
            EXPERIMENT_DIR / 'dataset.py',
            EXPERIMENT_DIR / 'ct_encoder.py',
            EXPERIMENT_DIR / 'validate_ct_encoder.py',
        )
        production_source = '\n'.join(path.read_text(encoding='utf-8').lower() for path in production_paths)
        for term in prohibited:
            self.assertNotIn(term, production_source)
        self.assertNotIn('gt_transform', inspect.getsource(build_ct_support_20mm))

    @unittest.skipUnless(
        TORCH_AVAILABLE and SPCONV_AVAILABLE and torch is not None and torch.cuda.is_available(),
        'PyTorch + CUDA + spconv are required for the sparse GPU integration test.',
    )
    def test_sparse_ct_encoder_output_contract(self):
        volume = np.zeros((21, 21, 21), dtype=np.int16)
        sample = dict(self.sample)
        sample['ct_volume'] = volume
        sample['ct_shape'] = np.asarray(volume.shape, dtype=np.int64)
        collated = m2_ct_collate_fn([sample])
        ct_dict = collated['ct']
        device = torch.device('cuda')
        for name, dtype in (
            ('ct_context_features', torch.float32),
            ('ct_context_indices', torch.int32),
            ('ct_support_indices_20mm', torch.int32),
            ('ct_support_linear_20mm', torch.int64),
            ('ct_support_phys_20mm', torch.float64),
        ):
            ct_dict[name] = torch.as_tensor(ct_dict[name], dtype=dtype, device=device)

        encoder = CTEncoder(make_cfg()).to(device).eval()
        projection_calls = []
        hook = encoder.ct_proj.register_forward_hook(lambda *_args: projection_calls.append(1))
        with torch.no_grad():
            output = encoder(ct_dict)
        hook.remove()

        support_count = output['support_count']
        self.assertGreater(support_count, 0)
        self.assertEqual(tuple(output['V_raw'].shape), (support_count, 128))
        self.assertEqual(tuple(output['K'].shape), (support_count, 256))
        self.assertEqual(tuple(output['Xv_phys_coarse'].shape), (support_count, 3))
        self.assertEqual(len(projection_calls), 1)
        for name in ('V_raw', 'K', 'Xv_phys_coarse'):
            self.assertTrue(bool(torch.isfinite(output[name]).all().item()))


if __name__ == '__main__':
    unittest.main()
