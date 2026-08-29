import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch

    import train_m3_defect
    import training
    from matching import PointCTMatcher
    from matching_loss import compute_collision_aware_match_loss
else:
    torch = None


class M4SoftTrainingSourceContractTest(unittest.TestCase):
    def test_cli_and_development_hyperparameter_status_are_explicit(self):
        source = (EXPERIMENT_DIR / 'train_m3_defect.py').read_text(encoding='utf-8')
        self.assertIn("'--enable-m4-soft-modulation'", source)
        self.assertIn("'--m4-soft-sigma-mm'", source)
        self.assertIn("'--m4-soft-strength'", source)
        self.assertIn(
            'M4 DEVELOPMENT HYPERPARAMETERS - NOT FROZEN PAPER HYPERPARAMETERS',
            source,
        )

    def test_frozen_train_m3_entry_has_no_soft_modulation_edits(self):
        source = (EXPERIMENT_DIR / 'train_m3.py').read_text(encoding='utf-8')
        self.assertNotIn('m4_soft_modulation', source)
        self.assertNotIn('m4_soft_sigma', source)


if TORCH_AVAILABLE:
    class ToyPointEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 256, bias=False)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.weight[:3].copy_(torch.eye(3))

        def forward(self, point_input):
            output = {
                'Q': self.projection(point_input['features']),
                'Xp_phys_coarse': point_input['points'][0],
            }
            if point_input.get('m4_defect_mapping_enabled', False):
                for field in (
                    'point_intact_coarse',
                    'point_raw_total_count_coarse',
                    'point_raw_defect_count_coarse',
                ):
                    output[field] = point_input[field]
            return output


    class ToyCTEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 256, bias=False)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.weight[:3].copy_(torch.eye(3))

        def forward(self, ct_input):
            output = {
                'K': self.projection(ct_input['ct_context_features']),
                'Xv_phys_coarse': ct_input['ct_support_phys_20mm'],
            }
            if ct_input.get('m4_defect_mapping_enabled', False):
                for field in (
                    'ct_intact_coarse',
                    'ct_raw_total_count_coarse',
                    'ct_raw_defect_count_coarse',
                ):
                    output[field] = ct_input[field]
            return output


    class RecordingMatcher(PointCTMatcher):
        def __init__(self):
            super().__init__(
                projected_dim=256,
                temperature=0.2,
                sinkhorn_iterations=40,
                alpha_init=1.0,
            )
            self.history = []

        def forward(self, q, k, point_valid_mask=None, ct_valid_mask=None):
            output = super().forward(
                q,
                k,
                point_valid_mask=point_valid_mask,
                ct_valid_mask=ct_valid_mask,
            )
            self.history.append(
                {
                    'q': q.detach().clone(),
                    'k': k.detach().clone(),
                    'log_assignment': output['log_assignment'].detach().clone(),
                    'point_valid_mask': point_valid_mask,
                    'ct_valid_mask': ct_valid_mask,
                }
            )
            return output


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for soft training tests.')
class M4SoftTrainingConfigTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            'enable_m4_defect_mapping': True,
            'enable_m4_soft_modulation': True,
            'm4_soft_sigma_mm': 4.0,
            'm4_soft_strength': 2.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_soft_requires_hard_mapping(self):
        with self.assertRaisesRegex(
            train_m3_defect.DefectTrainingContractError,
            'requires --enable-m4-defect-mapping',
        ):
            train_m3_defect._resolve_m4_soft_config(
                self._args(enable_m4_defect_mapping=False)
            )

    def test_sigma_and_strength_fail_closed(self):
        for field, values in (
            ('m4_soft_sigma_mm', (None, 0.0, -1.0, float('nan'), float('inf'))),
            ('m4_soft_strength', (None, -1.0, float('nan'), float('inf'))),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(
                        train_m3_defect.DefectTrainingContractError
                    ):
                        train_m3_defect._resolve_m4_soft_config(
                            self._args(**{field: value})
                        )

    def test_hard_only_defaults_remain_valid(self):
        resolved = train_m3_defect._resolve_m4_soft_config(
            self._args(
                enable_m4_soft_modulation=False,
                m4_soft_sigma_mm=None,
                m4_soft_strength=None,
            )
        )
        self.assertEqual(resolved, (True, False, None, None))


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for soft training tests.')
class M4SoftTrainingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.point_encoder = ToyPointEncoder()
        self.ct_encoder = ToyCTEncoder()
        self.matcher = RecordingMatcher()

    @staticmethod
    def _sample(mapping_enabled=True):
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return {
            'subject_id': 'ToySoft',
            'mapping_enabled': mapping_enabled,
            'point_features': torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            'ct_features': torch.tensor(
                [
                    [0.8, 0.6, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.8, 0.6],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            'point_phys': coordinates,
            'ct_phys': coordinates.clone(),
            'point_intact': torch.tensor([True, False, True, True]),
            'ct_intact': torch.tensor([True, False, True, True]),
            'gt_transform': np.eye(4, dtype=np.float64),
            'gt_transform_direction': 'Point Cloud -> CT',
        }

    @staticmethod
    def _point_collate(samples):
        sample = samples[0]
        branch = {
            'features': sample['point_features'],
            'point_network_scale_mm_to_m': 1.0,
            'points': [sample['point_phys']],
            'neighbors': [],
            'subsampling': [],
            'upsampling': [],
        }
        if sample['mapping_enabled']:
            intact = sample['point_intact']
            branch.update(
                {
                    'm4_defect_mapping_enabled': True,
                    'point_intact_coarse': intact,
                    'point_raw_total_count_coarse': torch.ones(
                        intact.shape, dtype=torch.int64
                    ),
                    'point_raw_defect_count_coarse': (~intact).to(torch.int64),
                }
            )
        return {'subject_id': sample['subject_id'], 'point': branch}

    @staticmethod
    def _ct_collate(samples):
        sample = samples[0]
        count = sample['ct_phys'].shape[0]
        branch = {
            'ct_context_features': sample['ct_features'],
            'ct_context_indices': torch.zeros((count, 4), dtype=torch.int64),
            'ct_support_indices_20mm': torch.zeros((count, 4), dtype=torch.int64),
            'ct_support_linear_20mm': torch.arange(count, dtype=torch.int64),
            'ct_support_phys_20mm': sample['ct_phys'],
            'ct_context_spatial_shape': (1, 1, count),
            'ct_support_spatial_shape_20mm': (1, 1, count),
            'physical_unit': 'mm',
            'coordinate_system': 'LPS',
        }
        if sample['mapping_enabled']:
            intact = sample['ct_intact']
            branch.update(
                {
                    'm4_defect_mapping_enabled': True,
                    'ct_intact_coarse': intact,
                    'ct_raw_total_count_coarse': torch.ones(
                        intact.shape, dtype=torch.int64
                    ),
                    'ct_raw_defect_count_coarse': (~intact).to(torch.int64),
                }
            )
        return {'subject_id': sample['subject_id'], 'ct': branch}

    @classmethod
    def _step_kwargs(cls):
        return {
            'point_collate_fn': cls._point_collate,
            'ct_collate_fn': cls._ct_collate,
            'primary_max_distance_mm': 100.0,
            'high_confidence_distance_mm': 1.0,
            'device': torch.device('cpu'),
        }

    def _validate(self, *, soft_enabled, strength, mapping_enabled=True):
        captures = {'loss_inputs': [], 'gt': [], 'soft': []}
        original_soft = training.run_m4_soft_modulated_matching
        original_gt = training.build_m4_mask_aware_gt

        def soft_spy(**kwargs):
            output = original_soft(**kwargs)
            captures['soft'].append((kwargs, output))
            return output

        def gt_spy(*args, **kwargs):
            output = original_gt(*args, **kwargs)
            captures['gt'].append(
                {
                    name: np.array(output[name], copy=True)
                    for name in ('gt_primary_ct_index', 'gt_primary_valid')
                }
            )
            return output

        def loss_spy(*args, **kwargs):
            captures['loss_inputs'].append(args[0].detach().clone())
            return compute_collision_aware_match_loss(*args, **kwargs)

        with mock.patch.object(
            training,
            'run_m4_soft_modulated_matching',
            side_effect=soft_spy,
        ), mock.patch.object(
            training,
            'build_m4_mask_aware_gt',
            side_effect=gt_spy,
        ), mock.patch.object(
            training,
            'compute_collision_aware_match_loss',
            side_effect=loss_spy,
        ):
            result = training.run_validation_step(
                self._sample(mapping_enabled),
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                m4_soft_modulation_enabled=soft_enabled,
                m4_soft_sigma_mm=4.0 if soft_enabled else None,
                m4_soft_strength=strength if soft_enabled else None,
                **self._step_kwargs(),
            )
        return result, captures

    def test_soft_off_is_hard_only_and_reports_inactive(self):
        result, captures = self._validate(soft_enabled=False, strength=0.0)
        self.assertFalse(result['m4_soft_modulation_enabled'])
        self.assertFalse(result['m4_soft_modulation_active'])
        self.assertFalse(result['soft_similarity_changed'])
        self.assertEqual(captures['soft'], [])
        self.assertEqual(len(self.matcher.history), 1)

    def test_lambda_zero_matches_hard_assignment_loss_gt_and_q_k(self):
        hard_result, hard_capture = self._validate(
            soft_enabled=False,
            strength=0.0,
        )
        hard_match = self.matcher.history[-1]
        zero_result, zero_capture = self._validate(
            soft_enabled=True,
            strength=0.0,
        )
        soft_kwargs, soft_output = zero_capture['soft'][0]
        self.assertTrue(
            torch.equal(
                hard_capture['loss_inputs'][0],
                zero_capture['loss_inputs'][0],
            )
        )
        torch.testing.assert_close(hard_result['loss'], zero_result['loss'])
        self.assertFalse(zero_result['soft_similarity_changed'])
        torch.testing.assert_close(hard_match['q'], soft_kwargs['q'])
        torch.testing.assert_close(hard_match['k'], soft_kwargs['k'])
        self.assertTrue(
            torch.equal(
                hard_match['log_assignment'],
                soft_output['log_assignment'],
            )
        )
        self.assertIs(
            soft_output['point_valid_mask'],
            soft_kwargs['point_valid_mask'],
        )
        self.assertIs(
            soft_output['ct_valid_mask'],
            soft_kwargs['ct_valid_mask'],
        )
        self.assertTrue(
            torch.equal(
                hard_match['point_valid_mask'],
                soft_output['point_valid_mask'],
            )
        )
        self.assertTrue(
            torch.equal(
                hard_match['ct_valid_mask'],
                soft_output['ct_valid_mask'],
            )
        )
        self.assertTrue(
            np.array_equal(
                hard_capture['gt'][0]['gt_primary_ct_index'],
                zero_capture['gt'][0]['gt_primary_ct_index'],
            )
        )
        self.assertTrue(
            np.array_equal(
                hard_capture['gt'][0]['gt_primary_valid'],
                zero_capture['gt'][0]['gt_primary_valid'],
            )
        )

    def test_positive_lambda_changes_similarity_assignment_and_loss_not_gt(self):
        hard_result, hard_capture = self._validate(
            soft_enabled=False,
            strength=0.0,
        )
        hard_match = self.matcher.history[-1]
        soft_result, soft_capture = self._validate(
            soft_enabled=True,
            strength=12.0,
        )
        soft_kwargs, soft_output = soft_capture['soft'][0]
        self.assertTrue(soft_result['soft_similarity_changed'])
        self.assertFalse(
            torch.equal(
                soft_output['base_similarity'],
                soft_output['modulated_similarity'],
            )
        )
        self.assertFalse(
            torch.allclose(
                hard_match['log_assignment'],
                soft_output['log_assignment'],
                atol=1e-7,
                rtol=1e-7,
            )
        )
        self.assertFalse(torch.allclose(hard_result['loss'], soft_result['loss']))
        torch.testing.assert_close(hard_match['q'], soft_kwargs['q'])
        torch.testing.assert_close(hard_match['k'], soft_kwargs['k'])
        self.assertTrue(
            np.array_equal(
                hard_capture['gt'][0]['gt_primary_ct_index'],
                soft_capture['gt'][0]['gt_primary_ct_index'],
            )
        )
        self.assertTrue(
            np.array_equal(
                hard_capture['gt'][0]['gt_primary_valid'],
                soft_capture['gt'][0]['gt_primary_valid'],
            )
        )
        self.assertTrue(torch.isfinite(soft_result['loss']))

    def test_soft_fails_closed_without_hard_mapping(self):
        with self.assertRaisesRegex(
            training.M3TrainingContractError,
            'requires active M4 defect mapping',
        ):
            self._validate(
                soft_enabled=True,
                strength=2.0,
                mapping_enabled=False,
            )

    def test_soft_training_backward_and_matcher_alpha_gradient_are_finite(self):
        optimizer = training.create_optimizer(
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
        )
        result = training.run_training_step(
            self._sample(True),
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            optimizer,
            m4_soft_modulation_enabled=True,
            m4_soft_sigma_mm=4.0,
            m4_soft_strength=4.0,
            **self._step_kwargs(),
        )
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertTrue(result['loss'].requires_grad)
        for module in (self.point_encoder, self.ct_encoder, self.matcher):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertIsNotNone(self.matcher.transport.alpha.grad)
        self.assertTrue(torch.isfinite(self.matcher.transport.alpha.grad))


if __name__ == '__main__':
    unittest.main()
