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


class M4OsseousTrainingSourceContractTest(unittest.TestCase):
    def test_cli_flags_and_development_status_are_explicit(self):
        source = (EXPERIMENT_DIR / 'train_m3_defect.py').read_text(encoding='utf-8')
        self.assertIn("'--m4-osseous-prior'", source)
        self.assertIn("'--m4-osseous-strength'", source)
        self.assertIn('DEVELOPMENT_ONLY - NOT FROZEN PAPER HYPERPARAMETER', source)

    def test_frozen_entries_and_score_definition_are_not_edited(self):
        for filename in ('train_m3.py', 'evaluate_m3.py'):
            source = (EXPERIMENT_DIR / filename).read_text(encoding='utf-8')
            self.assertNotIn('m4_osseous', source)
        score_source = (
            EXPERIMENT_DIR / 'm4_continuous_osseous_prior.py'
        ).read_text(encoding='utf-8')
        self.assertNotIn('m4_osseous_strength', score_source)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for training tests.')
class M4OsseousTrainingConfigTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            'enable_m4_defect_mapping': True,
            'enable_m4_soft_modulation': True,
            'm4_soft_sigma_mm': 4.0,
            'm4_soft_strength': 2.0,
            'm4_osseous_prior': True,
            'm4_osseous_strength': 1.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_prior_requires_both_mapping_and_soft_modulation(self):
        for overrides in (
            {'enable_m4_defect_mapping': False},
            {
                'enable_m4_soft_modulation': False,
                'm4_soft_sigma_mm': None,
                'm4_soft_strength': None,
            },
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    train_m3_defect.DefectTrainingContractError,
                    'requires --enable-m4-defect-mapping',
                ):
                    train_m3_defect._resolve_m4_osseous_config(
                        self._args(**overrides)
                    )

    def test_strength_is_required_finite_and_nonnegative(self):
        for value in (None, True, -1.0, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(train_m3_defect.DefectTrainingContractError):
                    train_m3_defect._resolve_m4_osseous_config(
                        self._args(m4_osseous_strength=value)
                    )
        resolved = train_m3_defect._resolve_m4_osseous_config(
            self._args(m4_osseous_strength=0.0)
        )
        self.assertEqual(resolved[-2:], (True, 0.0))

    def test_strength_without_prior_fails_closed(self):
        with self.assertRaisesRegex(
            train_m3_defect.DefectTrainingContractError,
            'requires --m4-osseous-prior',
        ):
            train_m3_defect._resolve_m4_osseous_config(
                self._args(m4_osseous_prior=False)
            )


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
        def __init__(self, *, reverse_locations=False):
            super().__init__()
            self.projection = torch.nn.Linear(3, 256, bias=False)
            self.reverse_locations = reverse_locations
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.weight[:3].copy_(torch.eye(3))

        def forward(self, ct_input):
            locations = ct_input['ct_support_phys_20mm']
            if self.reverse_locations:
                locations = locations.flip(0)
            output = {
                'K': self.projection(ct_input['ct_context_features']),
                'Xv_phys_coarse': locations,
            }
            if ct_input.get('m4_defect_mapping_enabled', False):
                for field in (
                    'ct_intact_coarse',
                    'ct_raw_total_count_coarse',
                    'ct_raw_defect_count_coarse',
                ):
                    output[field] = ct_input[field]
            return output


    class ToyMatcher(PointCTMatcher):
        def __init__(self):
            super().__init__(
                projected_dim=256,
                temperature=0.2,
                sinkhorn_iterations=40,
                alpha_init=1.0,
            )


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for training tests.')
class M4OsseousTrainingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.point_encoder = ToyPointEncoder()
        self.ct_encoder = ToyCTEncoder()
        self.matcher = ToyMatcher()

    @staticmethod
    def _sample(mapping_enabled=True, missing_ct_field=None):
        coordinates = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [45.0, 0.0, 0.0],
                [65.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        volume = np.full((1, 1, 81), -1000, dtype=np.int16)
        volume[:, :, 38:] = 1000
        return {
            'subject_id': 'ToyOsseous',
            'mapping_enabled': mapping_enabled,
            'missing_ct_field': missing_ct_field,
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
            'ct_volume': volume,
            'ct_spacing': np.ones(3, dtype=np.float64),
            'ct_origin': np.zeros(3, dtype=np.float64),
            'ct_direction': np.eye(3, dtype=np.float64),
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
                    'point_raw_total_count_coarse': torch.ones_like(
                        intact, dtype=torch.int64
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
            'ct_volume': sample['ct_volume'],
            'ct_spacing': sample['ct_spacing'],
            'ct_origin': sample['ct_origin'],
            'ct_direction': sample['ct_direction'],
        }
        if sample['mapping_enabled']:
            intact = sample['ct_intact']
            branch.update(
                {
                    'm4_defect_mapping_enabled': True,
                    'ct_intact_coarse': intact,
                    'ct_raw_total_count_coarse': torch.ones_like(
                        intact, dtype=torch.int64
                    ),
                    'ct_raw_defect_count_coarse': (~intact).to(torch.int64),
                }
            )
        if sample['missing_ct_field'] is not None:
            branch.pop(sample['missing_ct_field'])
        return {'subject_id': sample['subject_id'], 'ct': branch}

    @classmethod
    def _step_kwargs(cls):
        return {
            'point_collate_fn': cls._point_collate,
            'ct_collate_fn': cls._ct_collate,
            'primary_max_distance_mm': 100.0,
            'high_confidence_distance_mm': 1.0,
            'device': torch.device('cpu'),
            'm4_soft_modulation_enabled': True,
            'm4_soft_sigma_mm': 4.0,
            'm4_soft_strength': 2.0,
        }

    def _validate(self, *, osseous_enabled, osseous_strength=None, sample=None):
        captures = {'soft': [], 'osseous': [], 'loss': [], 'gt': []}
        original_soft = training.run_m4_soft_modulated_matching
        original_osseous = training.run_m4_osseous_integrated_matching
        original_gt = training.build_m4_mask_aware_gt

        def soft_spy(**kwargs):
            output = original_soft(**kwargs)
            captures['soft'].append((kwargs, output))
            return output

        def osseous_spy(**kwargs):
            output = original_osseous(**kwargs)
            captures['osseous'].append((kwargs, output))
            return output

        def gt_spy(*args, **kwargs):
            output = original_gt(*args, **kwargs)
            captures['gt'].append(
                {
                    field: np.array(output[field], copy=True)
                    for field in ('gt_primary_ct_index', 'gt_primary_valid')
                }
            )
            return output

        def loss_spy(*args, **kwargs):
            captures['loss'].append(args[0].detach().clone())
            return compute_collision_aware_match_loss(*args, **kwargs)

        with mock.patch.object(
            training, 'run_m4_soft_modulated_matching', side_effect=soft_spy
        ), mock.patch.object(
            training, 'run_m4_osseous_integrated_matching', side_effect=osseous_spy
        ), mock.patch.object(
            training, 'build_m4_mask_aware_gt', side_effect=gt_spy
        ), mock.patch.object(
            training, 'compute_collision_aware_match_loss', side_effect=loss_spy
        ):
            result = training.run_validation_step(
                self._sample() if sample is None else sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                m4_osseous_prior_enabled=osseous_enabled,
                m4_osseous_strength=osseous_strength,
                **self._step_kwargs(),
            )
        return result, captures

    def test_prior_off_uses_the_frozen_m4_2a_route(self):
        result, captures = self._validate(osseous_enabled=False)
        self.assertEqual(len(captures['soft']), 1)
        self.assertEqual(captures['osseous'], [])
        self.assertFalse(result['m4_osseous_prior_enabled'])
        self.assertFalse(result['m4_osseous_prior_active'])
        self.assertIsNone(result['m4_osseous_strength'])

    def test_zero_strength_matches_m4_2a_q_k_similarity_assignment_loss_and_gt(self):
        baseline, baseline_capture = self._validate(osseous_enabled=False)
        zero, zero_capture = self._validate(
            osseous_enabled=True,
            osseous_strength=0.0,
        )
        soft_kwargs, soft_output = baseline_capture['soft'][0]
        osseous_kwargs, osseous_output = zero_capture['osseous'][0]
        torch.testing.assert_close(soft_kwargs['q'], osseous_kwargs['q'], rtol=0, atol=0)
        torch.testing.assert_close(soft_kwargs['k'], osseous_kwargs['k'], rtol=0, atol=0)
        torch.testing.assert_close(
            soft_output['base_similarity'],
            osseous_output['base_similarity'],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            soft_output['modulated_similarity'],
            osseous_output['soft_similarity'],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            soft_output['log_assignment'],
            osseous_output['log_assignment'],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(baseline['loss'], zero['loss'], rtol=0, atol=0)
        self.assertTrue(torch.equal(baseline_capture['loss'][0], zero_capture['loss'][0]))
        for field in ('gt_primary_ct_index', 'gt_primary_valid'):
            self.assertTrue(
                np.array_equal(
                    baseline_capture['gt'][0][field],
                    zero_capture['gt'][0][field],
                )
            )
        self.assertFalse(zero['osseous_similarity_changed'])

    def test_only_explicit_defective_ct_and_token_fields_reach_core_api(self):
        _, captures = self._validate(osseous_enabled=True, osseous_strength=1.0)
        kwargs = captures['osseous'][0][0]
        self.assertEqual(
            set(kwargs),
            {
                'q',
                'k',
                'point_coordinates_mm',
                'ct_token_locations_mm',
                'point_valid_mask',
                'ct_valid_mask',
                'sigma_mm',
                'soft_strength',
                'osseous_strength',
                'matcher',
                'defective_ct_volume',
                'ct_spacing',
                'ct_origin',
                'ct_direction',
                'support_locations_mm',
            },
        )
        self.assertNotIn('sample', kwargs)
        self.assertNotIn('gt_transform', kwargs)
        self.assertNotIn('subject_id', kwargs)
        self.assertNotIn('defect_id', kwargs)

    def test_positive_strength_changes_assignment_and_reports_diagnostics(self):
        baseline, baseline_capture = self._validate(osseous_enabled=False)
        changed, changed_capture = self._validate(
            osseous_enabled=True,
            osseous_strength=3.0,
        )
        soft_output = baseline_capture['soft'][0][1]
        osseous_output = changed_capture['osseous'][0][1]
        self.assertTrue(changed['osseous_similarity_changed'])
        self.assertFalse(
            torch.allclose(
                soft_output['log_assignment'],
                osseous_output['log_assignment'],
                rtol=1e-7,
                atol=1e-7,
            )
        )
        self.assertTrue(changed['m4_osseous_prior_active'])
        self.assertEqual(changed['m4_osseous_strength'], 3.0)
        self.assertLess(changed['osseous_score_min'], changed['osseous_score_max'])
        self.assertTrue(torch.isfinite(changed['loss']))
        self.assertFalse(torch.allclose(baseline['loss'], changed['loss']))

    def test_missing_defective_ct_header_and_mapping_fail_closed(self):
        for field in ('ct_volume', 'ct_spacing', 'ct_origin', 'ct_direction'):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    training.M3TrainingContractError,
                    'M4 osseous defective CT input',
                ):
                    self._validate(
                        osseous_enabled=True,
                        osseous_strength=1.0,
                        sample=self._sample(missing_ct_field=field),
                    )
        with self.assertRaisesRegex(
            training.M3TrainingContractError,
            'requires active M4 defect mapping',
        ):
            self._validate(
                osseous_enabled=True,
                osseous_strength=1.0,
                sample=self._sample(mapping_enabled=False),
            )

    def test_wrong_support_token_order_fails_closed(self):
        self.ct_encoder = ToyCTEncoder(reverse_locations=True)
        with self.assertRaisesRegex(
            training.M3TrainingContractError,
            'must exactly match ct_token_locations_mm',
        ):
            self._validate(osseous_enabled=True, osseous_strength=1.0)

    def test_training_backward_gradients_are_finite_and_state_is_unchanged(self):
        state_keys_before = {
            name: tuple(module.state_dict())
            for name, module in (
                ('point', self.point_encoder),
                ('ct', self.ct_encoder),
                ('matcher', self.matcher),
            )
        }
        optimizer = training.create_optimizer(
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
        )
        result = training.run_training_step(
            self._sample(),
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            optimizer,
            m4_osseous_prior_enabled=True,
            m4_osseous_strength=1.0,
            **self._step_kwargs(),
        )
        self.assertTrue(result['loss'].requires_grad)
        self.assertTrue(torch.isfinite(result['loss']))
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
        state_keys_after = {
            name: tuple(module.state_dict())
            for name, module in (
                ('point', self.point_encoder),
                ('ct', self.ct_encoder),
                ('matcher', self.matcher),
            )
        }
        self.assertEqual(state_keys_before, state_keys_after)


if __name__ == '__main__':
    unittest.main()
