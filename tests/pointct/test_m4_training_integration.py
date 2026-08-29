import ast
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

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    import training
    from gt_correspondence import build_coarse_gt_correspondence
    from m4_hard_constraint import build_m4_mask_aware_gt
    from matching import PointCTMatcher
    from matching_loss import compute_collision_aware_match_loss


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
                temperature=0.1,
                sinkhorn_iterations=20,
                alpha_init=1.0,
            )
            self.history = []

        def forward(self, q, k, point_valid_mask=None, ct_valid_mask=None):
            output = super().forward(
                q=q,
                k=k,
                point_valid_mask=point_valid_mask,
                ct_valid_mask=ct_valid_mask,
            )
            self.history.append(
                {
                    'q': q.detach().clone(),
                    'k': k.detach().clone(),
                    'point_valid_mask': point_valid_mask,
                    'ct_valid_mask': ct_valid_mask,
                    'log_assignment': output['log_assignment'].detach().clone(),
                }
            )
            return output


def _function_source(path, function_name):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return ast.get_source_segment(source, function)


class M4TrainingProvenanceSourceTest(unittest.TestCase):
    def test_explicit_flag_controls_config_log_and_result_provenance(self):
        path = EXPERIMENT_DIR / 'train_m3_defect.py'
        adapter_source = _function_source(path, '_install_defect_adapter')
        run_source = _function_source(path, 'run_defect_training')

        self.assertIn(
            'm4_hard_constraint_active = enable_m4_defect_mapping',
            adapter_source,
        )
        self.assertIn(
            "record['m4_hard_constraint_active'] = m4_hard_constraint_active",
            adapter_source,
        )
        self.assertIn(
            "enriched['m4_hard_constraint_active'] = m4_hard_constraint_active",
            adapter_source,
        )
        self.assertIn("output['m4_hard_constraint_active'] = enabled", run_source)


@unittest.skipIf(torch is None, 'PyTorch is not installed locally.')
class M4TrainingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.point_encoder = ToyPointEncoder()
        self.ct_encoder = ToyCTEncoder()
        self.matcher = RecordingMatcher()

    @staticmethod
    def _sample(mapping_enabled):
        point_phys = torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        ct_phys = point_phys.clone()
        return {
            'subject_id': 'ToySubject',
            'mapping_enabled': mapping_enabled,
            'point_features': torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            # Point 0 has a high descriptor score for defect CT token 1, while
            # its physical GT target is intact CT token 0.
            'ct_features': torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            'point_phys': point_phys,
            'ct_phys': ct_phys,
            'point_intact': torch.tensor([True, False, True]),
            'ct_intact': torch.tensor([True, False, True]),
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

    def _validate(self, mapping_enabled):
        return training.run_validation_step(
            self._sample(mapping_enabled),
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            **self._step_kwargs(),
        )

    def test_mapping_off_is_exact_all_true_m3_fallback(self):
        with mock.patch.object(
            training,
            'build_m4_mask_aware_gt',
            side_effect=AssertionError('M4 GT wrapper must not run on M3 fallback.'),
        ):
            result = self._validate(False)
        recorded = self.matcher.history[-1]
        expected_point_mask = torch.ones(3, dtype=torch.bool)
        expected_ct_mask = torch.ones(3, dtype=torch.bool)
        torch.testing.assert_close(recorded['point_valid_mask'], expected_point_mask)
        torch.testing.assert_close(recorded['ct_valid_mask'], expected_ct_mask)

        direct_match = self.matcher(
            recorded['q'],
            recorded['k'],
            point_valid_mask=expected_point_mask,
            ct_valid_mask=expected_ct_mask,
        )
        sample = self._sample(False)
        correspondence = build_coarse_gt_correspondence(
            Xp_phys_coarse=sample['point_phys'].numpy(),
            Xv_phys_coarse=sample['ct_phys'].numpy(),
            gt_transform=sample['gt_transform'],
            gt_transform_direction=sample['gt_transform_direction'],
            primary_max_distance_mm=100.0,
            high_confidence_distance_mm=1.0,
        )
        expected_loss = compute_collision_aware_match_loss(
            direct_match['log_assignment'],
            torch.as_tensor(correspondence['gt_primary_ct_index']),
            torch.as_tensor(correspondence['gt_primary_valid']),
            point_valid_mask=expected_point_mask,
            ct_valid_mask=expected_ct_mask,
        )['loss']
        torch.testing.assert_close(result['loss'], expected_loss)
        self.assertFalse(result['m4_hard_constraint_enabled'])
        self.assertEqual(result['point_total_tokens'], 3)
        self.assertEqual(result['point_intact_tokens'], 3)
        self.assertEqual(result['point_excluded_tokens'], 0)
        self.assertEqual(result['ct_total_tokens'], 3)
        self.assertEqual(result['ct_intact_tokens'], 3)
        self.assertEqual(result['ct_excluded_tokens'], 0)
        self.assertEqual(result['num_supervised_points_after_mask'], 3)

    def test_mapping_on_changes_assignment_supervision_and_loss_not_q_or_k(self):
        off_result = self._validate(False)
        off_recorded = self.matcher.history[-1]
        loss_masks = {}
        gt_output = {}

        def gt_spy(*args, **kwargs):
            output = build_m4_mask_aware_gt(*args, **kwargs)
            gt_output.update(output)
            return output

        def loss_spy(*args, **kwargs):
            loss_masks.update(
                {
                    'point_valid_mask': kwargs['point_valid_mask'],
                    'ct_valid_mask': kwargs['ct_valid_mask'],
                }
            )
            return compute_collision_aware_match_loss(*args, **kwargs)

        with mock.patch.object(
            training,
            'build_m4_mask_aware_gt',
            side_effect=gt_spy,
        ) as gt_wrapper, mock.patch.object(
            training,
            'compute_collision_aware_match_loss',
            side_effect=loss_spy,
        ):
            on_result = self._validate(True)
        on_recorded = self.matcher.history[-1]

        expected_mask = torch.tensor([True, False, True])
        torch.testing.assert_close(on_recorded['point_valid_mask'], expected_mask)
        torch.testing.assert_close(on_recorded['ct_valid_mask'], expected_mask)
        self.assertIs(loss_masks['point_valid_mask'], on_recorded['point_valid_mask'])
        self.assertIs(loss_masks['ct_valid_mask'], on_recorded['ct_valid_mask'])
        torch.testing.assert_close(off_recorded['q'], on_recorded['q'])
        torch.testing.assert_close(off_recorded['k'], on_recorded['k'])
        self.assertFalse(
            torch.equal(
                off_recorded['log_assignment'],
                on_recorded['log_assignment'],
            )
        )
        self.assertTrue(torch.isneginf(on_recorded['log_assignment'][1, :-1]).all())
        self.assertTrue(torch.isneginf(on_recorded['log_assignment'][:-1, 1]).all())
        self.assertFalse(torch.allclose(off_result['loss'], on_result['loss']))
        self.assertTrue(torch.isfinite(on_result['loss']))

        self.assertEqual(gt_wrapper.call_count, 1)
        self.assertFalse(bool(gt_output['gt_primary_valid'][1]))
        supervised_targets = gt_output['gt_primary_ct_index'][
            gt_output['gt_primary_valid']
        ]
        self.assertTrue(np.all(np.asarray([True, False, True])[supervised_targets]))
        self.assertTrue(on_result['m4_hard_constraint_enabled'])
        self.assertEqual(on_result['point_total_tokens'], 3)
        self.assertEqual(on_result['point_intact_tokens'], 2)
        self.assertEqual(on_result['point_excluded_tokens'], 1)
        self.assertEqual(on_result['ct_total_tokens'], 3)
        self.assertEqual(on_result['ct_intact_tokens'], 2)
        self.assertEqual(on_result['ct_excluded_tokens'], 1)
        self.assertEqual(on_result['num_supervised_points_after_mask'], 2)

    def test_mapping_on_training_backward_has_finite_gradients(self):
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


if __name__ == '__main__':
    unittest.main()
