import functools
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch

    import dataset
    import evaluate_m4_hard_constraint
    import evaluate_m4_soft_modulation
    import matching_filter
    import registration
    import training
    from evaluation import M3EvaluationContractError
    from matching import PointCTMatcher
else:
    torch = None


class M4SoftEvaluationSourceContractTest(unittest.TestCase):
    def test_soft_evaluator_is_independent_and_reuses_stable_components(self):
        source = (
            EXPERIMENT_DIR / 'evaluate_m4_soft_modulation.py'
        ).read_text(encoding='utf-8')
        self.assertNotIn('_run_inference_case', source)
        self.assertIn('resolve_m4_matching_masks', source)
        self.assertIn('run_m4_soft_modulated_matching', source)
        self.assertIn('extract_dustbin_aware_mutual_correspondences', source)
        self.assertIn('estimate_weighted_point_to_ct_transform', source)


if TORCH_AVAILABLE:
    class StaticEncoder(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output
            self.grad_modes = []

        def forward(self, _input):
            self.grad_modes.append(torch.is_grad_enabled())
            return self.output


def _encoder_outputs(with_mapping=True):
    q = torch.zeros((4, 256), dtype=torch.float32)
    k = torch.zeros((4, 256), dtype=torch.float32)
    q[:, :4] = torch.eye(4, dtype=torch.float32)
    k[:, :4] = torch.tensor(
        [
            [0.8, 0.6, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.0, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    point_output = {'Q': q, 'Xp_phys_coarse': coordinates}
    ct_output = {'K': k, 'Xv_phys_coarse': coordinates.clone()}
    if with_mapping:
        valid = torch.tensor([True, False, True, True])
        point_output.update(
            {
                'point_intact_coarse': valid,
                'point_raw_total_count_coarse': torch.ones(4, dtype=torch.int64),
                'point_raw_defect_count_coarse': (~valid).to(torch.int64),
            }
        )
        ct_valid = valid.clone()
        ct_output.update(
            {
                'ct_intact_coarse': ct_valid,
                'ct_raw_total_count_coarse': torch.ones(4, dtype=torch.int64),
                'ct_raw_defect_count_coarse': (~ct_valid).to(torch.int64),
            }
        )
    return point_output, ct_output


def _prepare_recorder(storage):
    def prepare(sample, point_collate_fn, ct_collate_fn, device):
        storage.update(
            {
                'sample': sample,
                'point_collate_fn': point_collate_fn,
                'ct_collate_fn': ct_collate_fn,
                'device': device,
            }
        )
        return object(), object()

    return prepare


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for soft evaluation tests.')
class M4SoftEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.protocol = {'matching_filter_min_confidence': None}
        self.matcher = PointCTMatcher(
            projected_dim=256,
            temperature=0.2,
            sinkhorn_iterations=40,
            alpha_init=1.0,
        )

    def _run_soft(self, *, strength, with_mapping=True):
        point_output, ct_output = _encoder_outputs(with_mapping=with_mapping)
        prepared = {}
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(prepared),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            wraps=matching_filter.extract_dustbin_aware_mutual_correspondences,
        ) as mutual_filter, mock.patch.object(
            registration,
            'estimate_weighted_point_to_ct_transform',
            wraps=registration.estimate_weighted_point_to_ct_transform,
        ) as weighted_registration:
            result = evaluate_m4_soft_modulation.run_m4_soft_modulation_inference(
                sample={'subject_id': 'ToySoftEval'},
                point_encoder=StaticEncoder(point_output),
                ct_encoder=StaticEncoder(ct_output),
                matcher=self.matcher,
                evaluation_protocol=self.protocol,
                device=self.device,
                sigma_mm=4.0,
                strength=strength,
            )
        return result, prepared, mutual_filter, weighted_registration, point_output, ct_output

    def test_complete_hard_soft_filter_registration_chain_and_diagnostics(self):
        (
            result,
            prepared,
            mutual_filter,
            weighted_registration,
            point_output,
            ct_output,
        ) = self._run_soft(strength=8.0)
        point_collate = prepared['point_collate_fn']
        ct_collate = prepared['ct_collate_fn']
        self.assertIsInstance(point_collate, functools.partial)
        self.assertIsInstance(ct_collate, functools.partial)
        self.assertIs(point_collate.func, dataset.m2_point_collate_fn)
        self.assertIs(ct_collate.func, dataset.m2_ct_collate_fn)
        self.assertTrue(point_collate.keywords['enable_m4_defect_mapping'])
        self.assertTrue(ct_collate.keywords['enable_m4_defect_mapping'])

        point_mask = point_output['point_intact_coarse']
        ct_mask = ct_output['ct_intact_coarse']
        self.assertIs(result['point_valid_mask'], point_mask)
        self.assertIs(result['ct_valid_mask'], ct_mask)
        self.assertIs(
            mutual_filter.call_args.kwargs['point_valid_mask'],
            point_mask,
        )
        self.assertIs(
            mutual_filter.call_args.kwargs['ct_valid_mask'],
            ct_mask,
        )
        registration_args = weighted_registration.call_args.args
        self.assertTrue(torch.all(point_mask[registration_args[2]]))
        self.assertTrue(torch.all(ct_mask[registration_args[3]]))

        self.assertTrue(result['m4_hard_constraint_active'])
        self.assertTrue(result['m4_soft_modulation_active'])
        self.assertEqual(result['m4_soft_sigma_mm'], 4.0)
        self.assertEqual(result['m4_soft_strength'], 8.0)
        self.assertTrue(result['soft_similarity_changed'])
        self.assertTrue(torch.all(result['point_reliability'][~point_mask] == 0))
        self.assertTrue(torch.all(result['ct_reliability'][~ct_mask] == 0))
        self.assertEqual(result['point_excluded_tokens'], 1)
        self.assertEqual(result['ct_excluded_tokens'], 1)

    def test_lambda_zero_matches_m4_hard_only_log_assignment(self):
        point_output, ct_output = _encoder_outputs(with_mapping=True)
        prepared = {}
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(prepared),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            wraps=matching_filter.extract_dustbin_aware_mutual_correspondences,
        ) as hard_filter:
            hard_result = evaluate_m4_hard_constraint.run_m4_hard_constraint_inference(
                sample={'subject_id': 'ToySoftEval'},
                point_encoder=StaticEncoder(point_output),
                ct_encoder=StaticEncoder(ct_output),
                matcher=self.matcher,
                evaluation_protocol=self.protocol,
                device=self.device,
            )
        hard_log_assignment = hard_filter.call_args.args[0].detach().clone()
        soft_result, _, _, _, _, _ = self._run_soft(strength=0.0)
        self.assertTrue(
            torch.equal(hard_log_assignment, soft_result['log_assignment'])
        )
        self.assertFalse(soft_result['soft_similarity_changed'])
        self.assertEqual(
            hard_result['registration_output']['num_correspondences'],
            soft_result['registration_output']['num_correspondences'],
        )

    def test_positive_lambda_changes_log_assignment_without_mutating_q_k(self):
        point_output, ct_output = _encoder_outputs(with_mapping=True)
        q_before = point_output['Q'].clone()
        k_before = ct_output['K'].clone()
        point_encoder = StaticEncoder(point_output)
        ct_encoder = StaticEncoder(ct_output)
        prepared = {}
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(prepared),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            wraps=matching_filter.extract_dustbin_aware_mutual_correspondences,
        ) as hard_filter:
            evaluate_m4_hard_constraint.run_m4_hard_constraint_inference(
                sample={'subject_id': 'ToySoftEval'},
                point_encoder=point_encoder,
                ct_encoder=ct_encoder,
                matcher=self.matcher,
                evaluation_protocol=self.protocol,
                device=self.device,
            )
        hard_log_assignment = hard_filter.call_args.args[0].detach().clone()
        soft_result, _, _, _, _, _ = self._run_soft(strength=8.0)
        self.assertFalse(
            torch.allclose(
                hard_log_assignment,
                soft_result['log_assignment'],
                atol=1e-7,
                rtol=1e-7,
            )
        )
        torch.testing.assert_close(point_output['Q'], q_before)
        torch.testing.assert_close(ct_output['K'], k_before)

    def test_missing_mapping_fails_before_matching(self):
        point_output, ct_output = _encoder_outputs(with_mapping=False)
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            return_value=(object(), object()),
        ), mock.patch.object(
            self.matcher.transport,
            'forward',
            wraps=self.matcher.transport.forward,
        ) as transport:
            with self.assertRaisesRegex(
                M3EvaluationContractError,
                'requires complete Point and CT',
            ):
                evaluate_m4_soft_modulation.run_m4_soft_modulation_inference(
                    sample={'subject_id': 'ToySoftEval'},
                    point_encoder=StaticEncoder(point_output),
                    ct_encoder=StaticEncoder(ct_output),
                    matcher=self.matcher,
                    evaluation_protocol=self.protocol,
                    device=self.device,
                    sigma_mm=4.0,
                    strength=2.0,
                )
        transport.assert_not_called()

    def test_invalid_soft_hyperparameters_fail_closed(self):
        point_output, ct_output = _encoder_outputs(with_mapping=True)
        for sigma_mm, strength in (
            (0.0, 1.0),
            (float('nan'), 1.0),
            (4.0, -1.0),
            (4.0, float('inf')),
        ):
            with self.subTest(sigma_mm=sigma_mm, strength=strength):
                with mock.patch.object(
                    training,
                    '_prepare_model_inputs',
                    return_value=(object(), object()),
                ):
                    with self.assertRaisesRegex(
                        M3EvaluationContractError,
                        'soft-modulated matching failed',
                    ):
                        evaluate_m4_soft_modulation.run_m4_soft_modulation_inference(
                            sample={'subject_id': 'ToySoftEval'},
                            point_encoder=StaticEncoder(point_output),
                            ct_encoder=StaticEncoder(ct_output),
                            matcher=self.matcher,
                            evaluation_protocol=self.protocol,
                            device=self.device,
                            sigma_mm=sigma_mm,
                            strength=strength,
                        )


if __name__ == '__main__':
    unittest.main()
