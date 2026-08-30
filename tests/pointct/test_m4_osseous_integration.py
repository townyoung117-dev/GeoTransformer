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

    import m4_osseous_integration as osseous
    from m4_soft_modulation import run_m4_soft_modulated_matching
    from matching import PointCTMatcher
else:
    torch = None


class M4OsseousSourceContractTest(unittest.TestCase):
    def test_formula_fixed_score_and_no_trainable_state_are_explicit(self):
        source = (EXPERIMENT_DIR / 'm4_osseous_integration.py').read_text(
            encoding='utf-8'
        )
        self.assertIn('pair_reliability * osseous_score.unsqueeze(0)', source)
        self.assertIn(
            'final_similarity = soft_similarity - osseous_penalty', source
        )
        self.assertIn('OSSEOUS_CENTER_HU = 300.0', source)
        self.assertIn('OSSEOUS_TAU_HU = 100.0', source)
        self.assertIn('OSSEOUS_RADIUS_MM = 20.0', source)
        self.assertNotIn('nn.Parameter', source)
        self.assertNotIn('torch.nn', source)

    def test_score_boundary_accepts_only_declared_ct_and_alignment_inputs(self):
        if not TORCH_AVAILABLE:
            self.skipTest('PyTorch is required to import the integration module.')
        self.assertEqual(
            tuple(inspect.signature(osseous.compute_aligned_osseous_score).parameters),
            (
                'defective_ct_volume',
                'ct_spacing',
                'ct_origin',
                'ct_direction',
                'support_locations_mm',
                'ct_token_locations_mm',
            ),
        )
        forbidden = {
            'complete_ct',
            'complete_point',
            'gt_transform',
            'patient_id',
            'defect_id',
            'registration_success',
            'test_statistics',
            'anchor',
            'mirror',
            'anatomical_axes',
            'atlas',
            'anatomical_labels',
            'mp',
            'mv',
        }
        parameters = {
            name.lower()
            for name in inspect.signature(
                osseous.compute_aligned_osseous_score
            ).parameters
        }
        self.assertFalse(parameters & forbidden)

    def test_frozen_modules_do_not_import_new_integration(self):
        for filename in (
            'm4_hard_constraint.py',
            'm4_soft_modulation.py',
            'm4_continuous_osseous_prior.py',
            'evaluate_m3.py',
        ):
            source = (EXPERIMENT_DIR / filename).read_text(encoding='utf-8')
            self.assertNotIn('m4_osseous_integration', source, filename)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for numerical tests.')
class M4OsseousIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.matcher = PointCTMatcher(
            projected_dim=256,
            temperature=0.2,
            sinkhorn_iterations=60,
            alpha_init=1.0,
        )

    @staticmethod
    def _descriptors(*, requires_grad=False):
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
        q.requires_grad_(requires_grad)
        k.requires_grad_(requires_grad)
        return q, k

    @staticmethod
    def _geometry():
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [50.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [150.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        valid = torch.tensor([True, False, True, True])
        return coordinates, valid

    @staticmethod
    def _ct_inputs():
        return {
            'defective_ct_volume': np.asarray(
                [[[-1000, -1000, 1000, 1000]]], dtype=np.int16
            ),
            'ct_spacing': np.asarray([50.0, 1.0, 1.0]),
            'ct_origin': np.zeros(3, dtype=np.float64),
            'ct_direction': np.eye(3, dtype=np.float64),
        }

    def _run(self, osseous_strength, *, requires_grad=False):
        q, k = self._descriptors(requires_grad=requires_grad)
        coordinates, valid = self._geometry()
        output = osseous.run_m4_osseous_integrated_matching(
            q=q,
            k=k,
            point_coordinates_mm=coordinates,
            ct_token_locations_mm=coordinates.clone(),
            point_valid_mask=valid,
            ct_valid_mask=valid.clone(),
            sigma_mm=60.0,
            soft_strength=2.0,
            osseous_strength=osseous_strength,
            matcher=self.matcher,
            support_locations_mm=coordinates.clone(),
            **self._ct_inputs(),
        )
        return output, q, k, coordinates, valid

    def test_aligned_score_shape_finite_range_and_frozen_parameters(self):
        coordinates, _ = self._geometry()
        with mock.patch.object(
            osseous,
            'compute_continuous_osseous_support_score',
            wraps=osseous.compute_continuous_osseous_support_score,
        ) as frozen_score:
            score = osseous.compute_aligned_osseous_score(
                support_locations_mm=coordinates,
                ct_token_locations_mm=coordinates.clone(),
                **self._ct_inputs(),
            )
        self.assertEqual(tuple(score.shape), (4,))
        self.assertEqual(score.dtype, torch.float32)
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.all((score >= 0.0) & (score <= 1.0)))
        self.assertLess(float(score[0]), float(score[2]))
        self.assertFalse(score.requires_grad)
        args = frozen_score.call_args.args
        self.assertEqual(args[-3:], (300.0, 100.0, 20.0))

    def test_alignment_count_order_nan_and_score_contract_fail_closed(self):
        coordinates, _ = self._geometry()
        common = self._ct_inputs()
        with self.assertRaisesRegex(
            osseous.M4OsseousIntegrationError, 'token counts differ'
        ):
            osseous.compute_aligned_osseous_score(
                support_locations_mm=coordinates[:-1],
                ct_token_locations_mm=coordinates,
                **common,
            )
        with self.assertRaisesRegex(
            osseous.M4OsseousIntegrationError, 'including token order'
        ):
            osseous.compute_aligned_osseous_score(
                support_locations_mm=coordinates.flip(0),
                ct_token_locations_mm=coordinates,
                **common,
            )
        nonfinite = coordinates.clone()
        nonfinite[0, 0] = float('nan')
        with self.assertRaises(osseous.M4OsseousIntegrationError):
            osseous.compute_aligned_osseous_score(
                support_locations_mm=coordinates,
                ct_token_locations_mm=nonfinite,
                **common,
            )
        for returned in (
            np.zeros(3, dtype=np.float64),
            np.asarray([0.1, 0.2, np.nan, 0.4]),
            np.asarray([0.1, -0.1, 0.4, 0.8]),
            np.asarray([0.1, 0.2, 1.1, 0.8]),
        ):
            with self.subTest(returned=returned):
                with mock.patch.object(
                    osseous,
                    'compute_continuous_osseous_support_score',
                    return_value=returned,
                ):
                    with self.assertRaises(osseous.M4OsseousIntegrationError):
                        osseous.compute_aligned_osseous_score(
                            support_locations_mm=coordinates,
                            ct_token_locations_mm=coordinates.clone(),
                            **common,
                        )

    def test_candidate_c_is_pairwise_and_not_row_column_separable(self):
        soft = torch.zeros((2, 2), dtype=torch.float32)
        pair = torch.tensor([[0.81, 0.72], [0.36, 0.32]])
        score = torch.tensor([0.10, 0.95])
        output = osseous.integrate_osseous_prior(soft, pair, score, 1.0)
        expected_joint = pair * score.unsqueeze(0)
        torch.testing.assert_close(
            output['osseous_pair_reliability'], expected_joint, rtol=0, atol=0
        )
        effect = output['final_similarity'] - soft
        self.assertNotEqual(float(effect[0, 0]), float(effect[1, 0]))
        mixed = effect[0, 0] - effect[0, 1] - effect[1, 0] + effect[1, 1]
        self.assertGreater(abs(float(mixed)), 1e-5)

    def test_nonzero_effect_survives_sinkhorn_and_changes_expected_ranking(self):
        soft = torch.tensor([[1.2, 1.1], [0.2, 1.0]], dtype=torch.float32)
        pair = torch.tensor([[0.81, 0.72], [0.36, 0.32]])
        score = torch.tensor([0.10, 0.95])
        final = osseous.integrate_osseous_prior(
            soft, pair, score, 1.0
        )['final_similarity']
        mask = torch.ones((1, 2), dtype=torch.bool)
        soft_assignment = self.matcher.transport(soft.unsqueeze(0), mask, mask)[0]
        final_assignment = self.matcher.transport(final.unsqueeze(0), mask, mask)[0]
        self.assertFalse(torch.equal(soft, final))
        self.assertFalse(
            torch.allclose(soft_assignment, final_assignment, rtol=1e-7, atol=1e-7)
        )
        soft_probability = soft_assignment.exp()
        final_probability = final_assignment.exp()
        self.assertGreater(
            float(soft_probability[0, 0].detach()),
            float(soft_probability[0, 1].detach()),
        )
        self.assertLess(
            float(final_probability[0, 0].detach()),
            float(final_probability[0, 1].detach()),
        )
        self.assertLess(
            float(final_probability[0, 0].detach()),
            float(soft_probability[0, 0].detach()),
        )
        self.assertGreater(
            float(final_probability[0, 1].detach()),
            float(soft_probability[0, 1].detach()),
        )

    def test_zero_strength_is_exact_frozen_m4_2a_degeneration(self):
        q, k = self._descriptors()
        coordinates, valid = self._geometry()
        q_before = q.clone()
        k_before = k.clone()
        soft = run_m4_soft_modulated_matching(
            q=q,
            k=k,
            point_coordinates_mm=coordinates,
            ct_coordinates_mm=coordinates.clone(),
            point_valid_mask=valid,
            ct_valid_mask=valid.clone(),
            sigma_mm=60.0,
            strength=2.0,
            matcher=self.matcher,
        )
        integrated = osseous.run_m4_osseous_integrated_matching(
            q=q,
            k=k,
            point_coordinates_mm=coordinates,
            ct_token_locations_mm=coordinates.clone(),
            point_valid_mask=valid,
            ct_valid_mask=valid.clone(),
            sigma_mm=60.0,
            soft_strength=2.0,
            osseous_strength=0.0,
            matcher=self.matcher,
            support_locations_mm=coordinates.clone(),
            **self._ct_inputs(),
        )
        self.assertTrue(torch.equal(q, q_before))
        self.assertTrue(torch.equal(k, k_before))
        self.assertTrue(torch.equal(soft['base_similarity'], integrated['base_similarity']))
        self.assertTrue(
            torch.equal(soft['modulated_similarity'], integrated['soft_similarity'])
        )
        self.assertIs(integrated['final_similarity'], integrated['soft_similarity'])
        self.assertTrue(torch.equal(soft['log_assignment'], integrated['log_assignment']))
        self.assertFalse(integrated['osseous_similarity_changed'])

    def test_hard_priority_nonpositive_penalty_gradients_and_state(self):
        state_keys = tuple(self.matcher.state_dict())
        output, q, k, _, valid = self._run(1.0, requires_grad=True)
        self.assertIs(output['point_valid_mask'], valid)
        self.assertTrue(torch.equal(output['ct_valid_mask'], valid))
        real_assignment = output['log_assignment'][:-1, :-1]
        self.assertTrue(torch.isneginf(real_assignment[~valid, :]).all())
        self.assertTrue(torch.isneginf(real_assignment[:, ~valid]).all())
        self.assertTrue(torch.all(output['osseous_penalty'] >= 0.0))
        self.assertTrue(torch.all(output['final_similarity'] <= output['soft_similarity']))
        self.assertFalse(output['osseous_score'].requires_grad)
        loss = -output['log_assignment'][
            torch.isfinite(output['log_assignment'])
        ].mean()
        loss.backward()
        for gradient in (q.grad, k.grad, self.matcher.transport.alpha.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(state_keys, tuple(self.matcher.state_dict()))

    def test_invalid_strength_score_shape_range_and_dtype_fail_closed(self):
        soft = torch.zeros((2, 2), dtype=torch.float32)
        pair = torch.ones((2, 2), dtype=torch.float32)
        valid_score = torch.ones(2, dtype=torch.float32)
        for strength in (-1.0, float('nan'), float('inf'), -float('inf'), True):
            with self.subTest(strength=strength):
                with self.assertRaises(osseous.M4OsseousIntegrationError):
                    osseous.integrate_osseous_prior(
                        soft, pair, valid_score, strength
                    )
        invalid_scores = (
            torch.ones(3, dtype=torch.float32),
            torch.tensor([0.2, float('nan')]),
            torch.tensor([-0.1, 0.2]),
            torch.tensor([0.2, 1.1]),
            torch.ones(2, dtype=torch.int64),
        )
        for score in invalid_scores:
            with self.subTest(score=score):
                with self.assertRaises(osseous.M4OsseousIntegrationError):
                    osseous.integrate_osseous_prior(soft, pair, score, 1.0)

    def test_diagnostics_are_complete_finite_and_fixed(self):
        output, _, _, _, _ = self._run(1.0)
        self.assertTrue(output['m4_hard_constraint_active'])
        self.assertTrue(output['m4_soft_modulation_active'])
        self.assertTrue(output['m4_osseous_prior_active'])
        self.assertEqual(output['osseous_center_hu'], 300.0)
        self.assertEqual(output['osseous_tau_hu'], 100.0)
        self.assertEqual(output['osseous_radius_mm'], 20.0)
        self.assertEqual(output['m4_osseous_strength'], 1.0)
        for name in (
            'osseous_score_min',
            'osseous_score_mean',
            'osseous_score_max',
            'osseous_pair_reliability_min',
            'osseous_pair_reliability_mean',
            'osseous_pair_reliability_max',
            'final_modulation_min',
            'final_modulation_mean',
            'final_modulation_max',
        ):
            self.assertTrue(np.isfinite(output[name]), name)


if __name__ == '__main__':
    unittest.main()
