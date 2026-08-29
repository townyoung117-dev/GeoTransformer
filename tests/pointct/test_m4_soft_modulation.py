import importlib.util
import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch

    from m4_soft_modulation import (
        M4SoftModulationError,
        compute_defect_proximity_reliability,
        modulate_cross_modal_similarity,
        run_m4_soft_modulated_matching,
    )
    from matching import PointCTMatcher, compute_cross_modal_similarity
else:
    torch = None


class M4SoftModulationSourceContractTest(unittest.TestCase):
    def test_core_responsibilities_are_separate_and_reuse_frozen_m3_math(self):
        source = (
            EXPERIMENT_DIR / 'm4_soft_modulation.py'
        ).read_text(encoding='utf-8')
        self.assertIn('def compute_defect_proximity_reliability(', source)
        self.assertIn('def modulate_cross_modal_similarity(', source)
        self.assertIn('def run_m4_soft_modulated_matching(', source)
        self.assertIn('compute_cross_modal_similarity(', source)
        self.assertIn("transport = getattr(matcher, 'transport', None)", source)

    def test_pairwise_cross_term_is_explicit_and_no_trainable_parameter_exists(self):
        source = (
            EXPERIMENT_DIR / 'm4_soft_modulation.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            'point_reliability.unsqueeze(1) * ct_reliability.unsqueeze(0)',
            source,
        )
        self.assertNotIn('nn.Parameter', source)
        self.assertNotIn('defect_count / total_count', source)
        self.assertNotIn('raw_defect_count', source)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M4 soft numerical tests.')
class M4SoftModulationTest(unittest.TestCase):
    def setUp(self):
        self.matcher = PointCTMatcher(
            projected_dim=256,
            temperature=0.2,
            sinkhorn_iterations=40,
            alpha_init=1.0,
        )

    @staticmethod
    def _descriptors():
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
        return q, k

    @staticmethod
    def _geometry():
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        valid_mask = torch.tensor([True, False, True, True])
        return coordinates, valid_mask

    def _run(self, strength):
        q, k = self._descriptors()
        point_coordinates, point_mask = self._geometry()
        ct_coordinates, ct_mask = self._geometry()
        return run_m4_soft_modulated_matching(
            q=q,
            k=k,
            point_coordinates_mm=point_coordinates,
            ct_coordinates_mm=ct_coordinates,
            point_valid_mask=point_mask,
            ct_valid_mask=ct_mask,
            sigma_mm=4.0,
            strength=strength,
            matcher=self.matcher,
        )

    def test_reliability_range_invalid_zero_and_distance_order(self):
        coordinates, valid_mask = self._geometry()
        reliability = compute_defect_proximity_reliability(
            coordinates,
            valid_mask,
            sigma_mm=4.0,
        )
        self.assertTrue(torch.all(reliability >= 0.0))
        self.assertTrue(torch.all(reliability <= 1.0))
        self.assertEqual(float(reliability[1]), 0.0)
        self.assertLess(float(reliability[0]), float(reliability[2]))
        self.assertLess(float(reliability[2]), float(reliability[3]))

    def test_no_excluded_token_returns_all_one_without_synthetic_defect(self):
        coordinates, _ = self._geometry()
        reliability = compute_defect_proximity_reliability(
            coordinates,
            torch.ones(4, dtype=torch.bool),
            sigma_mm=4.0,
        )
        torch.testing.assert_close(reliability, torch.ones(4, dtype=torch.float32))

    def test_sigma_contract_rejects_nonpositive_nan_and_inf(self):
        coordinates, valid_mask = self._geometry()
        for value in (0.0, -1.0, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(M4SoftModulationError):
                    compute_defect_proximity_reliability(
                        coordinates,
                        valid_mask,
                        sigma_mm=value,
                    )

    def test_strength_contract_rejects_negative_nan_and_inf(self):
        similarity = torch.zeros((2, 2), dtype=torch.float32)
        reliability = torch.ones(2, dtype=torch.float32)
        for value in (-1.0, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(M4SoftModulationError):
                    modulate_cross_modal_similarity(
                        similarity,
                        reliability,
                        reliability,
                        strength=value,
                    )

    def test_shape_and_device_mismatch_fail_closed(self):
        coordinates, valid_mask = self._geometry()
        with self.assertRaisesRegex(M4SoftModulationError, 'shape'):
            compute_defect_proximity_reliability(
                coordinates,
                valid_mask[:-1],
                sigma_mm=4.0,
            )
        meta_mask = torch.ones(4, dtype=torch.bool, device='meta')
        with self.assertRaisesRegex(M4SoftModulationError, 'same device'):
            compute_defect_proximity_reliability(
                coordinates,
                meta_mask,
                sigma_mm=4.0,
            )
        with self.assertRaisesRegex(M4SoftModulationError, 'shape'):
            modulate_cross_modal_similarity(
                torch.zeros((2, 3), dtype=torch.float32),
                torch.ones(3, dtype=torch.float32),
                torch.ones(3, dtype=torch.float32),
                strength=1.0,
            )

    def test_coordinate_and_reliability_nonfinite_fail_closed(self):
        coordinates, valid_mask = self._geometry()
        for value in (float('nan'), float('inf')):
            bad = coordinates.clone()
            bad[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(M4SoftModulationError):
                    compute_defect_proximity_reliability(
                        bad,
                        valid_mask,
                        sigma_mm=4.0,
                    )
        with self.assertRaises(M4SoftModulationError):
            modulate_cross_modal_similarity(
                torch.zeros((2, 2), dtype=torch.float32),
                torch.tensor([1.0, float('nan')]),
                torch.ones(2, dtype=torch.float32),
                strength=1.0,
            )

    def test_lambda_zero_is_strict_similarity_and_assignment_degeneration(self):
        q, k = self._descriptors()
        coordinates, valid_mask = self._geometry()
        base_similarity = compute_cross_modal_similarity(q, k, self.matcher.temperature)
        hard_output = self.matcher(
            q,
            k,
            point_valid_mask=valid_mask,
            ct_valid_mask=valid_mask,
        )
        soft_output = self._run(strength=0.0)
        self.assertIs(soft_output['base_similarity'], soft_output['modulated_similarity'])
        self.assertTrue(torch.equal(soft_output['base_similarity'], base_similarity))
        self.assertTrue(
            torch.equal(
                soft_output['log_assignment'],
                hard_output['log_assignment'],
            )
        )
        self.assertFalse(soft_output['soft_similarity_changed'])

    def test_positive_lambda_changes_similarity_and_sinkhorn_assignment(self):
        zero_output = self._run(strength=0.0)
        soft_output = self._run(strength=4.0)
        self.assertTrue(soft_output['soft_similarity_changed'])
        self.assertFalse(
            torch.equal(
                soft_output['modulated_similarity'],
                soft_output['base_similarity'],
            )
        )
        self.assertFalse(
            torch.allclose(
                soft_output['log_assignment'],
                zero_output['log_assignment'],
                atol=1e-7,
                rtol=1e-7,
            )
        )

    def test_pairwise_cross_term_is_not_row_column_separable(self):
        base = torch.zeros((2, 2), dtype=torch.float32)
        output = modulate_cross_modal_similarity(
            base,
            torch.tensor([0.2, 0.8], dtype=torch.float32),
            torch.tensor([0.1, 0.9], dtype=torch.float32),
            strength=2.0,
        )
        penalty = base - output['modulated_similarity']
        interaction = penalty[0, 0] + penalty[1, 1] - penalty[0, 1] - penalty[1, 0]
        self.assertGreater(abs(float(interaction)), 1e-5)

    def test_near_defect_high_score_wrong_pair_receives_larger_penalty(self):
        base = torch.tensor(
            [[5.0, 4.0], [4.0, 5.0]],
            dtype=torch.float32,
        )
        output = modulate_cross_modal_similarity(
            base,
            torch.tensor([0.1, 0.9], dtype=torch.float32),
            torch.tensor([0.2, 0.95], dtype=torch.float32),
            strength=3.0,
        )
        penalty = base - output['modulated_similarity']
        self.assertGreater(float(penalty[0, 0]), float(penalty[1, 1]))
        self.assertLess(
            float(output['modulated_similarity'][0, 0]),
            float(base[0, 0]),
        )

    def test_hard_masks_are_preserved_and_invalid_tokens_never_reactivated(self):
        output = self._run(strength=4.0)
        _, valid_mask = self._geometry()
        self.assertTrue(torch.equal(output['point_valid_mask'], valid_mask))
        self.assertTrue(torch.equal(output['ct_valid_mask'], valid_mask))
        self.assertTrue(torch.all(output['point_reliability'][~valid_mask] == 0.0))
        self.assertTrue(torch.all(output['ct_reliability'][~valid_mask] == 0.0))
        ordinary_assignment = output['log_assignment'][:-1, :-1]
        self.assertTrue(torch.isneginf(ordinary_assignment[~valid_mask, :]).all())
        self.assertTrue(torch.isneginf(ordinary_assignment[:, ~valid_mask]).all())

    def test_q_k_unchanged_and_backward_gradients_including_alpha_are_finite(self):
        q, k = self._descriptors()
        q.requires_grad_(True)
        k.requires_grad_(True)
        q_before = q.detach().clone()
        k_before = k.detach().clone()
        coordinates, valid_mask = self._geometry()
        output = run_m4_soft_modulated_matching(
            q=q,
            k=k,
            point_coordinates_mm=coordinates,
            ct_coordinates_mm=coordinates.clone(),
            point_valid_mask=valid_mask,
            ct_valid_mask=valid_mask.clone(),
            sigma_mm=4.0,
            strength=3.0,
            matcher=self.matcher,
        )
        torch.testing.assert_close(q.detach(), q_before)
        torch.testing.assert_close(k.detach(), k_before)
        finite_assignment = output['log_assignment'][
            torch.isfinite(output['log_assignment'])
        ]
        loss = -finite_assignment.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for gradient in (q.grad, k.grad, self.matcher.transport.alpha.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_diagnostic_fields_are_finite_and_complete(self):
        output = self._run(strength=4.0)
        self.assertTrue(output['m4_soft_modulation_enabled'])
        self.assertTrue(output['m4_soft_modulation_active'])
        self.assertEqual(output['m4_soft_sigma_mm'], 4.0)
        self.assertEqual(output['m4_soft_strength'], 4.0)
        for name in (
            'point_reliability_min',
            'point_reliability_mean',
            'point_reliability_max',
            'ct_reliability_min',
            'ct_reliability_mean',
            'ct_reliability_max',
            'pair_reliability_min',
            'pair_reliability_mean',
            'pair_reliability_max',
            'base_similarity_mean',
            'modulated_similarity_mean',
        ):
            self.assertTrue(math.isfinite(output[name]), name)


if __name__ == '__main__':
    unittest.main()
