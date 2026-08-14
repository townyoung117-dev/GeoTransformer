import importlib.util
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
    import torch.nn as nn

    from matching import (
        PointCTLogSinkhorn,
        PointCTMatcher,
        PointCTMatchingContractError,
        compute_marginal_residual,
    )
else:
    torch = None
    nn = None


FIXTURE_ITERATIONS = 200
FIXTURE_ALPHA_INIT = 0.3
FIXTURE_TEMPERATURE = 0.4
MARGINAL_ATOL = 2e-4


class M3SinkhornSourceContractTest(unittest.TestCase):
    def test_sinkhorn_source_has_no_cuda_hardcode(self):
        source = (EXPERIMENT_DIR / 'matching.py').read_text(encoding='utf-8').lower()
        self.assertNotIn('.cuda(', source.replace(' ', ''))


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M3-2 Sinkhorn numerical tests.')
class M3SinkhornTest(unittest.TestCase):
    @staticmethod
    def _scores(batch_size, num_point, num_ct, device=None, dtype=None):
        if dtype is None:
            dtype = torch.float32
        values = torch.linspace(
            -0.7,
            0.9,
            steps=batch_size * num_point * num_ct,
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size, num_point, num_ct)
        return values.to(dtype=dtype)

    @staticmethod
    def _transport(device=None, iterations=FIXTURE_ITERATIONS):
        transport = PointCTLogSinkhorn(iterations, FIXTURE_ALPHA_INIT)
        if device is not None:
            transport = transport.to(device)
        return transport

    def _all_valid_output(self, num_point, num_ct):
        scores = self._scores(1, num_point, num_ct)
        point_mask = torch.ones((1, num_point), dtype=torch.bool)
        ct_mask = torch.ones((1, num_ct), dtype=torch.bool)
        output = self._transport()(scores, point_mask, ct_mask)
        return output, point_mask, ct_mask

    def assertMarginals(self, output, point_mask, ct_mask):
        probability = output.exp()
        num_point = point_mask.shape[1]
        num_ct = ct_mask.shape[1]
        for batch_index in range(output.shape[0]):
            valid_point = point_mask[batch_index]
            valid_ct = ct_mask[batch_index]
            m = int(valid_point.sum())
            n = int(valid_ct.sum())
            torch.testing.assert_close(
                probability[batch_index, :num_point, :].sum(dim=1)[valid_point],
                torch.ones(m),
                rtol=0.0,
                atol=MARGINAL_ATOL,
            )
            torch.testing.assert_close(
                probability[batch_index, :, :num_ct].sum(dim=0)[valid_ct],
                torch.ones(n),
                rtol=0.0,
                atol=MARGINAL_ATOL,
            )
            self.assertAlmostEqual(
                float(probability[batch_index, num_point, :].sum()),
                float(n),
                delta=MARGINAL_ATOL,
            )
            self.assertAlmostEqual(
                float(probability[batch_index, :, num_ct].sum()),
                float(m),
                delta=MARGINAL_ATOL,
            )
            self.assertAlmostEqual(
                float(probability[batch_index].sum()),
                float(m + n),
                delta=MARGINAL_ATOL,
            )

    def test_all_valid_three_by_five_rectangular_transport(self):
        output, point_mask, ct_mask = self._all_valid_output(3, 5)
        self.assertMarginals(output, point_mask, ct_mask)

    def test_all_valid_five_by_three_rectangular_transport(self):
        output, point_mask, ct_mask = self._all_valid_output(5, 3)
        self.assertMarginals(output, point_mask, ct_mask)

    def test_dustbin_output_shape_is_rectangular_plus_one_per_side(self):
        output, _, _ = self._all_valid_output(3, 5)
        self.assertEqual(tuple(output.shape), (1, 4, 6))

    def test_valid_ordinary_point_rows_have_unit_mass(self):
        output, _, _ = self._all_valid_output(3, 5)
        torch.testing.assert_close(
            output.exp()[0, :3, :].sum(dim=1),
            torch.ones(3),
            rtol=0.0,
            atol=MARGINAL_ATOL,
        )

    def test_valid_ordinary_ct_columns_have_unit_mass(self):
        output, _, _ = self._all_valid_output(3, 5)
        torch.testing.assert_close(
            output.exp()[0, :, :5].sum(dim=0),
            torch.ones(5),
            rtol=0.0,
            atol=MARGINAL_ATOL,
        )

    def test_point_dustbin_row_has_ct_valid_count_mass(self):
        output, _, _ = self._all_valid_output(3, 5)
        self.assertAlmostEqual(float(output.exp()[0, 3, :].sum()), 5.0, delta=MARGINAL_ATOL)

    def test_ct_dustbin_column_has_point_valid_count_mass(self):
        output, _, _ = self._all_valid_output(3, 5)
        self.assertAlmostEqual(float(output.exp()[0, :, 5].sum()), 3.0, delta=MARGINAL_ATOL)

    def test_total_transport_mass_is_sum_of_valid_counts(self):
        output, _, _ = self._all_valid_output(3, 5)
        self.assertAlmostEqual(float(output.exp().sum()), 8.0, delta=MARGINAL_ATOL)

    def test_padding_rows_and_columns_have_exact_zero_mass(self):
        scores = self._scores(1, 4, 6)
        point_mask = torch.tensor([[True, True, True, False]])
        ct_mask = torch.tensor([[True, True, True, True, True, False]])
        output = self._transport()(scores, point_mask, ct_mask)
        probability = output.exp()
        self.assertEqual(float(probability[0, 3, :].sum()), 0.0)
        self.assertEqual(float(probability[0, :, 5].sum()), 0.0)
        self.assertTrue(torch.isneginf(output[0, 3, :]).all())
        self.assertTrue(torch.isneginf(output[0, :, 5]).all())
        self.assertMarginals(output, point_mask, ct_mask)

    def test_all_point_tokens_invalid_fails_closed(self):
        with self.assertRaisesRegex(PointCTMatchingContractError, 'at least one valid'):
            self._transport()(
                self._scores(1, 4, 6),
                torch.zeros((1, 4), dtype=torch.bool),
                torch.ones((1, 6), dtype=torch.bool),
            )

    def test_all_ct_tokens_invalid_fails_closed(self):
        with self.assertRaisesRegex(PointCTMatchingContractError, 'at least one valid'):
            self._transport()(
                self._scores(1, 4, 6),
                torch.ones((1, 4), dtype=torch.bool),
                torch.zeros((1, 6), dtype=torch.bool),
            )

    def test_mixed_batch_uses_each_items_own_valid_counts(self):
        scores = self._scores(2, 4, 5)
        point_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        )
        ct_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, False, False, False]],
            dtype=torch.bool,
        )
        output = self._transport()(scores, point_mask, ct_mask)
        self.assertMarginals(output, point_mask, ct_mask)

    def test_alpha_is_float32_learnable_parameter(self):
        transport = self._transport()
        self.assertIsInstance(transport.alpha, nn.Parameter)
        self.assertEqual(transport.alpha.dtype, torch.float32)
        self.assertTrue(transport.alpha.requires_grad)

    def test_alpha_gradient_is_finite(self):
        transport = self._transport()
        output = transport(self._scores(1, 3, 5))
        objective = output[0, 0, 0] + 0.25 * output[0, -1, -1]
        objective.backward()
        self.assertIsNotNone(transport.alpha.grad)
        self.assertTrue(torch.isfinite(transport.alpha.grad))

    def test_q_and_k_gradients_are_finite(self):
        matcher = PointCTMatcher(
            256,
            FIXTURE_TEMPERATURE,
            FIXTURE_ITERATIONS,
            FIXTURE_ALPHA_INIT,
        )
        q = torch.randn((3, 256), dtype=torch.float32, requires_grad=True)
        k = torch.randn((5, 256), dtype=torch.float32, requires_grad=True)
        output = matcher(q, k)
        objective = output['log_assignment'][:3, :5].exp().square().sum()
        objective.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())

    def test_extreme_finite_logits_remain_nan_free(self):
        scores = torch.tensor(
            [[[-10000.0, 10000.0, 0.0], [10000.0, -10000.0, 5000.0]]],
            dtype=torch.float32,
        )
        output = self._transport()(scores)
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(torch.isfinite(output).all())

    def test_float16_scores_are_iterated_and_returned_as_float32(self):
        output = self._transport()(self._scores(1, 3, 5, dtype=torch.float16))
        self.assertEqual(output.dtype, torch.float32)

    def test_cpu_device_is_preserved(self):
        transport = self._transport(device=torch.device('cpu'))
        output = transport(self._scores(1, 3, 5, device=torch.device('cpu')))
        self.assertEqual(output.device.type, 'cpu')
        self.assertEqual(transport.alpha.device.type, 'cpu')

    @unittest.skipUnless(
        TORCH_AVAILABLE and torch.cuda.is_available(),
        'CUDA is required for the M3-2 Sinkhorn CUDA device test.',
    )
    def test_current_cuda_device_is_preserved(self):
        device = torch.device('cuda', torch.cuda.current_device())
        transport = self._transport(device=device)
        output = transport(self._scores(1, 3, 5, device=device))
        self.assertEqual(output.device, device)
        self.assertEqual(transport.alpha.device, device)

    def test_non_positive_iterations_fail_closed(self):
        for iterations in (0, -1, 1.5):
            with self.subTest(iterations=iterations):
                with self.assertRaisesRegex(PointCTMatchingContractError, 'positive integer'):
                    PointCTLogSinkhorn(iterations, 0.0)

    def test_marginal_diagnostic_reports_small_residuals(self):
        output, point_mask, ct_mask = self._all_valid_output(3, 5)
        residual = compute_marginal_residual(output, point_mask, ct_mask)
        self.assertLessEqual(float(residual['max_abs_row_residual']), MARGINAL_ATOL)
        self.assertLessEqual(float(residual['max_abs_col_residual']), MARGINAL_ATOL)


if __name__ == '__main__':
    unittest.main()
