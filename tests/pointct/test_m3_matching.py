import ast
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

    from matching import (
        PointCTMatcher,
        PointCTMatchingContractError,
        compute_cross_modal_similarity,
    )
else:
    torch = None


FIXTURE_TEMPERATURE = 0.25
FIXTURE_ITERATIONS = 80
FIXTURE_ALPHA_INIT = -0.2


class M3MatchingSourceContractTest(unittest.TestCase):
    def test_matching_source_stops_at_m3_2_and_has_no_cuda_hardcode(self):
        source_path = EXPERIMENT_DIR / 'matching.py'
        source = source_path.read_text(encoding='utf-8')
        lowered = source.lower()
        prohibited_text = {
            'gt_correspondence',
            'gt_transform',
            'gt_primary',
            'procrustes',
            'svd',
            'ransac',
            'geometrictransformer',
            'localglobalregistration',
            'confidence filtering',
            'mutual matching',
            'training loss',
        }
        for text in prohibited_text:
            self.assertNotIn(text, lowered)
        self.assertNotIn('.cuda(', lowered.replace(' ', ''))

        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
        self.assertEqual(imported_modules, {'math', 'torch', 'torch.nn'})


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M3-2 matching numerical tests.')
class M3MatchingTest(unittest.TestCase):
    @staticmethod
    def _descriptors(count, offset=0, dtype=None, device=None):
        if dtype is None:
            dtype = torch.float32
        descriptors = torch.zeros((count, 256), dtype=dtype, device=device)
        for index in range(count):
            descriptors[index, (index + offset) % 256] = index + 1.0
            descriptors[index, (index + offset + 17) % 256] = 0.5
        return descriptors

    @staticmethod
    def _matcher(device=None):
        matcher = PointCTMatcher(
            projected_dim=256,
            temperature=FIXTURE_TEMPERATURE,
            sinkhorn_iterations=FIXTURE_ITERATIONS,
            alpha_init=FIXTURE_ALPHA_INIT,
        )
        if device is not None:
            matcher = matcher.to(device)
        return matcher

    def test_exact_l2_normalization_and_cosine_over_temperature(self):
        q = torch.zeros((2, 256), dtype=torch.float32)
        k = torch.zeros((2, 256), dtype=torch.float32)
        q[0, 0] = 2.0
        q[1, 1] = -3.0
        k[0, 0] = 4.0
        k[1, 1] = 5.0
        similarity = compute_cross_modal_similarity(q, k, temperature=0.5)
        expected = torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.float32)
        torch.testing.assert_close(similarity, expected, rtol=0.0, atol=1e-7)

    def test_cosine_similarity_has_no_sqrt_projected_dim_scaling(self):
        q = torch.zeros((1, 256), dtype=torch.float32)
        q[0, 23] = 7.0
        similarity = compute_cross_modal_similarity(q, q.clone(), temperature=0.5)
        self.assertEqual(float(similarity[0, 0]), 2.0)

    def test_rectangular_three_point_by_five_ct(self):
        output = self._matcher()(self._descriptors(3), self._descriptors(5, offset=31))
        self.assertEqual(tuple(output['similarity'].shape), (3, 5))
        self.assertEqual(tuple(output['log_assignment'].shape), (4, 6))

    def test_rectangular_five_point_by_three_ct(self):
        output = self._matcher()(self._descriptors(5), self._descriptors(3, offset=31))
        self.assertEqual(tuple(output['similarity'].shape), (5, 3))
        self.assertEqual(tuple(output['log_assignment'].shape), (6, 4))

    def test_output_shapes_and_default_structural_masks(self):
        output = self._matcher()(self._descriptors(2), self._descriptors(4, offset=41))
        self.assertEqual(tuple(output['similarity'].shape), (2, 4))
        self.assertEqual(tuple(output['log_assignment'].shape), (3, 5))
        self.assertEqual(output['point_valid_mask'].dtype, torch.bool)
        self.assertEqual(output['ct_valid_mask'].dtype, torch.bool)
        self.assertTrue(output['point_valid_mask'].all())
        self.assertTrue(output['ct_valid_mask'].all())

    def test_wrong_projected_descriptor_dimension_fails_closed(self):
        with self.assertRaises(PointCTMatchingContractError):
            compute_cross_modal_similarity(torch.ones((2, 255)), torch.ones((3, 256)), 0.5)
        with self.assertRaises(PointCTMatchingContractError):
            compute_cross_modal_similarity(torch.ones((2, 256)), torch.ones((3, 257)), 0.5)
        with self.assertRaises(PointCTMatchingContractError):
            PointCTMatcher(128, 0.5, 10, 0.0)

    def test_empty_point_input_fails_closed(self):
        with self.assertRaisesRegex(PointCTMatchingContractError, 'Point token'):
            compute_cross_modal_similarity(torch.empty((0, 256)), torch.ones((1, 256)), 0.5)

    def test_empty_ct_input_fails_closed(self):
        with self.assertRaisesRegex(PointCTMatchingContractError, 'CT token'):
            compute_cross_modal_similarity(torch.ones((1, 256)), torch.empty((0, 256)), 0.5)

    def test_nan_descriptor_fails_closed(self):
        q = torch.ones((2, 256))
        q[0, 0] = float('nan')
        with self.assertRaisesRegex(PointCTMatchingContractError, 'finite'):
            compute_cross_modal_similarity(q, torch.ones((3, 256)), 0.5)

    def test_infinite_descriptor_fails_closed(self):
        k = torch.ones((3, 256))
        k[1, 4] = float('inf')
        with self.assertRaisesRegex(PointCTMatchingContractError, 'finite'):
            compute_cross_modal_similarity(torch.ones((2, 256)), k, 0.5)

    def test_zero_norm_descriptor_fails_closed(self):
        with self.assertRaisesRegex(PointCTMatchingContractError, 'L2 norm'):
            compute_cross_modal_similarity(torch.zeros((1, 256)), torch.ones((1, 256)), 0.5)

    def test_non_positive_temperature_fails_closed(self):
        q = torch.ones((1, 256))
        for temperature in (0.0, -0.1):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(PointCTMatchingContractError, 'greater than zero'):
                    compute_cross_modal_similarity(q, q, temperature)

    def test_non_finite_temperature_fails_closed(self):
        q = torch.ones((1, 256))
        for temperature in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(PointCTMatchingContractError, 'finite scalar'):
                    compute_cross_modal_similarity(q, q, temperature)

    def test_half_and_bfloat_inputs_use_float32_numerical_path(self):
        tested_dtypes = 0
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                if dtype == torch.bfloat16:
                    try:
                        torch.isfinite(torch.ones((1,), dtype=dtype))
                    except RuntimeError:
                        continue
                output = self._matcher()(
                    self._descriptors(2, dtype=dtype),
                    self._descriptors(3, offset=33, dtype=dtype),
                )
                tested_dtypes += 1
                self.assertEqual(output['similarity'].dtype, torch.float32)
                self.assertEqual(output['log_assignment'].dtype, torch.float32)
        self.assertGreaterEqual(tested_dtypes, 1)

    def test_cpu_execution_has_no_cuda_dependency(self):
        matcher = self._matcher(device=torch.device('cpu'))
        output = matcher(self._descriptors(2), self._descriptors(3, offset=51))
        self.assertEqual(output['similarity'].device.type, 'cpu')
        self.assertEqual(output['log_assignment'].device.type, 'cpu')
        self.assertEqual(matcher.transport.alpha.device.type, 'cpu')

    @unittest.skipUnless(
        TORCH_AVAILABLE and torch.cuda.is_available(),
        'CUDA is required for the M3-2 CUDA device test.',
    )
    def test_current_cuda_device_is_preserved(self):
        device = torch.device('cuda', torch.cuda.current_device())
        matcher = self._matcher(device=device)
        output = matcher(
            self._descriptors(2, device=device),
            self._descriptors(3, offset=61, device=device),
        )
        self.assertEqual(output['similarity'].device, device)
        self.assertEqual(output['log_assignment'].device, device)
        self.assertEqual(matcher.transport.alpha.device, device)

    def test_forward_is_deterministic_for_identical_input_and_state(self):
        matcher = self._matcher()
        q = self._descriptors(3)
        k = self._descriptors(4, offset=71)
        first = matcher(q, k)
        second = matcher(q, k)
        torch.testing.assert_close(first['similarity'], second['similarity'], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            first['log_assignment'],
            second['log_assignment'],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == '__main__':
    unittest.main()
