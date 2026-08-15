import ast
import importlib.util
import math
import re
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

    from registration import (
        PointCTRegistrationContractError,
        estimate_weighted_point_to_ct_transform,
    )
else:
    torch = None


class M3RegistrationSourceContractTest(unittest.TestCase):
    def test_source_has_only_registration_stage_inputs(self):
        source = (EXPERIMENT_DIR / 'registration.py').read_text(encoding='utf-8')
        lowered = source.lower()
        prohibited = {
            'gt_transform',
            'r_gt',
            't_gt',
            'gt_primary',
            'gt_high_confidence',
            'ransac',
            'icp',
            'localglobalregistration',
            'sinkhorn',
            'dustbin',
            'softmax',
            'similarity',
            'matching',
            'loss',
            'optimizer',
            'training',
            'defect',
            'anatomical',
            'entropy',
            'fusion',
            'residual feedback',
        }
        for token in prohibited:
            self.assertNotIn(token, lowered)
        self.assertIsNone(re.search(r'\brre\b', lowered))
        self.assertIsNone(re.search(r'\brte\b', lowered))

        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'estimate_weighted_point_to_ct_transform'
        )
        self.assertEqual(
            [argument.arg for argument in function.args.args],
            ['point_physical', 'ct_physical', 'point_indices', 'ct_indices', 'weights'],
        )

    def test_source_preserves_tensor_computation(self):
        source = (EXPERIMENT_DIR / 'registration.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn('detach', called_attributes)
        self.assertNotIn('cpu', called_attributes)
        self.assertNotIn('numpy', called_attributes)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M3-5 registration tests.')
class M3RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        angle = math.radians(37.0)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        self.rotation = torch.tensor(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.translation = torch.tensor([10.0, -5.0, 20.0], dtype=torch.float32)
        self.targets = self.points @ self.rotation.T + self.translation
        self.indices = torch.arange(self.points.shape[0], dtype=torch.long)
        self.weights = torch.ones(self.points.shape[0], dtype=torch.float32)

    def _estimate(self, points=None, targets=None, point_indices=None, ct_indices=None, weights=None):
        return estimate_weighted_point_to_ct_transform(
            self.points if points is None else points,
            self.targets if targets is None else targets,
            self.indices if point_indices is None else point_indices,
            self.indices if ct_indices is None else ct_indices,
            self.weights if weights is None else weights,
        )

    def assert_successful_transform(self, result, rotation=None, translation=None):
        self.assertTrue(result['success'])
        self.assertIsNone(result['failure_reason'])
        torch.testing.assert_close(
            result['rotation'],
            self.rotation if rotation is None else rotation,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            result['translation'],
            self.translation if translation is None else translation,
            rtol=1e-4,
            atol=1e-4,
        )

    def assert_closed_failure(self, result, reason):
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], reason)
        self.assertIsNone(result['rotation'])
        self.assertIsNone(result['translation'])

    def test_identity_transform(self):
        result = self._estimate(targets=self.points)
        self.assert_successful_transform(
            result,
            rotation=torch.eye(3, dtype=torch.float32),
            translation=torch.zeros(3, dtype=torch.float32),
        )

    def test_nontrivial_point_to_ct_transform(self):
        self.assert_successful_transform(self._estimate())

    def test_row_vector_convention_maps_point_to_ct(self):
        result = self._estimate()
        transformed = self.points @ result['rotation'].T + result['translation']
        torch.testing.assert_close(transformed, self.targets, rtol=1e-4, atol=1e-4)

    def test_positive_nonuniform_weights_recover_exact_transform(self):
        weights = torch.tensor([0.1, 0.5, 2.0, 7.0, 3.0], dtype=torch.float32)
        self.assert_successful_transform(self._estimate(weights=weights))

    def test_common_positive_weight_scale_does_not_change_transform(self):
        weights = torch.tensor([0.1, 0.5, 2.0, 7.0, 3.0], dtype=torch.float32)
        first = self._estimate(weights=weights)
        second = self._estimate(weights=weights * 19.0)
        torch.testing.assert_close(first['rotation'], second['rotation'], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            first['translation'], second['translation'], rtol=1e-4, atol=1e-4
        )

    def test_zero_weight_bad_correspondence_has_no_effect(self):
        bad_target = torch.cat(
            (self.targets, torch.tensor([[900.0, -700.0, 500.0]], dtype=torch.float32))
        )
        points = torch.cat((self.points, torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)))
        indices = torch.arange(6, dtype=torch.long)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        result = self._estimate(
            points=points,
            targets=bad_target,
            point_indices=indices,
            ct_indices=indices,
            weights=weights,
        )
        self.assert_successful_transform(result)

    def test_planar_rank_two_geometry_succeeds(self):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        targets = points @ self.rotation.T + self.translation
        indices = torch.arange(4, dtype=torch.long)
        result = self._estimate(
            points=points,
            targets=targets,
            point_indices=indices,
            ct_indices=indices,
            weights=torch.ones(4, dtype=torch.float32),
        )
        self.assert_successful_transform(result)
        self.assertEqual(result['source_rank'], 2)
        self.assertEqual(result['target_rank'], 2)
        self.assertEqual(result['covariance_rank'], 2)

    def test_collinear_source_geometry_fails_closed(self):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        targets = points @ self.rotation.T + self.translation
        indices = torch.arange(4, dtype=torch.long)
        result = self._estimate(
            points=points,
            targets=targets,
            point_indices=indices,
            ct_indices=indices,
            weights=torch.ones(4, dtype=torch.float32),
        )
        self.assert_closed_failure(result, 'degenerate_source_geometry')
        self.assertEqual(result['source_rank'], 1)

    def test_collinear_target_geometry_fails_closed(self):
        targets = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        indices = torch.arange(4, dtype=torch.long)
        result = self._estimate(
            points=self.points[:4],
            targets=targets,
            point_indices=indices,
            ct_indices=indices,
            weights=torch.ones(4, dtype=torch.float32),
        )
        self.assert_closed_failure(result, 'degenerate_target_geometry')
        self.assertGreaterEqual(result['source_rank'], 2)
        self.assertEqual(result['target_rank'], 1)

    def test_rank_one_covariance_fails_closed(self):
        source = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        indices = torch.arange(4, dtype=torch.long)
        result = self._estimate(
            points=source,
            targets=target,
            point_indices=indices,
            ct_indices=indices,
            weights=torch.ones(4, dtype=torch.float32),
        )
        self.assert_closed_failure(result, 'degenerate_covariance')
        self.assertEqual(result['source_rank'], 2)
        self.assertEqual(result['target_rank'], 2)
        self.assertEqual(result['covariance_rank'], 1)

    def test_reflection_is_corrected_to_proper_rotation(self):
        reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32))
        targets = self.points @ reflection.T + self.translation
        result = self._estimate(targets=targets)
        self.assertTrue(result['success'])
        self.assertGreater(result['det_rotation'], 0.0)
        self.assertAlmostEqual(result['det_rotation'], 1.0, places=4)

    def test_zero_one_and_two_correspondences_fail_closed(self):
        for count in range(3):
            with self.subTest(count=count):
                indices = torch.arange(count, dtype=torch.long)
                result = self._estimate(
                    point_indices=indices,
                    ct_indices=indices,
                    weights=torch.ones(count, dtype=torch.float32),
                )
                self.assert_closed_failure(result, 'insufficient_correspondences')
                self.assertEqual(result['num_correspondences'], count)

    def test_fewer_than_three_positive_weights_fail_closed(self):
        result = self._estimate(weights=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
        self.assert_closed_failure(result, 'insufficient_positive_weights')
        self.assertEqual(result['num_positive_weights'], 2)

    def test_all_zero_weights_fail_closed(self):
        result = self._estimate(weights=torch.zeros(5, dtype=torch.float32))
        self.assert_closed_failure(result, 'insufficient_positive_weights')
        self.assertEqual(result['weight_sum'], 0.0)

    def test_numerically_nonpositive_weight_sum_fails_closed(self):
        epsilon = torch.finfo(torch.float32).eps
        weights = torch.full((5,), epsilon / 8.0, dtype=torch.float32)
        result = self._estimate(weights=weights)
        self.assert_closed_failure(result, 'nonpositive_weight_sum')

    def test_finite_negative_weight_raises_contract_error(self):
        weights = self.weights.clone()
        weights[0] = -0.1
        with self.assertRaisesRegex(PointCTRegistrationContractError, 'nonnegative'):
            self._estimate(weights=weights)

    def test_nonfinite_weights_raise_contract_error(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                weights = self.weights.clone()
                weights[0] = value
                with self.assertRaisesRegex(PointCTRegistrationContractError, 'finite'):
                    self._estimate(weights=weights)

    def test_duplicate_point_indices_fail_closed(self):
        result = self._estimate(point_indices=torch.tensor([0, 0, 1, 2, 3], dtype=torch.long))
        self.assert_closed_failure(result, 'duplicate_point_indices')

    def test_duplicate_ct_indices_fail_closed(self):
        result = self._estimate(ct_indices=torch.tensor([0, 0, 1, 2, 3], dtype=torch.long))
        self.assert_closed_failure(result, 'duplicate_ct_indices')

    def test_out_of_range_indices_raise_contract_error(self):
        cases = (
            (torch.tensor([-1, 0, 1], dtype=torch.long), torch.tensor([0, 1, 2], dtype=torch.long)),
            (torch.tensor([0, 1, 5], dtype=torch.long), torch.tensor([0, 1, 2], dtype=torch.long)),
            (torch.tensor([0, 1, 2], dtype=torch.long), torch.tensor([-1, 0, 1], dtype=torch.long)),
            (torch.tensor([0, 1, 2], dtype=torch.long), torch.tensor([0, 1, 5], dtype=torch.long)),
        )
        for point_indices, ct_indices in cases:
            with self.subTest(point_indices=point_indices, ct_indices=ct_indices):
                with self.assertRaisesRegex(PointCTRegistrationContractError, 'out-of-range'):
                    self._estimate(
                        point_indices=point_indices,
                        ct_indices=ct_indices,
                        weights=torch.ones(3, dtype=torch.float32),
                    )

    def test_coordinate_shape_dtype_and_type_errors_raise(self):
        invalid_coordinates = (
            torch.ones(3, dtype=torch.float32),
            torch.ones((3, 2), dtype=torch.float32),
            torch.ones((0, 3), dtype=torch.float32),
            torch.ones((3, 3), dtype=torch.float64),
            [[0.0, 0.0, 0.0]],
        )
        for value in invalid_coordinates:
            with self.subTest(point_value=value):
                with self.assertRaises(PointCTRegistrationContractError):
                    self._estimate(points=value)
            with self.subTest(ct_value=value):
                with self.assertRaises(PointCTRegistrationContractError):
                    self._estimate(targets=value)

    def test_index_shape_dtype_and_type_errors_raise(self):
        invalid_indices = (
            torch.arange(5, dtype=torch.long).reshape(5, 1),
            torch.arange(5, dtype=torch.int32),
            [0, 1, 2, 3, 4],
        )
        for value in invalid_indices:
            with self.subTest(point_value=value):
                with self.assertRaises(PointCTRegistrationContractError):
                    self._estimate(point_indices=value)
            with self.subTest(ct_value=value):
                with self.assertRaises(PointCTRegistrationContractError):
                    self._estimate(ct_indices=value)

    def test_weight_shape_dtype_type_and_length_errors_raise(self):
        invalid_weights = (
            torch.ones((5, 1), dtype=torch.float32),
            torch.ones(5, dtype=torch.float64),
            [1.0] * 5,
            torch.ones(4, dtype=torch.float32),
        )
        for value in invalid_weights:
            with self.subTest(value=value):
                with self.assertRaises(PointCTRegistrationContractError):
                    self._estimate(weights=value)

    def test_nonfinite_physical_coordinates_raise_contract_error(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                points = self.points.clone()
                points[0, 0] = value
                with self.assertRaisesRegex(PointCTRegistrationContractError, 'finite'):
                    self._estimate(points=points)
                targets = self.targets.clone()
                targets[0, 0] = value
                with self.assertRaisesRegex(PointCTRegistrationContractError, 'finite'):
                    self._estimate(targets=targets)

    def test_correspondence_permutation_invariance(self):
        permutation = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
        weights = torch.tensor([0.1, 0.5, 2.0, 7.0, 3.0], dtype=torch.float32)
        first = self._estimate(weights=weights)
        second = self._estimate(
            point_indices=self.indices[permutation],
            ct_indices=self.indices[permutation],
            weights=weights[permutation],
        )
        torch.testing.assert_close(first['rotation'], second['rotation'], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            first['translation'], second['translation'], rtol=1e-4, atol=1e-4
        )

    def test_output_contract_and_diagnostics(self):
        result = self._estimate()
        self.assertEqual(tuple(result['rotation'].shape), (3, 3))
        self.assertEqual(tuple(result['translation'].shape), (3,))
        self.assertEqual(result['rotation'].dtype, torch.float32)
        self.assertEqual(result['translation'].dtype, torch.float32)
        self.assertEqual(result['rotation'].device, self.points.device)
        self.assertEqual(result['translation'].device, self.points.device)
        self.assertEqual(result['num_correspondences'], 5)
        self.assertEqual(result['num_positive_weights'], 5)
        self.assertEqual(result['weight_sum'], 5.0)
        self.assertGreaterEqual(result['source_rank'], 2)
        self.assertGreaterEqual(result['target_rank'], 2)
        self.assertGreaterEqual(result['covariance_rank'], 2)

    def test_svd_runtime_failure_fails_closed(self):
        with mock.patch('registration.torch.linalg.svd', side_effect=RuntimeError('failed')):
            result = self._estimate()
        self.assert_closed_failure(result, 'svd_failure')

    def test_invalid_svd_rotation_fails_closed(self):
        invalid_right_vectors_h = torch.diag(torch.tensor([2.0, 1.0, 1.0]))
        svd_output = (
            torch.eye(3, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
            invalid_right_vectors_h,
        )
        with mock.patch('registration.torch.linalg.svd', return_value=svd_output):
            result = self._estimate()
        self.assert_closed_failure(result, 'invalid_rotation')

    def test_autograd_is_preserved_on_success(self):
        points = self.points.clone().requires_grad_(True)
        targets = self.targets.clone().requires_grad_(True)
        weights = self.weights.clone().requires_grad_(True)
        result = self._estimate(points=points, targets=targets, weights=weights)
        self.assertTrue(result['rotation'].requires_grad)
        self.assertTrue(result['translation'].requires_grad)

    @unittest.skipUnless(
        TORCH_AVAILABLE and torch.cuda.is_available(),
        'CUDA is required for the M3-5 device-contract tests.',
    )
    def test_cuda_device_preservation_and_device_mismatch(self):
        device = torch.device('cuda', torch.cuda.current_device())
        points = self.points.to(device)
        targets = self.targets.to(device)
        indices = self.indices.to(device)
        weights = self.weights.to(device)
        result = self._estimate(
            points=points,
            targets=targets,
            point_indices=indices,
            ct_indices=indices,
            weights=weights,
        )
        self.assertEqual(result['rotation'].device, device)
        self.assertEqual(result['translation'].device, device)
        mismatch_cases = (
            {'targets': self.targets},
            {'point_indices': self.indices},
            {'ct_indices': self.indices},
            {'weights': self.weights},
        )
        for override in mismatch_cases:
            with self.subTest(override=tuple(override)):
                arguments = {
                    'points': points,
                    'targets': targets,
                    'point_indices': indices,
                    'ct_indices': indices,
                    'weights': weights,
                }
                arguments.update(override)
                with self.assertRaisesRegex(PointCTRegistrationContractError, 'same device'):
                    self._estimate(**arguments)


if __name__ == '__main__':
    unittest.main()
