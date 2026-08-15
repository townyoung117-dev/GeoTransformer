import ast
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

    from matching_filter import (
        DustbinAwareMatchingFilterError,
        extract_dustbin_aware_mutual_correspondences,
    )
else:
    torch = None


class M3MatchingFilterSourceContractTest(unittest.TestCase):
    def test_source_has_no_supervision_geometry_or_later_stage_inputs(self):
        source_path = EXPERIMENT_DIR / 'matching_filter.py'
        source = source_path.read_text(encoding='utf-8')
        lowered = source.lower()
        prohibited = {
            'gt_primary',
            'gt_high_confidence',
            'gt_transform',
            'physical',
            'xp_phys',
            'xv_phys',
            'svd',
            'procrustes',
            'ransac',
            'localglobalregistration',
            'registration',
            'rigid',
            'loss',
            'backward',
            'optimizer',
            'training',
            'supervision',
            'defect',
            'anatomical',
            'entropy',
            'residual feedback',
            'fusion',
            'random',
        }
        for token in prohibited:
            self.assertNotIn(token, lowered)

        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'extract_dustbin_aware_mutual_correspondences'
        )
        argument_names = [argument.arg.lower() for argument in function.args.args]
        self.assertEqual(
            argument_names,
            ['log_assignment', 'point_valid_mask', 'ct_valid_mask', 'min_confidence'],
        )

    def test_source_does_not_renormalize_or_recompute_assignment(self):
        source = (EXPERIMENT_DIR / 'matching_filter.py').read_text(encoding='utf-8')
        lowered = source.lower()
        for token in ('softmax', 'sinkhorn', 'similarity', 'matmul', 'einsum'):
            self.assertNotIn(token, lowered)

        tree = ast.parse(source)
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id.lower())
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr.lower())
        self.assertNotIn('cuda', called_names)
        self.assertNotIn('cpu', called_names)
        self.assertNotIn('numpy', called_names)
        self.assertNotIn('set', called_names)

    def test_source_has_no_dense_correspondence_construction(self):
        source = (EXPERIMENT_DIR / 'matching_filter.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id.lower())
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr.lower())
        self.assertNotIn('one_hot', called_names)
        self.assertNotIn('scatter', called_names)
        self.assertNotIn('scatter_', called_names)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M3-4 matching-filter tests.')
class M3MatchingFilterTest(unittest.TestCase):
    @staticmethod
    def _log(value):
        return math.log(value)

    @staticmethod
    def _assignment(ordinary, point_dustbin=None, ct_dustbin=None, corner=-9.0):
        ordinary = torch.as_tensor(ordinary, dtype=torch.float32)
        num_point, num_ct = ordinary.shape
        assignment = torch.full(
            (num_point + 1, num_ct + 1),
            -9.0,
            dtype=torch.float32,
            device=ordinary.device,
        )
        assignment[:num_point, :num_ct] = ordinary
        if point_dustbin is not None:
            assignment[:num_point, num_ct] = torch.as_tensor(
                point_dustbin,
                dtype=torch.float32,
                device=ordinary.device,
            )
        if ct_dustbin is not None:
            assignment[num_point, :num_ct] = torch.as_tensor(
                ct_dustbin,
                dtype=torch.float32,
                device=ordinary.device,
            )
        assignment[num_point, num_ct] = corner
        return assignment

    @staticmethod
    def _extract(assignment, point_mask=None, ct_mask=None, threshold=None):
        return extract_dustbin_aware_mutual_correspondences(
            assignment,
            point_valid_mask=point_mask,
            ct_valid_mask=ct_mask,
            min_confidence=threshold,
        )

    def test_basic_mutual_fixture_and_direct_confidence(self):
        # Synthetic selection fixture only; it is not a complete transport-marginal fixture.
        assignment = self._assignment(
            [
                [self._log(0.6), self._log(0.1)],
                [self._log(0.2), self._log(0.7)],
            ],
            point_dustbin=[self._log(0.2), self._log(0.1)],
            ct_dustbin=[self._log(0.2), self._log(0.1)],
        )
        output = self._extract(assignment)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0, 1]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0, 1]))
        torch.testing.assert_close(
            output['log_confidence'],
            assignment[torch.tensor([0, 1]), torch.tensor([0, 1])],
        )
        torch.testing.assert_close(
            output['confidence'],
            torch.tensor([0.6, 0.7], dtype=torch.float32),
        )
        self.assertEqual(output['num_valid_points'], 2)
        self.assertEqual(output['num_valid_ct'], 2)
        self.assertEqual(output['num_point_dustbin_wins'], 0)
        self.assertEqual(output['num_ct_dustbin_wins'], 0)
        self.assertEqual(output['num_point_ordinary_wins'], 2)
        self.assertEqual(output['num_ct_ordinary_wins'], 2)
        self.assertEqual(output['num_mutual_before_confidence'], 2)
        self.assertEqual(output['num_correspondences'], 2)

    def test_non_mutual_pair_is_not_retained(self):
        assignment = self._assignment(
            [[self._log(0.7), self._log(0.1)], [self._log(0.8), self._log(0.1)]],
            point_dustbin=[self._log(0.1), self._log(0.1)],
            ct_dustbin=[self._log(0.1), self._log(0.2)],
        )
        output = self._extract(assignment)
        torch.testing.assert_close(output['point_indices'], torch.tensor([1]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0]))

    def test_point_dustbin_win_prevents_ordinary_match(self):
        assignment = self._assignment(
            [[-0.8]],
            point_dustbin=[-0.2],
            ct_dustbin=[-1.0],
        )
        output = self._extract(assignment)
        self.assertEqual(output['num_point_dustbin_wins'], 1)
        self.assertEqual(output['num_correspondences'], 0)

    def test_ct_dustbin_win_prevents_ordinary_match(self):
        assignment = self._assignment(
            [[-0.2]],
            point_dustbin=[-1.0],
            ct_dustbin=[-0.1],
        )
        output = self._extract(assignment)
        self.assertEqual(output['num_ct_dustbin_wins'], 1)
        self.assertEqual(output['num_correspondences'], 0)

    def test_either_dustbin_direction_rejects_pair(self):
        assignment = self._assignment(
            [[-0.2, -2.0], [-2.0, -0.2]],
            point_dustbin=[-0.1, -1.0],
            ct_dustbin=[-1.0, -0.1],
        )
        output = self._extract(assignment)
        self.assertEqual(output['num_point_dustbin_wins'], 1)
        self.assertEqual(output['num_ct_dustbin_wins'], 1)
        self.assertEqual(output['num_correspondences'], 0)

    def test_confidence_is_exponentiated_selected_assignment(self):
        assignment = self._assignment([[-0.37]], point_dustbin=[-2.0], ct_dustbin=[-2.0])
        output = self._extract(assignment)
        torch.testing.assert_close(output['log_confidence'], assignment[0, 0].reshape(1))
        torch.testing.assert_close(output['confidence'], torch.exp(assignment[0, 0]).reshape(1))

    def test_confidence_below_threshold_is_filtered(self):
        assignment = self._assignment(
            [[self._log(0.4)]],
            point_dustbin=[self._log(0.1)],
            ct_dustbin=[self._log(0.1)],
        )
        output = self._extract(assignment, threshold=0.5)
        self.assertEqual(output['num_mutual_before_confidence'], 1)
        self.assertEqual(output['num_correspondences'], 0)

    def test_confidence_threshold_boundary_is_inclusive(self):
        assignment = self._assignment([[0.0]], point_dustbin=[-1.0], ct_dustbin=[-1.0])
        output = self._extract(assignment, threshold=1.0)
        self.assertEqual(output['num_correspondences'], 1)
        torch.testing.assert_close(output['confidence'], torch.ones((1,), dtype=torch.float32))

    def test_none_threshold_does_not_filter(self):
        assignment = self._assignment([[-20.0]], point_dustbin=[-30.0], ct_dustbin=[-30.0])
        output = self._extract(assignment, threshold=None)
        self.assertEqual(output['num_mutual_before_confidence'], 1)
        self.assertEqual(output['num_correspondences'], 1)

    def test_invalid_confidence_threshold_fails_closed(self):
        assignment = self._assignment([[-0.2]], point_dustbin=[-1.0], ct_dustbin=[-1.0])
        for threshold in (float('nan'), float('inf'), float('-inf'), -0.01, 1.01, True):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'finite scalar'):
                    self._extract(assignment, threshold=threshold)

    def test_invalid_point_does_not_participate_in_either_direction(self):
        assignment = self._assignment(
            [[-0.01], [-0.2]],
            point_dustbin=[-1.0, -1.0],
            ct_dustbin=[-2.0],
        )
        point_mask = torch.tensor([False, True], dtype=torch.bool)
        output = self._extract(assignment, point_mask=point_mask)
        torch.testing.assert_close(output['point_indices'], torch.tensor([1]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0]))
        self.assertEqual(output['num_valid_points'], 1)

    def test_invalid_ct_does_not_participate_in_either_direction(self):
        assignment = self._assignment(
            [[-0.01, -0.2]],
            point_dustbin=[-1.0],
            ct_dustbin=[-2.0, -2.0],
        )
        ct_mask = torch.tensor([False, True], dtype=torch.bool)
        output = self._extract(assignment, ct_mask=ct_mask)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([1]))
        self.assertEqual(output['num_valid_ct'], 1)

    def test_invalid_padding_negative_infinity_is_allowed(self):
        assignment = torch.full((3, 3), float('-inf'), dtype=torch.float32)
        assignment[0, 0] = -0.2
        assignment[0, 2] = -1.0
        assignment[2, 0] = -1.0
        point_mask = torch.tensor([True, False], dtype=torch.bool)
        ct_mask = torch.tensor([True, False], dtype=torch.bool)
        output = self._extract(assignment, point_mask=point_mask, ct_mask=ct_mask)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0]))

    def test_nan_in_valid_candidate_fails_closed(self):
        assignments = (
            self._assignment([[float('nan')]], point_dustbin=[-1.0], ct_dustbin=[-1.0]),
            self._assignment([[-1.0]], point_dustbin=[float('nan')], ct_dustbin=[-1.0]),
            self._assignment([[-1.0]], point_dustbin=[-1.0], ct_dustbin=[float('nan')]),
        )
        for assignment in assignments:
            with self.subTest(assignment=assignment):
                with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'NaN'):
                    self._extract(assignment)

    def test_positive_infinity_in_valid_candidate_fails_closed(self):
        assignments = (
            self._assignment([[float('inf')]], point_dustbin=[-1.0], ct_dustbin=[-1.0]),
            self._assignment([[-1.0]], point_dustbin=[float('inf')], ct_dustbin=[-1.0]),
            self._assignment([[-1.0]], point_dustbin=[-1.0], ct_dustbin=[float('inf')]),
        )
        for assignment in assignments:
            with self.subTest(assignment=assignment):
                with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'positive infinity'):
                    self._extract(assignment)

    def test_point_with_only_negative_infinity_candidates_fails_closed(self):
        assignment = self._assignment(
            [[float('-inf')]],
            point_dustbin=[float('-inf')],
            ct_dustbin=[-1.0],
        )
        with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'Point.*all candidates'):
            self._extract(assignment)

    def test_ct_with_only_negative_infinity_candidates_fails_closed(self):
        assignment = self._assignment(
            [[float('-inf')]],
            point_dustbin=[-1.0],
            ct_dustbin=[float('-inf')],
        )
        with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'CT.*all candidates'):
            self._extract(assignment)

    def test_rectangular_three_by_five(self):
        ordinary = torch.full((3, 5), -8.0, dtype=torch.float32)
        ordinary[0, 0] = -0.1
        ordinary[1, 1] = -0.2
        ordinary[2, 2] = -0.3
        assignment = self._assignment(
            ordinary,
            point_dustbin=[-2.0, -2.0, -2.0],
            ct_dustbin=[-2.0, -2.0, -2.0, -1.0, -1.0],
        )
        output = self._extract(assignment)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0, 1, 2]))

    def test_rectangular_five_by_three(self):
        ordinary = torch.full((5, 3), -8.0, dtype=torch.float32)
        ordinary[0, 0] = -0.1
        ordinary[1, 1] = -0.2
        ordinary[2, 2] = -0.3
        assignment = self._assignment(
            ordinary,
            point_dustbin=[-2.0, -2.0, -2.0, -1.0, -1.0],
            ct_dustbin=[-2.0, -2.0, -2.0],
        )
        output = self._extract(assignment)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0, 1, 2]))

    def test_empty_correspondence_is_valid_sparse_output(self):
        assignment = self._assignment(
            [[-2.0, -3.0], [-3.0, -2.0]],
            point_dustbin=[-1.0, -1.0],
            ct_dustbin=[-1.0, -1.0],
        )
        output = self._extract(assignment)
        self.assertEqual(output['num_correspondences'], 0)
        self.assertEqual(tuple(output['point_indices'].shape), (0,))
        self.assertEqual(tuple(output['ct_indices'].shape), (0,))
        self.assertEqual(tuple(output['confidence'].shape), (0,))

    def test_output_dtype_device_and_sparse_rank(self):
        assignment = self._assignment([[-0.2]], point_dustbin=[-1.0], ct_dustbin=[-1.0])
        output = self._extract(assignment)
        self.assertEqual(output['point_indices'].dtype, torch.long)
        self.assertEqual(output['ct_indices'].dtype, torch.long)
        self.assertEqual(output['log_confidence'].dtype, torch.float32)
        self.assertEqual(output['confidence'].dtype, torch.float32)
        for key in ('point_indices', 'ct_indices', 'log_confidence', 'confidence'):
            self.assertEqual(output[key].device, assignment.device)
            self.assertEqual(output[key].ndim, 1)

    @unittest.skipUnless(
        TORCH_AVAILABLE and torch.cuda.is_available(),
        'CUDA is required for the M3-4 device-preservation test.',
    )
    def test_cuda_device_is_preserved_without_mask_transfer(self):
        device = torch.device('cuda', torch.cuda.current_device())
        assignment = self._assignment([[-0.2]], point_dustbin=[-1.0], ct_dustbin=[-1.0]).to(
            device
        )
        point_mask = torch.ones((1,), dtype=torch.bool, device=device)
        ct_mask = torch.ones((1,), dtype=torch.bool, device=device)
        output = self._extract(assignment, point_mask=point_mask, ct_mask=ct_mask)
        for key in ('point_indices', 'ct_indices', 'log_confidence', 'confidence'):
            self.assertEqual(output[key].device, device)

    def test_correspondences_are_deterministically_sorted_by_point_index(self):
        assignment = self._assignment(
            [[-8.0, -8.0, -0.2], [-8.0, -8.0, -8.0], [-0.1, -8.0, -8.0]],
            point_dustbin=[-2.0, -1.0, -2.0],
            ct_dustbin=[-2.0, -1.0, -2.0],
        )
        output = self._extract(assignment)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0, 2]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([2, 0]))

    def test_wrong_assignment_shape_fails_closed(self):
        for assignment in (
            torch.ones((2,), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((1, 2, 2), dtype=torch.float32),
        ):
            with self.subTest(shape=tuple(assignment.shape)):
                with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'shape'):
                    self._extract(assignment)

    def test_wrong_assignment_type_or_dtype_fails_closed(self):
        with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'torch.Tensor'):
            self._extract([[0.0, -1.0], [-1.0, -2.0]])
        for dtype in (torch.float64, torch.float16, torch.int64):
            with self.subTest(dtype=dtype):
                assignment = torch.ones((2, 2), dtype=dtype)
                with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'float32'):
                    self._extract(assignment)

    def test_mask_type_shape_and_dtype_mismatch_fail_closed(self):
        assignment = self._assignment([[-0.2]], point_dustbin=[-1.0], ct_dustbin=[-1.0])
        cases = (
            ([True], None),
            (torch.tensor([True, False]), None),
            (torch.tensor([1]), None),
            (None, [True]),
            (None, torch.tensor([True, False])),
            (None, torch.tensor([1])),
        )
        for point_mask, ct_mask in cases:
            with self.subTest(point_mask=point_mask, ct_mask=ct_mask):
                with self.assertRaises(DustbinAwareMatchingFilterError):
                    self._extract(assignment, point_mask=point_mask, ct_mask=ct_mask)

    @unittest.skipUnless(
        TORCH_AVAILABLE and torch.cuda.is_available(),
        'CUDA is required for the M3-4 mask-device mismatch test.',
    )
    def test_mask_device_mismatch_fails_closed(self):
        device = torch.device('cuda', torch.cuda.current_device())
        assignment = self._assignment([[-0.2]], point_dustbin=[-1.0], ct_dustbin=[-1.0]).to(
            device
        )
        with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'same device'):
            self._extract(assignment, point_mask=torch.tensor([True], dtype=torch.bool))
        with self.assertRaisesRegex(DustbinAwareMatchingFilterError, 'same device'):
            self._extract(assignment, ct_mask=torch.tensor([True], dtype=torch.bool))

    def test_ties_use_first_index_with_dustbin_last(self):
        # Ordinary index 0 precedes the last-position dustbin in both decision directions.
        assignment = self._assignment([[0.0]], point_dustbin=[0.0], ct_dustbin=[0.0])
        output = self._extract(assignment)
        torch.testing.assert_close(output['point_indices'], torch.tensor([0]))
        torch.testing.assert_close(output['ct_indices'], torch.tensor([0]))

    def test_all_invalid_points_return_empty_result(self):
        assignment = self._assignment([[-0.2]], point_dustbin=[-1.0], ct_dustbin=[-0.1])
        output = self._extract(
            assignment,
            point_mask=torch.tensor([False], dtype=torch.bool),
        )
        self.assertEqual(output['num_valid_points'], 0)
        self.assertEqual(output['num_ct_dustbin_wins'], 1)
        self.assertEqual(output['num_correspondences'], 0)

    def test_all_invalid_ct_return_empty_result(self):
        assignment = self._assignment([[-0.2]], point_dustbin=[-0.1], ct_dustbin=[-1.0])
        output = self._extract(
            assignment,
            ct_mask=torch.tensor([False], dtype=torch.bool),
        )
        self.assertEqual(output['num_valid_ct'], 0)
        self.assertEqual(output['num_point_dustbin_wins'], 1)
        self.assertEqual(output['num_correspondences'], 0)


if __name__ == '__main__':
    unittest.main()
