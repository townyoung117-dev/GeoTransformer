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

    from matching_loss import (
        CollisionAwareMatchLossError,
        compute_collision_aware_match_loss,
    )
else:
    torch = None


class M3MatchingLossSourceContractTest(unittest.TestCase):
    def test_source_uses_only_sparse_primary_supervision(self):
        source_path = EXPERIMENT_DIR / 'matching_loss.py'
        source = source_path.read_text(encoding='utf-8')
        lowered = source.lower()
        prohibited = {
            'gt_high_confidence',
            'high_confidence',
            'gt_primary_distance',
            'xp_phys',
            'xv_phys',
            'gt_transform',
            'one_hot',
            'dense label',
            'pairwise gt',
        }
        for token in prohibited:
            self.assertNotIn(token, lowered)

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
        self.assertNotIn('zeros', called_names)
        self.assertNotIn('zeros_like', called_names)
        self.assertNotIn('detach', called_names)
        self.assertNotIn('cpu', called_names)
        self.assertNotIn('numpy', called_names)

    def test_source_stops_at_m3_3_boundary(self):
        source = (EXPERIMENT_DIR / 'matching_loss.py').read_text(encoding='utf-8').lower()
        prohibited = {
            'similarity',
            'sinkhorn',
            'mutual',
            'argmax',
            'procrustes',
            'svd',
            'registration',
            'ransac',
            'optimizer',
            'defect',
            'anatomical',
            'entropy',
            'residual feedback',
            'fusion',
        }
        for token in prohibited:
            self.assertNotIn(token, source)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for M3-3 matching-loss numerical tests.')
class M3MatchingLossTest(unittest.TestCase):
    @staticmethod
    def _assignment(ordinary_scores, fill=-7.0):
        ordinary_scores = torch.as_tensor(ordinary_scores, dtype=torch.float32)
        num_point, num_ct = ordinary_scores.shape
        assignment = torch.full(
            (num_point + 1, num_ct + 1),
            fill,
            dtype=torch.float32,
        )
        assignment[:num_point, :num_ct] = ordinary_scores
        return assignment

    @staticmethod
    def _compute(assignment, primary, valid=None, point_mask=None, ct_mask=None):
        primary = torch.as_tensor(primary, dtype=torch.long)
        if valid is None:
            valid = torch.ones(primary.shape, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid, dtype=torch.bool)
        return compute_collision_aware_match_loss(
            assignment,
            primary,
            valid,
            point_valid_mask=point_mask,
            ct_valid_mask=ct_mask,
        )

    def test_singleton_groups_reduce_to_positive_nll(self):
        assignment = self._assignment([[-0.2, -4.0], [-3.0, -0.7]])
        output = self._compute(assignment, [0, 1])
        expected = torch.stack([-assignment[0, 0], -assignment[1, 1]]).mean()
        torch.testing.assert_close(output['loss'], expected, rtol=0.0, atol=1e-7)

    def test_two_point_collision_uses_negative_log_set_mass(self):
        assignment = self._assignment(
            [[math.log(0.4)], [math.log(0.6)]],
        )
        output = self._compute(assignment, [0, 0])
        expected = -torch.logsumexp(assignment[:2, 0], dim=0)
        naive_point_mean = -assignment[:2, 0].mean()
        torch.testing.assert_close(output['loss'], expected, rtol=0.0, atol=1e-7)
        self.assertFalse(torch.isclose(output['loss'], naive_point_mean))

    def test_collision_set_is_invariant_to_point_score_permutation(self):
        first = self._assignment([[math.log(0.2)], [math.log(0.8)]])
        second = self._assignment([[math.log(0.8)], [math.log(0.2)]])
        first_loss = self._compute(first, [0, 0])['loss']
        second_loss = self._compute(second, [0, 0])['loss']
        torch.testing.assert_close(first_loss, second_loss, rtol=0.0, atol=1e-7)

    def test_equal_group_total_mass_has_equal_set_objective(self):
        # These synthetic log scores isolate the objective; they are not a full marginal fixture.
        first = self._assignment([[math.log(0.4)], [math.log(0.6)]])
        second = self._assignment([[math.log(0.5)], [math.log(0.5)]])
        first_loss = self._compute(first, [0, 0])['loss']
        second_loss = self._compute(second, [0, 0])['loss']
        torch.testing.assert_close(first_loss, second_loss, rtol=0.0, atol=1e-7)

    def test_unique_ct_groups_receive_equal_weight(self):
        assignment = self._assignment(
            [
                [math.log(0.1), -7.0],
                [math.log(0.1), -7.0],
                [math.log(0.1), -7.0],
                [-7.0, math.log(0.8)],
            ]
        )
        output = self._compute(assignment, [0, 0, 0, 1])
        group_a = -torch.logsumexp(assignment[:3, 0], dim=0)
        group_b = -assignment[3, 1]
        expected = torch.stack([group_a, group_b]).mean()
        point_weighted = torch.stack(
            [-assignment[0, 0], -assignment[1, 0], -assignment[2, 0], -assignment[3, 1]]
        ).mean()
        torch.testing.assert_close(output['loss'], expected, rtol=0.0, atol=1e-7)
        self.assertFalse(torch.isclose(output['loss'], point_weighted))

    def test_collision_diagnostics_are_exact(self):
        assignment = self._assignment(torch.full((6, 3), -1.0))
        output = self._compute(assignment, [0, 0, 1, 2, 2, 2])
        self.assertEqual(output['num_supervised_points'], 6)
        self.assertEqual(output['num_supervised_groups'], 3)
        self.assertEqual(output['num_collision_groups'], 2)
        self.assertEqual(output['num_collision_points'], 5)
        self.assertEqual(output['num_excess_collision_points'], 3)
        self.assertEqual(output['max_group_size'], 3)

    def test_gt_primary_invalid_point_is_excluded(self):
        assignment = self._assignment([[-0.2, -3.0], [-0.4, -3.0], [-3.0, -0.8]])
        output = self._compute(assignment, [0, 0, 1], valid=[True, False, True])
        self.assertEqual(output['num_supervised_points'], 2)
        self.assertEqual(output['num_supervised_groups'], 2)
        self.assertEqual(output['num_collision_groups'], 0)

    def test_invalid_point_sentinel_index_is_ignored(self):
        assignment = self._assignment([[-0.2], [-0.5]])
        output = self._compute(assignment, [0, -1], valid=[True, False])
        torch.testing.assert_close(output['loss'], -assignment[0, 0], rtol=0.0, atol=1e-7)
        self.assertEqual(output['num_supervised_points'], 1)

    def test_supervised_out_of_range_ct_target_fails_closed(self):
        assignment = self._assignment([[-0.2, -1.0]])
        for primary in ([-1], [2]):
            with self.subTest(primary=primary):
                with self.assertRaisesRegex(CollisionAwareMatchLossError, 'out-of-range'):
                    self._compute(assignment, primary)

    def test_supervised_structurally_invalid_ct_target_fails_closed(self):
        assignment = self._assignment([[-0.2, -1.0]])
        ct_mask = torch.tensor([False, True], dtype=torch.bool)
        with self.assertRaisesRegex(CollisionAwareMatchLossError, 'structurally invalid CT'):
            self._compute(assignment, [0], ct_mask=ct_mask)

    def test_structurally_invalid_point_is_not_supervised(self):
        assignment = self._assignment([[-5.0], [-0.3]])
        point_mask = torch.tensor([False, True], dtype=torch.bool)
        output = self._compute(assignment, [-999, 0], point_mask=point_mask)
        torch.testing.assert_close(output['loss'], -assignment[1, 0], rtol=0.0, atol=1e-7)
        self.assertEqual(output['num_supervised_points'], 1)

    def test_zero_supervision_fails_closed(self):
        assignment = self._assignment([[-0.2], [-0.3]])
        with self.assertRaisesRegex(CollisionAwareMatchLossError, 'zero valid Points'):
            self._compute(assignment, [-1, -1], valid=[False, False])
        with self.assertRaisesRegex(CollisionAwareMatchLossError, 'zero valid Points'):
            self._compute(
                assignment,
                [0, 0],
                point_mask=torch.tensor([False, False], dtype=torch.bool),
            )

    def test_wrong_log_assignment_shape_fails_closed(self):
        primary = torch.tensor([0], dtype=torch.long)
        valid = torch.tensor([True], dtype=torch.bool)
        for assignment in (
            torch.ones((2,), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((1, 2, 2), dtype=torch.float32),
        ):
            with self.subTest(shape=tuple(assignment.shape)):
                with self.assertRaisesRegex(CollisionAwareMatchLossError, 'shape'):
                    compute_collision_aware_match_loss(assignment, primary, valid)

    def test_gt_and_structural_mask_shape_mismatch_fails_closed(self):
        assignment = self._assignment([[-0.2], [-0.3]])
        cases = (
            (torch.tensor([0]), torch.tensor([True, True]), None, None),
            (torch.tensor([0, 0]), torch.tensor([True]), None, None),
            (
                torch.tensor([0, 0]),
                torch.tensor([True, True]),
                torch.tensor([True]),
                None,
            ),
            (
                torch.tensor([0, 0]),
                torch.tensor([True, True]),
                None,
                torch.tensor([True, True]),
            ),
        )
        for primary, valid, point_mask, ct_mask in cases:
            with self.subTest(
                primary_shape=tuple(primary.shape),
                valid_shape=tuple(valid.shape),
            ):
                with self.assertRaisesRegex(CollisionAwareMatchLossError, 'shape'):
                    compute_collision_aware_match_loss(
                        assignment,
                        primary,
                        valid,
                        point_valid_mask=point_mask,
                        ct_valid_mask=ct_mask,
                    )

    def test_wrong_input_dtypes_fail_closed(self):
        assignment = self._assignment([[-0.2]])
        valid = torch.tensor([True], dtype=torch.bool)
        primary = torch.tensor([0], dtype=torch.long)
        cases = (
            (assignment.to(torch.float64), primary, valid, None, None),
            (assignment, primary.to(torch.float32), valid, None, None),
            (assignment, primary, valid.to(torch.int64), None, None),
            (assignment, primary, valid, torch.tensor([1]), None),
            (assignment, primary, valid, None, torch.tensor([1])),
        )
        for values in cases:
            with self.subTest(dtypes=[value.dtype if value is not None else None for value in values]):
                with self.assertRaisesRegex(CollisionAwareMatchLossError, 'dtype'):
                    compute_collision_aware_match_loss(
                        values[0],
                        values[1],
                        values[2],
                        point_valid_mask=values[3],
                        ct_valid_mask=values[4],
                    )

    def test_selected_nan_fails_closed(self):
        assignment = self._assignment([[float('nan')]])
        with self.assertRaisesRegex(CollisionAwareMatchLossError, 'NaN or Inf'):
            self._compute(assignment, [0])

    def test_selected_positive_or_negative_infinity_fails_closed(self):
        for value in (float('inf'), float('-inf')):
            with self.subTest(value=value):
                assignment = self._assignment([[value]])
                with self.assertRaisesRegex(CollisionAwareMatchLossError, 'NaN or Inf'):
                    self._compute(assignment, [0])

    def test_irrelevant_padding_negative_infinity_is_allowed(self):
        assignment = torch.full((4, 4), float('-inf'), dtype=torch.float32)
        assignment[0, 0] = -0.25
        point_mask = torch.tensor([True, False, False], dtype=torch.bool)
        ct_mask = torch.tensor([True, False, False], dtype=torch.bool)
        output = self._compute(
            assignment,
            [0, -1, -1],
            valid=[True, False, False],
            point_mask=point_mask,
            ct_mask=ct_mask,
        )
        torch.testing.assert_close(output['loss'], -assignment[0, 0], rtol=0.0, atol=1e-7)

    def test_loss_backward_produces_finite_assignment_gradient(self):
        assignment = self._assignment([[-0.2, -2.0], [-0.5, -3.0], [-4.0, -0.7]])
        assignment.requires_grad_()
        output = self._compute(assignment, [0, 0, 1])
        output['loss'].backward()
        self.assertIsNotNone(assignment.grad)
        self.assertTrue(torch.isfinite(assignment.grad).all())

    def test_all_selected_collision_entries_receive_finite_nonzero_gradient(self):
        assignment = self._assignment([[math.log(0.4)], [math.log(0.6)]])
        assignment.requires_grad_()
        output = self._compute(assignment, [0, 0])
        output['loss'].backward()
        selected_gradient = assignment.grad[:2, 0]
        self.assertTrue(torch.isfinite(selected_gradient).all())
        self.assertTrue(torch.all(selected_gradient != 0.0))


if __name__ == '__main__':
    unittest.main()
