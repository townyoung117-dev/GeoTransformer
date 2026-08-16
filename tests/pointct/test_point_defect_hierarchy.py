import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


try:
    import torch
except ModuleNotFoundError:
    torch = None
    RUNTIME_READY = False
    RUNTIME_SKIP_REASON = 'PyTorch is not installed in this local environment.'
else:
    try:
        from geotransformer.modules.ops.point_defect_hierarchy import (
            PointDefectHierarchyError,
            aggregate_point_defect_hierarchy,
        )
    except (ImportError, RuntimeError) as error:
        RUNTIME_READY = False
        RUNTIME_SKIP_REASON = str(error)
    else:
        RUNTIME_READY = True
        RUNTIME_SKIP_REASON = ''


class PointDefectHierarchySourceContractTests(unittest.TestCase):
    def test_default_precompute_does_not_call_point_aggregation(self):
        source = (REPO_ROOT / 'geotransformer/utils/data.py').read_text(encoding='utf-8')
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == 'precompute_data_stack_mode'
        )
        called_names = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn('grid_subsample', called_names)
        self.assertNotIn('grid_subsample_with_parent', called_names)
        self.assertNotIn('aggregate_point_defect_hierarchy', called_names)

    def test_aggregation_source_uses_parent_validation_and_integer_sums(self):
        source = (
            REPO_ROOT / 'geotransformer/modules/ops/point_defect_hierarchy.py'
        ).read_text(encoding='utf-8')
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == 'aggregate_point_defect_hierarchy'
        )
        function_source = ast.get_source_segment(source, function)
        self.assertIn('validate_grid_subsampling_parent(', function_source)
        self.assertEqual(function_source.count('.index_add_('), 2)
        self.assertNotIn('grid_subsample(', function_source)
        self.assertNotIn('.mean(', function_source)
        self.assertNotIn('.sort(', function_source)


@unittest.skipUnless(RUNTIME_READY, RUNTIME_SKIP_REASON)
class PointDefectHierarchyRuntimeTests(unittest.TestCase):
    @staticmethod
    def _hierarchy(counts, parents, length_values=None):
        points = [torch.zeros((count, 3), dtype=torch.float32) for count in counts]
        if length_values is None:
            lengths = [torch.tensor([count], dtype=torch.int64) for count in counts]
        else:
            lengths = [torch.tensor(values, dtype=torch.int64) for values in length_values]
        parent_indices = [torch.tensor(parent, dtype=torch.int64) for parent in parents]
        return points, lengths, parent_indices

    @staticmethod
    def _two_output_hierarchy():
        return PointDefectHierarchyRuntimeTests._hierarchy(
            counts=(5, 2, 2, 2),
            parents=(
                [0, 0, 1, 1, 1],
                [0, 1],
                [0, 1],
            ),
        )

    @staticmethod
    def _unequal_child_hierarchy():
        parent0 = [0] * 10 + [1] * 7 + [2] * 2 + [3]
        return PointDefectHierarchyRuntimeTests._hierarchy(
            counts=(20, 4, 2, 1),
            parents=(
                parent0,
                [0, 0, 1, 1],
                [0, 0],
            ),
        )

    def test_single_level_grouping_counts_raw_contributors(self):
        points, lengths, parents = self._two_output_hierarchy()
        raw_intact = torch.tensor([True, False, True, True, False])

        output = aggregate_point_defect_hierarchy(raw_intact, points, lengths, parents)

        self.assertTrue(
            torch.equal(output['point_raw_total_count_coarse'], torch.tensor([2, 3], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(output['point_raw_defect_count_coarse'], torch.tensor([1, 1], dtype=torch.int64))
        )
        self.assertTrue(torch.equal(output['point_intact_coarse'], torch.tensor([False, False])))

    def test_three_level_hierarchy_preserves_unequal_raw_child_counts(self):
        points, lengths, parents = self._unequal_child_hierarchy()
        raw_intact = torch.ones(20, dtype=torch.bool)
        raw_intact[0] = False
        raw_intact[19] = False

        output = aggregate_point_defect_hierarchy(raw_intact, points, lengths, parents)

        self.assertTrue(
            torch.equal(output['point_raw_total_count_coarse'], torch.tensor([20], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(output['point_raw_defect_count_coarse'], torch.tensor([2], dtype=torch.int64))
        )
        self.assertTrue(torch.equal(output['point_intact_coarse'], torch.tensor([False])))

    def test_any_raw_defect_makes_final_output_not_intact(self):
        points, lengths, parents = self._hierarchy(
            counts=(4, 2, 1, 1),
            parents=([0, 0, 1, 1], [0, 0], [0]),
        )
        raw_intact = torch.tensor([True, True, False, True])

        output = aggregate_point_defect_hierarchy(raw_intact, points, lengths, parents)

        self.assertEqual(output['point_raw_defect_count_coarse'].item(), 1)
        self.assertFalse(output['point_intact_coarse'].item())

    def test_all_intact(self):
        points, lengths, parents = self._two_output_hierarchy()
        output = aggregate_point_defect_hierarchy(
            torch.ones(5, dtype=torch.bool),
            points,
            lengths,
            parents,
        )

        self.assertTrue(torch.equal(output['point_raw_total_count_coarse'], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(output['point_raw_defect_count_coarse'], torch.tensor([0, 0])))
        self.assertTrue(torch.equal(output['point_intact_coarse'], torch.tensor([True, True])))

    def test_all_defect(self):
        points, lengths, parents = self._two_output_hierarchy()
        output = aggregate_point_defect_hierarchy(
            torch.zeros(5, dtype=torch.bool),
            points,
            lengths,
            parents,
        )

        self.assertTrue(torch.equal(output['point_raw_total_count_coarse'], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(output['point_raw_defect_count_coarse'], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(output['point_intact_coarse'], torch.tensor([False, False])))

    def test_stacked_two_clouds_remain_in_separate_global_segments(self):
        points, lengths, parents = self._hierarchy(
            counts=(7, 4, 3, 2),
            parents=(
                [0, 0, 1, 1, 2, 3, 3],
                [0, 0, 1, 2],
                [0, 1, 1],
            ),
            length_values=(
                [4, 3],
                [2, 2],
                [1, 2],
                [1, 1],
            ),
        )
        raw_intact = torch.tensor([True, True, True, True, True, False, True])

        output = aggregate_point_defect_hierarchy(raw_intact, points, lengths, parents)

        self.assertTrue(torch.equal(output['point_raw_total_count_coarse'], torch.tensor([4, 3])))
        self.assertTrue(torch.equal(output['point_raw_defect_count_coarse'], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(output['point_intact_coarse'], torch.tensor([True, False])))

    def test_final_counts_conserve_raw_total_and_raw_defect_count(self):
        points, lengths, parents = self._unequal_child_hierarchy()
        raw_intact = torch.tensor(([True, False, True, False, True] * 4), dtype=torch.bool)

        output = aggregate_point_defect_hierarchy(raw_intact, points, lengths, parents)

        self.assertEqual(output['point_raw_total_count_coarse'].sum().item(), raw_intact.numel())
        self.assertEqual(
            output['point_raw_defect_count_coarse'].sum().item(),
            (~raw_intact).sum().item(),
        )

    def test_explicit_binary_uint8_is_accepted_without_semantic_inversion(self):
        points, lengths, parents = self._two_output_hierarchy()
        raw_intact_uint8 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.uint8)

        output = aggregate_point_defect_hierarchy(raw_intact_uint8, points, lengths, parents)

        self.assertTrue(torch.equal(output['point_raw_defect_count_coarse'], torch.tensor([1, 1])))
        self.assertTrue(torch.equal(output['point_intact_coarse'], torch.tensor([False, False])))

    def test_invalid_masks_fail_closed(self):
        points, lengths, parents = self._two_output_hierarchy()
        invalid_masks = (
            torch.ones((1, 5), dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
            torch.tensor([1, 0, 2, 1, 0], dtype=torch.uint8),
            torch.tensor([1.0, 0.0, 1.0, 1.0, 0.5], dtype=torch.float32),
        )

        for invalid_mask in invalid_masks:
            with self.subTest(dtype=invalid_mask.dtype, shape=tuple(invalid_mask.shape)):
                with self.assertRaises(PointDefectHierarchyError):
                    aggregate_point_defect_hierarchy(invalid_mask, points, lengths, parents)

    def test_invalid_parent_length_range_and_coverage_fail_closed(self):
        points, lengths, parents = self._two_output_hierarchy()
        raw_intact = torch.ones(5, dtype=torch.bool)

        invalid_parent_sets = []
        wrong_length = [parent.clone() for parent in parents]
        wrong_length[0] = wrong_length[0][:-1]
        invalid_parent_sets.append((points, lengths, wrong_length))

        negative = [parent.clone() for parent in parents]
        negative[0][0] = -1
        invalid_parent_sets.append((points, lengths, negative))

        out_of_range = [parent.clone() for parent in parents]
        out_of_range[0][0] = points[1].shape[0]
        invalid_parent_sets.append((points, lengths, out_of_range))

        uncovered_points, uncovered_lengths, uncovered_parents = self._hierarchy(
            counts=(5, 3, 2, 2),
            parents=(
                [0, 0, 1, 1, 1],
                [0, 1, 1],
                [0, 1],
            ),
        )
        invalid_parent_sets.append((uncovered_points, uncovered_lengths, uncovered_parents))

        for invalid_points, invalid_lengths, invalid_parents in invalid_parent_sets:
            with self.subTest(parent_shapes=[tuple(parent.shape) for parent in invalid_parents]):
                with self.assertRaises(PointDefectHierarchyError):
                    aggregate_point_defect_hierarchy(
                        raw_intact,
                        invalid_points,
                        invalid_lengths,
                        invalid_parents,
                    )

    def test_stacked_cross_segment_parent_fails_closed(self):
        points, lengths, parents = self._hierarchy(
            counts=(7, 4, 3, 2),
            parents=(
                [2, 0, 1, 1, 0, 3, 3],
                [0, 0, 1, 2],
                [0, 1, 1],
            ),
            length_values=(
                [4, 3],
                [2, 2],
                [1, 2],
                [1, 1],
            ),
        )

        with self.assertRaises(PointDefectHierarchyError):
            aggregate_point_defect_hierarchy(
                torch.ones(7, dtype=torch.bool),
                points,
                lengths,
                parents,
            )


if __name__ == '__main__':
    unittest.main()
