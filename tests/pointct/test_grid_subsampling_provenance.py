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
        import geotransformer.ext as ext_module

        if not hasattr(ext_module, 'grid_subsampling_with_parent'):
            raise RuntimeError('geotransformer.ext has not been rebuilt with the provenance entry point.')
        from geotransformer.modules.ops.grid_subsample import (
            GridSubsamplingProvenanceError,
            grid_subsample,
            grid_subsample_with_parent,
            validate_grid_subsampling_parent,
        )
    except (ImportError, RuntimeError) as error:
        RUNTIME_READY = False
        RUNTIME_SKIP_REASON = str(error)
    else:
        RUNTIME_READY = True
        RUNTIME_SKIP_REASON = ''


class GridSubsamplingSourceContractTests(unittest.TestCase):
    def test_old_python_wrapper_keeps_two_value_return_contract(self):
        source = (REPO_ROOT / 'geotransformer/modules/ops/grid_subsample.py').read_text(encoding='utf-8')
        module = ast.parse(source)
        function = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'grid_subsample'
        )
        self.assertEqual([argument.arg for argument in function.args.args], ['points', 'lengths', 'voxel_size'])
        returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsInstance(returns[0].value, ast.Tuple)
        self.assertEqual(len(returns[0].value.elts), 2)

    def test_default_precompute_still_calls_only_old_grid_wrapper(self):
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


@unittest.skipUnless(RUNTIME_READY, RUNTIME_SKIP_REASON)
class GridSubsamplingProvenanceRuntimeTests(unittest.TestCase):
    @staticmethod
    def _single_cloud_points():
        return torch.tensor(
            [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [1.25, 0.25, 0.25],
                [1.75, 0.75, 0.75],
                [2.50, 0.50, 0.50],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _assert_parent_centroids(test_case, points, s_points, parent_indices):
        for output_row in range(s_points.shape[0]):
            contributor_rows = torch.nonzero(parent_indices == output_row, as_tuple=False).flatten()
            test_case.assertGreater(contributor_rows.numel(), 0)
            expected = points.index_select(0, contributor_rows).sum(dim=0) / contributor_rows.numel()
            test_case.assertTrue(torch.equal(expected, s_points[output_row]))

    def test_old_wrapper_return_arity_is_unchanged(self):
        points = self._single_cloud_points()
        lengths = torch.tensor([points.shape[0]], dtype=torch.int64)
        result = grid_subsample(points, lengths, 1.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_single_cloud_parent_rows_match_actual_output_centroids(self):
        points = self._single_cloud_points()
        lengths = torch.tensor([points.shape[0]], dtype=torch.int64)

        old_points, old_lengths = grid_subsample(points, lengths, 1.0)
        s_points, s_lengths, parent_indices = grid_subsample_with_parent(points, lengths, 1.0)

        self.assertTrue(torch.equal(s_points, old_points))
        self.assertTrue(torch.equal(s_lengths, old_lengths))
        self.assertEqual(parent_indices.dtype, torch.int64)
        self.assertEqual(tuple(parent_indices.shape), (points.shape[0],))
        self.assertEqual(parent_indices[0].item(), parent_indices[1].item())
        self.assertEqual(parent_indices[2].item(), parent_indices[3].item())
        self.assertNotEqual(parent_indices[0].item(), parent_indices[2].item())
        self._assert_parent_centroids(self, points, s_points, parent_indices)

    def test_stacked_cloud_parents_use_global_output_rows(self):
        cloud0 = torch.tensor(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.25, 0.25, 0.25]],
            dtype=torch.float32,
        )
        cloud1 = torch.tensor(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.25, 0.25, 0.25], [1.75, 0.75, 0.75]],
            dtype=torch.float32,
        )
        points = torch.cat((cloud0, cloud1), dim=0)
        lengths = torch.tensor([cloud0.shape[0], cloud1.shape[0]], dtype=torch.int64)

        s_points, s_lengths, parent_indices = grid_subsample_with_parent(points, lengths, 1.0)

        output0 = int(s_lengths[0].item())
        output1 = int(s_lengths[1].item())
        cloud0_parents = parent_indices[: cloud0.shape[0]]
        cloud1_parents = parent_indices[cloud0.shape[0] :]
        self.assertTrue(bool(torch.all((cloud0_parents >= 0) & (cloud0_parents < output0))))
        self.assertTrue(bool(torch.all((cloud1_parents >= output0) & (cloud1_parents < output0 + output1))))
        self._assert_parent_centroids(self, points, s_points, parent_indices)

    def test_old_and_provenance_coordinates_match_exactly_for_multiple_inputs(self):
        cases = (
            (self._single_cloud_points(), torch.tensor([5], dtype=torch.int64), 1.0),
            (
                torch.tensor(
                    [
                        [-1.75, -0.75, 0.25],
                        [-1.25, -0.25, 0.75],
                        [-0.75, -0.75, 0.25],
                        [0.25, 0.25, 0.25],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([4], dtype=torch.int64),
                1.0,
            ),
            (
                torch.tensor(
                    [
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.75],
                        [1.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [1.25, 0.25, 0.25],
                        [1.75, 0.75, 0.75],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([3, 3], dtype=torch.int64),
                1.0,
            ),
        )

        for points, lengths, voxel_size in cases:
            with self.subTest(lengths=lengths.tolist(), voxel_size=voxel_size):
                old_points, old_lengths = grid_subsample(points, lengths, voxel_size)
                s_points, s_lengths, _ = grid_subsample_with_parent(points, lengths, voxel_size)
                self.assertTrue(torch.equal(s_points, old_points))
                self.assertTrue(torch.equal(s_lengths, old_lengths))

    def test_validator_rejects_invalid_parent_dtype_length_and_range(self):
        points = self._single_cloud_points()
        lengths = torch.tensor([points.shape[0]], dtype=torch.int64)
        s_points, s_lengths, parent_indices = grid_subsample_with_parent(points, lengths, 1.0)

        invalid_parents = (
            parent_indices.to(dtype=torch.float32),
            parent_indices.unsqueeze(0),
            parent_indices[:-1],
            parent_indices.clone().index_fill(0, torch.tensor([0]), -1),
            parent_indices.clone().index_fill(0, torch.tensor([0]), s_points.shape[0]),
        )
        for invalid_parent in invalid_parents:
            with self.subTest(dtype=invalid_parent.dtype, shape=tuple(invalid_parent.shape)):
                with self.assertRaises(GridSubsamplingProvenanceError):
                    validate_grid_subsampling_parent(
                        points,
                        lengths,
                        s_points,
                        s_lengths,
                        invalid_parent,
                    )

    def test_validator_rejects_uncovered_output_and_cross_cloud_assignment(self):
        points = self._single_cloud_points()
        lengths = torch.tensor([points.shape[0]], dtype=torch.int64)
        s_points, s_lengths, parent_indices = grid_subsample_with_parent(points, lengths, 1.0)
        extra_s_points = torch.cat((s_points, torch.zeros((1, 3), dtype=torch.float32)), dim=0)
        extra_s_lengths = s_lengths + 1
        with self.assertRaises(GridSubsamplingProvenanceError):
            validate_grid_subsampling_parent(
                points,
                lengths,
                extra_s_points,
                extra_s_lengths,
                parent_indices,
            )

        stacked_points = torch.tensor(
            [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
            dtype=torch.float32,
        )
        stacked_lengths = torch.tensor([1, 1], dtype=torch.int64)
        stacked_s_points = stacked_points.clone()
        stacked_s_lengths = torch.tensor([1, 1], dtype=torch.int64)
        crossed_parents = torch.tensor([1, 0], dtype=torch.int64)
        with self.assertRaises(GridSubsamplingProvenanceError):
            validate_grid_subsampling_parent(
                stacked_points,
                stacked_lengths,
                stacked_s_points,
                stacked_s_lengths,
                crossed_parents,
            )


if __name__ == '__main__':
    unittest.main()
