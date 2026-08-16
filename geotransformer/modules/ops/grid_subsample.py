import importlib
import math

import torch


ext_module = importlib.import_module('geotransformer.ext')


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths


class GridSubsamplingProvenanceError(RuntimeError):
    pass


_INTEGER_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_grid_subsampling_inputs(points, lengths, voxel_size):
    if not torch.is_tensor(points) or points.ndim != 2 or points.shape[1] != 3:
        raise GridSubsamplingProvenanceError('points must be a torch tensor with shape [N,3].')
    if points.device.type != 'cpu' or points.dtype != torch.float32 or not points.is_contiguous():
        raise GridSubsamplingProvenanceError('points must be a contiguous CPU float32 tensor.')
    if not torch.is_tensor(lengths) or lengths.ndim != 1:
        raise GridSubsamplingProvenanceError('lengths must be a torch tensor with shape [B].')
    if lengths.device.type != 'cpu' or lengths.dtype != torch.int64 or not lengths.is_contiguous():
        raise GridSubsamplingProvenanceError('lengths must be a contiguous CPU int64 tensor.')
    length_values = [int(value) for value in lengths.tolist()]
    if not length_values or any(value <= 0 for value in length_values):
        raise GridSubsamplingProvenanceError('lengths must contain one positive value per stacked cloud.')
    if sum(length_values) != points.shape[0]:
        raise GridSubsamplingProvenanceError(
            f'input lengths sum {sum(length_values)} does not match input point count {points.shape[0]}.'
        )
    try:
        voxel_size_value = float(voxel_size)
    except (TypeError, ValueError) as error:
        raise GridSubsamplingProvenanceError('voxel_size must be finite and positive.') from error
    if not math.isfinite(voxel_size_value) or voxel_size_value <= 0:
        raise GridSubsamplingProvenanceError('voxel_size must be finite and positive.')


def validate_grid_subsampling_parent(points, lengths, s_points, s_lengths, parent_indices):
    """Validate same-pass stacked parent provenance against actual output rows."""
    if not torch.is_tensor(s_points) or s_points.ndim != 2 or s_points.shape[1] != 3:
        raise GridSubsamplingProvenanceError('subsampled points must be a torch tensor with shape [M,3].')
    if s_points.device != points.device or s_points.dtype != points.dtype:
        raise GridSubsamplingProvenanceError('subsampled points must preserve the input device and dtype.')
    if not torch.is_tensor(s_lengths) or s_lengths.ndim != 1:
        raise GridSubsamplingProvenanceError('subsampled lengths must be a torch tensor with shape [B].')
    if s_lengths.device != lengths.device or s_lengths.dtype != lengths.dtype:
        raise GridSubsamplingProvenanceError('subsampled lengths must preserve the input device and dtype.')
    if s_lengths.shape[0] != lengths.shape[0]:
        raise GridSubsamplingProvenanceError('input and output lengths must describe the same cloud count.')

    input_lengths = [int(value) for value in lengths.tolist()]
    output_lengths = [int(value) for value in s_lengths.tolist()]
    input_count = int(points.shape[0])
    output_count = int(s_points.shape[0])
    if not input_lengths or any(value <= 0 for value in input_lengths):
        raise GridSubsamplingProvenanceError('input lengths must contain one positive value per stacked cloud.')
    if sum(input_lengths) != input_count:
        raise GridSubsamplingProvenanceError(
            f'input lengths sum {sum(input_lengths)} does not match input point count {input_count}.'
        )
    if any(value <= 0 for value in output_lengths):
        raise GridSubsamplingProvenanceError('every input cloud must produce at least one output point.')
    if sum(output_lengths) != output_count:
        raise GridSubsamplingProvenanceError(
            f'output lengths sum {sum(output_lengths)} does not match output point count {output_count}.'
        )

    if not torch.is_tensor(parent_indices) or parent_indices.ndim != 1:
        raise GridSubsamplingProvenanceError('parent_indices must be a one-dimensional integer tensor.')
    if parent_indices.dtype not in _INTEGER_DTYPES:
        raise GridSubsamplingProvenanceError('parent_indices must use an integer dtype.')
    if parent_indices.device != points.device:
        raise GridSubsamplingProvenanceError('parent_indices must be on the input point device.')
    if parent_indices.shape[0] != input_count:
        raise GridSubsamplingProvenanceError(
            f'parent_indices length {parent_indices.shape[0]} does not match input point count {input_count}.'
        )
    if bool(torch.any(parent_indices < 0)):
        raise GridSubsamplingProvenanceError('parent_indices contains a negative output row id.')
    if bool(torch.any(parent_indices >= output_count)):
        raise GridSubsamplingProvenanceError('parent_indices contains an out-of-range output row id.')

    contributor_counts = torch.bincount(parent_indices.to(dtype=torch.int64), minlength=output_count)
    if contributor_counts.shape[0] != output_count or bool(torch.any(contributor_counts == 0)):
        raise GridSubsamplingProvenanceError('every output row must have at least one input contributor.')

    input_offset = 0
    output_offset = 0
    for cloud_index, (input_length, output_length) in enumerate(zip(input_lengths, output_lengths)):
        cloud_parents = parent_indices[input_offset : input_offset + input_length]
        in_segment = (cloud_parents >= output_offset) & (cloud_parents < output_offset + output_length)
        if not bool(torch.all(in_segment)):
            raise GridSubsamplingProvenanceError(
                f'parent_indices crosses the stacked output segment for cloud {cloud_index}.'
            )
        input_offset += input_length
        output_offset += output_length


def grid_subsample_with_parent(points, lengths, voxel_size):
    """Grid subsampling with input-row to actual output-row provenance."""
    _validate_grid_subsampling_inputs(points, lengths, voxel_size)
    if not hasattr(ext_module, 'grid_subsampling_with_parent'):
        raise GridSubsamplingProvenanceError(
            'geotransformer.ext does not expose grid_subsampling_with_parent; rebuild the project extension.'
        )
    s_points, s_lengths, parent_indices = ext_module.grid_subsampling_with_parent(
        points,
        lengths,
        voxel_size,
    )
    validate_grid_subsampling_parent(points, lengths, s_points, s_lengths, parent_indices)
    return s_points, s_lengths, parent_indices
