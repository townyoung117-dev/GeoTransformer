from typing import Sequence

import torch

from geotransformer.modules.ops.grid_subsample import (
    GridSubsamplingProvenanceError,
    validate_grid_subsampling_parent,
)


class PointDefectHierarchyError(RuntimeError):
    pass


def _require_hierarchy_sequence(name, value, expected_length):
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise PointDefectHierarchyError(
            f'{name} must be a list or tuple containing exactly {expected_length} stages.'
        )


def _validate_raw_intact_mask(point_defect_mask, raw_points):
    if not torch.is_tensor(raw_points) or raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise PointDefectHierarchyError('points[0] must be a torch tensor with shape [Nraw,3].')
    if not torch.is_tensor(point_defect_mask) or point_defect_mask.ndim != 1:
        raise PointDefectHierarchyError('point_defect_mask must be a one-dimensional torch tensor.')
    if point_defect_mask.shape[0] != raw_points.shape[0]:
        raise PointDefectHierarchyError(
            f'point_defect_mask length {point_defect_mask.shape[0]} does not match '
            f'raw point count {raw_points.shape[0]}.'
        )
    if point_defect_mask.device != raw_points.device:
        raise PointDefectHierarchyError('point_defect_mask and points[0] must be on the same device.')

    if point_defect_mask.dtype == torch.bool:
        return point_defect_mask
    if point_defect_mask.dtype == torch.uint8:
        binary = (point_defect_mask == 0) | (point_defect_mask == 1)
        if not bool(torch.all(binary)):
            raise PointDefectHierarchyError('uint8 point_defect_mask may contain only binary values {0,1}.')
        return point_defect_mask.to(dtype=torch.bool)
    raise PointDefectHierarchyError(
        'point_defect_mask must use torch.bool or explicitly binary torch.uint8; probabilities are forbidden.'
    )


def aggregate_point_defect_hierarchy(
    point_defect_mask,
    points: Sequence,
    lengths: Sequence,
    parent_indices: Sequence,
):
    """Propagate raw Point intact state and exact raw contributor counts through three parent maps."""
    _require_hierarchy_sequence('points', points, 4)
    _require_hierarchy_sequence('lengths', lengths, 4)
    _require_hierarchy_sequence('parent_indices', parent_indices, 3)

    raw_intact = _validate_raw_intact_mask(point_defect_mask, points[0])
    raw_count = int(points[0].shape[0])
    stage_total_count = torch.ones(raw_count, dtype=torch.int64, device=raw_intact.device)
    stage_defect_count = (~raw_intact).to(dtype=torch.int64)
    raw_defect_count = int(stage_defect_count.sum().item())

    for stage_index in range(3):
        try:
            validate_grid_subsampling_parent(
                points[stage_index],
                lengths[stage_index],
                points[stage_index + 1],
                lengths[stage_index + 1],
                parent_indices[stage_index],
            )
        except GridSubsamplingProvenanceError as error:
            raise PointDefectHierarchyError(
                f'parent provenance is invalid at hierarchy stage {stage_index}: {error}'
            ) from error

        input_count = int(points[stage_index].shape[0])
        output_count = int(points[stage_index + 1].shape[0])
        if stage_total_count.shape != (input_count,) or stage_defect_count.shape != (input_count,):
            raise PointDefectHierarchyError(
                f'raw contributor count shape is inconsistent at hierarchy stage {stage_index}.'
            )

        parent_i64 = parent_indices[stage_index].to(dtype=torch.int64)
        next_total_count = torch.zeros(output_count, dtype=torch.int64, device=stage_total_count.device)
        next_defect_count = torch.zeros(output_count, dtype=torch.int64, device=stage_defect_count.device)
        next_total_count.index_add_(0, parent_i64, stage_total_count)
        next_defect_count.index_add_(0, parent_i64, stage_defect_count)

        if bool(torch.any(next_total_count <= 0)):
            raise PointDefectHierarchyError(
                f'every output row must retain at least one raw contributor at hierarchy stage {stage_index}.'
            )
        if bool(torch.any(next_defect_count < 0)) or bool(torch.any(next_defect_count > next_total_count)):
            raise PointDefectHierarchyError(
                f'raw defect counts are outside total contributor counts at hierarchy stage {stage_index}.'
            )
        if int(next_total_count.sum().item()) != raw_count:
            raise PointDefectHierarchyError(
                f'raw total contributor conservation failed at hierarchy stage {stage_index}.'
            )
        if int(next_defect_count.sum().item()) != raw_defect_count:
            raise PointDefectHierarchyError(
                f'raw defect contributor conservation failed at hierarchy stage {stage_index}.'
            )

        stage_total_count = next_total_count
        stage_defect_count = next_defect_count

    point_intact_coarse = stage_defect_count == 0
    coarse_count = int(points[3].shape[0])
    if point_intact_coarse.shape != (coarse_count,):
        raise PointDefectHierarchyError('point_intact_coarse shape does not match points[3].')
    if stage_total_count.shape != (coarse_count,) or stage_defect_count.shape != (coarse_count,):
        raise PointDefectHierarchyError('coarse raw contributor count shapes do not match points[3].')

    return {
        'point_intact_coarse': point_intact_coarse,
        'point_raw_total_count_coarse': stage_total_count,
        'point_raw_defect_count_coarse': stage_defect_count,
    }
