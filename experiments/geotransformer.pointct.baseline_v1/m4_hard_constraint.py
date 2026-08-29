"""M4-1 defect-aware hard constraints for coarse matching and supervision.

The module is intentionally independent from the Point/CT encoders and the
matcher.  Encoder descriptors are never changed here: the only matching
intervention is the pair of structural masks returned by
``resolve_m4_matching_masks``.
"""

from collections.abc import Mapping
from typing import Dict

import numpy as np
import torch

from gt_correspondence import build_coarse_gt_correspondence


class M4HardConstraintError(RuntimeError):
    """Raised when M4 hard-constraint artifacts violate their contract."""


_POINT_MAPPING_FIELDS = (
    'point_intact_coarse',
    'point_raw_total_count_coarse',
    'point_raw_defect_count_coarse',
)
_CT_MAPPING_FIELDS = (
    'ct_intact_coarse',
    'ct_raw_total_count_coarse',
    'ct_raw_defect_count_coarse',
)


def _require_device(device):
    try:
        return torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise M4HardConstraintError('device must identify a valid torch device.') from error


def _require_encoder_output(value, name):
    if not isinstance(value, Mapping):
        raise M4HardConstraintError(f'{name} must be a mapping.')
    return value


def _require_descriptor(output, field, branch_name, device):
    if field not in output:
        raise M4HardConstraintError(f'{branch_name} encoder output is missing {field}.')
    descriptor = output[field]
    if not torch.is_tensor(descriptor):
        raise M4HardConstraintError(f'{field} must be a torch.Tensor.')
    if descriptor.ndim != 2 or descriptor.shape[0] == 0:
        raise M4HardConstraintError(f'{field} must be a non-empty descriptor matrix.')
    if descriptor.device != device:
        raise M4HardConstraintError(f'{field} must be on the requested device {device}.')
    return descriptor


def _present_fields(output, fields):
    return tuple(name for name in fields if name in output)


def _validate_branch_mapping(
    output,
    *,
    branch_name,
    prefix,
    fields,
    token_count,
    device,
):
    missing = [name for name in fields if name not in output]
    if missing:
        raise M4HardConstraintError(
            f'{branch_name} M4 coarse mapping is incomplete; missing fields: {missing}.'
        )

    intact_name, total_name, defect_name = fields
    intact = output[intact_name]
    total = output[total_name]
    defect = output[defect_name]
    for field, tensor, dtype in (
        (intact_name, intact, torch.bool),
        (total_name, total, torch.int64),
        (defect_name, defect, torch.int64),
    ):
        if not torch.is_tensor(tensor):
            raise M4HardConstraintError(f'{field} must be a torch.Tensor.')
        if tensor.dtype != dtype:
            raise M4HardConstraintError(f'{field} must use dtype {dtype}.')
        if tuple(tensor.shape) != (token_count,):
            raise M4HardConstraintError(
                f'{field} must have exact token shape ({token_count},); '
                f'got {tuple(tensor.shape)}.'
            )
        if tensor.device != device:
            raise M4HardConstraintError(
                f'{field} must be on the requested token device {device}.'
            )

    if bool(torch.any(total <= 0)):
        raise M4HardConstraintError(f'{branch_name} raw total counts must all be positive.')
    if bool(torch.any(defect < 0)) or bool(torch.any(defect > total)):
        raise M4HardConstraintError(
            f'{branch_name} raw defect counts must lie in [0, raw total count].'
        )
    if not torch.equal(intact, defect == 0):
        raise M4HardConstraintError(
            f'{intact_name} must equal ({defect_name} == 0).'
        )

    intact_count = int(torch.count_nonzero(intact).item())
    if intact_count == 0:
        raise M4HardConstraintError(
            f'M4 hard constraint requires at least one intact {branch_name} token.'
        )
    return {
        f'{prefix}_valid_mask': intact,
        f'{prefix}_total_count': token_count,
        f'{prefix}_intact_count': intact_count,
        f'{prefix}_excluded_count': token_count - intact_count,
    }


def resolve_m4_matching_masks(point_encoder_output, ct_encoder_output, device):
    """Resolve strict coarse matching masks from Point and CT encoder outputs.

    The only legacy fallback is the complete absence of all six M4 mapping
    artifacts.  Any partial or single-branch presence raises instead of
    silently reverting to the frozen all-valid M3 behavior.
    """
    point_output = _require_encoder_output(point_encoder_output, 'point_encoder_output')
    ct_output = _require_encoder_output(ct_encoder_output, 'ct_encoder_output')
    device = _require_device(device)
    q = _require_descriptor(point_output, 'Q', 'Point', device)
    k = _require_descriptor(ct_output, 'K', 'CT', device)
    point_count = int(q.shape[0])
    ct_count = int(k.shape[0])

    point_present = _present_fields(point_output, _POINT_MAPPING_FIELDS)
    ct_present = _present_fields(ct_output, _CT_MAPPING_FIELDS)
    if not point_present and not ct_present:
        return {
            'enabled': False,
            'point_valid_mask': torch.ones(
                (point_count,),
                dtype=torch.bool,
                device=device,
            ),
            'ct_valid_mask': torch.ones(
                (ct_count,),
                dtype=torch.bool,
                device=device,
            ),
            'point_total_count': point_count,
            'point_intact_count': point_count,
            'point_excluded_count': 0,
            'ct_total_count': ct_count,
            'ct_intact_count': ct_count,
            'ct_excluded_count': 0,
        }

    if not point_present or not ct_present:
        present_branch = 'Point' if point_present else 'CT'
        missing_branch = 'CT' if point_present else 'Point'
        raise M4HardConstraintError(
            f'{present_branch}-only M4 coarse mapping is forbidden; '
            f'{missing_branch} mapping artifacts are absent.'
        )

    point_result = _validate_branch_mapping(
        point_output,
        branch_name='Point',
        prefix='point',
        fields=_POINT_MAPPING_FIELDS,
        token_count=point_count,
        device=device,
    )
    ct_result = _validate_branch_mapping(
        ct_output,
        branch_name='CT',
        prefix='ct',
        fields=_CT_MAPPING_FIELDS,
        token_count=ct_count,
        device=device,
    )
    return {
        'enabled': True,
        **point_result,
        **ct_result,
    }


def _coordinates_as_numpy(value, name):
    original_device = None
    if torch.is_tensor(value):
        original_device = value.device
        if value.device.type == 'meta':
            raise M4HardConstraintError(f'{name} cannot be a meta tensor.')
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise M4HardConstraintError(
            f'{name} must be a finite numeric array with shape [N,3].'
        ) from error
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise M4HardConstraintError(
            f'{name} must be a non-empty array with shape [N,3]; got {array.shape}.'
        )
    if not np.issubdtype(array.dtype, np.number):
        raise M4HardConstraintError(f'{name} must contain numeric coordinates.')
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise M4HardConstraintError(f'{name} contains NaN or Inf.')
    return array, original_device


def _bool_mask_as_numpy(value, length, name, coordinate_device):
    if torch.is_tensor(value):
        if value.dtype != torch.bool:
            raise M4HardConstraintError(f'{name} must use bool dtype.')
        if coordinate_device is not None and value.device != coordinate_device:
            raise M4HardConstraintError(
                f'{name} must share the corresponding coordinate tensor device.'
            )
        if value.device.type == 'meta':
            raise M4HardConstraintError(f'{name} cannot be a meta tensor.')
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise M4HardConstraintError(f'{name} must be a bool vector.') from error
        if array.dtype != np.bool_:
            raise M4HardConstraintError(f'{name} must use bool dtype.')
    if array.shape != (length,):
        raise M4HardConstraintError(
            f'{name} must have exact token shape ({length},); got {array.shape}.'
        )
    return array


def _require_subset_correspondence(correspondence, point_count, ct_count):
    if not isinstance(correspondence, Mapping):
        raise M4HardConstraintError('Subset GT correspondence output must be a mapping.')
    required = (
        'Xp_gt_ct_phys',
        'gt_primary_ct_index',
        'gt_primary_distance_mm',
        'gt_primary_valid',
        'gt_high_confidence',
    )
    missing = [name for name in required if name not in correspondence]
    if missing:
        raise M4HardConstraintError(
            f'Subset GT correspondence output is missing fields: {missing}.'
        )

    transformed = np.asarray(correspondence['Xp_gt_ct_phys'])
    target = np.asarray(correspondence['gt_primary_ct_index'])
    distance = np.asarray(correspondence['gt_primary_distance_mm'])
    primary_valid = np.asarray(correspondence['gt_primary_valid'])
    high_confidence = np.asarray(correspondence['gt_high_confidence'])
    if transformed.shape != (point_count, 3) or not np.issubdtype(transformed.dtype, np.number):
        raise M4HardConstraintError('Subset Xp_gt_ct_phys has an invalid shape or dtype.')
    if not np.all(np.isfinite(transformed)):
        raise M4HardConstraintError('Subset Xp_gt_ct_phys contains NaN or Inf.')
    if target.shape != (point_count,) or not np.issubdtype(target.dtype, np.integer):
        raise M4HardConstraintError('Subset gt_primary_ct_index must be an integer vector.')
    if np.any(target < 0) or np.any(target >= ct_count):
        raise M4HardConstraintError('Subset GT contains an out-of-range intact CT target.')
    if distance.shape != (point_count,) or not np.issubdtype(distance.dtype, np.number):
        raise M4HardConstraintError('Subset gt_primary_distance_mm must be a numeric vector.')
    if not np.all(np.isfinite(distance)):
        raise M4HardConstraintError('Subset gt_primary_distance_mm contains NaN or Inf.')
    for name, array in (
        ('gt_primary_valid', primary_valid),
        ('gt_high_confidence', high_confidence),
    ):
        if array.shape != (point_count,) or array.dtype != np.bool_:
            raise M4HardConstraintError(f'Subset {name} must be a bool vector.')
    return (
        transformed.astype(np.float64, copy=False),
        target.astype(np.int64, copy=False),
        distance.astype(np.float64, copy=False),
        primary_valid,
        high_confidence,
    )


def build_m4_mask_aware_gt(
    Xp_phys_coarse,
    Xv_phys_coarse,
    gt_transform,
    gt_transform_direction,
    primary_max_distance_mm,
    high_confidence_distance_mm,
    point_valid_mask,
    ct_valid_mask,
) -> Dict[str, np.ndarray]:
    """Build full-length GT labels using only intact Point and CT subsets.

    Excluded Point rows are deliberately unsupervised.  Their target index is
    the sentinel ``-1`` and their distance is ``inf``; neither value is consumed
    when ``gt_primary_valid`` is respected.  Intact-subset CT targets are
    remapped to indices in the original full CT token order.
    """
    point_phys, point_coordinate_device = _coordinates_as_numpy(
        Xp_phys_coarse,
        'Xp_phys_coarse',
    )
    ct_phys, ct_coordinate_device = _coordinates_as_numpy(
        Xv_phys_coarse,
        'Xv_phys_coarse',
    )
    point_mask = _bool_mask_as_numpy(
        point_valid_mask,
        point_phys.shape[0],
        'point_valid_mask',
        point_coordinate_device,
    )
    ct_mask = _bool_mask_as_numpy(
        ct_valid_mask,
        ct_phys.shape[0],
        'ct_valid_mask',
        ct_coordinate_device,
    )
    point_indices = np.flatnonzero(point_mask).astype(np.int64, copy=False)
    ct_indices = np.flatnonzero(ct_mask).astype(np.int64, copy=False)
    if point_indices.size == 0:
        raise M4HardConstraintError(
            'Mask-aware GT requires at least one intact Point token.'
        )
    if ct_indices.size == 0:
        raise M4HardConstraintError(
            'Mask-aware GT requires at least one intact CT token.'
        )

    subset = build_coarse_gt_correspondence(
        Xp_phys_coarse=point_phys[point_indices],
        Xv_phys_coarse=ct_phys[ct_indices],
        gt_transform=gt_transform,
        gt_transform_direction=gt_transform_direction,
        primary_max_distance_mm=primary_max_distance_mm,
        high_confidence_distance_mm=high_confidence_distance_mm,
    )
    (
        subset_transformed,
        subset_target,
        subset_distance,
        subset_primary_valid,
        subset_high_confidence,
    ) = _require_subset_correspondence(
        subset,
        point_indices.size,
        ct_indices.size,
    )

    full_target = np.full((point_phys.shape[0],), -1, dtype=np.int64)
    full_distance = np.full((point_phys.shape[0],), np.inf, dtype=np.float64)
    full_primary_valid = np.zeros((point_phys.shape[0],), dtype=np.bool_)
    full_high_confidence = np.zeros((point_phys.shape[0],), dtype=np.bool_)
    full_transformed = np.empty_like(point_phys, dtype=np.float64)

    try:
        transform = np.asarray(gt_transform, dtype=np.float64)
        full_transformed[:] = point_phys @ transform[:3, :3].T + transform[:3, 3]
    except (TypeError, ValueError, IndexError) as error:
        raise M4HardConstraintError('gt_transform could not be applied to full Point coordinates.') from error
    if not np.all(np.isfinite(full_transformed)):
        raise M4HardConstraintError('Full transformed Point coordinates contain NaN or Inf.')
    full_transformed[point_indices] = subset_transformed

    remapped_target = ct_indices[subset_target]
    full_target[point_indices] = remapped_target
    full_distance[point_indices] = subset_distance
    full_primary_valid[point_indices] = subset_primary_valid
    full_high_confidence[point_indices] = subset_high_confidence

    supervised_targets = full_target[full_primary_valid]
    if np.any(supervised_targets < 0) or not np.all(ct_mask[supervised_targets]):
        raise M4HardConstraintError(
            'A supervised Point targets a defect or out-of-range CT token.'
        )
    if np.any(full_primary_valid[~point_mask]) or np.any(full_high_confidence[~point_mask]):
        raise M4HardConstraintError('An excluded Point received GT supervision.')

    return {
        'Xp_phys_coarse': point_phys,
        'Xp_gt_ct_phys': full_transformed,
        'Xv_phys_coarse': ct_phys,
        'gt_primary_ct_index': full_target,
        'gt_primary_distance_mm': full_distance,
        'gt_primary_valid': full_primary_valid,
        'gt_high_confidence': full_high_confidence,
    }


__all__ = [
    'M4HardConstraintError',
    'build_m4_mask_aware_gt',
    'resolve_m4_matching_masks',
]
