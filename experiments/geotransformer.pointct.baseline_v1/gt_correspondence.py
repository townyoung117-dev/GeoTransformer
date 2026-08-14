"""M2-3 physical-coordinate Point-to-CT coarse GT label contract."""

from typing import Dict

import numpy as np


POINT_TO_CT_DIRECTION = 'Point Cloud -> CT'


class GTCorrespondenceContractError(ValueError):
    pass


def _require_coordinates(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise GTCorrespondenceContractError(
            f'{name} must be a finite numeric array with shape [N,3].'
        ) from error
    if array.ndim != 2 or array.shape[1] != 3:
        raise GTCorrespondenceContractError(f'{name} must have shape [N,3]; got {array.shape}.')
    if array.shape[0] == 0:
        raise GTCorrespondenceContractError(f'{name} must contain at least one coarse token.')
    if not np.issubdtype(array.dtype, np.number):
        raise GTCorrespondenceContractError(f'{name} must contain numeric physical coordinates.')
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise GTCorrespondenceContractError(f'{name} contains NaN or Inf.')
    return array


def _require_rigid_transform(value) -> np.ndarray:
    try:
        transform = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise GTCorrespondenceContractError('gt_transform must be a finite rigid [4,4] matrix.') from error
    if transform.shape != (4, 4) or not np.issubdtype(transform.dtype, np.number):
        raise GTCorrespondenceContractError(
            f'gt_transform must be a numeric matrix with shape [4,4]; got {transform.shape}.'
        )
    transform = transform.astype(np.float64, copy=False)
    if not np.all(np.isfinite(transform)):
        raise GTCorrespondenceContractError('gt_transform contains NaN or Inf.')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8):
        raise GTCorrespondenceContractError('gt_transform must have homogeneous last row [0,0,0,1].')

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-6, atol=1e-6):
        raise GTCorrespondenceContractError('gt_transform rotation must be orthonormal.')
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-6, atol=1e-6):
        raise GTCorrespondenceContractError('gt_transform rotation must have determinant +1.')
    return transform


def _require_threshold(value, name: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise GTCorrespondenceContractError(f'{name} must be finite and non-negative.') from error
    if not np.isfinite(threshold) or threshold < 0.0:
        raise GTCorrespondenceContractError(f'{name} must be finite and non-negative.')
    return threshold


def build_coarse_gt_correspondence(
    Xp_phys_coarse,
    Xv_phys_coarse,
    gt_transform,
    gt_transform_direction,
    primary_max_distance_mm,
    high_confidence_distance_mm,
) -> Dict[str, np.ndarray]:
    """Build one nearest CT support label for each physical Point token.

    Inputs and returned coordinates are in millimetres. Point coordinates are
    transformed into the CT/LPS frame with the row-vector form
    ``Xp_phys_coarse @ R_gt.T + t_gt``. A full squared-distance array is used
    only as a temporary nearest-neighbour calculation; ``np.argmin`` returns
    the first occurrence and therefore the smallest CT support index on exact
    distance ties. No pairwise label array is returned.
    """
    if gt_transform_direction != POINT_TO_CT_DIRECTION:
        raise GTCorrespondenceContractError(
            f'gt_transform_direction must be exactly {POINT_TO_CT_DIRECTION!r}; '
            f'got {gt_transform_direction!r}. The transform is not inverted automatically.'
        )

    point_phys = _require_coordinates(Xp_phys_coarse, 'Xp_phys_coarse')
    ct_phys = _require_coordinates(Xv_phys_coarse, 'Xv_phys_coarse')
    transform = _require_rigid_transform(gt_transform)
    primary_threshold = _require_threshold(primary_max_distance_mm, 'primary_max_distance_mm')
    high_confidence_threshold = _require_threshold(
        high_confidence_distance_mm,
        'high_confidence_distance_mm',
    )
    if high_confidence_threshold > primary_threshold:
        raise GTCorrespondenceContractError(
            'high_confidence_distance_mm must not exceed primary_max_distance_mm.'
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    point_gt_ct_phys = point_phys @ rotation.T + translation
    if not np.all(np.isfinite(point_gt_ct_phys)):
        raise GTCorrespondenceContractError('Transformed Point physical coordinates contain NaN or Inf.')

    try:
        with np.errstate(over='raise', invalid='raise'):
            difference = point_gt_ct_phys[:, np.newaxis, :] - ct_phys[np.newaxis, :, :]
            distance_squared = np.sum(difference * difference, axis=2)
    except FloatingPointError as error:
        raise GTCorrespondenceContractError('Physical nearest-distance calculation overflowed.') from error
    if not np.all(np.isfinite(distance_squared)):
        raise GTCorrespondenceContractError('Physical nearest-distance calculation produced NaN or Inf.')

    primary_index = np.argmin(distance_squared, axis=1).astype(np.int64, copy=False)
    row_index = np.arange(point_phys.shape[0], dtype=np.int64)
    primary_distance = np.sqrt(distance_squared[row_index, primary_index])
    primary_valid = primary_distance <= primary_threshold
    high_confidence = primary_distance <= high_confidence_threshold

    return {
        'Xp_phys_coarse': point_phys,
        'Xp_gt_ct_phys': point_gt_ct_phys,
        'Xv_phys_coarse': ct_phys,
        'gt_primary_ct_index': primary_index,
        'gt_primary_distance_mm': primary_distance,
        'gt_primary_valid': primary_valid,
        'gt_high_confidence': high_confidence,
    }


__all__ = [
    'GTCorrespondenceContractError',
    'POINT_TO_CT_DIRECTION',
    'build_coarse_gt_correspondence',
]
