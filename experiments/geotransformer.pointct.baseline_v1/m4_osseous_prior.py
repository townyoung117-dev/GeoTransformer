"""Input-only CT-derived osseous support score computation.

This module is deliberately independent of training, matching, registration,
and defect annotations.  It samples original HU values in a physical-mm sphere
around each supplied CT support location.
"""

import math
from typing import Dict, Sequence, Tuple

import numpy as np


class OsseousPriorContractError(RuntimeError):
    """Raised when an osseous-score input violates the fail-closed contract."""


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise OsseousPriorContractError(f'{name} must be a finite scalar.')
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise OsseousPriorContractError(f'{name} must be a finite scalar.') from error
    if not math.isfinite(scalar):
        raise OsseousPriorContractError(f'{name} must be a finite scalar.')
    return scalar


def _finite_vector(value, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise OsseousPriorContractError(
            f'{name} must be a finite vector with shape (3,).'
        ) from error
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise OsseousPriorContractError(
            f'{name} must be a finite vector with shape (3,).'
        )
    if positive and np.any(array <= 0.0):
        raise OsseousPriorContractError(f'{name} must contain positive values.')
    return array


def _direction_matrix(value) -> np.ndarray:
    try:
        direction = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise OsseousPriorContractError(
            'ct_direction must be a finite orthonormal matrix with shape (3,3).'
        ) from error
    if direction.shape == (9,):
        direction = direction.reshape(3, 3)
    if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
        raise OsseousPriorContractError(
            'ct_direction must be a finite orthonormal matrix with shape (3,3).'
        )
    gram = direction.T @ direction
    determinant = float(np.linalg.det(direction))
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-6) or not np.isclose(
        abs(determinant), 1.0, rtol=0.0, atol=1e-6
    ):
        raise OsseousPriorContractError(
            'ct_direction must be orthonormal with determinant magnitude one.'
        )
    return direction


def _validate_inputs(
    ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
):
    volume = np.asarray(ct_volume)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise OsseousPriorContractError(
            'ct_volume must be a non-empty 3D array in [z,y,x] order.'
        )
    if isinstance(volume, np.ma.MaskedArray) or volume.dtype == np.dtype(bool):
        raise OsseousPriorContractError('ct_volume must contain real numeric HU values.')
    if not (
        np.issubdtype(volume.dtype, np.integer)
        or np.issubdtype(volume.dtype, np.floating)
    ):
        raise OsseousPriorContractError('ct_volume must contain real numeric HU values.')
    if np.issubdtype(volume.dtype, np.floating) and not np.all(np.isfinite(volume)):
        raise OsseousPriorContractError('ct_volume contains NaN or Inf HU values.')

    spacing = _finite_vector(ct_spacing, 'ct_spacing', positive=True)
    origin = _finite_vector(ct_origin, 'ct_origin')
    direction = _direction_matrix(ct_direction)
    try:
        support = np.asarray(support_locations_mm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise OsseousPriorContractError(
            'support_locations_mm must have finite non-empty shape [N,3].'
        ) from error
    if (
        support.ndim != 2
        or support.shape[0] == 0
        or support.shape[1] != 3
        or not np.all(np.isfinite(support))
    ):
        raise OsseousPriorContractError(
            'support_locations_mm must have finite non-empty shape [N,3].'
        )
    return volume, spacing, origin, direction, support


def _parameter_grid(values: Sequence, name: str, *, positive: bool) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise OsseousPriorContractError(f'{name} must be a non-empty sequence.')
    try:
        scalars = tuple(_finite_scalar(value, name) for value in values)
    except TypeError as error:
        raise OsseousPriorContractError(f'{name} must be a non-empty sequence.') from error
    if not scalars:
        raise OsseousPriorContractError(f'{name} must be a non-empty sequence.')
    if positive and any(value <= 0.0 for value in scalars):
        raise OsseousPriorContractError(f'{name} values must be greater than zero.')
    if len(set(scalars)) != len(scalars):
        raise OsseousPriorContractError(f'{name} values must be unique.')
    return scalars


def compute_osseous_support_score_grid(
    ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
    *,
    hu_thresholds: Sequence,
    radius_values_mm: Sequence,
) -> Dict[Tuple[float, float], np.ndarray]:
    """Compute HU fractions for a small declared threshold/radius grid.

    For support token ``j``, threshold ``h``, and radius ``r``::

        score[j] = count(||x_voxel - x_j||_2 <= r and HU_voxel >= h)
                   / count(||x_voxel - x_j||_2 <= r)

    Only in-volume voxel centres contribute to the denominator.  NumPy storage
    is [z,y,x], while spacing, origin, direction, and support locations use
    image/physical [x,y,z] semantics.
    """
    volume, spacing, origin, direction, support = _validate_inputs(
        ct_volume,
        ct_spacing,
        ct_origin,
        ct_direction,
        support_locations_mm,
    )
    thresholds = _parameter_grid(hu_thresholds, 'hu_thresholds', positive=False)
    radii = _parameter_grid(radius_values_mm, 'radius_values_mm', positive=True)

    image_affine = direction @ np.diag(spacing)
    try:
        support_index_xyz = np.linalg.solve(
            image_affine,
            (support - origin[np.newaxis, :]).T,
        ).T
    except np.linalg.LinAlgError as error:
        raise OsseousPriorContractError('CT image geometry is not invertible.') from error
    if not np.all(np.isfinite(support_index_xyz)):
        raise OsseousPriorContractError('Support-to-image coordinate conversion is not finite.')

    scores = {
        (threshold, radius): np.empty(support.shape[0], dtype=np.float64)
        for radius in radii
        for threshold in thresholds
    }
    max_radius = max(radii)
    max_index_delta = max_radius / spacing
    size_xyz = np.asarray(volume.shape[::-1], dtype=np.int64)
    distance_tolerance = max(1.0, max_radius * max_radius) * 1e-12

    for row, centre_xyz in enumerate(support_index_xyz):
        lower_xyz = np.floor(centre_xyz - max_index_delta).astype(np.int64)
        upper_xyz = np.ceil(centre_xyz + max_index_delta).astype(np.int64)
        lower_xyz = np.maximum(lower_xyz, 0)
        upper_xyz = np.minimum(upper_xyz, size_xyz - 1)
        if np.any(lower_xyz > upper_xyz):
            raise OsseousPriorContractError(
                f'Support token {row} has an empty physical neighborhood.'
            )

        x = np.arange(lower_xyz[0], upper_xyz[0] + 1, dtype=np.float64)
        y = np.arange(lower_xyz[1], upper_xyz[1] + 1, dtype=np.float64)
        z = np.arange(lower_xyz[2], upper_xyz[2] + 1, dtype=np.float64)
        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')
        delta_index_xyz = np.stack(
            (
                grid_x - centre_xyz[0],
                grid_y - centre_xyz[1],
                grid_z - centre_xyz[2],
            ),
            axis=-1,
        )
        delta_physical = delta_index_xyz @ image_affine.T
        distance_squared = np.einsum(
            '...i,...i->...',
            delta_physical,
            delta_physical,
        )
        local_hu = volume[
            lower_xyz[2] : upper_xyz[2] + 1,
            lower_xyz[1] : upper_xyz[1] + 1,
            lower_xyz[0] : upper_xyz[0] + 1,
        ]
        if local_hu.shape != distance_squared.shape:
            raise OsseousPriorContractError('Internal CT neighborhood shape mismatch.')

        for radius in radii:
            in_sphere = distance_squared <= radius * radius + distance_tolerance
            denominator = int(np.count_nonzero(in_sphere))
            if denominator <= 0:
                raise OsseousPriorContractError(
                    f'Support token {row} has an empty physical neighborhood at radius {radius} mm.'
                )
            sphere_hu = local_hu[in_sphere]
            for threshold in thresholds:
                numerator = int(np.count_nonzero(sphere_hu >= threshold))
                scores[(threshold, radius)][row] = numerator / denominator

    for key, score in scores.items():
        if score.shape != (support.shape[0],) or not np.all(np.isfinite(score)):
            raise OsseousPriorContractError(f'Osseous score {key} violates its numeric contract.')
        if np.any(score < 0.0) or np.any(score > 1.0):
            raise OsseousPriorContractError(f'Osseous score {key} lies outside [0,1].')
    return scores


def compute_osseous_support_score(
    ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
    *,
    hu_threshold,
    radius_mm,
) -> np.ndarray:
    """Compute one CT-derived osseous support score per supplied token."""
    threshold = _finite_scalar(hu_threshold, 'hu_threshold')
    radius = _finite_scalar(radius_mm, 'radius_mm')
    return compute_osseous_support_score_grid(
        ct_volume,
        ct_spacing,
        ct_origin,
        ct_direction,
        support_locations_mm,
        hu_thresholds=(threshold,),
        radius_values_mm=(radius,),
    )[(threshold, radius)]


__all__ = [
    'OsseousPriorContractError',
    'compute_osseous_support_score',
    'compute_osseous_support_score_grid',
]
