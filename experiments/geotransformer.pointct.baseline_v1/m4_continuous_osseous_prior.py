"""Input-only CT-derived continuous osseous support evidence.

The public score API is deliberately limited to defective CT HU/header data,
physical support locations, and the three declared score parameters. It has no
training, matching, registration, identity, or defect-annotation dependency.
"""

import math
from typing import Dict, Tuple

import numpy as np


class ContinuousOsseousPriorContractError(RuntimeError):
    """Raised when a score input violates the fail-closed contract."""


_OPEN_UNIT_LOWER = np.nextafter(np.float64(0.0), np.float64(1.0))
_OPEN_UNIT_UPPER = np.nextafter(np.float64(1.0), np.float64(0.0))


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ContinuousOsseousPriorContractError(f'{name} must be a finite scalar.')
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContinuousOsseousPriorContractError(
            f'{name} must be a finite scalar.'
        ) from error
    if not math.isfinite(scalar):
        raise ContinuousOsseousPriorContractError(f'{name} must be a finite scalar.')
    return scalar


def _finite_vector(value, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContinuousOsseousPriorContractError(
            f'{name} must be a finite vector with shape (3,).'
        ) from error
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ContinuousOsseousPriorContractError(
            f'{name} must be a finite vector with shape (3,).'
        )
    if positive and np.any(array <= 0.0):
        raise ContinuousOsseousPriorContractError(
            f'{name} must contain positive values.'
        )
    return array


def _direction_matrix(value) -> np.ndarray:
    try:
        direction = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContinuousOsseousPriorContractError(
            'ct_direction must be a finite orthonormal matrix with shape (3,3).'
        ) from error
    if direction.shape == (9,):
        direction = direction.reshape(3, 3)
    if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
        raise ContinuousOsseousPriorContractError(
            'ct_direction must be a finite orthonormal matrix with shape (3,3).'
        )
    gram = direction.T @ direction
    determinant = float(np.linalg.det(direction))
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-6) or not np.isclose(
        abs(determinant), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ContinuousOsseousPriorContractError(
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
    if isinstance(ct_volume, np.ma.MaskedArray):
        raise ContinuousOsseousPriorContractError(
            'ct_volume must contain unmasked real numeric HU values.'
        )
    volume = np.asarray(ct_volume)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise ContinuousOsseousPriorContractError(
            'ct_volume must be a non-empty 3D array in [z,y,x] order.'
        )
    if volume.dtype == np.dtype(bool) or not (
        np.issubdtype(volume.dtype, np.integer)
        or np.issubdtype(volume.dtype, np.floating)
    ):
        raise ContinuousOsseousPriorContractError(
            'ct_volume must contain real numeric HU values.'
        )
    if np.issubdtype(volume.dtype, np.floating) and not np.all(np.isfinite(volume)):
        raise ContinuousOsseousPriorContractError(
            'ct_volume contains NaN or Inf HU values.'
        )

    spacing = _finite_vector(ct_spacing, 'ct_spacing', positive=True)
    origin = _finite_vector(ct_origin, 'ct_origin')
    direction = _direction_matrix(ct_direction)
    try:
        support = np.asarray(support_locations_mm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContinuousOsseousPriorContractError(
            'support_locations_mm must have finite non-empty shape [N,3].'
        ) from error
    if (
        support.ndim != 2
        or support.shape[0] == 0
        or support.shape[1] != 3
        or not np.all(np.isfinite(support))
    ):
        raise ContinuousOsseousPriorContractError(
            'support_locations_mm must have finite non-empty shape [N,3].'
        )
    return volume, spacing, origin, direction, support


def _parameter_grid(value, name: str) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ContinuousOsseousPriorContractError(
            f'{name} must be a positive scalar or non-empty sequence.'
        )
    if np.isscalar(value):
        scalars = (_finite_scalar(value, name),)
    else:
        try:
            scalars = tuple(_finite_scalar(item, name) for item in value)
        except TypeError as error:
            raise ContinuousOsseousPriorContractError(
                f'{name} must be a positive scalar or non-empty sequence.'
            ) from error
    if not scalars:
        raise ContinuousOsseousPriorContractError(
            f'{name} must be a positive scalar or non-empty sequence.'
        )
    if any(item <= 0.0 for item in scalars):
        raise ContinuousOsseousPriorContractError(
            f'{name} values must be greater than zero.'
        )
    if len(set(scalars)) != len(scalars):
        raise ContinuousOsseousPriorContractError(f'{name} values must be unique.')
    return scalars


def continuous_hu_evidence(hu_values, center_hu, tau_hu) -> np.ndarray:
    """Evaluate a numerically stable open-unit sigmoid evidence function."""
    centre = _finite_scalar(center_hu, 'center_hu')
    tau = _finite_scalar(tau_hu, 'tau_hu')
    if tau <= 0.0:
        raise ContinuousOsseousPriorContractError('tau_hu must be greater than zero.')
    try:
        hu = np.asarray(hu_values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContinuousOsseousPriorContractError(
            'hu_values must contain finite real values.'
        ) from error
    if hu.size == 0 or not np.all(np.isfinite(hu)):
        raise ContinuousOsseousPriorContractError(
            'hu_values must contain finite real values.'
        )

    scaled = (hu - centre) / tau
    evidence = np.empty_like(scaled, dtype=np.float64)
    nonnegative = scaled >= 0.0
    evidence[nonnegative] = 1.0 / (1.0 + np.exp(-scaled[nonnegative]))
    exp_scaled = np.exp(scaled[~nonnegative])
    evidence[~nonnegative] = exp_scaled / (1.0 + exp_scaled)
    np.clip(evidence, _OPEN_UNIT_LOWER, _OPEN_UNIT_UPPER, out=evidence)
    if not np.all(np.isfinite(evidence)) or np.any(evidence <= 0.0) or np.any(
        evidence >= 1.0
    ):
        raise ContinuousOsseousPriorContractError(
            'Continuous HU evidence violates its open-unit numeric contract.'
        )
    return evidence


def compute_continuous_osseous_support_score_grid(
    ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
    center_hu,
    tau_hu,
    radius_mm,
) -> Dict[Tuple[float, float], np.ndarray]:
    """Compute a declared tau/radius grid using physical-mm spheres.

    ``tau_hu`` and ``radius_mm`` may each be either one positive scalar or a
    non-empty sequence. The parameter names remain identical to the singular
    leakage-controlled score API.
    """
    volume, spacing, origin, direction, support = _validate_inputs(
        ct_volume,
        ct_spacing,
        ct_origin,
        ct_direction,
        support_locations_mm,
    )
    centre = _finite_scalar(center_hu, 'center_hu')
    taus = _parameter_grid(tau_hu, 'tau_hu')
    radii = _parameter_grid(radius_mm, 'radius_mm')

    image_affine = direction @ np.diag(spacing)
    try:
        support_index_xyz = np.linalg.solve(
            image_affine,
            (support - origin[np.newaxis, :]).T,
        ).T
    except np.linalg.LinAlgError as error:
        raise ContinuousOsseousPriorContractError(
            'CT image geometry is not invertible.'
        ) from error
    if not np.all(np.isfinite(support_index_xyz)):
        raise ContinuousOsseousPriorContractError(
            'Support-to-image coordinate conversion is not finite.'
        )

    scores = {
        (tau, radius): np.empty(support.shape[0], dtype=np.float64)
        for radius in radii
        for tau in taus
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
            raise ContinuousOsseousPriorContractError(
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
            '...i,...i->...', delta_physical, delta_physical
        )
        local_hu = volume[
            lower_xyz[2] : upper_xyz[2] + 1,
            lower_xyz[1] : upper_xyz[1] + 1,
            lower_xyz[0] : upper_xyz[0] + 1,
        ]
        if local_hu.shape != distance_squared.shape:
            raise ContinuousOsseousPriorContractError(
                'Internal CT neighborhood shape mismatch.'
            )

        evidence_by_tau = {
            tau: continuous_hu_evidence(local_hu, centre, tau) for tau in taus
        }
        for radius in radii:
            in_sphere = distance_squared <= radius * radius + distance_tolerance
            denominator = int(np.count_nonzero(in_sphere))
            if denominator <= 0:
                raise ContinuousOsseousPriorContractError(
                    f'Support token {row} has an empty physical neighborhood '
                    f'at radius {radius} mm.'
                )
            for tau in taus:
                value = float(np.mean(evidence_by_tau[tau][in_sphere], dtype=np.float64))
                scores[(tau, radius)][row] = float(
                    np.clip(value, _OPEN_UNIT_LOWER, _OPEN_UNIT_UPPER)
                )

    expected_shape = (support.shape[0],)
    for key, score in scores.items():
        if score.shape != expected_shape or not np.all(np.isfinite(score)):
            raise ContinuousOsseousPriorContractError(
                f'Continuous osseous score {key} violates its numeric contract.'
            )
        if np.any(score <= 0.0) or np.any(score >= 1.0):
            raise ContinuousOsseousPriorContractError(
                f'Continuous osseous score {key} lies outside the open unit interval.'
            )
    return scores


def compute_continuous_osseous_support_score(
    ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
    center_hu,
    tau_hu,
    radius_mm,
) -> np.ndarray:
    """Compute one continuous CT-derived osseous score per support token."""
    centre = _finite_scalar(center_hu, 'center_hu')
    tau = _finite_scalar(tau_hu, 'tau_hu')
    radius = _finite_scalar(radius_mm, 'radius_mm')
    return compute_continuous_osseous_support_score_grid(
        ct_volume,
        ct_spacing,
        ct_origin,
        ct_direction,
        support_locations_mm,
        centre,
        tau,
        radius,
    )[(tau, radius)]


__all__ = [
    'ContinuousOsseousPriorContractError',
    'compute_continuous_osseous_support_score',
    'compute_continuous_osseous_support_score_grid',
    'continuous_hu_evidence',
]
