"""Deterministic M3-6B physical-LPS-mm Point rigid perturbation primitives."""

import hashlib
import json
import math
from typing import Mapping

import numpy as np


SEED_SCHEME_VERSION = 'm3_6b_seed_v1'
POINT_TO_CT_DIRECTION = 'Point Cloud -> CT'
AXIS_NORM_EPSILON = 1e-12
NORMAL_NORM_EPSILON = 1e-12
AXIS_RETRY_LIMIT = 16
SO3_ATOL = 1e-7
PROVENANCE_FIELD = 'm3_6b_perturbation'
UINT64_MAX = (1 << 64) - 1


class M3PerturbationContractError(ValueError):
    pass


def _require_nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise M3PerturbationContractError(f'{name} must be a non-empty string.')
    return value.strip()


def _require_identifier(value, name: str, *, optional: bool):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        qualifier = 'None or ' if optional else ''
        raise M3PerturbationContractError(
            f'{name} must be {qualifier}a non-empty string or integer.'
        )
    if isinstance(value, str):
        return _require_nonempty_string(value, name)
    return int(value)


def _require_nonnegative_integer(value, name: str, *, maximum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise M3PerturbationContractError(f'{name} must be a non-negative integer.')
    integer = int(value)
    if integer < 0 or (maximum is not None and integer > maximum):
        suffix = f' no greater than {maximum}.' if maximum is not None else '.'
        raise M3PerturbationContractError(f'{name} must be a non-negative integer{suffix}')
    return integer


def _require_nonnegative_bound(value, name: str) -> float:
    if isinstance(value, bool) or isinstance(value, np.ndarray):
        raise M3PerturbationContractError(f'{name} must be a finite non-negative scalar.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3PerturbationContractError(
            f'{name} must be a finite non-negative scalar.'
        ) from error
    if not math.isfinite(number) or number < 0.0:
        raise M3PerturbationContractError(f'{name} must be a finite non-negative scalar.')
    return number


def derive_perturbation_seed(
    *,
    scheme_version=SEED_SCHEME_VERSION,
    protocol_hash,
    fold_id,
    root_seed,
    purpose,
    epoch,
    subject_id,
    severity,
    variant_id,
) -> int:
    """Derive one big-endian uint64 seed from a complete canonical payload."""
    scheme_version = _require_nonempty_string(scheme_version, 'scheme_version')
    if scheme_version != SEED_SCHEME_VERSION:
        raise M3PerturbationContractError(
            f'unsupported seed scheme_version: {scheme_version!r}.'
        )
    protocol_hash = _require_nonempty_string(protocol_hash, 'protocol_hash')
    fold_id = _require_identifier(fold_id, 'fold_id', optional=False)
    root_seed = _require_nonnegative_integer(root_seed, 'root_seed')
    purpose = _require_nonempty_string(purpose, 'purpose')
    if epoch is not None:
        epoch = _require_nonnegative_integer(epoch, 'epoch')
    subject_id = _require_nonempty_string(subject_id, 'subject_id')
    severity = _require_identifier(severity, 'severity', optional=True)
    variant_id = _require_identifier(variant_id, 'variant_id', optional=True)

    payload = {
        'scheme_version': scheme_version,
        'protocol_hash': protocol_hash,
        'fold_id': fold_id,
        'root_seed': root_seed,
        'purpose': purpose,
        'epoch': epoch,
        'subject_id': subject_id,
        'severity': severity,
        'variant_id': variant_id,
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical_json.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


def _require_seed(seed) -> int:
    return _require_nonnegative_integer(seed, 'seed', maximum=UINT64_MAX)


def _require_float_vector(value, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3PerturbationContractError(f'{name} must be a finite float64 3-vector.') from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise M3PerturbationContractError(f'{name} must be a finite float64 3-vector.')
    return vector


def _require_rotation(value, name: str = 'R_aug') -> np.ndarray:
    try:
        rotation = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3PerturbationContractError(f'{name} must be a finite SO(3) matrix.') from error
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise M3PerturbationContractError(f'{name} must be a finite SO(3) matrix.')
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=SO3_ATOL,
    ):
        raise M3PerturbationContractError(f'{name} must be orthonormal.')
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or not np.isclose(
        determinant,
        1.0,
        rtol=0.0,
        atol=SO3_ATOL,
    ):
        raise M3PerturbationContractError(f'{name} must have determinant +1.')
    return rotation


def _axis_angle_rotation(axis, angle_deg: float) -> np.ndarray:
    axis = _require_float_vector(axis, 'axis')
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= AXIS_NORM_EPSILON:
        raise M3PerturbationContractError('axis norm must be greater than 1e-12.')
    axis = axis / axis_norm
    angle_rad = np.deg2rad(float(angle_deg))
    if not np.isfinite(angle_rad):
        raise M3PerturbationContractError('sampled rotation angle is not finite in radians.')
    x, y, z = axis
    skew = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    sine = math.sin(float(angle_rad))
    cosine = math.cos(float(angle_rad))
    rotation = np.eye(3, dtype=np.float64) + sine * skew + (1.0 - cosine) * (skew @ skew)
    return _require_rotation(rotation)


def sample_rigid_perturbation(seed, max_rotation_deg, max_translation_mm):
    """Sample one deterministic float64 Point perturbation with local PCG64."""
    seed = _require_seed(seed)
    max_rotation_deg = _require_nonnegative_bound(max_rotation_deg, 'max_rotation_deg')
    max_translation_mm = _require_nonnegative_bound(
        max_translation_mm,
        'max_translation_mm',
    )
    generator = np.random.Generator(np.random.PCG64(seed))

    axis = None
    for _ in range(AXIS_RETRY_LIMIT):
        candidate = generator.normal(0.0, 1.0, size=3).astype(np.float64, copy=False)
        candidate_norm = float(np.linalg.norm(candidate))
        if math.isfinite(candidate_norm) and candidate_norm > AXIS_NORM_EPSILON:
            axis = candidate / candidate_norm
            break
    if axis is None:
        raise M3PerturbationContractError(
            f'axis sampling failed after {AXIS_RETRY_LIMIT} attempts.'
        )

    try:
        angle_deg = float(generator.uniform(-max_rotation_deg, max_rotation_deg))
        translation_mm = generator.uniform(
            -max_translation_mm,
            max_translation_mm,
            size=3,
        ).astype(np.float64, copy=False)
    except (FloatingPointError, OverflowError, ValueError) as error:
        raise M3PerturbationContractError('rigid perturbation sampling failed.') from error
    if not math.isfinite(angle_deg) or not np.all(np.isfinite(translation_mm)):
        raise M3PerturbationContractError('rigid perturbation sampling produced NaN or Inf.')
    if abs(angle_deg) > max_rotation_deg or np.any(np.abs(translation_mm) > max_translation_mm):
        raise M3PerturbationContractError('rigid perturbation sampling exceeded its bounds.')

    rotation = _axis_angle_rotation(axis, angle_deg)
    return {
        'seed': seed,
        'axis': axis.copy(),
        'angle_deg': angle_deg,
        'translation_mm': translation_mm.copy(),
        'R_aug': rotation.copy(),
    }


def _require_coordinate_matrix(value, name: str):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise M3PerturbationContractError(f'{name} must have shape [N,3], N>0.')
    if not np.issubdtype(array.dtype, np.floating):
        raise M3PerturbationContractError(f'{name} must use a floating dtype.')
    try:
        float64 = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3PerturbationContractError(f'{name} cannot be represented in float64.') from error
    if not np.all(np.isfinite(float64)):
        raise M3PerturbationContractError(f'{name} contains NaN or Inf.')
    return array, float64


def _cast_preserving_dtype(value: np.ndarray, dtype, name: str) -> np.ndarray:
    try:
        result = value.astype(dtype, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3PerturbationContractError(f'{name} cannot be cast back to its input dtype.') from error
    if not np.all(np.isfinite(result)):
        raise M3PerturbationContractError(
            f'{name} became NaN or Inf when cast back to its input dtype.'
        )
    return result


def apply_point_rigid_perturbation(
    point_xyz_phys,
    point_normal,
    R_aug,
    translation_mm,
):
    """Apply a centroid-centered perturbation in physical LPS millimetres."""
    xyz_input, xyz = _require_coordinate_matrix(point_xyz_phys, 'point_xyz_phys')
    rotation = _require_rotation(R_aug)
    translation = _require_float_vector(translation_mm, 'translation_mm')
    centroid = np.mean(xyz, axis=0, dtype=np.float64)
    if not np.all(np.isfinite(centroid)):
        raise M3PerturbationContractError('point centroid contains NaN or Inf.')
    with np.errstate(over='ignore', invalid='ignore'):
        xyz_aug = (xyz - centroid) @ rotation.T + centroid + translation
        b_aug = centroid + translation - rotation @ centroid
    if not np.all(np.isfinite(xyz_aug)) or not np.all(np.isfinite(b_aug)):
        raise M3PerturbationContractError('augmented physical Point coordinates contain NaN or Inf.')
    xyz_aug = _cast_preserving_dtype(xyz_aug, xyz_input.dtype, 'point_xyz_phys')

    if point_normal is None:
        normal_aug = None
    else:
        normal_input, normals = _require_coordinate_matrix(point_normal, 'point_normal')
        if normals.shape != xyz.shape:
            raise M3PerturbationContractError(
                'point_normal must have the same shape as point_xyz_phys.'
            )
        with np.errstate(over='ignore', invalid='ignore'):
            rotated_normals = normals @ rotation.T
            normal_norms = np.linalg.norm(rotated_normals, axis=1, keepdims=True)
        if not np.all(np.isfinite(rotated_normals)) or not np.all(np.isfinite(normal_norms)):
            raise M3PerturbationContractError('augmented point normals contain NaN or Inf.')
        if np.any(normal_norms <= NORMAL_NORM_EPSILON):
            raise M3PerturbationContractError('point normal norm must be greater than 1e-12.')
        normal_aug_float64 = rotated_normals / normal_norms
        normal_aug = _cast_preserving_dtype(
            normal_aug_float64,
            normal_input.dtype,
            'point_normal',
        )
        cast_norms = np.linalg.norm(normal_aug.astype(np.float64), axis=1)
        if not np.all(np.isfinite(cast_norms)) or np.any(cast_norms <= NORMAL_NORM_EPSILON):
            raise M3PerturbationContractError('cast augmented point normals are degenerate.')
        dtype_tolerance = max(1e-6, 8.0 * float(np.finfo(normal_input.dtype).eps))
        if np.any(np.abs(cast_norms - 1.0) > dtype_tolerance):
            raise M3PerturbationContractError(
                'cast augmented point normals are not unit length within dtype tolerance.'
            )

    return {
        'point_xyz_phys': xyz_aug,
        'point_normal': normal_aug,
        'centroid_phys_mm': centroid.copy(),
        'b_aug': b_aug.copy(),
    }


def _require_rigid_transform(value, name: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3PerturbationContractError(f'{name} must be a finite rigid [4,4] matrix.') from error
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise M3PerturbationContractError(f'{name} must be a finite rigid [4,4] matrix.')
    if not np.allclose(
        transform[3],
        [0.0, 0.0, 0.0, 1.0],
        rtol=0.0,
        atol=SO3_ATOL,
    ):
        raise M3PerturbationContractError(f'{name} must have homogeneous last row [0,0,0,1].')
    _require_rotation(transform[:3, :3], f'{name} rotation')
    return transform


def compose_effective_point_to_ct_gt(
    gt_transform,
    R_aug,
    b_aug,
    *,
    gt_transform_direction,
) -> np.ndarray:
    """Return float64 ``T_gt @ inverse(T_aug)`` for Point-to-CT supervision."""
    if gt_transform_direction != POINT_TO_CT_DIRECTION:
        raise M3PerturbationContractError(
            f'gt_transform_direction must be exactly {POINT_TO_CT_DIRECTION!r}.'
        )
    original = _require_rigid_transform(gt_transform, 'gt_transform')
    rotation_aug = _require_rotation(R_aug)
    translation_aug = _require_float_vector(b_aug, 'b_aug')
    rotation_effective = original[:3, :3] @ rotation_aug.T
    translation_effective = original[:3, 3] - rotation_effective @ translation_aug
    effective = np.eye(4, dtype=np.float64)
    effective[:3, :3] = rotation_effective
    effective[:3, 3] = translation_effective
    return _require_rigid_transform(effective, 'effective gt_transform').copy()


def augment_point_sample(
    raw_sample,
    *,
    seed,
    max_rotation_deg,
    max_translation_mm,
    scheme_version=SEED_SCHEME_VERSION,
):
    """Return a copied sample with deterministic Point-only physical augmentation."""
    if not isinstance(raw_sample, Mapping):
        raise M3PerturbationContractError('raw_sample must be a mapping.')
    required = (
        'point_xyz_phys',
        'point_normal',
        'gt_transform',
        'gt_transform_direction',
    )
    missing = [field for field in required if field not in raw_sample]
    if missing:
        raise M3PerturbationContractError(f'raw_sample is missing fields: {missing}.')
    if PROVENANCE_FIELD in raw_sample:
        raise M3PerturbationContractError('raw_sample is already marked as perturbed.')
    scheme_version = _require_nonempty_string(scheme_version, 'scheme_version')
    if scheme_version != SEED_SCHEME_VERSION:
        raise M3PerturbationContractError(
            f'unsupported seed scheme_version: {scheme_version!r}.'
        )

    perturbation = sample_rigid_perturbation(
        seed,
        max_rotation_deg,
        max_translation_mm,
    )
    applied = apply_point_rigid_perturbation(
        raw_sample['point_xyz_phys'],
        raw_sample['point_normal'],
        perturbation['R_aug'],
        perturbation['translation_mm'],
    )
    effective_gt = compose_effective_point_to_ct_gt(
        raw_sample['gt_transform'],
        perturbation['R_aug'],
        applied['b_aug'],
        gt_transform_direction=raw_sample['gt_transform_direction'],
    )

    augmented = dict(raw_sample)
    augmented['point_xyz_phys'] = applied['point_xyz_phys']
    augmented['point_normal'] = applied['point_normal']
    augmented['gt_transform'] = effective_gt
    augmented[PROVENANCE_FIELD] = {
        'scheme_version': scheme_version,
        'seed': perturbation['seed'],
        'axis': perturbation['axis'].tolist(),
        'angle_deg': perturbation['angle_deg'],
        'translation_mm': perturbation['translation_mm'].tolist(),
        'centroid_phys_mm': applied['centroid_phys_mm'].tolist(),
        'R_aug': perturbation['R_aug'].tolist(),
        'b_aug': applied['b_aug'].tolist(),
        'max_rotation_deg': float(max_rotation_deg),
        'max_translation_mm': float(max_translation_mm),
    }
    return augmented


__all__ = [
    'AXIS_NORM_EPSILON',
    'AXIS_RETRY_LIMIT',
    'M3PerturbationContractError',
    'NORMAL_NORM_EPSILON',
    'POINT_TO_CT_DIRECTION',
    'PROVENANCE_FIELD',
    'SEED_SCHEME_VERSION',
    'apply_point_rigid_perturbation',
    'augment_point_sample',
    'compose_effective_point_to_ct_gt',
    'derive_perturbation_seed',
    'sample_rigid_perturbation',
]
