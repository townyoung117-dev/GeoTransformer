"""M4-2C continuous osseous-support integration for Point-CT matching.

This module composes the frozen M4-2A soft modulation with the frozen
CT-derived continuous osseous-support score.  It introduces no module, state,
or trainable parameter.  The formal V1 interaction is

``R_oss[i,j] = R_soft[i,j] * o[j]``

and

``S_final[i,j] = S_soft[i,j] - lambda_oss * (1 - R_oss[i,j])``.

The score is computed only from the defective CT image/header and the declared
support locations.  Exact float32 location equality ties those locations to
the CT tokens actually supplied to the matcher before the score is accepted.
"""

import math

import numpy as np
import torch

from m4_continuous_osseous_prior import (
    ContinuousOsseousPriorContractError,
    compute_continuous_osseous_support_score,
)
from m4_soft_modulation import (
    M4SoftModulationError,
    compute_defect_proximity_reliability,
    modulate_cross_modal_similarity,
)
from matching import PointCTMatchingContractError, compute_cross_modal_similarity


OSSEOUS_CENTER_HU = 300.0
OSSEOUS_TAU_HU = 100.0
OSSEOUS_RADIUS_MM = 20.0


class M4OsseousIntegrationError(RuntimeError):
    """Raised when an M4-2C integration input violates its contract."""


def _require_finite_nonnegative_scalar(value, name: str) -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise M4OsseousIntegrationError(
            f'{name} must be a finite scalar greater than or equal to zero.'
        )
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M4OsseousIntegrationError(
            f'{name} must be a finite scalar greater than or equal to zero.'
        ) from error
    if not math.isfinite(scalar) or scalar < 0.0:
        raise M4OsseousIntegrationError(
            f'{name} must be a finite scalar greater than or equal to zero.'
        )
    return scalar


def _require_finite_positive_scalar(value, name: str) -> float:
    scalar = _require_finite_nonnegative_scalar(value, name)
    if scalar <= 0.0:
        raise M4OsseousIntegrationError(
            f'{name} must be a finite scalar greater than zero.'
        )
    return scalar


def _require_float_matrix(tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise M4OsseousIntegrationError(f'{name} must be a torch.Tensor.')
    if tensor.ndim != 2 or min(tensor.shape) <= 0:
        raise M4OsseousIntegrationError(
            f'{name} must have non-empty shape [Np,Nv]; got {tuple(tensor.shape)}.'
        )
    if not torch.is_floating_point(tensor):
        raise M4OsseousIntegrationError(
            f'{name} must use a real floating-point dtype.'
        )
    if tensor.device.type == 'meta':
        raise M4OsseousIntegrationError(f'{name} cannot be a meta tensor.')
    if not bool(torch.isfinite(tensor).all()):
        raise M4OsseousIntegrationError(f'{name} contains NaN or Inf.')
    return tensor


def _require_unit_tensor(
    tensor,
    shape,
    *,
    device,
    dtype,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise M4OsseousIntegrationError(f'{name} must be a torch.Tensor.')
    if not torch.is_floating_point(tensor):
        raise M4OsseousIntegrationError(
            f'{name} must use a real floating-point dtype.'
        )
    if tuple(tensor.shape) != tuple(shape):
        raise M4OsseousIntegrationError(
            f'{name} must have exact shape {tuple(shape)}; got {tuple(tensor.shape)}.'
        )
    if tensor.device != device:
        raise M4OsseousIntegrationError(
            f'{name} must be on the same device as soft_similarity.'
        )
    if tensor.dtype != dtype:
        raise M4OsseousIntegrationError(
            f'{name} must use the same dtype as soft_similarity.'
        )
    if not bool(torch.isfinite(tensor).all()):
        raise M4OsseousIntegrationError(f'{name} contains NaN or Inf.')
    if bool(torch.any(tensor < 0.0)) or bool(torch.any(tensor > 1.0)):
        raise M4OsseousIntegrationError(f'{name} values must lie in [0,1].')
    return tensor


def _require_token_locations(token_locations, name: str) -> torch.Tensor:
    if not torch.is_tensor(token_locations):
        raise M4OsseousIntegrationError(f'{name} must be a torch.Tensor.')
    if token_locations.ndim != 2 or token_locations.shape[0] == 0 or token_locations.shape[1] != 3:
        raise M4OsseousIntegrationError(
            f'{name} must have finite non-empty shape [Nv,3]; '
            f'got {tuple(token_locations.shape)}.'
        )
    if not torch.is_floating_point(token_locations):
        raise M4OsseousIntegrationError(
            f'{name} must use a real floating-point dtype.'
        )
    if token_locations.device.type == 'meta':
        raise M4OsseousIntegrationError(f'{name} cannot be a meta tensor.')
    locations_float32 = token_locations.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(locations_float32).all()):
        raise M4OsseousIntegrationError(f'{name} contains NaN or Inf.')
    return locations_float32


def _support_locations_numpy(support_locations_mm):
    if isinstance(support_locations_mm, np.ma.MaskedArray):
        raise M4OsseousIntegrationError(
            'support_locations_mm must contain unmasked physical-mm locations.'
        )
    if torch.is_tensor(support_locations_mm):
        if support_locations_mm.device.type == 'meta':
            raise M4OsseousIntegrationError(
                'support_locations_mm cannot be a meta tensor.'
            )
        if not torch.is_floating_point(support_locations_mm):
            raise M4OsseousIntegrationError(
                'support_locations_mm must use a real floating-point dtype.'
            )
        support_float64 = (
            support_locations_mm.detach()
            .to(device='cpu', dtype=torch.float64)
            .numpy()
        )
    else:
        try:
            support_float64 = np.asarray(support_locations_mm, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise M4OsseousIntegrationError(
                'support_locations_mm must have finite non-empty shape [Nv,3].'
            ) from error
    if (
        support_float64.ndim != 2
        or support_float64.shape[0] == 0
        or support_float64.shape[1] != 3
        or not np.all(np.isfinite(support_float64))
    ):
        raise M4OsseousIntegrationError(
            'support_locations_mm must have finite non-empty shape [Nv,3].'
        )
    support_float32 = support_float64.astype(np.float32, copy=True)
    return support_float64, support_float32


def _summary(tensor: torch.Tensor, prefix: str) -> dict:
    detached = tensor.detach()
    return {
        f'{prefix}_min': float(detached.amin().item()),
        f'{prefix}_mean': float(detached.mean().item()),
        f'{prefix}_max': float(detached.amax().item()),
    }


def compute_aligned_osseous_score(
    *,
    defective_ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
    ct_token_locations_mm: torch.Tensor,
) -> torch.Tensor:
    """Compute the frozen score after exact support-to-token alignment.

    Only defective CT image/header fields, declared support locations, and the
    actual CT matcher-token locations are accepted.  Both location arrays are
    converted to float32 and must then be exactly equal in length, order, and
    value.  No nearest-neighbour matching, truncation, or broadcasting occurs.
    """
    token_locations_float32 = _require_token_locations(
        ct_token_locations_mm,
        'ct_token_locations_mm',
    )
    support_float64, support_float32 = _support_locations_numpy(
        support_locations_mm
    )
    token_float32_cpu = token_locations_float32.to(device='cpu').numpy()
    expected_shape = tuple(token_float32_cpu.shape)
    if tuple(support_float32.shape) != expected_shape:
        raise M4OsseousIntegrationError(
            'support_locations_mm and ct_token_locations_mm token counts differ: '
            f'{support_float32.shape[0]} versus {token_float32_cpu.shape[0]}.'
        )
    if not np.array_equal(support_float32, token_float32_cpu):
        raise M4OsseousIntegrationError(
            'support_locations_mm must exactly match ct_token_locations_mm after '
            'float32 conversion, including token order.'
        )

    try:
        score_array = compute_continuous_osseous_support_score(
            defective_ct_volume,
            ct_spacing,
            ct_origin,
            ct_direction,
            support_float64,
            OSSEOUS_CENTER_HU,
            OSSEOUS_TAU_HU,
            OSSEOUS_RADIUS_MM,
        )
    except ContinuousOsseousPriorContractError as error:
        raise M4OsseousIntegrationError(
            f'continuous osseous-support scoring failed: {error}'
        ) from error

    if isinstance(score_array, np.ma.MaskedArray):
        raise M4OsseousIntegrationError(
            'continuous osseous score must be an unmasked floating array.'
        )
    try:
        score_float32 = np.asarray(score_array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise M4OsseousIntegrationError(
            'continuous osseous score must be a floating vector.'
        ) from error
    expected_score_shape = (int(token_locations_float32.shape[0]),)
    if score_float32.shape != expected_score_shape:
        raise M4OsseousIntegrationError(
            'continuous osseous score and CT matcher-token counts differ: '
            f'{tuple(score_float32.shape)} versus {expected_score_shape}.'
        )
    if not np.all(np.isfinite(score_float32)):
        raise M4OsseousIntegrationError(
            'continuous osseous score contains NaN or Inf.'
        )
    if np.any(score_float32 < 0.0) or np.any(score_float32 > 1.0):
        raise M4OsseousIntegrationError(
            'continuous osseous score values must lie in [0,1].'
        )
    score = torch.from_numpy(np.ascontiguousarray(score_float32)).to(
        device=ct_token_locations_mm.device
    )
    return score


def integrate_osseous_prior(
    soft_similarity: torch.Tensor,
    pair_reliability: torch.Tensor,
    osseous_score: torch.Tensor,
    strength,
) -> dict:
    """Apply the parameter-free Candidate C osseous pair interaction."""
    soft_similarity = _require_float_matrix(soft_similarity, 'soft_similarity')
    num_point, num_ct = soft_similarity.shape
    pair_reliability = _require_unit_tensor(
        pair_reliability,
        (num_point, num_ct),
        device=soft_similarity.device,
        dtype=soft_similarity.dtype,
        name='pair_reliability',
    )
    osseous_score = _require_unit_tensor(
        osseous_score,
        (num_ct,),
        device=soft_similarity.device,
        dtype=soft_similarity.dtype,
        name='osseous_score',
    )
    strength = _require_finite_nonnegative_scalar(strength, 'strength')

    osseous_pair_reliability = pair_reliability * osseous_score.unsqueeze(0)
    if strength == 0.0:
        # Preserve tensor identity and the exact frozen M4-2A autograd path.
        final_similarity = soft_similarity
        osseous_penalty = torch.zeros_like(soft_similarity)
    else:
        osseous_penalty = strength * (1.0 - osseous_pair_reliability)
        final_similarity = soft_similarity - osseous_penalty
    if not bool(torch.isfinite(final_similarity).all()):
        raise M4OsseousIntegrationError('final_similarity contains NaN or Inf.')

    return {
        'soft_similarity': soft_similarity,
        'final_similarity': final_similarity,
        'osseous_score': osseous_score,
        'osseous_pair_reliability': osseous_pair_reliability,
        'osseous_penalty': osseous_penalty,
        'osseous_similarity_changed': not torch.equal(
            final_similarity,
            soft_similarity,
        ),
    }


def run_m4_osseous_integrated_matching(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    point_coordinates_mm: torch.Tensor,
    ct_token_locations_mm: torch.Tensor,
    point_valid_mask: torch.Tensor,
    ct_valid_mask: torch.Tensor,
    sigma_mm,
    soft_strength,
    osseous_strength,
    matcher,
    defective_ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
) -> dict:
    """Run frozen similarity/soft math, M4-2C, and one masked Sinkhorn."""
    if not hasattr(matcher, 'temperature') or not hasattr(matcher, 'norm_epsilon'):
        raise M4OsseousIntegrationError(
            'matcher must expose the frozen temperature and norm_epsilon settings.'
        )
    transport = getattr(matcher, 'transport', None)
    if transport is None or not callable(transport):
        raise M4OsseousIntegrationError(
            'matcher must expose callable masked Sinkhorn transport.'
        )
    if not torch.is_tensor(q) or q.ndim != 2 or q.shape[0] == 0:
        raise M4OsseousIntegrationError('Q must be a non-empty rank-2 torch.Tensor.')
    if not torch.is_tensor(k) or k.ndim != 2 or k.shape[0] == 0:
        raise M4OsseousIntegrationError('K must be a non-empty rank-2 torch.Tensor.')
    if not torch.is_tensor(point_coordinates_mm) or point_coordinates_mm.shape[0] != q.shape[0]:
        raise M4OsseousIntegrationError(
            'Q and Point physical-coordinate token counts differ.'
        )
    if not torch.is_tensor(ct_token_locations_mm) or ct_token_locations_mm.shape[0] != k.shape[0]:
        raise M4OsseousIntegrationError(
            'K and CT physical-coordinate token counts differ.'
        )
    if point_coordinates_mm.device != q.device:
        raise M4OsseousIntegrationError(
            'Point physical coordinates and Q must share a device.'
        )
    if ct_token_locations_mm.device != k.device:
        raise M4OsseousIntegrationError(
            'CT physical coordinates and K must share a device.'
        )

    sigma_value = _require_finite_positive_scalar(sigma_mm, 'sigma_mm')
    soft_strength_value = _require_finite_nonnegative_scalar(
        soft_strength,
        'soft_strength',
    )
    osseous_strength_value = _require_finite_nonnegative_scalar(
        osseous_strength,
        'osseous_strength',
    )
    try:
        point_reliability = compute_defect_proximity_reliability(
            point_coordinates_mm,
            point_valid_mask,
            sigma_value,
        )
        ct_reliability = compute_defect_proximity_reliability(
            ct_token_locations_mm,
            ct_valid_mask,
            sigma_value,
        )
    except M4SoftModulationError as error:
        raise M4OsseousIntegrationError(
            f'frozen defect-proximity reliability failed: {error}'
        ) from error
    if not bool(torch.any(point_valid_mask)) or not bool(torch.any(ct_valid_mask)):
        raise M4OsseousIntegrationError(
            'osseous-integrated matching requires at least one hard-valid token '
            'per modality.'
        )

    try:
        base_similarity = compute_cross_modal_similarity(
            q,
            k,
            matcher.temperature,
            norm_epsilon=matcher.norm_epsilon,
        )
    except PointCTMatchingContractError as error:
        raise M4OsseousIntegrationError(
            f'frozen cross-modal similarity failed: {error}'
        ) from error
    try:
        soft_modulation = modulate_cross_modal_similarity(
            base_similarity,
            point_reliability,
            ct_reliability,
            soft_strength_value,
        )
    except M4SoftModulationError as error:
        raise M4OsseousIntegrationError(
            f'frozen M4-2A soft modulation failed: {error}'
        ) from error

    osseous_score = compute_aligned_osseous_score(
        defective_ct_volume=defective_ct_volume,
        ct_spacing=ct_spacing,
        ct_origin=ct_origin,
        ct_direction=ct_direction,
        support_locations_mm=support_locations_mm,
        ct_token_locations_mm=ct_token_locations_mm,
    )
    osseous_modulation = integrate_osseous_prior(
        soft_modulation['modulated_similarity'],
        soft_modulation['pair_reliability'],
        osseous_score,
        osseous_strength_value,
    )
    final_similarity = osseous_modulation['final_similarity']
    try:
        # The original hard masks are forwarded unchanged; M4-2C never edits or
        # derives replacement masks and therefore cannot reactivate a token.
        log_assignment = transport(
            final_similarity.unsqueeze(0),
            point_valid_mask.unsqueeze(0),
            ct_valid_mask.unsqueeze(0),
        ).squeeze(0)
    except PointCTMatchingContractError as error:
        raise M4OsseousIntegrationError(
            f'masked Sinkhorn transport failed: {error}'
        ) from error
    expected_assignment_shape = (int(q.shape[0]) + 1, int(k.shape[0]) + 1)
    if tuple(log_assignment.shape) != expected_assignment_shape:
        raise M4OsseousIntegrationError(
            f'masked Sinkhorn returned shape {tuple(log_assignment.shape)}; '
            f'expected {expected_assignment_shape}.'
        )

    osseous_penalty = osseous_modulation['osseous_penalty']
    output = {
        'base_similarity': base_similarity,
        'soft_similarity': soft_modulation['modulated_similarity'],
        'final_similarity': final_similarity,
        'similarity': final_similarity,
        'modulated_similarity': final_similarity,
        'pair_reliability': soft_modulation['pair_reliability'],
        'soft_pair_reliability': soft_modulation['pair_reliability'],
        'osseous_pair_reliability': osseous_modulation[
            'osseous_pair_reliability'
        ],
        'osseous_penalty': osseous_penalty,
        'osseous_score': osseous_score,
        'log_assignment': log_assignment,
        'point_reliability': point_reliability,
        'ct_reliability': ct_reliability,
        'point_valid_mask': point_valid_mask,
        'ct_valid_mask': ct_valid_mask,
        'soft_similarity_changed': soft_modulation['soft_similarity_changed'],
        'osseous_similarity_changed': osseous_modulation[
            'osseous_similarity_changed'
        ],
        'm4_hard_constraint_active': True,
        'm4_soft_modulation_enabled': True,
        'm4_soft_modulation_active': True,
        'm4_osseous_prior_enabled': True,
        'm4_osseous_prior_active': True,
        'm4_soft_sigma_mm': sigma_value,
        'm4_soft_strength': soft_strength_value,
        'osseous_strength': osseous_strength_value,
        'm4_osseous_strength': osseous_strength_value,
        'osseous_center_hu': OSSEOUS_CENTER_HU,
        'osseous_tau_hu': OSSEOUS_TAU_HU,
        'osseous_radius_mm': OSSEOUS_RADIUS_MM,
        **_summary(point_reliability, 'point_reliability'),
        **_summary(ct_reliability, 'ct_reliability'),
        **_summary(soft_modulation['pair_reliability'], 'pair_reliability'),
        **_summary(osseous_score, 'osseous_score'),
        **_summary(
            osseous_modulation['osseous_pair_reliability'],
            'osseous_pair_reliability',
        ),
        **_summary(osseous_penalty, 'osseous_penalty'),
        **_summary(osseous_penalty, 'final_modulation'),
        'base_similarity_mean': float(base_similarity.detach().mean().item()),
        'soft_similarity_mean': float(
            soft_modulation['modulated_similarity'].detach().mean().item()
        ),
        'final_similarity_mean': float(final_similarity.detach().mean().item()),
    }
    return output


# Compatibility names used by the training and evaluator integration layers.
compute_aligned_continuous_osseous_score = compute_aligned_osseous_score
modulate_osseous_similarity = integrate_osseous_prior
run_m4_osseous_prior_matching = run_m4_osseous_integrated_matching


__all__ = [
    'M4OsseousIntegrationError',
    'OSSEOUS_CENTER_HU',
    'OSSEOUS_TAU_HU',
    'OSSEOUS_RADIUS_MM',
    'compute_aligned_osseous_score',
    'compute_aligned_continuous_osseous_score',
    'integrate_osseous_prior',
    'modulate_osseous_similarity',
    'run_m4_osseous_integrated_matching',
    'run_m4_osseous_prior_matching',
]
