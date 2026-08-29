"""M4-2A defect-proximity soft modulation for Point-CT matching.

This module leaves the frozen M3 descriptor similarity and matcher forward
definitions untouched.  It derives continuous reliabilities from the same
M4-1 hard masks, applies a genuinely pairwise penalty to the existing cosine
similarity, and then calls the existing masked Sinkhorn transport.
"""

import math
import torch

from matching import PointCTMatchingContractError, compute_cross_modal_similarity


class M4SoftModulationError(RuntimeError):
    """Raised when an M4 soft-modulation input violates its contract."""


def _require_finite_scalar(value, name: str) -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise M4SoftModulationError(f'{name} must be a finite scalar.')
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M4SoftModulationError(f'{name} must be a finite scalar.') from error
    if not math.isfinite(scalar):
        raise M4SoftModulationError(f'{name} must be a finite scalar.')
    return scalar


def _require_coordinates(coordinates, name: str) -> torch.Tensor:
    if not torch.is_tensor(coordinates):
        raise M4SoftModulationError(f'{name} must be a torch.Tensor.')
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] != 3:
        raise M4SoftModulationError(
            f'{name} must have non-empty shape [N,3]; got {tuple(coordinates.shape)}.'
        )
    if not torch.is_floating_point(coordinates):
        raise M4SoftModulationError(f'{name} must use a real floating-point dtype.')
    if coordinates.device.type == 'meta':
        raise M4SoftModulationError(f'{name} cannot be a meta tensor.')
    coordinates_float = coordinates.to(dtype=torch.float32)
    if not bool(torch.isfinite(coordinates_float).all()):
        raise M4SoftModulationError(f'{name} must contain finite physical-mm coordinates.')
    return coordinates_float


def _require_valid_mask(valid_mask, token_count: int, device, name: str) -> torch.Tensor:
    if not torch.is_tensor(valid_mask):
        raise M4SoftModulationError(f'{name} must be a torch.Tensor.')
    if valid_mask.dtype != torch.bool:
        raise M4SoftModulationError(f'{name} must use bool dtype.')
    if tuple(valid_mask.shape) != (token_count,):
        raise M4SoftModulationError(
            f'{name} must have exact token shape ({token_count},); '
            f'got {tuple(valid_mask.shape)}.'
        )
    if valid_mask.device != device:
        raise M4SoftModulationError(
            f'{name} must be on the same device as its physical coordinates.'
        )
    return valid_mask


def _require_reliability(reliability, length: int, device, dtype, name: str) -> torch.Tensor:
    if not torch.is_tensor(reliability):
        raise M4SoftModulationError(f'{name} must be a torch.Tensor.')
    if not torch.is_floating_point(reliability):
        raise M4SoftModulationError(f'{name} must use a real floating-point dtype.')
    if tuple(reliability.shape) != (length,):
        raise M4SoftModulationError(
            f'{name} must have exact token shape ({length},); got {tuple(reliability.shape)}.'
        )
    if reliability.device != device:
        raise M4SoftModulationError(f'{name} must be on the same device as base_similarity.')
    if reliability.dtype != dtype:
        raise M4SoftModulationError(f'{name} must use the same dtype as base_similarity.')
    if not bool(torch.isfinite(reliability).all()):
        raise M4SoftModulationError(f'{name} contains NaN or Inf.')
    if bool(torch.any(reliability < 0.0)) or bool(torch.any(reliability > 1.0)):
        raise M4SoftModulationError(f'{name} values must lie in [0,1].')
    return reliability


def _summary(tensor: torch.Tensor, prefix: str) -> dict:
    detached = tensor.detach()
    return {
        f'{prefix}_min': float(detached.amin().item()),
        f'{prefix}_mean': float(detached.mean().item()),
        f'{prefix}_max': float(detached.amax().item()),
    }


def compute_defect_proximity_reliability(
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma_mm,
) -> torch.Tensor:
    """Return reliability based on distance to the nearest hard-excluded token.

    For every valid token ``i``, ``r_i = 1 - exp(-0.5*(d_i/sigma_mm)^2)``.
    Hard-excluded tokens receive zero.  If no token is excluded, every valid
    token receives one; no synthetic defect location is introduced.
    """
    coordinates = _require_coordinates(coordinates, 'coordinates')
    valid_mask = _require_valid_mask(
        valid_mask,
        int(coordinates.shape[0]),
        coordinates.device,
        'valid_mask',
    )
    sigma_mm = _require_finite_scalar(sigma_mm, 'sigma_mm')
    if sigma_mm <= 0.0:
        raise M4SoftModulationError('sigma_mm must be greater than zero.')

    reliability = torch.zeros_like(coordinates[:, 0], dtype=torch.float32)
    if not bool(torch.any(~valid_mask)):
        return reliability.masked_fill(valid_mask, 1.0)
    if not bool(torch.any(valid_mask)):
        return reliability

    valid_coordinates = coordinates[valid_mask]
    excluded_coordinates = coordinates[~valid_mask]
    nearest_distance_mm = torch.cdist(
        valid_coordinates,
        excluded_coordinates,
        p=2.0,
    ).amin(dim=1)
    scaled_squared = torch.square(nearest_distance_mm / sigma_mm)
    valid_reliability = -torch.expm1(-0.5 * scaled_squared)
    valid_reliability = torch.clamp(valid_reliability, min=0.0, max=1.0)
    reliability = reliability.masked_scatter(valid_mask, valid_reliability)
    if not bool(torch.isfinite(reliability).all()):
        raise M4SoftModulationError('defect proximity reliability is not finite.')
    return reliability


def modulate_cross_modal_similarity(
    base_similarity: torch.Tensor,
    point_reliability: torch.Tensor,
    ct_reliability: torch.Tensor,
    strength,
) -> dict:
    """Apply ``S_soft = S_base - strength * (1 - r_p r_v^T)``."""
    if not torch.is_tensor(base_similarity):
        raise M4SoftModulationError('base_similarity must be a torch.Tensor.')
    if base_similarity.ndim != 2 or min(base_similarity.shape) <= 0:
        raise M4SoftModulationError(
            f'base_similarity must have non-empty shape [Np,Nv]; '
            f'got {tuple(base_similarity.shape)}.'
        )
    if not torch.is_floating_point(base_similarity):
        raise M4SoftModulationError('base_similarity must use a real floating-point dtype.')
    if base_similarity.device.type == 'meta':
        raise M4SoftModulationError('base_similarity cannot be a meta tensor.')
    if not bool(torch.isfinite(base_similarity).all()):
        raise M4SoftModulationError('base_similarity contains NaN or Inf.')
    point_reliability = _require_reliability(
        point_reliability,
        int(base_similarity.shape[0]),
        base_similarity.device,
        base_similarity.dtype,
        'point_reliability',
    )
    ct_reliability = _require_reliability(
        ct_reliability,
        int(base_similarity.shape[1]),
        base_similarity.device,
        base_similarity.dtype,
        'ct_reliability',
    )
    strength = _require_finite_scalar(strength, 'strength')
    if strength < 0.0:
        raise M4SoftModulationError('strength must be greater than or equal to zero.')

    pair_reliability = point_reliability.unsqueeze(1) * ct_reliability.unsqueeze(0)
    if strength == 0.0:
        # Preserve strict equality and the original autograd path at the
        # Hard-only degeneration point.
        modulated_similarity = base_similarity
    else:
        modulated_similarity = base_similarity - strength * (1.0 - pair_reliability)
    if not bool(torch.isfinite(modulated_similarity).all()):
        raise M4SoftModulationError('modulated similarity contains NaN or Inf.')
    return {
        'base_similarity': base_similarity,
        'modulated_similarity': modulated_similarity,
        'pair_reliability': pair_reliability,
        'soft_similarity_changed': not torch.equal(
            modulated_similarity,
            base_similarity,
        ),
    }


def run_m4_soft_modulated_matching(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    point_coordinates_mm: torch.Tensor,
    ct_coordinates_mm: torch.Tensor,
    point_valid_mask: torch.Tensor,
    ct_valid_mask: torch.Tensor,
    sigma_mm,
    strength,
    matcher,
) -> dict:
    """Run frozen M3 similarity, M4 soft modulation, and masked Sinkhorn."""
    if not hasattr(matcher, 'temperature') or not hasattr(matcher, 'norm_epsilon'):
        raise M4SoftModulationError(
            'matcher must expose the frozen temperature and norm_epsilon settings.'
        )
    transport = getattr(matcher, 'transport', None)
    if transport is None or not callable(transport):
        raise M4SoftModulationError('matcher must expose callable masked Sinkhorn transport.')

    point_coordinates_mm = _require_coordinates(
        point_coordinates_mm,
        'point_coordinates_mm',
    )
    ct_coordinates_mm = _require_coordinates(
        ct_coordinates_mm,
        'ct_coordinates_mm',
    )
    if not torch.is_tensor(q) or point_coordinates_mm.shape[0] != q.shape[0]:
        raise M4SoftModulationError('Q and Point physical-coordinate token counts differ.')
    if not torch.is_tensor(k) or ct_coordinates_mm.shape[0] != k.shape[0]:
        raise M4SoftModulationError('K and CT physical-coordinate token counts differ.')
    if point_coordinates_mm.device != q.device:
        raise M4SoftModulationError('Point physical coordinates and Q must share a device.')
    if ct_coordinates_mm.device != k.device:
        raise M4SoftModulationError('CT physical coordinates and K must share a device.')
    point_valid_mask = _require_valid_mask(
        point_valid_mask,
        int(q.shape[0]),
        q.device,
        'point_valid_mask',
    )
    ct_valid_mask = _require_valid_mask(
        ct_valid_mask,
        int(k.shape[0]),
        k.device,
        'ct_valid_mask',
    )
    if not bool(torch.any(point_valid_mask)) or not bool(torch.any(ct_valid_mask)):
        raise M4SoftModulationError(
            'soft-modulated matching requires at least one hard-valid token per modality.'
        )

    point_reliability = compute_defect_proximity_reliability(
        point_coordinates_mm,
        point_valid_mask,
        sigma_mm,
    )
    ct_reliability = compute_defect_proximity_reliability(
        ct_coordinates_mm,
        ct_valid_mask,
        sigma_mm,
    )
    try:
        base_similarity = compute_cross_modal_similarity(
            q,
            k,
            matcher.temperature,
            norm_epsilon=matcher.norm_epsilon,
        )
    except PointCTMatchingContractError as error:
        raise M4SoftModulationError(
            f'frozen cross-modal similarity failed: {error}'
        ) from error
    modulation = modulate_cross_modal_similarity(
        base_similarity,
        point_reliability,
        ct_reliability,
        strength,
    )
    try:
        log_assignment = transport(
            modulation['modulated_similarity'].unsqueeze(0),
            point_valid_mask.unsqueeze(0),
            ct_valid_mask.unsqueeze(0),
        ).squeeze(0)
    except PointCTMatchingContractError as error:
        raise M4SoftModulationError(f'masked Sinkhorn transport failed: {error}') from error
    expected_shape = (int(q.shape[0]) + 1, int(k.shape[0]) + 1)
    if tuple(log_assignment.shape) != expected_shape:
        raise M4SoftModulationError(
            f'masked Sinkhorn returned shape {tuple(log_assignment.shape)}; '
            f'expected {expected_shape}.'
        )

    sigma_mm = _require_finite_scalar(sigma_mm, 'sigma_mm')
    strength = _require_finite_scalar(strength, 'strength')
    output = {
        **modulation,
        'similarity': modulation['modulated_similarity'],
        'log_assignment': log_assignment,
        'point_reliability': point_reliability,
        'ct_reliability': ct_reliability,
        'point_valid_mask': point_valid_mask,
        'ct_valid_mask': ct_valid_mask,
        'm4_soft_modulation_enabled': True,
        'm4_soft_modulation_active': True,
        'm4_soft_sigma_mm': sigma_mm,
        'm4_soft_strength': strength,
        **_summary(point_reliability, 'point_reliability'),
        **_summary(ct_reliability, 'ct_reliability'),
        **_summary(modulation['pair_reliability'], 'pair_reliability'),
        'base_similarity_mean': float(base_similarity.detach().mean().item()),
        'modulated_similarity_mean': float(
            modulation['modulated_similarity'].detach().mean().item()
        ),
    }
    return output


__all__ = [
    'M4SoftModulationError',
    'compute_defect_proximity_reliability',
    'modulate_cross_modal_similarity',
    'run_m4_soft_modulated_matching',
]
