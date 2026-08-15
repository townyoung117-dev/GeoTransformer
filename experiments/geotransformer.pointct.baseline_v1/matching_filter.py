"""M3-4 dustbin-aware bidirectional mutual Point-CT token selection."""

import math

import torch


class DustbinAwareMatchingFilterError(RuntimeError):
    pass


def _require_structural_mask(mask, length: int, device, name: str) -> torch.Tensor:
    if mask is None:
        return torch.ones((length,), dtype=torch.bool, device=device)
    if not torch.is_tensor(mask):
        raise DustbinAwareMatchingFilterError(f'{name} must be a torch.Tensor.')
    if mask.dtype != torch.bool:
        raise DustbinAwareMatchingFilterError(
            f'{name} must use bool dtype with True=valid and False=padding.'
        )
    if tuple(mask.shape) != (length,):
        raise DustbinAwareMatchingFilterError(
            f'{name} must have shape [{length}]; got {tuple(mask.shape)}.'
        )
    if mask.device != device:
        raise DustbinAwareMatchingFilterError(
            f'{name} must be on the same device as log_assignment.'
        )
    return mask


def _require_min_confidence(value):
    if value is None:
        return None
    if isinstance(value, bool) or torch.is_tensor(value):
        raise DustbinAwareMatchingFilterError(
            'min_confidence must be a finite scalar in [0,1] or None.'
        )
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DustbinAwareMatchingFilterError(
            'min_confidence must be a finite scalar in [0,1] or None.'
        ) from error
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise DustbinAwareMatchingFilterError(
            'min_confidence must be a finite scalar in [0,1] or None.'
        )
    return threshold


def _validate_decision_scores(scores: torch.Tensor, direction: str):
    if bool(torch.isnan(scores).any()) or bool((scores == float('inf')).any()):
        raise DustbinAwareMatchingFilterError(
            f'{direction} decision candidates contain NaN or positive infinity.'
        )
    if scores.shape[0] > 0 and bool(torch.isneginf(scores).all(dim=1).any()):
        raise DustbinAwareMatchingFilterError(
            f'A valid {direction} token has no defined winner because all candidates are -Inf.'
        )


def extract_dustbin_aware_mutual_correspondences(
    log_assignment,
    point_valid_mask=None,
    ct_valid_mask=None,
    min_confidence=None,
):
    """Return sparse, point-index-sorted mutual ordinary token pairs.

    Each candidate list contains ascending valid ordinary indices followed by its
    dustbin, so ``torch.argmax`` ties use the deterministic first-index rule.
    """
    if not torch.is_tensor(log_assignment):
        raise DustbinAwareMatchingFilterError('log_assignment must be a torch.Tensor.')
    if (
        log_assignment.ndim != 2
        or log_assignment.shape[0] <= 1
        or log_assignment.shape[1] <= 1
    ):
        raise DustbinAwareMatchingFilterError(
            'log_assignment must have shape [Np+1,Nv+1] with Np>0 and Nv>0.'
        )
    if log_assignment.dtype != torch.float32:
        raise DustbinAwareMatchingFilterError('log_assignment must use float32 dtype.')

    num_point = int(log_assignment.shape[0] - 1)
    num_ct = int(log_assignment.shape[1] - 1)
    device = log_assignment.device
    point_valid_mask = _require_structural_mask(
        point_valid_mask,
        num_point,
        device,
        'point_valid_mask',
    )
    ct_valid_mask = _require_structural_mask(
        ct_valid_mask,
        num_ct,
        device,
        'ct_valid_mask',
    )
    min_confidence = _require_min_confidence(min_confidence)

    valid_point_indices = torch.nonzero(point_valid_mask, as_tuple=False).flatten()
    valid_ct_indices = torch.nonzero(ct_valid_mask, as_tuple=False).flatten()

    point_dustbin_index = valid_point_indices.new_tensor([num_point])
    ct_dustbin_index = valid_ct_indices.new_tensor([num_ct])
    point_candidate_indices = torch.cat([valid_point_indices, point_dustbin_index])
    ct_candidate_indices = torch.cat([valid_ct_indices, ct_dustbin_index])

    point_decision_scores = log_assignment.index_select(0, valid_point_indices).index_select(
        1,
        ct_candidate_indices,
    )
    _validate_decision_scores(point_decision_scores, 'Point')
    point_best_positions = torch.argmax(point_decision_scores, dim=1)
    point_best_indices = ct_candidate_indices[point_best_positions]

    ct_decision_scores = (
        log_assignment.index_select(0, point_candidate_indices)
        .index_select(1, valid_ct_indices)
        .transpose(0, 1)
    )
    _validate_decision_scores(ct_decision_scores, 'CT')
    ct_best_positions = torch.argmax(ct_decision_scores, dim=1)
    ct_best_indices = point_candidate_indices[ct_best_positions]

    point_ordinary_win_mask = point_best_indices != num_ct
    ordinary_point_indices = valid_point_indices[point_ordinary_win_mask]
    ordinary_ct_indices = point_best_indices[point_ordinary_win_mask]

    ct_best_by_index = torch.full(
        (num_ct,),
        num_point,
        dtype=torch.long,
        device=device,
    )
    ct_best_by_index[valid_ct_indices] = ct_best_indices
    mutual_mask = ct_best_by_index[ordinary_ct_indices] == ordinary_point_indices
    point_indices = ordinary_point_indices[mutual_mask]
    ct_indices = ordinary_ct_indices[mutual_mask]

    log_confidence = log_assignment[point_indices, ct_indices]
    confidence = torch.exp(log_confidence)
    if not bool(torch.isfinite(confidence).all()):
        raise DustbinAwareMatchingFilterError('Selected correspondence confidence is not finite.')

    num_mutual_before_confidence = int(point_indices.numel())
    if min_confidence is not None:
        confidence_mask = confidence >= min_confidence
        point_indices = point_indices[confidence_mask]
        ct_indices = ct_indices[confidence_mask]
        log_confidence = log_confidence[confidence_mask]
        confidence = confidence[confidence_mask]

    return {
        'point_indices': point_indices,
        'ct_indices': ct_indices,
        'log_confidence': log_confidence,
        'confidence': confidence,
        'num_valid_points': int(valid_point_indices.numel()),
        'num_valid_ct': int(valid_ct_indices.numel()),
        'num_point_dustbin_wins': int((point_best_indices == num_ct).sum().item()),
        'num_ct_dustbin_wins': int((ct_best_indices == num_point).sum().item()),
        'num_point_ordinary_wins': int(point_ordinary_win_mask.sum().item()),
        'num_ct_ordinary_wins': int((ct_best_indices != num_point).sum().item()),
        'num_mutual_before_confidence': num_mutual_before_confidence,
        'num_correspondences': int(point_indices.numel()),
    }


__all__ = [
    'DustbinAwareMatchingFilterError',
    'extract_dustbin_aware_mutual_correspondences',
]
