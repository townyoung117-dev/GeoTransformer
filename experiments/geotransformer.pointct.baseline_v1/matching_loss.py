"""M3-3 collision-aware group/set supervision for Point-CT assignment."""

import torch


class CollisionAwareMatchLossError(RuntimeError):
    pass


def _require_tensor(value, name: str):
    if not torch.is_tensor(value):
        raise CollisionAwareMatchLossError(f'{name} must be a torch.Tensor.')


def _require_structural_mask(mask, length: int, device, name: str):
    if mask is None:
        return torch.ones((length,), dtype=torch.bool, device=device)
    _require_tensor(mask, name)
    if mask.dtype != torch.bool:
        raise CollisionAwareMatchLossError(
            f'{name} must use bool dtype with True=valid and False=padding.'
        )
    if tuple(mask.shape) != (length,):
        raise CollisionAwareMatchLossError(
            f'{name} must have shape [{length}]; got {tuple(mask.shape)}.'
        )
    if mask.device != device:
        raise CollisionAwareMatchLossError(
            f'{name} must be on the same device as log_assignment.'
        )
    return mask


def compute_collision_aware_match_loss(
    log_assignment,
    gt_primary_ct_index,
    gt_primary_valid,
    point_valid_mask=None,
    ct_valid_mask=None,
):
    """Average one set-mass objective for every uniquely supervised CT token."""
    _require_tensor(log_assignment, 'log_assignment')
    if log_assignment.ndim != 2 or log_assignment.shape[0] <= 1 or log_assignment.shape[1] <= 1:
        raise CollisionAwareMatchLossError(
            'log_assignment must have shape [Np+1,Nv+1] with Np>0 and Nv>0.'
        )
    if log_assignment.dtype != torch.float32:
        raise CollisionAwareMatchLossError('log_assignment must use float32 dtype.')
    num_point = int(log_assignment.shape[0] - 1)
    num_ct = int(log_assignment.shape[1] - 1)

    _require_tensor(gt_primary_ct_index, 'gt_primary_ct_index')
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if gt_primary_ct_index.dtype not in integer_dtypes:
        raise CollisionAwareMatchLossError('gt_primary_ct_index must use an integer dtype.')
    if tuple(gt_primary_ct_index.shape) != (num_point,):
        raise CollisionAwareMatchLossError(
            f'gt_primary_ct_index must have shape [{num_point}]; '
            f'got {tuple(gt_primary_ct_index.shape)}.'
        )
    if gt_primary_ct_index.device != log_assignment.device:
        raise CollisionAwareMatchLossError(
            'gt_primary_ct_index must be on the same device as log_assignment.'
        )

    _require_tensor(gt_primary_valid, 'gt_primary_valid')
    if gt_primary_valid.dtype != torch.bool:
        raise CollisionAwareMatchLossError('gt_primary_valid must use bool dtype.')
    if tuple(gt_primary_valid.shape) != (num_point,):
        raise CollisionAwareMatchLossError(
            f'gt_primary_valid must have shape [{num_point}]; got {tuple(gt_primary_valid.shape)}.'
        )
    if gt_primary_valid.device != log_assignment.device:
        raise CollisionAwareMatchLossError(
            'gt_primary_valid must be on the same device as log_assignment.'
        )

    point_valid_mask = _require_structural_mask(
        point_valid_mask,
        num_point,
        log_assignment.device,
        'point_valid_mask',
    )
    ct_valid_mask = _require_structural_mask(
        ct_valid_mask,
        num_ct,
        log_assignment.device,
        'ct_valid_mask',
    )

    supervision_mask = gt_primary_valid & point_valid_mask
    supervised_point_indices = torch.nonzero(supervision_mask, as_tuple=False).flatten()
    if supervised_point_indices.numel() == 0:
        raise CollisionAwareMatchLossError('Collision-aware supervision contains zero valid Points.')

    primary_indices = gt_primary_ct_index[supervised_point_indices].to(dtype=torch.int64)
    if torch.any(primary_indices < 0) or torch.any(primary_indices >= num_ct):
        raise CollisionAwareMatchLossError(
            'A supervised Point has an out-of-range ordinary CT target index.'
        )
    if not bool(torch.all(ct_valid_mask[primary_indices])):
        raise CollisionAwareMatchLossError(
            'A supervised Point targets a structurally invalid CT token.'
        )

    ordinary_assignment = log_assignment[:num_point, :num_ct]
    selected_scores = ordinary_assignment[supervised_point_indices, primary_indices]
    if not bool(torch.isfinite(selected_scores).all()):
        raise CollisionAwareMatchLossError(
            'A log_assignment entry selected by valid supervision is NaN or Inf.'
        )

    unique_ct_indices, inverse_groups, group_sizes = torch.unique(
        primary_indices,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    if unique_ct_indices.numel() == 0:
        raise CollisionAwareMatchLossError('Collision-aware supervision contains zero CT groups.')

    group_losses = []
    for group_index in range(int(unique_ct_indices.numel())):
        group_scores = selected_scores[inverse_groups == group_index]
        group_losses.append(-torch.logsumexp(group_scores, dim=0))
    loss = torch.stack(group_losses).mean()
    if not bool(torch.isfinite(loss)):
        raise CollisionAwareMatchLossError('Collision-aware matching loss is not finite.')

    collision_group_sizes = group_sizes[group_sizes >= 2]
    num_supervised_points = int(supervised_point_indices.numel())
    num_supervised_groups = int(unique_ct_indices.numel())
    num_collision_groups = int(collision_group_sizes.numel())
    num_collision_points = int(collision_group_sizes.sum().item())
    num_excess_collision_points = int((group_sizes - 1).sum().item())
    max_group_size = int(group_sizes.max().item())

    return {
        'loss': loss,
        'num_supervised_points': num_supervised_points,
        'num_supervised_groups': num_supervised_groups,
        'num_collision_groups': num_collision_groups,
        'num_collision_points': num_collision_points,
        'num_excess_collision_points': num_excess_collision_points,
        'max_group_size': max_group_size,
    }


__all__ = [
    'CollisionAwareMatchLossError',
    'compute_collision_aware_match_loss',
]
