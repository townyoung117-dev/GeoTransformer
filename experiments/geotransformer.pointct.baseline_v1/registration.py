"""M3-5 fail-closed weighted Point-to-CT rigid registration."""

import torch


# Floating-point guards only; neither constant is a registration-quality threshold.
FLOAT32_WEIGHT_SUM_EPSILON = torch.finfo(torch.float32).eps
ROTATION_SANITY_TOLERANCE = 1e-4


class PointCTRegistrationContractError(RuntimeError):
    """Raised when an input violates the M3-5 registration contract."""


def _require_physical_coordinates(value, name: str) -> None:
    if not torch.is_tensor(value):
        raise PointCTRegistrationContractError(f'{name} must be a torch.Tensor.')
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] <= 0:
        raise PointCTRegistrationContractError(
            f'{name} must have shape [N,3] with N>0; got {tuple(value.shape)}.'
        )
    if value.dtype != torch.float32:
        raise PointCTRegistrationContractError(f'{name} must use float32 dtype.')
    if not bool(torch.isfinite(value).all()):
        raise PointCTRegistrationContractError(f'{name} must contain only finite values.')


def _require_index_vector(value, name: str) -> None:
    if not torch.is_tensor(value):
        raise PointCTRegistrationContractError(f'{name} must be a torch.Tensor.')
    if value.ndim != 1:
        raise PointCTRegistrationContractError(
            f'{name} must have shape [C]; got {tuple(value.shape)}.'
        )
    if value.dtype != torch.long:
        raise PointCTRegistrationContractError(f'{name} must use torch.long dtype.')


def _require_weight_vector(value) -> None:
    if not torch.is_tensor(value):
        raise PointCTRegistrationContractError('weights must be a torch.Tensor.')
    if value.ndim != 1:
        raise PointCTRegistrationContractError(
            f'weights must have shape [C]; got {tuple(value.shape)}.'
        )
    if value.dtype != torch.float32:
        raise PointCTRegistrationContractError('weights must use float32 dtype.')
    if not bool(torch.isfinite(value).all()):
        raise PointCTRegistrationContractError('weights must contain only finite values.')
    if bool((value < 0).any()):
        raise PointCTRegistrationContractError('weights must be nonnegative.')


def _result(
    success: bool,
    failure_reason,
    num_correspondences: int,
    num_positive_weights: int,
    weight_sum: float,
    source_rank=None,
    target_rank=None,
    covariance_rank=None,
    rotation=None,
    translation=None,
    det_rotation=None,
):
    return {
        'success': success,
        'rotation': rotation,
        'translation': translation,
        'failure_reason': failure_reason,
        'num_correspondences': num_correspondences,
        'num_positive_weights': num_positive_weights,
        'weight_sum': weight_sum,
        'source_rank': source_rank,
        'target_rank': target_rank,
        'covariance_rank': covariance_rank,
        'det_rotation': det_rotation,
    }


def estimate_weighted_point_to_ct_transform(
    point_physical,
    ct_physical,
    point_indices,
    ct_indices,
    weights,
):
    """Estimate ``Y = X @ R.T + t`` in the physical LPS-mm coordinate frame."""
    _require_physical_coordinates(point_physical, 'point_physical')
    _require_physical_coordinates(ct_physical, 'ct_physical')
    _require_index_vector(point_indices, 'point_indices')
    _require_index_vector(ct_indices, 'ct_indices')
    _require_weight_vector(weights)

    num_correspondences = int(point_indices.numel())
    if ct_indices.numel() != num_correspondences or weights.numel() != num_correspondences:
        raise PointCTRegistrationContractError(
            'point_indices, ct_indices, and weights must have the same shape [C].'
        )

    device = point_physical.device
    inputs = (ct_physical, point_indices, ct_indices, weights)
    if any(value.device != device for value in inputs):
        raise PointCTRegistrationContractError('All inputs must be on the same device.')

    if bool((point_indices < 0).any()) or bool((point_indices >= point_physical.shape[0]).any()):
        raise PointCTRegistrationContractError('point_indices contains an out-of-range index.')
    if bool((ct_indices < 0).any()) or bool((ct_indices >= ct_physical.shape[0]).any()):
        raise PointCTRegistrationContractError('ct_indices contains an out-of-range index.')

    positive_mask = weights > 0
    num_positive_weights = int(positive_mask.sum().item())
    weight_sum_tensor = weights.sum()
    weight_sum = float(weight_sum_tensor.item())
    diagnostics = {
        'num_correspondences': num_correspondences,
        'num_positive_weights': num_positive_weights,
        'weight_sum': weight_sum,
    }

    if num_correspondences < 3:
        return _result(False, 'insufficient_correspondences', **diagnostics)
    if torch.unique(point_indices).numel() != num_correspondences:
        return _result(False, 'duplicate_point_indices', **diagnostics)
    if torch.unique(ct_indices).numel() != num_correspondences:
        return _result(False, 'duplicate_ct_indices', **diagnostics)
    if num_positive_weights < 3:
        return _result(False, 'insufficient_positive_weights', **diagnostics)
    if (
        not bool(torch.isfinite(weight_sum_tensor))
        or not bool(weight_sum_tensor > FLOAT32_WEIGHT_SUM_EPSILON)
    ):
        return _result(False, 'nonpositive_weight_sum', **diagnostics)

    source = point_physical[point_indices]
    target = ct_physical[ct_indices]
    source_centroid = (weights[:, None] * source).sum(dim=0) / weight_sum_tensor
    target_centroid = (weights[:, None] * target).sum(dim=0) / weight_sum_tensor
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    if not bool(torch.isfinite(source_centroid).all()) or not bool(
        torch.isfinite(source_centered).all()
    ):
        return _result(False, 'degenerate_source_geometry', **diagnostics)
    if not bool(torch.isfinite(target_centroid).all()) or not bool(
        torch.isfinite(target_centered).all()
    ):
        return _result(False, 'degenerate_target_geometry', **diagnostics)

    positive_source_centered = source_centered[positive_mask]
    positive_target_centered = target_centered[positive_mask]
    try:
        source_rank = int(torch.linalg.matrix_rank(positive_source_centered).item())
    except RuntimeError:
        return _result(False, 'degenerate_source_geometry', **diagnostics)
    try:
        target_rank = int(torch.linalg.matrix_rank(positive_target_centered).item())
    except RuntimeError:
        return _result(
            False,
            'degenerate_target_geometry',
            **diagnostics,
            source_rank=source_rank,
        )

    covariance = source_centered.transpose(0, 1) @ (weights[:, None] * target_centered)
    covariance_rank = None
    if bool(torch.isfinite(covariance).all()):
        try:
            covariance_rank = int(torch.linalg.matrix_rank(covariance).item())
        except RuntimeError:
            covariance_rank = None

    rank_diagnostics = {
        **diagnostics,
        'source_rank': source_rank,
        'target_rank': target_rank,
        'covariance_rank': covariance_rank,
    }
    if source_rank < 2:
        return _result(False, 'degenerate_source_geometry', **rank_diagnostics)
    if target_rank < 2:
        return _result(False, 'degenerate_target_geometry', **rank_diagnostics)
    if covariance_rank is None or covariance_rank < 2:
        return _result(False, 'degenerate_covariance', **rank_diagnostics)

    try:
        left_vectors, _, right_vectors_h = torch.linalg.svd(covariance)
    except RuntimeError:
        return _result(False, 'svd_failure', **rank_diagnostics)

    right_vectors = right_vectors_h.transpose(0, 1)
    uncorrected_rotation = right_vectors @ left_vectors.transpose(0, 1)
    determinant_sign = torch.sign(torch.linalg.det(uncorrected_rotation))
    correction = torch.diag(
        torch.stack(
            (
                determinant_sign.new_ones(()),
                determinant_sign.new_ones(()),
                determinant_sign,
            )
        )
    )
    rotation = right_vectors @ correction @ left_vectors.transpose(0, 1)
    translation = target_centroid - source_centroid @ rotation.transpose(0, 1)

    det_rotation_tensor = torch.linalg.det(rotation)
    det_rotation = (
        float(det_rotation_tensor.item()) if bool(torch.isfinite(det_rotation_tensor)) else None
    )
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    rotation_is_valid = (
        bool(torch.isfinite(rotation).all())
        and torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            identity,
            rtol=ROTATION_SANITY_TOLERANCE,
            atol=ROTATION_SANITY_TOLERANCE,
        )
        and det_rotation is not None
        and abs(det_rotation - 1.0) <= ROTATION_SANITY_TOLERANCE
    )
    if not rotation_is_valid:
        return _result(
            False,
            'invalid_rotation',
            **rank_diagnostics,
            det_rotation=det_rotation,
        )
    if not bool(torch.isfinite(translation).all()):
        return _result(
            False,
            'invalid_translation',
            **rank_diagnostics,
            det_rotation=det_rotation,
        )

    return _result(
        True,
        None,
        **rank_diagnostics,
        rotation=rotation,
        translation=translation,
        det_rotation=det_rotation,
    )


__all__ = [
    'PointCTRegistrationContractError',
    'estimate_weighted_point_to_ct_transform',
]
