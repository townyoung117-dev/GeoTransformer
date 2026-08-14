"""M3-2 Point-CT descriptor similarity and rectangular log-domain transport."""

import math

import torch
import torch.nn as nn


POINT_CT_PROJECTED_DIM = 256
DEFAULT_NORM_EPSILON = 1e-12


class PointCTMatchingContractError(RuntimeError):
    pass


def _require_finite_scalar(value, name: str) -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise PointCTMatchingContractError(f'{name} must be a finite scalar.')
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PointCTMatchingContractError(f'{name} must be a finite scalar.') from error
    if not math.isfinite(scalar):
        raise PointCTMatchingContractError(f'{name} must be a finite scalar.')
    return scalar


def _require_positive_integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise PointCTMatchingContractError(f'{name} must be a positive integer.')
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PointCTMatchingContractError(f'{name} must be a positive integer.') from error
    if integer <= 0 or integer != value:
        raise PointCTMatchingContractError(f'{name} must be a positive integer; got {value!r}.')
    return integer


def _require_structural_mask(mask, shape, device, name: str) -> torch.Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not torch.is_tensor(mask):
        raise PointCTMatchingContractError(f'{name} must be a torch.Tensor.')
    if mask.dtype != torch.bool:
        raise PointCTMatchingContractError(f'{name} must use bool dtype with True=valid and False=padding.')
    if tuple(mask.shape) != tuple(shape):
        raise PointCTMatchingContractError(f'{name} must have shape {tuple(shape)}; got {tuple(mask.shape)}.')
    if mask.device != device:
        raise PointCTMatchingContractError(f'{name} must be on the same device as its scores/descriptors.')
    return mask


def compute_cross_modal_similarity(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature,
    *,
    norm_epsilon=DEFAULT_NORM_EPSILON,
) -> torch.Tensor:
    """Return float32 cosine logits ``q_hat @ k_hat.T / temperature``."""
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise PointCTMatchingContractError('Q and K must be torch.Tensor inputs.')
    if q.ndim != 2 or q.shape[1] != POINT_CT_PROJECTED_DIM:
        raise PointCTMatchingContractError(
            f'Q must have shape [Np,{POINT_CT_PROJECTED_DIM}]; got {tuple(q.shape)}.'
        )
    if k.ndim != 2 or k.shape[1] != POINT_CT_PROJECTED_DIM:
        raise PointCTMatchingContractError(
            f'K must have shape [Nv,{POINT_CT_PROJECTED_DIM}]; got {tuple(k.shape)}.'
        )
    if q.shape[0] == 0:
        raise PointCTMatchingContractError('Q must contain at least one Point token.')
    if k.shape[0] == 0:
        raise PointCTMatchingContractError('K must contain at least one CT token.')
    if not torch.is_floating_point(q) or not torch.is_floating_point(k):
        raise PointCTMatchingContractError('Q and K must be real floating-point tensors.')
    if q.device != k.device:
        raise PointCTMatchingContractError('Q and K must be on the same device.')

    temperature = _require_finite_scalar(temperature, 'temperature')
    if temperature <= 0.0:
        raise PointCTMatchingContractError('temperature must be greater than zero.')
    norm_epsilon = _require_finite_scalar(norm_epsilon, 'norm_epsilon')
    if norm_epsilon <= 0.0:
        raise PointCTMatchingContractError('norm_epsilon must be greater than zero.')

    q_float = q.to(dtype=torch.float32)
    k_float = k.to(dtype=torch.float32)
    if not torch.isfinite(q_float).all() or not torch.isfinite(k_float).all():
        raise PointCTMatchingContractError('Q and K must be finite in the float32 numerical path.')

    q_norm = torch.linalg.vector_norm(q_float, ord=2, dim=1, keepdim=True)
    k_norm = torch.linalg.vector_norm(k_float, ord=2, dim=1, keepdim=True)
    if not torch.isfinite(q_norm).all() or not torch.isfinite(k_norm).all():
        raise PointCTMatchingContractError('Q or K has a non-finite float32 L2 norm.')
    if torch.any(q_norm < norm_epsilon) or torch.any(k_norm < norm_epsilon):
        raise PointCTMatchingContractError(
            f'Every Q/K descriptor must have L2 norm >= {norm_epsilon:g}.'
        )

    q_normalized = q_float / q_norm
    k_normalized = k_float / k_norm
    similarity = torch.matmul(q_normalized, k_normalized.transpose(0, 1)) / temperature
    if similarity.dtype != torch.float32 or not torch.isfinite(similarity).all():
        raise PointCTMatchingContractError('Cross-modal similarity must be finite float32.')
    return similarity


class PointCTLogSinkhorn(nn.Module):
    """Device-safe rectangular log-domain transport with two dustbins."""

    def __init__(self, sinkhorn_iterations, alpha_init):
        super().__init__()
        self.sinkhorn_iterations = _require_positive_integer(
            sinkhorn_iterations,
            'sinkhorn_iterations',
        )
        alpha_init = _require_finite_scalar(alpha_init, 'alpha_init')
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def _log_sinkhorn_normalization(
        self,
        scores: torch.Tensor,
        log_mu: torch.Tensor,
        log_nu: torch.Tensor,
        row_valid: torch.Tensor,
        col_valid: torch.Tensor,
    ) -> torch.Tensor:
        negative_infinity = scores.new_tensor(float('-inf'))
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(self.sinkhorn_iterations):
            row_logsumexp = torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            row_logsumexp = torch.where(
                row_valid,
                row_logsumexp,
                torch.zeros_like(row_logsumexp),
            )
            u = torch.where(row_valid, log_mu - row_logsumexp, negative_infinity)

            col_logsumexp = torch.logsumexp(scores + u.unsqueeze(2), dim=1)
            col_logsumexp = torch.where(
                col_valid,
                col_logsumexp,
                torch.zeros_like(col_logsumexp),
            )
            v = torch.where(col_valid, log_nu - col_logsumexp, negative_infinity)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(
        self,
        scores: torch.Tensor,
        point_valid_mask: torch.Tensor = None,
        ct_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(scores):
            raise PointCTMatchingContractError('scores must be a torch.Tensor.')
        if scores.ndim != 3:
            raise PointCTMatchingContractError(
                f'scores must have shape [B,M,N]; got {tuple(scores.shape)}.'
            )
        batch_size, num_point, num_ct = scores.shape
        if batch_size <= 0 or num_point <= 0 or num_ct <= 0:
            raise PointCTMatchingContractError('scores requires B>=1, M>=1, and N>=1.')
        if not torch.is_floating_point(scores):
            raise PointCTMatchingContractError('scores must be a real floating-point tensor.')
        scores = scores.to(dtype=torch.float32)
        if not torch.isfinite(scores).all():
            raise PointCTMatchingContractError('scores must be finite in the float32 numerical path.')
        if self.alpha.dtype != torch.float32:
            raise PointCTMatchingContractError('alpha must remain a float32 parameter.')
        if self.alpha.device != scores.device:
            raise PointCTMatchingContractError('Sinkhorn module and scores must be on the same device.')
        if not torch.isfinite(self.alpha):
            raise PointCTMatchingContractError('alpha must remain finite.')

        point_valid_mask = _require_structural_mask(
            point_valid_mask,
            (batch_size, num_point),
            scores.device,
            'point_valid_mask',
        )
        ct_valid_mask = _require_structural_mask(
            ct_valid_mask,
            (batch_size, num_ct),
            scores.device,
            'ct_valid_mask',
        )
        num_valid_point = point_valid_mask.sum(dim=1, dtype=torch.float32)
        num_valid_ct = ct_valid_mask.sum(dim=1, dtype=torch.float32)
        if torch.any(num_valid_point == 0) or torch.any(num_valid_ct == 0):
            raise PointCTMatchingContractError(
                'Every batch item requires at least one valid Point token and one valid CT token.'
            )

        alpha = self.alpha.reshape(1, 1, 1)
        dustbin_col = alpha.expand(batch_size, num_point, 1)
        dustbin_row = alpha.expand(batch_size, 1, num_ct + 1)
        padded_scores = torch.cat(
            [torch.cat([scores, dustbin_col], dim=2), dustbin_row],
            dim=1,
        )

        dustbin_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=scores.device)
        padded_point_valid = torch.cat([point_valid_mask, dustbin_valid], dim=1)
        padded_ct_valid = torch.cat([ct_valid_mask, dustbin_valid], dim=1)
        transport_valid = padded_point_valid.unsqueeze(2) & padded_ct_valid.unsqueeze(1)
        negative_infinity = scores.new_tensor(float('-inf'))
        padded_scores = padded_scores.masked_fill(~transport_valid, negative_infinity)

        normalizer = -torch.log(num_valid_point + num_valid_ct)
        log_mu = torch.cat(
            [
                torch.where(
                    point_valid_mask,
                    normalizer.unsqueeze(1),
                    negative_infinity,
                ),
                (torch.log(num_valid_ct) + normalizer).unsqueeze(1),
            ],
            dim=1,
        )
        log_nu = torch.cat(
            [
                torch.where(
                    ct_valid_mask,
                    normalizer.unsqueeze(1),
                    negative_infinity,
                ),
                (torch.log(num_valid_point) + normalizer).unsqueeze(1),
            ],
            dim=1,
        )

        log_assignment = self._log_sinkhorn_normalization(
            padded_scores,
            log_mu,
            log_nu,
            padded_point_valid,
            padded_ct_valid,
        )
        log_assignment = log_assignment - normalizer[:, None, None]
        log_assignment = log_assignment.masked_fill(~transport_valid, negative_infinity)
        if log_assignment.dtype != torch.float32:
            raise PointCTMatchingContractError('log_assignment must be float32.')
        if not torch.isfinite(log_assignment.masked_select(transport_valid)).all():
            raise PointCTMatchingContractError('Sinkhorn produced a non-finite valid transport entry.')
        return log_assignment


def compute_marginal_residual(
    log_assignment: torch.Tensor,
    point_valid_mask: torch.Tensor,
    ct_valid_mask: torch.Tensor,
):
    """Return numerical QA residuals for the rescaled transport marginals."""
    if not torch.is_tensor(log_assignment) or log_assignment.ndim != 3:
        raise PointCTMatchingContractError('log_assignment must have shape [B,M+1,N+1].')
    if not torch.is_floating_point(log_assignment):
        raise PointCTMatchingContractError('log_assignment must be floating point.')
    batch_size, padded_point, padded_ct = log_assignment.shape
    if batch_size <= 0 or padded_point <= 1 or padded_ct <= 1:
        raise PointCTMatchingContractError('log_assignment must have B>=1, M>=1, and N>=1.')
    num_point = padded_point - 1
    num_ct = padded_ct - 1
    point_valid_mask = _require_structural_mask(
        point_valid_mask,
        (batch_size, num_point),
        log_assignment.device,
        'point_valid_mask',
    )
    ct_valid_mask = _require_structural_mask(
        ct_valid_mask,
        (batch_size, num_ct),
        log_assignment.device,
        'ct_valid_mask',
    )
    if torch.any(point_valid_mask.sum(dim=1) == 0) or torch.any(ct_valid_mask.sum(dim=1) == 0):
        raise PointCTMatchingContractError('Marginal diagnostics require a non-empty valid set on both sides.')
    if torch.isnan(log_assignment).any() or torch.any(log_assignment == float('inf')):
        raise PointCTMatchingContractError('log_assignment contains NaN or positive infinity.')

    probability = torch.exp(log_assignment.to(dtype=torch.float32))
    num_valid_point = point_valid_mask.sum(dim=1, dtype=torch.float32)
    num_valid_ct = ct_valid_mask.sum(dim=1, dtype=torch.float32)

    ordinary_point_residual = torch.abs(probability[:, :num_point, :].sum(dim=2) - 1.0)
    ordinary_point_residual = torch.where(
        point_valid_mask,
        ordinary_point_residual,
        torch.zeros_like(ordinary_point_residual),
    )
    dustbin_point_residual = torch.abs(probability[:, num_point, :].sum(dim=1) - num_valid_ct)
    row_residual = torch.maximum(
        ordinary_point_residual.amax(dim=1),
        dustbin_point_residual,
    )

    ordinary_ct_residual = torch.abs(probability[:, :, :num_ct].sum(dim=1) - 1.0)
    ordinary_ct_residual = torch.where(
        ct_valid_mask,
        ordinary_ct_residual,
        torch.zeros_like(ordinary_ct_residual),
    )
    dustbin_ct_residual = torch.abs(probability[:, :, num_ct].sum(dim=1) - num_valid_point)
    col_residual = torch.maximum(
        ordinary_ct_residual.amax(dim=1),
        dustbin_ct_residual,
    )
    return {
        'max_abs_row_residual': row_residual.amax(),
        'max_abs_col_residual': col_residual.amax(),
    }


class PointCTMatcher(nn.Module):
    """M3-2 unbatched Point-CT similarity and assignment wrapper."""

    def __init__(
        self,
        projected_dim,
        temperature,
        sinkhorn_iterations,
        alpha_init,
        *,
        norm_epsilon=DEFAULT_NORM_EPSILON,
    ):
        super().__init__()
        projected_dim = _require_positive_integer(projected_dim, 'projected_dim')
        if projected_dim != POINT_CT_PROJECTED_DIM:
            raise PointCTMatchingContractError(
                f'Point-CT projected_dim must remain {POINT_CT_PROJECTED_DIM}; got {projected_dim}.'
            )
        self.projected_dim = projected_dim
        self.temperature = _require_finite_scalar(temperature, 'temperature')
        if self.temperature <= 0.0:
            raise PointCTMatchingContractError('temperature must be greater than zero.')
        self.norm_epsilon = _require_finite_scalar(norm_epsilon, 'norm_epsilon')
        if self.norm_epsilon <= 0.0:
            raise PointCTMatchingContractError('norm_epsilon must be greater than zero.')
        self.transport = PointCTLogSinkhorn(sinkhorn_iterations, alpha_init)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        point_valid_mask: torch.Tensor = None,
        ct_valid_mask: torch.Tensor = None,
    ):
        similarity = compute_cross_modal_similarity(
            q,
            k,
            self.temperature,
            norm_epsilon=self.norm_epsilon,
        )
        point_valid_mask = _require_structural_mask(
            point_valid_mask,
            (q.shape[0],),
            q.device,
            'point_valid_mask',
        )
        ct_valid_mask = _require_structural_mask(
            ct_valid_mask,
            (k.shape[0],),
            k.device,
            'ct_valid_mask',
        )
        log_assignment = self.transport(
            similarity.unsqueeze(0),
            point_valid_mask.unsqueeze(0),
            ct_valid_mask.unsqueeze(0),
        ).squeeze(0)
        return {
            'similarity': similarity,
            'log_assignment': log_assignment,
            'point_valid_mask': point_valid_mask,
            'ct_valid_mask': ct_valid_mask,
        }
