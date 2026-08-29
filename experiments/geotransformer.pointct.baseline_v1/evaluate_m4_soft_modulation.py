"""Independent M4 hard-plus-soft inference entry.

The byte-frozen M3 evaluator and stable M4-1 hard-only evaluator remain
unchanged.  This entry reuses their preprocessing/validation boundaries and
the existing matching filter and weighted registration implementations.
"""

import math
import time
from functools import partial

from evaluate_m4_hard_constraint import (
    _require_intact_filter_output,
    _require_mapping,
    _synchronize,
    _temporary_evaluation_mode,
)
from evaluation import M3EvaluationContractError


def _require_soft_matching_output(
    torch,
    soft_output,
    *,
    point_valid_mask,
    ct_valid_mask,
    point_count,
    ct_count,
    device,
):
    soft_output = _require_mapping(soft_output, 'M4 soft matching output')
    required = (
        'log_assignment',
        'point_valid_mask',
        'ct_valid_mask',
        'point_reliability',
        'ct_reliability',
        'pair_reliability',
        'base_similarity',
        'modulated_similarity',
    )
    missing = [field for field in required if field not in soft_output]
    if missing:
        raise M3EvaluationContractError(
            f'M4 soft matching output is missing fields: {missing}.'
        )
    if soft_output['point_valid_mask'] is not point_valid_mask or not torch.equal(
        soft_output['point_valid_mask'],
        point_valid_mask,
    ):
        raise M3EvaluationContractError(
            'M4 soft matching did not preserve the resolved Point hard mask.'
        )
    if soft_output['ct_valid_mask'] is not ct_valid_mask or not torch.equal(
        soft_output['ct_valid_mask'],
        ct_valid_mask,
    ):
        raise M3EvaluationContractError(
            'M4 soft matching did not preserve the resolved CT hard mask.'
        )

    tensor_contracts = (
        ('base_similarity', (point_count, ct_count)),
        ('modulated_similarity', (point_count, ct_count)),
        ('point_reliability', (point_count,)),
        ('ct_reliability', (ct_count,)),
        ('pair_reliability', (point_count, ct_count)),
        ('log_assignment', (point_count + 1, ct_count + 1)),
    )
    for name, shape in tensor_contracts:
        value = soft_output[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise M3EvaluationContractError(
                f'M4 soft matching {name} has an invalid shape.'
            )
        if value.device != device or value.dtype != torch.float32:
            raise M3EvaluationContractError(
                f'M4 soft matching {name} must be float32 on the requested device.'
            )
        if name == 'log_assignment':
            if bool(torch.isnan(value).any()) or bool((value == float('inf')).any()):
                raise M3EvaluationContractError(
                    'M4 soft matching log_assignment contains NaN or positive infinity.'
                )
        elif not bool(torch.isfinite(value).all()):
            raise M3EvaluationContractError(
                f'M4 soft matching {name} contains NaN or Inf.'
            )
    if not bool(torch.all(soft_output['point_reliability'][~point_valid_mask] == 0)):
        raise M3EvaluationContractError(
            'An excluded Point token received nonzero soft reliability.'
        )
    if not bool(torch.all(soft_output['ct_reliability'][~ct_valid_mask] == 0)):
        raise M3EvaluationContractError(
            'An excluded CT token received nonzero soft reliability.'
        )
    return soft_output['log_assignment']


def run_m4_soft_modulation_inference(
    *,
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    evaluation_protocol,
    device,
    sigma_mm,
    strength,
) -> dict:
    """Run one strict M4 hard-plus-soft inference case."""
    import torch

    from dataset import m2_ct_collate_fn, m2_point_collate_fn
    from m4_hard_constraint import M4HardConstraintError, resolve_m4_matching_masks
    from m4_soft_modulation import (
        M4SoftModulationError,
        run_m4_soft_modulated_matching,
    )
    from matching_filter import extract_dustbin_aware_mutual_correspondences
    from registration import estimate_weighted_point_to_ct_transform
    from training import _prepare_model_inputs, _require_encoder_outputs

    evaluation_protocol = _require_mapping(
        evaluation_protocol,
        'evaluation_protocol',
    )
    if 'matching_filter_min_confidence' not in evaluation_protocol:
        raise M3EvaluationContractError(
            'evaluation_protocol is missing matching_filter_min_confidence.'
        )
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise M3EvaluationContractError(
            'device must identify a valid torch device.'
        ) from error

    point_collate_fn = partial(
        m2_point_collate_fn,
        enable_m4_defect_mapping=True,
    )
    ct_collate_fn = partial(
        m2_ct_collate_fn,
        enable_m4_defect_mapping=True,
    )
    point_input, ct_input = _prepare_model_inputs(
        sample,
        point_collate_fn,
        ct_collate_fn,
        device,
    )

    with torch.no_grad(), _temporary_evaluation_mode(
        torch,
        point_encoder,
        ct_encoder,
        matcher,
    ):
        _synchronize(torch, device)
        start = time.perf_counter()
        point_output = point_encoder(point_input)
        ct_output = ct_encoder(ct_input)
        q, k, point_physical, ct_physical = _require_encoder_outputs(
            point_output,
            ct_output,
            device,
        )
        try:
            hard_constraint = resolve_m4_matching_masks(
                point_output,
                ct_output,
                device,
            )
        except M4HardConstraintError as error:
            raise M3EvaluationContractError(
                f'M4 hard-constraint mask resolution failed: {error}'
            ) from error
        hard_constraint = _require_mapping(
            hard_constraint,
            'M4 hard-constraint mask resolution',
        )
        if hard_constraint.get('enabled') is not True:
            raise M3EvaluationContractError(
                'M4 soft evaluation requires complete Point and CT coarse '
                'mapping artifacts.'
            )
        point_valid_mask = hard_constraint['point_valid_mask']
        ct_valid_mask = hard_constraint['ct_valid_mask']
        try:
            soft_output = run_m4_soft_modulated_matching(
                q=q,
                k=k,
                point_coordinates_mm=point_physical,
                ct_coordinates_mm=ct_physical,
                point_valid_mask=point_valid_mask,
                ct_valid_mask=ct_valid_mask,
                sigma_mm=sigma_mm,
                strength=strength,
                matcher=matcher,
            )
        except M4SoftModulationError as error:
            raise M3EvaluationContractError(
                f'M4 soft-modulated matching failed: {error}'
            ) from error
        log_assignment = _require_soft_matching_output(
            torch,
            soft_output,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            point_count=int(q.shape[0]),
            ct_count=int(k.shape[0]),
            device=device,
        )

        filter_output = extract_dustbin_aware_mutual_correspondences(
            log_assignment,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            min_confidence=evaluation_protocol['matching_filter_min_confidence'],
        )
        (
            point_indices,
            ct_indices,
            confidence,
            correspondence_count,
        ) = _require_intact_filter_output(
            torch,
            filter_output,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            device=device,
        )
        registration_output = estimate_weighted_point_to_ct_transform(
            point_physical,
            ct_physical,
            point_indices,
            ct_indices,
            confidence,
        )
        registration_output = _require_mapping(
            registration_output,
            'weighted registration output',
        )
        if 'num_correspondences' not in registration_output or int(
            registration_output['num_correspondences']
        ) != correspondence_count:
            raise M3EvaluationContractError(
                'weighted registration correspondence count does not match '
                'mutual filtering.'
            )
        _synchronize(torch, device)
        runtime_ms = (time.perf_counter() - start) * 1000.0

    if not math.isfinite(runtime_ms) or runtime_ms < 0.0:
        raise M3EvaluationContractError('inference runtime is invalid.')
    diagnostic_fields = (
        'm4_soft_modulation_enabled',
        'm4_soft_modulation_active',
        'm4_soft_sigma_mm',
        'm4_soft_strength',
        'point_reliability_min',
        'point_reliability_mean',
        'point_reliability_max',
        'ct_reliability_min',
        'ct_reliability_mean',
        'ct_reliability_max',
        'pair_reliability_min',
        'pair_reliability_mean',
        'pair_reliability_max',
        'base_similarity_mean',
        'modulated_similarity_mean',
        'soft_similarity_changed',
    )
    return {
        'point_physical': point_physical,
        'ct_physical': ct_physical,
        'filter_output': filter_output,
        'registration_output': registration_output,
        'inference_runtime_ms': runtime_ms,
        'm4_hard_constraint_enabled': True,
        'm4_hard_constraint_active': True,
        **{field: soft_output[field] for field in diagnostic_fields},
        'point_valid_mask': point_valid_mask,
        'ct_valid_mask': ct_valid_mask,
        'point_reliability': soft_output['point_reliability'],
        'ct_reliability': soft_output['ct_reliability'],
        'pair_reliability': soft_output['pair_reliability'],
        'base_similarity': soft_output['base_similarity'],
        'modulated_similarity': soft_output['modulated_similarity'],
        'log_assignment': soft_output['log_assignment'],
        'point_total_tokens': hard_constraint['point_total_count'],
        'point_intact_tokens': hard_constraint['point_intact_count'],
        'point_excluded_tokens': hard_constraint['point_excluded_count'],
        'ct_total_tokens': hard_constraint['ct_total_count'],
        'ct_intact_tokens': hard_constraint['ct_intact_count'],
        'ct_excluded_tokens': hard_constraint['ct_excluded_count'],
        'm4_mask_provenance': {
            'point_source': 'point_encoder_output.point_intact_coarse',
            'ct_source': 'ct_encoder_output.ct_intact_coarse',
            'soft_point_mask_is_hard_point_mask': True,
            'soft_ct_mask_is_hard_ct_mask': True,
            'mutual_filter_uses_resolved_masks': True,
            'registration_correspondences_verified_intact': True,
        },
    }


__all__ = ['run_m4_soft_modulation_inference']
