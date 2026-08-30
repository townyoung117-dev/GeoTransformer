"""Independent M4 hard-soft-osseous inference entry.

The frozen M3 evaluator and the existing M4-1/M4-2A evaluators remain
unchanged.  This entry owns the complete M4-2C inference path while reusing
the established preprocessing, mask resolution, mutual filtering, intact
verification, and weighted registration boundaries.
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


_RAW_OSSEOUS_FIELDS = (
    'ct_volume',
    'ct_spacing',
    'ct_origin',
    'ct_direction',
)


def _require_raw_osseous_inputs(sample):
    """Select only the defective-CT fields allowed by the score contract."""
    sample = _require_mapping(sample, 'sample')
    missing = [field for field in _RAW_OSSEOUS_FIELDS if field not in sample]
    if missing:
        raise M3EvaluationContractError(
            f'M4 osseous evaluation sample is missing defective CT fields: {missing}.'
        )
    return {field: sample[field] for field in _RAW_OSSEOUS_FIELDS}


def _require_aligned_ct_support(torch, ct_input, ct_physical, device):
    """Require the collated score locations to be the actual matcher tokens."""
    ct_input = _require_mapping(ct_input, 'CT encoder input')
    if 'ct_support_phys_20mm' not in ct_input:
        raise M3EvaluationContractError(
            'CT encoder input is missing ct_support_phys_20mm.'
        )
    support = ct_input['ct_support_phys_20mm']
    if not torch.is_tensor(support):
        raise M3EvaluationContractError(
            'ct_support_phys_20mm must be a torch.Tensor.'
        )
    if support.device != device or not torch.is_floating_point(support):
        raise M3EvaluationContractError(
            'ct_support_phys_20mm must be floating point on the requested device.'
        )
    if tuple(support.shape) != tuple(ct_physical.shape):
        raise M3EvaluationContractError(
            'ct_support_phys_20mm and Xv_phys_coarse token counts differ.'
        )
    support_float32 = support.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(support_float32).all()):
        raise M3EvaluationContractError(
            'ct_support_phys_20mm contains NaN or Inf.'
        )
    if not torch.equal(support_float32, ct_physical.detach()):
        raise M3EvaluationContractError(
            'ct_support_phys_20mm must exactly match Xv_phys_coarse after '
            'float32 conversion, including token order.'
        )
    return support


def _require_bool_field(output, name):
    value = output[name]
    if not isinstance(value, bool):
        raise M3EvaluationContractError(
            f'M4 osseous matching {name} must be bool.'
        )
    return value


def _require_finite_diagnostic(output, name):
    value = output[name]
    if isinstance(value, bool):
        raise M3EvaluationContractError(
            f'M4 osseous matching {name} must be a finite scalar.'
        )
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3EvaluationContractError(
            f'M4 osseous matching {name} must be a finite scalar.'
        ) from error
    if not math.isfinite(value):
        raise M3EvaluationContractError(
            f'M4 osseous matching {name} must be a finite scalar.'
        )
    return value


def _require_osseous_matching_output(
    torch,
    output,
    *,
    point_valid_mask,
    ct_valid_mask,
    point_count,
    ct_count,
    device,
):
    """Validate the complete hard-soft-osseous matching result."""
    output = _require_mapping(output, 'M4 osseous matching output')
    tensor_shapes = {
        'base_similarity': (point_count, ct_count),
        'soft_similarity': (point_count, ct_count),
        'final_similarity': (point_count, ct_count),
        'similarity': (point_count, ct_count),
        'modulated_similarity': (point_count, ct_count),
        'point_reliability': (point_count,),
        'ct_reliability': (ct_count,),
        'pair_reliability': (point_count, ct_count),
        'soft_pair_reliability': (point_count, ct_count),
        'osseous_pair_reliability': (point_count, ct_count),
        'osseous_penalty': (point_count, ct_count),
        'osseous_score': (ct_count,),
        'log_assignment': (point_count + 1, ct_count + 1),
    }
    mask_fields = ('point_valid_mask', 'ct_valid_mask')
    bool_fields = (
        'm4_hard_constraint_active',
        'm4_soft_modulation_enabled',
        'm4_soft_modulation_active',
        'm4_osseous_prior_enabled',
        'm4_osseous_prior_active',
        'soft_similarity_changed',
        'osseous_similarity_changed',
    )
    scalar_fields = (
        'm4_soft_sigma_mm',
        'm4_soft_strength',
        'osseous_strength',
        'm4_osseous_strength',
        'osseous_center_hu',
        'osseous_tau_hu',
        'osseous_radius_mm',
        'osseous_score_min',
        'osseous_score_mean',
        'osseous_score_max',
        'osseous_pair_reliability_min',
        'osseous_pair_reliability_mean',
        'osseous_pair_reliability_max',
        'osseous_penalty_min',
        'osseous_penalty_mean',
        'osseous_penalty_max',
        'final_modulation_min',
        'final_modulation_mean',
        'final_modulation_max',
    )
    required = tuple(tensor_shapes) + mask_fields + bool_fields + scalar_fields
    missing = [field for field in required if field not in output]
    if missing:
        raise M3EvaluationContractError(
            f'M4 osseous matching output is missing fields: {missing}.'
        )

    for name, resolved_mask in (
        ('point_valid_mask', point_valid_mask),
        ('ct_valid_mask', ct_valid_mask),
    ):
        if output[name] is not resolved_mask or not torch.equal(
            output[name], resolved_mask
        ):
            raise M3EvaluationContractError(
                f'M4 osseous matching did not preserve the resolved {name}.'
            )

    for name, shape in tensor_shapes.items():
        value = output[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise M3EvaluationContractError(
                f'M4 osseous matching {name} has an invalid shape.'
            )
        if value.device != device or value.dtype != torch.float32:
            raise M3EvaluationContractError(
                f'M4 osseous matching {name} must be float32 on the requested device.'
            )
        if name == 'log_assignment':
            if bool(torch.isnan(value).any()) or bool(
                (value == float('inf')).any()
            ):
                raise M3EvaluationContractError(
                    'M4 osseous matching log_assignment contains NaN or '
                    'positive infinity.'
                )
        elif not bool(torch.isfinite(value).all()):
            raise M3EvaluationContractError(
                f'M4 osseous matching {name} contains NaN or Inf.'
            )

    for name in bool_fields:
        _require_bool_field(output, name)
    for name in scalar_fields:
        _require_finite_diagnostic(output, name)

    if not torch.equal(output['similarity'], output['final_similarity']) or not torch.equal(
        output['modulated_similarity'], output['final_similarity']
    ):
        raise M3EvaluationContractError(
            'M4 osseous matching final-similarity aliases are inconsistent.'
        )
    if not torch.equal(
        output['soft_pair_reliability'], output['pair_reliability']
    ):
        raise M3EvaluationContractError(
            'M4 osseous matching soft pair-reliability aliases are inconsistent.'
        )

    unit_tensors = (
        'point_reliability',
        'ct_reliability',
        'pair_reliability',
        'soft_pair_reliability',
        'osseous_pair_reliability',
        'osseous_score',
    )
    for name in unit_tensors:
        value = output[name]
        if bool(torch.any(value < 0.0)) or bool(torch.any(value > 1.0)):
            raise M3EvaluationContractError(
                f'M4 osseous matching {name} values must lie in [0,1].'
            )
    if not bool(torch.all(output['point_reliability'][~point_valid_mask] == 0)):
        raise M3EvaluationContractError(
            'An excluded Point token received nonzero soft reliability.'
        )
    if not bool(torch.all(output['ct_reliability'][~ct_valid_mask] == 0)):
        raise M3EvaluationContractError(
            'An excluded CT token received nonzero soft reliability.'
        )

    valid_pairs = point_valid_mask.unsqueeze(1) & ct_valid_mask.unsqueeze(0)
    if not bool(torch.all(output['pair_reliability'][~valid_pairs] == 0)) or not bool(
        torch.all(output['osseous_pair_reliability'][~valid_pairs] == 0)
    ):
        raise M3EvaluationContractError(
            'A hard-invalid pair received nonzero pair reliability.'
        )
    real_log_assignment = output['log_assignment'][:-1, :-1]
    if not bool(torch.all(torch.isneginf(real_log_assignment[~valid_pairs]))):
        raise M3EvaluationContractError(
            'A hard-invalid pair was restored by osseous-integrated Sinkhorn.'
        )

    if output['m4_hard_constraint_active'] is not True:
        raise M3EvaluationContractError(
            'M4 osseous matching must keep the hard constraint active.'
        )
    if output['m4_soft_modulation_enabled'] is not True or output[
        'm4_soft_modulation_active'
    ] is not True:
        raise M3EvaluationContractError(
            'M4 osseous matching must keep M4-2A soft modulation active.'
        )
    if output['m4_osseous_prior_enabled'] is not True or output[
        'm4_osseous_prior_active'
    ] is not True:
        raise M3EvaluationContractError(
            'M4 osseous matching did not keep the osseous path active.'
        )
    if output['osseous_center_hu'] != 300.0 or output[
        'osseous_tau_hu'
    ] != 100.0 or output['osseous_radius_mm'] != 20.0:
        raise M3EvaluationContractError(
            'M4 osseous matching changed the frozen score parameters.'
        )
    return output['log_assignment']


def run_m4_osseous_prior_inference(
    *,
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    evaluation_protocol,
    device,
    sigma_mm,
    soft_strength,
    osseous_strength,
) -> dict:
    """Run one strict hard-soft-osseous Point-CT inference case."""
    import torch

    from dataset import m2_ct_collate_fn, m2_point_collate_fn
    from m4_hard_constraint import M4HardConstraintError, resolve_m4_matching_masks
    from m4_osseous_integration import (
        M4OsseousIntegrationError,
        run_m4_osseous_integrated_matching,
    )
    from matching_filter import extract_dustbin_aware_mutual_correspondences
    from registration import estimate_weighted_point_to_ct_transform
    from training import _prepare_model_inputs, _require_encoder_outputs

    raw_osseous_inputs = _require_raw_osseous_inputs(sample)
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
        support_locations = _require_aligned_ct_support(
            torch,
            ct_input,
            ct_physical,
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
                'M4 osseous evaluation requires complete Point and CT coarse '
                'mapping artifacts.'
            )
        point_valid_mask = hard_constraint['point_valid_mask']
        ct_valid_mask = hard_constraint['ct_valid_mask']

        try:
            osseous_output = run_m4_osseous_integrated_matching(
                q=q,
                k=k,
                point_coordinates_mm=point_physical,
                ct_token_locations_mm=ct_physical,
                point_valid_mask=point_valid_mask,
                ct_valid_mask=ct_valid_mask,
                sigma_mm=sigma_mm,
                soft_strength=soft_strength,
                osseous_strength=osseous_strength,
                matcher=matcher,
                defective_ct_volume=raw_osseous_inputs['ct_volume'],
                ct_spacing=raw_osseous_inputs['ct_spacing'],
                ct_origin=raw_osseous_inputs['ct_origin'],
                ct_direction=raw_osseous_inputs['ct_direction'],
                support_locations_mm=support_locations,
            )
        except M4OsseousIntegrationError as error:
            raise M3EvaluationContractError(
                f'M4 osseous-integrated matching failed: {error}'
            ) from error
        log_assignment = _require_osseous_matching_output(
            torch,
            osseous_output,
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
        'm4_osseous_prior_enabled',
        'm4_osseous_prior_active',
        'm4_soft_sigma_mm',
        'm4_soft_strength',
        'osseous_strength',
        'm4_osseous_strength',
        'osseous_center_hu',
        'osseous_tau_hu',
        'osseous_radius_mm',
        'point_reliability_min',
        'point_reliability_mean',
        'point_reliability_max',
        'ct_reliability_min',
        'ct_reliability_mean',
        'ct_reliability_max',
        'pair_reliability_min',
        'pair_reliability_mean',
        'pair_reliability_max',
        'osseous_score_min',
        'osseous_score_mean',
        'osseous_score_max',
        'osseous_pair_reliability_min',
        'osseous_pair_reliability_mean',
        'osseous_pair_reliability_max',
        'osseous_penalty_min',
        'osseous_penalty_mean',
        'osseous_penalty_max',
        'final_modulation_min',
        'final_modulation_mean',
        'final_modulation_max',
        'base_similarity_mean',
        'soft_similarity_mean',
        'final_similarity_mean',
        'soft_similarity_changed',
        'osseous_similarity_changed',
    )
    return {
        'point_physical': point_physical,
        'ct_physical': ct_physical,
        'filter_output': filter_output,
        'registration_output': registration_output,
        'inference_runtime_ms': runtime_ms,
        'm4_hard_constraint_enabled': True,
        'm4_hard_constraint_active': True,
        **{field: osseous_output[field] for field in diagnostic_fields},
        'point_valid_mask': point_valid_mask,
        'ct_valid_mask': ct_valid_mask,
        'point_reliability': osseous_output['point_reliability'],
        'ct_reliability': osseous_output['ct_reliability'],
        'pair_reliability': osseous_output['pair_reliability'],
        'soft_pair_reliability': osseous_output['soft_pair_reliability'],
        'osseous_score': osseous_output['osseous_score'],
        'osseous_pair_reliability': osseous_output[
            'osseous_pair_reliability'
        ],
        'osseous_penalty': osseous_output['osseous_penalty'],
        'base_similarity': osseous_output['base_similarity'],
        'soft_similarity': osseous_output['soft_similarity'],
        'final_similarity': osseous_output['final_similarity'],
        'similarity': osseous_output['similarity'],
        'modulated_similarity': osseous_output['modulated_similarity'],
        'log_assignment': osseous_output['log_assignment'],
        'point_total_tokens': hard_constraint['point_total_count'],
        'point_intact_tokens': hard_constraint['point_intact_count'],
        'point_excluded_tokens': hard_constraint['point_excluded_count'],
        'ct_total_tokens': hard_constraint['ct_total_count'],
        'ct_intact_tokens': hard_constraint['ct_intact_count'],
        'ct_excluded_tokens': hard_constraint['ct_excluded_count'],
        'm4_mask_provenance': {
            'point_source': 'point_encoder_output.point_intact_coarse',
            'ct_source': 'ct_encoder_output.ct_intact_coarse',
            'osseous_point_mask_is_hard_point_mask': True,
            'osseous_ct_mask_is_hard_ct_mask': True,
            'mutual_filter_uses_resolved_masks': True,
            'registration_correspondences_verified_intact': True,
        },
        'm4_osseous_provenance': {
            'score_volume_source': 'sample.ct_volume (defective CT)',
            'score_header_sources': (
                'sample.ct_spacing',
                'sample.ct_origin',
                'sample.ct_direction',
            ),
            'support_source': 'ct_input.ct_support_phys_20mm',
            'matcher_token_source': 'ct_output.Xv_phys_coarse',
            'support_to_matcher_token_alignment': 'exact_float32',
            'support_to_matcher_token_alignment_verified': True,
            'complete_counterpart_used': False,
            'ground_truth_used': False,
            'identity_used': False,
        },
    }


run_m4_osseous_inference = run_m4_osseous_prior_inference


__all__ = [
    'run_m4_osseous_inference',
    'run_m4_osseous_prior_inference',
]
