"""Frozen M3-7 Point-to-CT evaluation metrics and aggregation.

This module contains no matcher or registration implementation.  It evaluates
outputs from the frozen M3 pipeline in physical LPS millimetres and keeps every
solver failure in the aggregation denominator.
"""

import hashlib
import hmac
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EVALUATION_PROTOCOL_VERSION = 'm3_7_eval_v1'
REQUIRED_TRAINING_PROTOCOL_VERSION = 'm3_6b_5fold_v1'
REQUIRED_CHECKPOINT_BASENAME = 'best_val_loss.pt'
FORMAL_BEST_CHECKPOINTS = {
    'Fold1': {'epoch': 11, 'best_val_loss': 2.187397321065267},
    'Fold2': {'epoch': 17, 'best_val_loss': 2.3390124638875327},
    'Fold3': {'epoch': 19, 'best_val_loss': 2.2600975036621094},
    'Fold4': {'epoch': 14, 'best_val_loss': 2.5588058630625405},
    'Fold5': {'epoch': 18, 'best_val_loss': 3.0356096691555448},
}
REQUIRED_FORMAL_TRAINING = {
    'seed': 20260815,
    'learning_rate': 0.0003,
    'weight_decay': 0.0001,
    'batch_size': 1,
    'precision': 'fp32',
    'temperature': 0.10,
    'sinkhorn_iterations': 20,
    'alpha_init': 1.0,
    # Provenance-only because existing formal checkpoints do not store this field.
    'max_epochs': 20,
}

_EVALUATION_PROTOCOL_FIELDS = {
    'evaluation_protocol_version',
    'evaluation_protocol_hash',
    'required_training_protocol_version',
    'registration_rre_threshold_deg',
    'registration_rte_threshold_mm',
    'correspondence_inlier_threshold_mm',
    'matching_filter_min_confidence',
    'required_checkpoint_basename',
    'formal_best_checkpoints',
    'required_formal_training',
    'solver_failure_metric_policy',
    'zero_correspondence_policy',
    'runtime_scope',
}
_EXPECTED_EVALUATION_PROTOCOL = {
    'evaluation_protocol_version': EVALUATION_PROTOCOL_VERSION,
    'required_training_protocol_version': REQUIRED_TRAINING_PROTOCOL_VERSION,
    'registration_rre_threshold_deg': 5.0,
    'registration_rte_threshold_mm': 10.0,
    'correspondence_inlier_threshold_mm': 15.0,
    'matching_filter_min_confidence': None,
    'required_checkpoint_basename': REQUIRED_CHECKPOINT_BASENAME,
    'formal_best_checkpoints': FORMAL_BEST_CHECKPOINTS,
    'required_formal_training': REQUIRED_FORMAL_TRAINING,
    'solver_failure_metric_policy': 'null_rre_rte',
    'zero_correspondence_policy': (
        'zero_inliers_zero_ratio_registration_fail_closed'
    ),
    'runtime_scope': (
        'model_forward+matching+mutual_filtering+weighted_registration'
    ),
}
_ROTATION_VALIDATION_ATOL = 1e-5


class M3EvaluationContractError(RuntimeError):
    """Raised when an M3-7 evaluation input violates the frozen contract."""


def _reject_duplicate_json_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise M3EvaluationContractError(f'duplicate JSON object field: {key!r}.')
        output[key] = value
    return output


def _canonical_protocol_json(protocol: Mapping) -> str:
    if not isinstance(protocol, Mapping):
        raise M3EvaluationContractError('evaluation protocol must be a mapping.')
    hash_input = dict(protocol)
    hash_input.pop('evaluation_protocol_hash', None)
    try:
        return json.dumps(
            hash_input,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M3EvaluationContractError(
            'evaluation protocol must be canonical-JSON serializable.'
        ) from error


def compute_evaluation_protocol_hash(protocol: Mapping) -> str:
    """Return the SHA-256 hash excluding only ``evaluation_protocol_hash``."""
    canonical_json = _canonical_protocol_json(protocol)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def validate_evaluation_protocol(protocol: Mapping) -> Mapping:
    """Fail closed on any deviation from the frozen M3-7 protocol."""
    if not isinstance(protocol, Mapping):
        raise M3EvaluationContractError('evaluation protocol must be a JSON object.')
    actual_fields = set(protocol)
    missing = sorted(_EVALUATION_PROTOCOL_FIELDS.difference(actual_fields))
    unexpected = sorted(actual_fields.difference(_EVALUATION_PROTOCOL_FIELDS))
    if missing or unexpected:
        raise M3EvaluationContractError(
            'evaluation protocol fields mismatch; '
            f'missing={missing}, unexpected={unexpected}.'
        )

    stored_hash = protocol['evaluation_protocol_hash']
    if (
        not isinstance(stored_hash, str)
        or len(stored_hash) != 64
        or any(character not in '0123456789abcdef' for character in stored_hash)
    ):
        raise M3EvaluationContractError(
            'evaluation_protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    computed_hash = compute_evaluation_protocol_hash(protocol)
    if not hmac.compare_digest(stored_hash, computed_hash):
        raise M3EvaluationContractError(
            'evaluation_protocol_hash mismatch: '
            f'stored={stored_hash}, computed={computed_hash}.'
        )

    for field, expected in _EXPECTED_EVALUATION_PROTOCOL.items():
        value = protocol[field]
        if value != expected or type(value) is not type(expected):
            raise M3EvaluationContractError(
                f'{field} must be exactly {expected!r} for '
                f'{EVALUATION_PROTOCOL_VERSION}.'
            )
    return protocol


def load_evaluation_protocol(path) -> Mapping:
    """Load and validate an M3-7 evaluation protocol JSON file."""
    path = Path(path)
    if not path.is_file():
        raise M3EvaluationContractError(
            f'evaluation protocol does not exist: {path}.'
        )
    try:
        with path.open('r', encoding='utf-8') as handle:
            protocol = json.load(handle, object_pairs_hook=_reject_duplicate_json_fields)
    except M3EvaluationContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3EvaluationContractError(
            f'cannot load evaluation protocol {path}: {error}'
        ) from error
    return validate_evaluation_protocol(protocol)


def _as_float64_array(value, shape, name: str) -> np.ndarray:
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3EvaluationContractError(
            f'{name} must be a finite float64-compatible array with shape {shape}.'
        ) from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise M3EvaluationContractError(
            f'{name} must be a finite float64-compatible array with shape {shape}.'
        )
    return array


def _as_coordinate_matrix(value, name: str) -> np.ndarray:
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3EvaluationContractError(
            f'{name} must be a finite coordinate matrix with shape [N,3].'
        ) from error
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise M3EvaluationContractError(
            f'{name} must be a finite coordinate matrix with shape [N,3].'
        )
    return array


def _as_index_vector(value, name: str) -> np.ndarray:
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in 'iu':
        raise M3EvaluationContractError(f'{name} must be an integer vector with shape [C].')
    return array.astype(np.int64, copy=False)


def _as_transform(value, name: str = 'gt_transform') -> np.ndarray:
    transform = _as_float64_array(value, (4, 4), name)
    if not np.allclose(
        transform[3],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise M3EvaluationContractError(f'{name} must be a homogeneous rigid transform.')
    if not _valid_rotation(transform[:3, :3]):
        raise M3EvaluationContractError(f'{name} rotation must be in SO(3).')
    return transform


def _require_finite_nonnegative(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise M3EvaluationContractError(f'{name} must be finite and non-negative.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3EvaluationContractError(
            f'{name} must be finite and non-negative.'
        ) from error
    if not math.isfinite(number) or number < 0.0:
        raise M3EvaluationContractError(f'{name} must be finite and non-negative.')
    return number


def rotation_error_degrees(predicted_rotation, ground_truth_rotation) -> float:
    """Compute float64 RRE from ``R_pred @ R_gt.T`` with acos clamping."""
    predicted = _as_float64_array(predicted_rotation, (3, 3), 'predicted_rotation')
    ground_truth = _as_float64_array(
        ground_truth_rotation,
        (3, 3),
        'ground_truth_rotation',
    )
    rotation_error = predicted @ ground_truth.T
    cosine = float((np.trace(rotation_error) - 1.0) / 2.0)
    cosine = min(1.0, max(-1.0, cosine))
    result = math.degrees(math.acos(cosine))
    if not math.isfinite(result):
        raise M3EvaluationContractError('RRE calculation produced a non-finite result.')
    return result


def translation_error_mm(predicted_translation, ground_truth_translation) -> float:
    """Compute float64 Euclidean translation error in millimetres."""
    predicted = _as_float64_array(predicted_translation, (3,), 'predicted_translation')
    ground_truth = _as_float64_array(
        ground_truth_translation,
        (3,),
        'ground_truth_translation',
    )
    result = float(np.linalg.norm(predicted - ground_truth))
    if not math.isfinite(result):
        raise M3EvaluationContractError('RTE calculation produced a non-finite result.')
    return result


def correspondence_inlier_metrics(
    point_physical,
    ct_physical,
    point_indices,
    ct_indices,
    ground_truth_transform,
    inlier_threshold_mm,
) -> dict:
    """Evaluate predicted Point/CT pairs using only the effective Point->CT GT."""
    points = _as_coordinate_matrix(point_physical, 'point_physical')
    ct_points = _as_coordinate_matrix(ct_physical, 'ct_physical')
    point_indices = _as_index_vector(point_indices, 'point_indices')
    ct_indices = _as_index_vector(ct_indices, 'ct_indices')
    if point_indices.shape != ct_indices.shape:
        raise M3EvaluationContractError(
            'point_indices and ct_indices must have the same shape [C].'
        )
    if np.any(point_indices < 0) or np.any(point_indices >= points.shape[0]):
        raise M3EvaluationContractError('point_indices contains an out-of-range index.')
    if np.any(ct_indices < 0) or np.any(ct_indices >= ct_points.shape[0]):
        raise M3EvaluationContractError('ct_indices contains an out-of-range index.')
    transform = _as_transform(ground_truth_transform)
    threshold = _require_finite_nonnegative(
        inlier_threshold_mm,
        'inlier_threshold_mm',
    )

    num_correspondences = int(point_indices.size)
    if num_correspondences == 0:
        return {
            'num_correspondences': 0,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
        }

    transformed_points = (
        points[point_indices] @ transform[:3, :3].T + transform[:3, 3]
    )
    distances = np.linalg.norm(transformed_points - ct_points[ct_indices], axis=1)
    if not np.all(np.isfinite(distances)):
        raise M3EvaluationContractError(
            'correspondence distance calculation produced NaN or Inf.'
        )
    num_inliers = int(np.count_nonzero(distances <= threshold))
    return {
        'num_correspondences': num_correspondences,
        'num_inliers': num_inliers,
        'inlier_ratio': float(num_inliers / num_correspondences),
    }


def _valid_rotation(rotation: np.ndarray) -> bool:
    identity = np.eye(3, dtype=np.float64)
    return bool(
        np.allclose(
            rotation.T @ rotation,
            identity,
            rtol=0.0,
            atol=_ROTATION_VALIDATION_ATOL,
        )
        and np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=_ROTATION_VALIDATION_ATOL,
        )
    )


def _registration_failure(status: str) -> dict:
    return {
        'solver_success': False,
        'solver_status': status,
        'rre_deg': None,
        'rte_mm': None,
        'registration_recall_hit': False,
    }


def evaluate_registration_result(
    registration_result,
    ground_truth_transform,
    rre_threshold_deg,
    rte_threshold_mm,
) -> dict:
    """Evaluate a fail-closed weighted-registration result.

    A solver-declared failure never receives a fabricated transform or zero
    error.  A solver-declared success with an invalid transform is converted to
    the explicit fail-closed status ``invalid_registration_output``.
    """
    if not isinstance(registration_result, Mapping):
        raise M3EvaluationContractError('registration_result must be a mapping.')
    success = registration_result.get('success')
    if not isinstance(success, (bool, np.bool_)):
        raise M3EvaluationContractError(
            'registration_result.success must be a boolean.'
        )
    if not bool(success):
        failure_reason = registration_result.get('failure_reason')
        status = (
            failure_reason
            if isinstance(failure_reason, str) and failure_reason.strip()
            else 'solver_failure'
        )
        return _registration_failure(status)

    transform = _as_transform(ground_truth_transform)
    try:
        rotation = _as_float64_array(
            registration_result.get('rotation'),
            (3, 3),
            'registration rotation',
        )
        translation = _as_float64_array(
            registration_result.get('translation'),
            (3,),
            'registration translation',
        )
    except M3EvaluationContractError:
        return _registration_failure('invalid_registration_output')
    if not _valid_rotation(rotation):
        return _registration_failure('invalid_registration_output')

    rre_threshold = _require_finite_nonnegative(
        rre_threshold_deg,
        'rre_threshold_deg',
    )
    rte_threshold = _require_finite_nonnegative(
        rte_threshold_mm,
        'rte_threshold_mm',
    )
    rre = rotation_error_degrees(rotation, transform[:3, :3])
    rte = translation_error_mm(translation, transform[:3, 3])
    return {
        'solver_success': True,
        'solver_status': 'success',
        'rre_deg': rre,
        'rte_mm': rte,
        'registration_recall_hit': bool(
            rre <= rre_threshold and rte <= rte_threshold
        ),
    }


def _mean_median(values: Sequence, *, sample_count_name: str) -> dict:
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise M3EvaluationContractError('metric aggregation received NaN or Inf.')
    return {
        sample_count_name: len(normalized),
        'mean': statistics.fmean(normalized) if normalized else None,
        'median': statistics.median(normalized) if normalized else None,
    }


def aggregate_runtime_metrics(cases: Sequence) -> dict:
    """Aggregate all case runtimes without excluding failures or slow cases."""
    runtimes = [
        _require_finite_nonnegative(case['inference_runtime_ms'], 'inference_runtime_ms')
        for case in cases
    ]
    return {
        'sample_count': len(runtimes),
        'mean_ms': statistics.fmean(runtimes) if runtimes else None,
        'median_ms': statistics.median(runtimes) if runtimes else None,
        'p95_ms': float(np.percentile(runtimes, 95)) if runtimes else None,
    }


def _aggregate_case_group(cases: Sequence) -> dict:
    cases = list(cases)
    successful = [case for case in cases if case['solver_success'] is True]
    failed = [case for case in cases if case['solver_success'] is False]
    if len(successful) + len(failed) != len(cases):
        raise M3EvaluationContractError('every case requires a boolean solver_success.')
    hits = [case for case in cases if case['registration_recall_hit'] is True]
    if any(
        case['registration_recall_hit'] not in (True, False)
        for case in cases
    ):
        raise M3EvaluationContractError(
            'every case requires a boolean registration_recall_hit.'
        )
    if any(case['registration_recall_hit'] is True for case in failed):
        raise M3EvaluationContractError(
            'solver-failure cases cannot be registration recall hits.'
        )
    for case in successful:
        if case.get('rre_deg') is None or case.get('rte_mm') is None:
            raise M3EvaluationContractError(
                'solver-success cases require numeric RRE and RTE.'
            )
        _require_finite_nonnegative(case['rre_deg'], 'rre_deg')
        _require_finite_nonnegative(case['rte_mm'], 'rte_mm')
    for case in failed:
        if case.get('rre_deg') is not None or case.get('rte_mm') is not None:
            raise M3EvaluationContractError(
                'solver-failure cases must keep RRE and RTE null.'
            )
    inlier_ratios = [
        _require_finite_nonnegative(case['inlier_ratio'], 'inlier_ratio')
        for case in cases
    ]
    if any(value > 1.0 for value in inlier_ratios):
        raise M3EvaluationContractError('inlier_ratio must be in [0,1].')

    total = len(cases)
    return {
        'total_cases': total,
        'solver_success_cases': len(successful),
        'solver_failure_cases': len(failed),
        'solver_success_rate': float(len(successful) / total) if total else 0.0,
        'solver_failure_rate': float(len(failed) / total) if total else 0.0,
        'registration_recall': float(len(hits) / total) if total else 0.0,
        'rre_deg': _mean_median(
            [case['rre_deg'] for case in successful],
            sample_count_name='success_sample_count',
        ),
        'rte_mm': _mean_median(
            [case['rte_mm'] for case in successful],
            sample_count_name='success_sample_count',
        ),
        'inlier_ratio': _mean_median(
            inlier_ratios,
            sample_count_name='sample_count',
        ),
        'runtime': aggregate_runtime_metrics(cases),
    }


def _aggregate_by_field(cases: Sequence, field: str) -> dict:
    groups = defaultdict(list)
    for case in cases:
        value = case.get(field)
        if not isinstance(value, str) or not value:
            raise M3EvaluationContractError(f'every case requires a non-empty {field}.')
        groups[value].append(case)
    return {
        key: _aggregate_case_group(groups[key])
        for key in sorted(groups)
    }


def aggregate_severity_metrics(cases: Sequence) -> dict:
    """Return one complete aggregate for each perturbation severity."""
    return _aggregate_by_field(cases, 'severity')


def aggregate_subject_metrics(cases: Sequence) -> dict:
    """Return one complete aggregate for each test subject."""
    return _aggregate_by_field(cases, 'subject_id')


def aggregate_case_metrics(cases: Sequence) -> dict:
    """Aggregate global, per-severity, and per-subject M3-7 metrics."""
    cases = list(cases)
    summary = _aggregate_case_group(cases)
    summary['per_severity'] = aggregate_severity_metrics(cases)
    summary['per_subject'] = aggregate_subject_metrics(cases)
    return summary


__all__ = [
    'EVALUATION_PROTOCOL_VERSION',
    'FORMAL_BEST_CHECKPOINTS',
    'M3EvaluationContractError',
    'REQUIRED_CHECKPOINT_BASENAME',
    'REQUIRED_FORMAL_TRAINING',
    'REQUIRED_TRAINING_PROTOCOL_VERSION',
    'aggregate_case_metrics',
    'aggregate_runtime_metrics',
    'aggregate_severity_metrics',
    'aggregate_subject_metrics',
    'compute_evaluation_protocol_hash',
    'correspondence_inlier_metrics',
    'evaluate_registration_result',
    'load_evaluation_protocol',
    'rotation_error_degrees',
    'translation_error_mm',
    'validate_evaluation_protocol',
]
