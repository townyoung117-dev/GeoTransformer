"""Independent, origin-invariant diagnostics for the frozen M3 defect baseline.

This module observes transforms and source coordinates emitted by the frozen
pipeline.  It does not perform inference, matching, registration, thresholding,
or prediction repair.  All metric arithmetic is deterministic NumPy float64.
"""

import hashlib
import hmac
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


DIAGNOSTIC_PROTOCOL_VERSION = 'm3_metric_diagnostic_v1'
DIAGNOSTIC_PROTOCOL_HASH = (
    '3807dee00a551840d5a293bbb715e40d281855a6fef375c4872b6a25fb3d016b'
)
CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION = 'm3_metric_diagnostic_clean10_v2'
CLEAN10_DIAGNOSTIC_PROTOCOL_HASH = (
    '5f40c322433c2274d356213969f9041d2931ff0e058efb0c541bcf5e1a9624bc'
)
SOURCE_TRAINING_PROTOCOL_VERSION = 'm3_6b_5fold_v1'
SOURCE_TRAINING_PROTOCOL_HASH = (
    'c0c635c1cb897ce33d15ba3ed58abdb12a8c5a8feb6a82fe51910146e8a99fac'
)
SOURCE_DEFECT_EVALUATION_PROTOCOL_VERSION = 'm3_defect_eval_v1'
SOURCE_DEFECT_EVALUATION_PROTOCOL_HASH = (
    '66b93d340a79947295ba1b75738291001192b9848f3cc48a8144492915fd49fe'
)
POINT_TRE_PERCENTILE = 95
POINT_TRE_PERCENTILE_METHOD = 'linear'
CATASTROPHIC_ROTATION_THRESHOLD_DEG = 45.0
EXPECTED_PATIENT_COUNT = 11
EXPECTED_DEFECT_CONDITION_COUNT = 5
EXPECTED_DEFECT_INSTANCE_COUNT = 55
EXPECTED_TEST_CASE_COUNT = 825
EXPECTED_PAT6_CASE_COUNT = 75
EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD = {
    'Fold1': 225,
    'Fold2': 150,
    'Fold3': 150,
    'Fold4': 150,
    'Fold5': 150,
}
CLEAN10_EXPECTED_PATIENT_COUNT = 10
CLEAN10_EXPECTED_DEFECT_CONDITION_COUNT = 5
CLEAN10_EXPECTED_DEFECT_INSTANCE_COUNT = 50
CLEAN10_EXPECTED_TEST_CASE_COUNT = 750
CLEAN10_EXPECTED_PAT6_CASE_COUNT = 0
CLEAN10_EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD = {
    'Fold1': 150,
    'Fold2': 150,
    'Fold3': 150,
    'Fold4': 150,
    'Fold5': 150,
}
SO3_ATOL = 1e-4
HOMOGENEOUS_ATOL = 1e-12

# Repeated execution of the same frozen float32 GPU inference should agree much
# more closely than a clinically meaningful millimetre or degree.  These named
# tolerances are used only to cross-check against the legacy formal JSONL.
LEGACY_CONTINUOUS_ABS_TOLERANCE = 1e-6
LEGACY_CONTINUOUS_REL_TOLERANCE = 1e-7

REQUIRED_CASE_IDENTITY_FIELDS = (
    'fold_id',
    'subject_id',
    'defect_id',
    'severity',
    'variant_id',
    'perturbation_seed',
)
DIAGNOSTIC_TRE_FIELDS = (
    'centroid_tre_mm',
    'point_tre_mean_mm',
    'point_tre_median_mm',
    'point_tre_rmse_mm',
    'point_tre_p95_mm',
    'point_tre_max_mm',
)
CONFIDENCE_FIELDS = (
    'confidence_mean',
    'confidence_median',
    'confidence_p05',
    'confidence_p95',
    'confidence_min',
    'confidence_max',
)

_EXPECTED_DIAGNOSTIC_PROTOCOL = {
    'diagnostic_protocol_version': DIAGNOSTIC_PROTOCOL_VERSION,
    'source_training_protocol_version': SOURCE_TRAINING_PROTOCOL_VERSION,
    'source_training_protocol_hash': SOURCE_TRAINING_PROTOCOL_HASH,
    'source_defect_evaluation_protocol_version': (
        SOURCE_DEFECT_EVALUATION_PROTOCOL_VERSION
    ),
    'source_defect_evaluation_protocol_hash': (
        SOURCE_DEFECT_EVALUATION_PROTOCOL_HASH
    ),
    'source_reference_points': (
        'augmented_defective_point_physical_coordinates'
    ),
    'centroid_tre_definition': (
        'norm((R_pred*c+t_pred)-(R_gt*c+t_gt)), '
        'c=mean(source_reference_points)'
    ),
    'point_tre_definition': (
        'd_i=norm((R_pred*x_i+t_pred)-(R_gt*x_i+t_gt)); '
        'report mean,median,rmse,p95,max over all source_reference_points'
    ),
    'legacy_parameter_rte_definition': 'norm(t_pred-t_gt)',
    'origin_rotation_lever_proxy_definition': (
        '2*norm(source_centroid)*sin(rre_rad/2)'
    ),
    'origin_rotation_lever_proxy_policy': (
        'diagnostic_proxy_only_NOT_registration_accuracy_metric'
    ),
    'point_tre_percentile': POINT_TRE_PERCENTILE,
    'point_tre_percentile_method': POINT_TRE_PERCENTILE_METHOD,
    'solver_failure_policy': 'null_diagnostic_metrics',
    'catastrophic_rotation_threshold_deg': (
        CATASTROPHIC_ROTATION_THRESHOLD_DEG
    ),
    'required_case_identity_fields': list(REQUIRED_CASE_IDENTITY_FIELDS),
    'legacy_continuous_abs_tolerance': LEGACY_CONTINUOUS_ABS_TOLERANCE,
    'legacy_continuous_rel_tolerance': LEGACY_CONTINUOUS_REL_TOLERANCE,
    'legacy_case_authoritative_source': (
        'strict_union_of_Fold1_through_Fold5_cases_jsonl'
    ),
    'legacy_root_cases_policy': (
        'optional_complete_union_or_exact_single_fold_invocation_artifact'
    ),
    'expected_patient_count': EXPECTED_PATIENT_COUNT,
    'expected_defect_condition_count': EXPECTED_DEFECT_CONDITION_COUNT,
    'expected_defect_instance_count': EXPECTED_DEFECT_INSTANCE_COUNT,
    'expected_test_case_count': EXPECTED_TEST_CASE_COUNT,
    'diagnostic_replaces_frozen_m3_metrics': False,
    'diagnostic_changes_registration_success_threshold': False,
    'diagnostic_changes_model_predictions': False,
}
_CLEAN10_EXPECTED_DIAGNOSTIC_PROTOCOL = {
    **_EXPECTED_DIAGNOSTIC_PROTOCOL,
    'diagnostic_protocol_version': CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION,
    'source_training_protocol_version': 'm3_6b_5fold_clean10_v2',
    'source_training_protocol_hash': (
        '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'
    ),
    'source_defect_evaluation_protocol_version': (
        'm3_defect_eval_clean10_v2'
    ),
    'source_defect_evaluation_protocol_hash': (
        'cfd519b923be3b623cffec8c8f5830cb5b1160859461180eddeb7134a0adb6b0'
    ),
    'expected_patient_count': CLEAN10_EXPECTED_PATIENT_COUNT,
    'expected_defect_condition_count': (
        CLEAN10_EXPECTED_DEFECT_CONDITION_COUNT
    ),
    'expected_defect_instance_count': CLEAN10_EXPECTED_DEFECT_INSTANCE_COUNT,
    'expected_test_case_count': CLEAN10_EXPECTED_TEST_CASE_COUNT,
}
_DIAGNOSTIC_PROTOCOL_CONTRACTS = {
    DIAGNOSTIC_PROTOCOL_VERSION: {
        'diagnostic_protocol_hash': DIAGNOSTIC_PROTOCOL_HASH,
        'expected': _EXPECTED_DIAGNOSTIC_PROTOCOL,
        'pat6_forensic_required': True,
        'expected_pat6_case_count': EXPECTED_PAT6_CASE_COUNT,
        'legacy_complete_union_status': 'complete_union_verified',
    },
    CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION: {
        'diagnostic_protocol_hash': CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
        'expected': _CLEAN10_EXPECTED_DIAGNOSTIC_PROTOCOL,
        'pat6_forensic_required': False,
        'expected_pat6_case_count': CLEAN10_EXPECTED_PAT6_CASE_COUNT,
        'legacy_complete_union_status': 'complete_union',
    },
}
SUPPORTED_DIAGNOSTIC_PROTOCOL_VERSIONS = tuple(
    _DIAGNOSTIC_PROTOCOL_CONTRACTS
)
_DIAGNOSTIC_PROTOCOL_FIELDS = set(_EXPECTED_DIAGNOSTIC_PROTOCOL) | {
    'diagnostic_protocol_hash'
}


class M3MetricDiagnosticContractError(RuntimeError):
    """Raised when an independent diagnostic contract is violated."""


def _reject_duplicate_json_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise M3MetricDiagnosticContractError(
                f'duplicate JSON object field: {key!r}.'
            )
        output[key] = value
    return output


def _canonical_protocol_json(protocol: Mapping) -> str:
    if not isinstance(protocol, Mapping):
        raise M3MetricDiagnosticContractError(
            'diagnostic protocol must be a mapping.'
        )
    hash_input = dict(protocol)
    hash_input.pop('diagnostic_protocol_hash', None)
    try:
        return json.dumps(
            hash_input,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M3MetricDiagnosticContractError(
            'diagnostic protocol must be canonical-JSON serializable.'
        ) from error


def compute_diagnostic_protocol_hash(protocol: Mapping) -> str:
    """Hash every diagnostic protocol field except its stored hash."""
    canonical = _canonical_protocol_json(protocol)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _diagnostic_contract_for_version(
    diagnostic_protocol_version: str,
) -> Mapping:
    try:
        return _DIAGNOSTIC_PROTOCOL_CONTRACTS[diagnostic_protocol_version]
    except (KeyError, TypeError) as error:
        raise M3MetricDiagnosticContractError(
            'unsupported diagnostic_protocol_version '
            f'{diagnostic_protocol_version!r}; expected one of '
            f'{list(SUPPORTED_DIAGNOSTIC_PROTOCOL_VERSIONS)}.'
        ) from error


def get_diagnostic_protocol_contract(protocol_or_version) -> dict:
    """Return a defensive runtime contract for one supported diagnostic."""
    if isinstance(protocol_or_version, Mapping):
        validate_diagnostic_protocol(protocol_or_version)
        version = protocol_or_version['diagnostic_protocol_version']
    else:
        version = protocol_or_version
    contract = _diagnostic_contract_for_version(version)
    expected = contract['expected']
    return {
        'diagnostic_protocol_version': version,
        'diagnostic_protocol_hash': contract['diagnostic_protocol_hash'],
        'expected_patient_count': expected['expected_patient_count'],
        'expected_defect_condition_count': expected[
            'expected_defect_condition_count'
        ],
        'expected_defect_instance_count': expected[
            'expected_defect_instance_count'
        ],
        'expected_test_case_count': expected['expected_test_case_count'],
        'pat6_forensic_required': contract['pat6_forensic_required'],
        'expected_pat6_case_count': contract['expected_pat6_case_count'],
        'legacy_complete_union_status': contract[
            'legacy_complete_union_status'
        ],
    }


def validate_diagnostic_protocol(protocol: Mapping) -> Mapping:
    """Fail closed on any deviation from the versioned diagnostic contract."""
    if not isinstance(protocol, Mapping):
        raise M3MetricDiagnosticContractError(
            'diagnostic protocol must be a JSON object.'
        )
    actual_fields = set(protocol)
    missing = sorted(_DIAGNOSTIC_PROTOCOL_FIELDS.difference(actual_fields))
    unexpected = sorted(actual_fields.difference(_DIAGNOSTIC_PROTOCOL_FIELDS))
    if missing or unexpected:
        raise M3MetricDiagnosticContractError(
            'diagnostic protocol fields mismatch; '
            f'missing={missing}, unexpected={unexpected}.'
        )
    contract = _diagnostic_contract_for_version(
        protocol.get('diagnostic_protocol_version')
    )
    stored_hash = protocol['diagnostic_protocol_hash']
    if (
        not isinstance(stored_hash, str)
        or len(stored_hash) != 64
        or any(character not in '0123456789abcdef' for character in stored_hash)
    ):
        raise M3MetricDiagnosticContractError(
            'diagnostic_protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    computed_hash = compute_diagnostic_protocol_hash(protocol)
    if not hmac.compare_digest(stored_hash, computed_hash):
        raise M3MetricDiagnosticContractError(
            'diagnostic_protocol_hash mismatch: '
            f'stored={stored_hash}, computed={computed_hash}.'
        )
    if not hmac.compare_digest(
        stored_hash,
        contract['diagnostic_protocol_hash'],
    ):
        raise M3MetricDiagnosticContractError(
            'diagnostic_protocol_hash is not the frozen hash for '
            f'{protocol["diagnostic_protocol_version"]}: '
            f'stored={stored_hash}, '
            f'expected={contract["diagnostic_protocol_hash"]}.'
        )
    for field, expected in contract['expected'].items():
        actual = protocol[field]
        if actual != expected or type(actual) is not type(expected):
            raise M3MetricDiagnosticContractError(
                f'{field} must be exactly {expected!r} for '
                f'{protocol["diagnostic_protocol_version"]}.'
            )
    return protocol


def load_diagnostic_protocol(path) -> Mapping:
    """Load diagnostic JSON with duplicate-key rejection and strict validation."""
    path = Path(path)
    if not path.is_file():
        raise M3MetricDiagnosticContractError(
            f'diagnostic protocol does not exist: {path}.'
        )
    try:
        with path.open('r', encoding='utf-8') as handle:
            protocol = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_fields,
            )
    except M3MetricDiagnosticContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3MetricDiagnosticContractError(
            f'cannot load diagnostic protocol {path}: {error}'
        ) from error
    return validate_diagnostic_protocol(protocol)


def validate_source_protocols(
    diagnostic_protocol: Mapping,
    training_protocol: Mapping,
    defect_evaluation_protocol: Mapping,
) -> None:
    """Bind the diagnostic to the exact frozen source protocol pair."""
    validate_diagnostic_protocol(diagnostic_protocol)
    if not isinstance(training_protocol, Mapping):
        raise M3MetricDiagnosticContractError(
            'source training protocol must be a mapping.'
        )
    if not isinstance(defect_evaluation_protocol, Mapping):
        raise M3MetricDiagnosticContractError(
            'source defect evaluation protocol must be a mapping.'
        )
    # A matching stored version/hash is insufficient: validate both source
    # manifests canonically and against their own frozen versioned contracts.
    try:
        from defect_evaluation import (
            validate_defect_evaluation_protocol,
            validate_protocol_pair,
        )
        from training_protocol import validate_training_protocol

        validate_training_protocol(training_protocol)
        validate_defect_evaluation_protocol(defect_evaluation_protocol)
        validate_protocol_pair(training_protocol, defect_evaluation_protocol)
    except Exception as error:
        if isinstance(error, M3MetricDiagnosticContractError):
            raise
        raise M3MetricDiagnosticContractError(
            f'source protocol canonical validation failed: {error}'
        ) from error
    comparisons = (
        (
            training_protocol.get('protocol_version'),
            diagnostic_protocol['source_training_protocol_version'],
            'source training protocol version',
        ),
        (
            training_protocol.get('protocol_hash'),
            diagnostic_protocol['source_training_protocol_hash'],
            'source training protocol hash',
        ),
        (
            defect_evaluation_protocol.get('evaluation_protocol_version'),
            diagnostic_protocol[
                'source_defect_evaluation_protocol_version'
            ],
            'source defect evaluation protocol version',
        ),
        (
            defect_evaluation_protocol.get('evaluation_protocol_hash'),
            diagnostic_protocol['source_defect_evaluation_protocol_hash'],
            'source defect evaluation protocol hash',
        ),
    )
    for actual, expected, name in comparisons:
        if actual != expected or type(actual) is not type(expected):
            raise M3MetricDiagnosticContractError(
                f'{name} mismatch: expected={expected!r}, actual={actual!r}.'
            )


def _to_float64_array(value, shape, name: str) -> np.ndarray:
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3MetricDiagnosticContractError(
            f'{name} must be a finite float64-compatible array with shape {shape}.'
        ) from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise M3MetricDiagnosticContractError(
            f'{name} must be a finite float64-compatible array with shape {shape}.'
        )
    return array


def _to_points(value, name: str = 'source_points') -> np.ndarray:
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3MetricDiagnosticContractError(
            f'{name} must be finite with shape [N,3], N>0.'
        ) from error
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or points.shape[0] <= 0
        or not np.all(np.isfinite(points))
    ):
        raise M3MetricDiagnosticContractError(
            f'{name} must be finite with shape [N,3], N>0.'
        )
    return points


def _valid_rotation(rotation: np.ndarray) -> bool:
    return bool(
        np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=SO3_ATOL,
        )
        and np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=SO3_ATOL,
        )
    )


def _to_transform(value, name: str) -> np.ndarray:
    transform = _to_float64_array(value, (4, 4), name)
    if not np.allclose(
        transform[3],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=HOMOGENEOUS_ATOL,
    ):
        raise M3MetricDiagnosticContractError(
            f'{name} must have homogeneous last row [0,0,0,1].'
        )
    if not _valid_rotation(transform[:3, :3]):
        raise M3MetricDiagnosticContractError(
            f'{name} rotation must be in SO(3); diagnostic refuses repair.'
        )
    return transform


def make_rigid_transform(rotation, translation, name: str = 'transform') -> np.ndarray:
    """Build and validate one float64 homogeneous rigid transform."""
    rotation_array = _to_float64_array(rotation, (3, 3), f'{name} rotation')
    translation_array = _to_float64_array(
        translation,
        (3,),
        f'{name} translation',
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return _to_transform(transform, name).copy()


def transform_points(source_points, transform) -> np.ndarray:
    """Apply ``y = R x + t`` to row-vector source coordinates."""
    points = _to_points(source_points)
    rigid = _to_transform(transform, 'transform')
    transformed = points @ rigid[:3, :3].T + rigid[:3, 3]
    if not np.all(np.isfinite(transformed)):
        raise M3MetricDiagnosticContractError(
            'transform application produced NaN or Inf.'
        )
    return transformed


def rotation_error_degrees(predicted_rotation, ground_truth_rotation) -> float:
    """Compute the diagnostic float64 relative rotation angle."""
    predicted = _to_float64_array(
        predicted_rotation,
        (3, 3),
        'predicted rotation',
    )
    ground_truth = _to_float64_array(
        ground_truth_rotation,
        (3, 3),
        'ground-truth rotation',
    )
    if not _valid_rotation(predicted) or not _valid_rotation(ground_truth):
        raise M3MetricDiagnosticContractError(
            'RRE inputs must both be in SO(3).'
        )
    relative = predicted @ ground_truth.T
    cosine = float((np.trace(relative) - 1.0) / 2.0)
    result = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
    if not math.isfinite(result):
        raise M3MetricDiagnosticContractError('RRE produced NaN or Inf.')
    return result


def legacy_parameter_rte_mm(predicted_translation, ground_truth_translation) -> float:
    """Return the frozen parameter-space reference ``||t_pred - t_gt||``."""
    predicted = _to_float64_array(
        predicted_translation,
        (3,),
        'predicted translation',
    )
    ground_truth = _to_float64_array(
        ground_truth_translation,
        (3,),
        'ground-truth translation',
    )
    result = float(np.linalg.norm(predicted - ground_truth))
    if not math.isfinite(result):
        raise M3MetricDiagnosticContractError(
            'legacy parameter RTE produced NaN or Inf.'
        )
    return result


def _percentile(values: np.ndarray, percentile: float) -> float:
    try:
        result = np.percentile(
            values,
            percentile,
            method=POINT_TRE_PERCENTILE_METHOD,
        )
    except TypeError as error:
        raise M3MetricDiagnosticContractError(
            'NumPy must support percentile(method="linear"); '
            'the diagnostic percentile contract cannot be downgraded.'
        ) from error
    return float(result)


def compute_tre_diagnostics(
    source_points,
    predicted_transform,
    effective_gt_transform,
    *,
    solver_success: bool,
) -> dict:
    """Compute origin-invariant TRE metrics for one frozen registration output.

    Solver failures retain source/GT provenance but all predicted-transform and
    TRE fields are null.  A solver-declared success with malformed geometry is
    a contract error; it is never projected to SO(3) or replaced by identity.
    """
    if not isinstance(solver_success, (bool, np.bool_)):
        raise M3MetricDiagnosticContractError(
            'solver_success must be a boolean.'
        )
    points = _to_points(source_points)
    ground_truth = _to_transform(
        effective_gt_transform,
        'effective_gt_transform',
    )
    centroid = np.mean(points, axis=0, dtype=np.float64)
    if not np.all(np.isfinite(centroid)):
        raise M3MetricDiagnosticContractError(
            'source centroid produced NaN or Inf.'
        )
    centroid_norm = float(np.linalg.norm(centroid))
    gt_translation_norm = float(np.linalg.norm(ground_truth[:3, 3]))
    common = {
        'effective_gt_transform_4x4': ground_truth.tolist(),
        'effective_gt_translation_norm_mm': gt_translation_norm,
        'source_centroid_mm': centroid.tolist(),
        'source_centroid_norm_mm': centroid_norm,
    }
    if not bool(solver_success):
        return {
            **common,
            'predicted_transform_4x4': None,
            'predicted_translation_norm_mm': None,
            'legacy_parameter_rte_mm': None,
            **{field: None for field in DIAGNOSTIC_TRE_FIELDS},
            # Diagnostic proxy only; NOT a registration accuracy metric.
            'origin_rotation_lever_proxy_mm': None,
        }

    predicted = _to_transform(predicted_transform, 'predicted_transform')
    predicted_points = transform_points(points, predicted)
    ground_truth_points = transform_points(points, ground_truth)
    distances = np.linalg.norm(predicted_points - ground_truth_points, axis=1)
    if not np.all(np.isfinite(distances)):
        raise M3MetricDiagnosticContractError(
            'point TRE calculation produced NaN or Inf.'
        )
    predicted_centroid = (
        predicted[:3, :3] @ centroid + predicted[:3, 3]
    )
    ground_truth_centroid = (
        ground_truth[:3, :3] @ centroid + ground_truth[:3, 3]
    )
    centroid_tre = float(np.linalg.norm(predicted_centroid - ground_truth_centroid))
    rre_deg = rotation_error_degrees(
        predicted[:3, :3],
        ground_truth[:3, :3],
    )
    # Diagnostic proxy only; NOT a registration accuracy metric.
    origin_rotation_lever_proxy = float(
        2.0 * centroid_norm * math.sin(math.radians(rre_deg) / 2.0)
    )
    result = {
        **common,
        'predicted_transform_4x4': predicted.tolist(),
        'predicted_translation_norm_mm': float(
            np.linalg.norm(predicted[:3, 3])
        ),
        'legacy_parameter_rte_mm': legacy_parameter_rte_mm(
            predicted[:3, 3],
            ground_truth[:3, 3],
        ),
        'centroid_tre_mm': centroid_tre,
        'point_tre_mean_mm': float(np.mean(distances, dtype=np.float64)),
        'point_tre_median_mm': float(np.median(distances)),
        'point_tre_rmse_mm': float(
            np.sqrt(np.mean(np.square(distances), dtype=np.float64))
        ),
        'point_tre_p95_mm': _percentile(distances, POINT_TRE_PERCENTILE),
        'point_tre_max_mm': float(np.max(distances)),
        'origin_rotation_lever_proxy_mm': origin_rotation_lever_proxy,
    }
    if any(
        value is not None and not math.isfinite(float(value))
        for key, value in result.items()
        if key.endswith('_mm') and key not in {'source_centroid_mm'}
    ):
        raise M3MetricDiagnosticContractError(
            'diagnostic metric calculation produced NaN or Inf.'
        )
    return result


def shift_transform_origin(transform, shift_mm) -> np.ndarray:
    """Express a rigid transform after the common origin shift ``x'=x-s``."""
    rigid = _to_transform(transform, 'transform')
    shift = _to_float64_array(shift_mm, (3,), 'origin shift')
    shifted = rigid.copy()
    shifted[:3, 3] = rigid[:3, :3] @ shift + rigid[:3, 3] - shift
    return _to_transform(shifted, 'origin-shifted transform').copy()


def summarize_confidence(confidence, num_correspondences: int) -> dict:
    """Summarize frozen correspondence confidence without changing pairs."""
    if isinstance(num_correspondences, bool) or not isinstance(
        num_correspondences,
        (int, np.integer),
    ):
        raise M3MetricDiagnosticContractError(
            'num_correspondences must be a non-negative integer.'
        )
    count = int(num_correspondences)
    if count < 0:
        raise M3MetricDiagnosticContractError(
            'num_correspondences must be a non-negative integer.'
        )
    if hasattr(confidence, 'detach') and hasattr(confidence, 'cpu'):
        confidence = confidence.detach().cpu().numpy()
    try:
        values = np.asarray(confidence, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3MetricDiagnosticContractError(
            'confidence must be a finite vector matching correspondences.'
        ) from error
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise M3MetricDiagnosticContractError(
            'confidence must be a finite vector matching correspondences.'
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise M3MetricDiagnosticContractError('confidence must be in [0,1].')
    if count == 0:
        return {field: None for field in CONFIDENCE_FIELDS}
    return {
        'confidence_mean': float(np.mean(values, dtype=np.float64)),
        'confidence_median': float(np.median(values)),
        'confidence_p05': _percentile(values, 5.0),
        'confidence_p95': _percentile(values, 95.0),
        'confidence_min': float(np.min(values)),
        'confidence_max': float(np.max(values)),
    }


def _finite_number(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise M3MetricDiagnosticContractError(f'{name} must be a finite number.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3MetricDiagnosticContractError(
            f'{name} must be a finite number.'
        ) from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise M3MetricDiagnosticContractError(f'{name} must be a finite number.')
    return result


def metric_statistics(values: Iterable[Optional[float]]) -> dict:
    """Aggregate finite non-null values; null values are never converted to zero."""
    normalized = []
    for value in values:
        if value is None:
            continue
        normalized.append(_finite_number(value, 'metric aggregation value'))
    array = np.asarray(normalized, dtype=np.float64)
    return {
        'sample_count': len(normalized),
        'mean': float(np.mean(array, dtype=np.float64)) if normalized else None,
        'median': float(np.median(array)) if normalized else None,
        'p95': _percentile(array, 95.0) if normalized else None,
    }


def safe_correlation(x_values: Sequence, y_values: Sequence) -> Optional[float]:
    """Return Pearson correlation, or null for fewer than two/zero-variance pairs."""
    if len(x_values) != len(y_values):
        raise M3MetricDiagnosticContractError(
            'correlation inputs must have the same length.'
        )
    pairs = []
    for x_value, y_value in zip(x_values, y_values):
        if x_value is None or y_value is None:
            continue
        pairs.append(
            (
                _finite_number(x_value, 'correlation x'),
                _finite_number(y_value, 'correlation y'),
            )
        )
    if len(pairs) < 2:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return None
    result = float(np.corrcoef(x, y)[0, 1])
    return result if math.isfinite(result) else None


def _validate_case_metric_contract(case: Mapping) -> None:
    if not isinstance(case, Mapping):
        raise M3MetricDiagnosticContractError('every case must be a mapping.')
    success = case.get('solver_success')
    if not isinstance(success, (bool, np.bool_)):
        raise M3MetricDiagnosticContractError(
            'every case requires boolean solver_success.'
        )
    recall = case.get('registration_recall_hit')
    if not isinstance(recall, (bool, np.bool_)):
        raise M3MetricDiagnosticContractError(
            'every case requires boolean registration_recall_hit.'
        )
    if not bool(success) and bool(recall):
        raise M3MetricDiagnosticContractError(
            'solver failures cannot be registration recall hits.'
        )
    nullable_success_fields = (
        'rre_deg',
        'legacy_parameter_rte_mm',
        *DIAGNOSTIC_TRE_FIELDS,
    )
    for field in nullable_success_fields:
        value = case.get(field)
        if bool(success):
            if value is None:
                raise M3MetricDiagnosticContractError(
                    f'solver-success case requires {field}.'
                )
            _finite_number(value, field, nonnegative=True)
        elif value is not None:
            raise M3MetricDiagnosticContractError(
                f'solver-failure case must keep {field} null.'
            )
    inlier_ratio = _finite_number(
        case.get('inlier_ratio'),
        'inlier_ratio',
        nonnegative=True,
    )
    if inlier_ratio > 1.0:
        raise M3MetricDiagnosticContractError('inlier_ratio must be in [0,1].')
    for field in ('num_correspondences', 'num_inliers'):
        value = case.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise M3MetricDiagnosticContractError(
                f'{field} must be a non-negative integer.'
            )
        if int(value) < 0:
            raise M3MetricDiagnosticContractError(
                f'{field} must be a non-negative integer.'
            )


def aggregate_case_group(cases: Sequence[Mapping]) -> dict:
    """Aggregate one diagnostic group while retaining failures in denominators."""
    rows = list(cases)
    for case in rows:
        _validate_case_metric_contract(case)
    successes = [case for case in rows if bool(case['solver_success'])]
    failures = [case for case in rows if not bool(case['solver_success'])]
    hits = sum(bool(case['registration_recall_hit']) for case in rows)
    total = len(rows)
    metric_fields = (
        'rre_deg',
        'legacy_parameter_rte_mm',
        'inlier_ratio',
        'centroid_tre_mm',
        'point_tre_mean_mm',
        'point_tre_median_mm',
        'point_tre_rmse_mm',
        'point_tre_p95_mm',
        'point_tre_max_mm',
        'num_correspondences',
        'num_inliers',
        *CONFIDENCE_FIELDS,
        'source_centroid_norm_mm',
        'origin_rotation_lever_proxy_mm',
    )
    return {
        'case_count': total,
        'solver_success_count': len(successes),
        'solver_failure_count': len(failures),
        'registration_recall': float(hits / total) if total else 0.0,
        **{
            field: metric_statistics(case.get(field) for case in rows)
            for field in metric_fields
        },
    }


def _aggregate_by(cases: Sequence[Mapping], fields: Tuple[str, ...]) -> dict:
    groups = defaultdict(list)
    for case in cases:
        values = []
        for field in fields:
            value = case.get(field)
            if not isinstance(value, str) or not value:
                raise M3MetricDiagnosticContractError(
                    f'every case requires non-empty {field}.'
                )
            values.append(value)
        groups[tuple(values)].append(case)
    return {
        '|'.join(key): aggregate_case_group(groups[key])
        for key in sorted(groups)
    }


def build_pat6_forensic(
    cases: Sequence[Mapping],
    *,
    expected_case_count: Optional[int] = None,
) -> dict:
    """Build Pat6-only catastrophic-rotation forensic summaries."""
    pat6 = [case for case in cases if case.get('subject_id') == 'Pat6']
    for case in pat6:
        _validate_case_metric_contract(case)
    if expected_case_count is not None and len(pat6) != expected_case_count:
        raise M3MetricDiagnosticContractError(
            f'Pat6 case count mismatch: expected={expected_case_count}, '
            f'actual={len(pat6)}.'
        )
    catastrophic = [
        case
        for case in pat6
        if case.get('rre_deg') is not None
        and float(case['rre_deg']) >= CATASTROPHIC_ROTATION_THRESHOLD_DEG
    ]
    by_defect = defaultdict(int)
    by_severity = defaultdict(int)
    for case in pat6:
        by_defect[case['defect_id']] += 0
        by_severity[case['severity']] += 0
    for case in catastrophic:
        by_defect[case['defect_id']] += 1
        by_severity[case['severity']] += 1
    worst = sorted(
        (case for case in pat6 if case.get('rre_deg') is not None),
        key=lambda case: (
            -float(case['rre_deg']),
            str(case.get('case_key', '')),
        ),
    )[:10]
    worst_fields = (
        'fold_id',
        'subject_id',
        'defect_id',
        'severity',
        'variant_id',
        'perturbation_seed',
        'rre_deg',
        'legacy_parameter_rte_mm',
        'centroid_tre_mm',
        'point_tre_mean_mm',
        'inlier_ratio',
        'num_correspondences',
        'confidence_mean',
    )
    case_count = len(pat6)
    return {
        'subject_id': 'Pat6',
        'case_count': case_count,
        'rre_deg': metric_statistics(case.get('rre_deg') for case in pat6),
        'legacy_parameter_rte_mm': metric_statistics(
            case.get('legacy_parameter_rte_mm') for case in pat6
        ),
        'centroid_tre_mm': metric_statistics(
            case.get('centroid_tre_mm') for case in pat6
        ),
        'point_tre_mean_mm': metric_statistics(
            case.get('point_tre_mean_mm') for case in pat6
        ),
        'inlier_ratio': metric_statistics(
            case.get('inlier_ratio') for case in pat6
        ),
        'num_correspondences': metric_statistics(
            case.get('num_correspondences') for case in pat6
        ),
        'confidence_mean': metric_statistics(
            case.get('confidence_mean') for case in pat6
        ),
        'catastrophic_rotation_threshold_deg': (
            CATASTROPHIC_ROTATION_THRESHOLD_DEG
        ),
        'catastrophic_rotation_case_count': len(catastrophic),
        'catastrophic_rotation_case_ratio': (
            float(len(catastrophic) / case_count) if case_count else 0.0
        ),
        'catastrophic_count_by_defect_id': {
            key: by_defect[key] for key in sorted(by_defect)
        },
        'catastrophic_count_by_severity': {
            key: by_severity[key] for key in sorted(by_severity)
        },
        'worst_10_cases': [
            {field: case.get(field) for field in worst_fields}
            for case in worst
        ],
        'correlations': {
            'rre_vs_legacy_parameter_rte': safe_correlation(
                [case.get('rre_deg') for case in pat6],
                [case.get('legacy_parameter_rte_mm') for case in pat6],
            ),
            'rre_vs_centroid_tre': safe_correlation(
                [case.get('rre_deg') for case in pat6],
                [case.get('centroid_tre_mm') for case in pat6],
            ),
            'rre_vs_point_tre_mean': safe_correlation(
                [case.get('rre_deg') for case in pat6],
                [case.get('point_tre_mean_mm') for case in pat6],
            ),
            'inlier_ratio_vs_rre': safe_correlation(
                [case.get('inlier_ratio') for case in pat6],
                [case.get('rre_deg') for case in pat6],
            ),
        },
    }


def aggregate_diagnostic_cases(
    cases: Sequence[Mapping],
    *,
    expected_case_count: Optional[int] = None,
    expected_pat6_case_count: Optional[int] = None,
    diagnostic_protocol: Optional[Mapping] = None,
) -> dict:
    """Build all required global and stratified diagnostic aggregates."""
    rows = list(cases)
    if expected_case_count is not None and len(rows) != expected_case_count:
        raise M3MetricDiagnosticContractError(
            f'diagnostic case count mismatch: expected={expected_case_count}, '
            f'actual={len(rows)}.'
        )
    if diagnostic_protocol is None:
        # Backward-compatible public behavior for callers of the original v1
        # API: retain the Pat6 forensic member unless a version is explicit.
        pat6_forensic_required = True
    else:
        contract = get_diagnostic_protocol_contract(diagnostic_protocol)
        pat6_forensic_required = contract['pat6_forensic_required']
        if expected_pat6_case_count is None and pat6_forensic_required:
            expected_pat6_case_count = contract['expected_pat6_case_count']
        if not pat6_forensic_required and expected_pat6_case_count not in (
            None,
            0,
        ):
            raise M3MetricDiagnosticContractError(
                'a nonzero Pat6 expectation is invalid for a diagnostic '
                'protocol where Pat6 forensic is not applicable.'
            )
    result = {
        'overall': aggregate_case_group(rows),
        'per_fold': _aggregate_by(rows, ('fold_id',)),
        'per_subject': _aggregate_by(rows, ('subject_id',)),
        'per_defect': _aggregate_by(rows, ('defect_id',)),
        'per_severity': _aggregate_by(rows, ('severity',)),
        'per_subject_defect': _aggregate_by(
            rows,
            ('subject_id', 'defect_id'),
        ),
    }
    if pat6_forensic_required:
        result['pat6_forensic'] = build_pat6_forensic(
            rows,
            expected_case_count=expected_pat6_case_count,
        )
    return result


def case_identity(case: Mapping) -> tuple:
    """Return the exact formal six-field identity, rejecting malformed values."""
    if not isinstance(case, Mapping):
        raise M3MetricDiagnosticContractError('case identity must be a mapping.')
    values = []
    for field in REQUIRED_CASE_IDENTITY_FIELDS[:4]:
        value = case.get(field)
        if not isinstance(value, str) or not value:
            raise M3MetricDiagnosticContractError(
                f'case identity field {field} must be a non-empty string.'
            )
        values.append(value)
    for field in REQUIRED_CASE_IDENTITY_FIELDS[4:]:
        value = case.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise M3MetricDiagnosticContractError(
                f'case identity field {field} must be an integer.'
            )
        values.append(int(value))
    return tuple(values)


def _unique_identity_map(
    rows: Sequence[Mapping],
    label: str,
) -> Mapping[tuple, Mapping]:
    result = {}
    for row in rows:
        identity = case_identity(row)
        if identity in result:
            raise M3MetricDiagnosticContractError(
                f'duplicate {label} case identity: {identity!r}.'
            )
        result[identity] = row
    return result


def validate_case_identity_set(
    rows: Sequence[Mapping],
    expected_rows: Sequence[Mapping],
    *,
    expected_count: int,
    label: str,
) -> None:
    """Require an exact, duplicate-free case identity set and count."""
    if len(rows) != expected_count:
        raise M3MetricDiagnosticContractError(
            f'{label} case count mismatch: expected={expected_count}, '
            f'actual={len(rows)}.'
        )
    actual_map = _unique_identity_map(rows, label)
    expected_map = _unique_identity_map(expected_rows, 'expected manifest')
    if len(expected_map) != expected_count:
        raise M3MetricDiagnosticContractError(
            'expected manifest case count contract is invalid: '
            f'expected={expected_count}, actual={len(expected_map)}.'
        )
    missing = sorted(set(expected_map).difference(actual_map))
    unexpected = sorted(set(actual_map).difference(expected_map))
    if missing or unexpected:
        raise M3MetricDiagnosticContractError(
            f'{label} identity mismatch; first_missing='
            f'{missing[0] if missing else None!r}, first_unexpected='
            f'{unexpected[0] if unexpected else None!r}.'
        )


def _continuous_matches(legacy_value, diagnostic_value) -> bool:
    if legacy_value is None or diagnostic_value is None:
        return legacy_value is None and diagnostic_value is None
    legacy = _finite_number(legacy_value, 'legacy continuous metric')
    diagnostic = _finite_number(
        diagnostic_value,
        'diagnostic continuous metric',
    )
    return math.isclose(
        legacy,
        diagnostic,
        rel_tol=LEGACY_CONTINUOUS_REL_TOLERANCE,
        abs_tol=LEGACY_CONTINUOUS_ABS_TOLERANCE,
    )


def cross_check_legacy_cases(
    legacy_rows: Sequence[Mapping],
    diagnostic_rows: Sequence[Mapping],
    *,
    expected_count: int,
) -> dict:
    """Fail closed unless every rerun case matches the frozen formal result."""
    if len(legacy_rows) != expected_count or len(diagnostic_rows) != expected_count:
        raise M3MetricDiagnosticContractError(
            'legacy/diagnostic case count mismatch: '
            f'expected={expected_count}, legacy={len(legacy_rows)}, '
            f'diagnostic={len(diagnostic_rows)}.'
        )
    legacy_by_id = _unique_identity_map(legacy_rows, 'legacy')
    diagnostic_by_id = _unique_identity_map(diagnostic_rows, 'diagnostic')
    if set(legacy_by_id) != set(diagnostic_by_id):
        missing = sorted(set(legacy_by_id).difference(diagnostic_by_id))
        unexpected = sorted(set(diagnostic_by_id).difference(legacy_by_id))
        raise M3MetricDiagnosticContractError(
            'legacy/diagnostic identity mismatch; first_missing_diagnostic='
            f'{missing[0] if missing else None!r}, first_unexpected_diagnostic='
            f'{unexpected[0] if unexpected else None!r}.'
        )
    discrete_fields = (
        'solver_success',
        'solver_status',
        'registration_recall_hit',
        'num_correspondences',
        'num_inliers',
    )
    continuous_fields = (
        ('rre_deg', 'rre_deg'),
        ('rte_mm', 'legacy_parameter_rte_mm'),
        ('inlier_ratio', 'inlier_ratio'),
    )
    for identity in sorted(legacy_by_id):
        legacy = legacy_by_id[identity]
        diagnostic = diagnostic_by_id[identity]
        for field in discrete_fields:
            if legacy.get(field) != diagnostic.get(field) or type(
                legacy.get(field)
            ) is not type(diagnostic.get(field)):
                raise M3MetricDiagnosticContractError(
                    f'legacy mismatch at case={identity!r}, field={field}: '
                    f'legacy={legacy.get(field)!r}, '
                    f'diagnostic={diagnostic.get(field)!r}.'
                )
        for legacy_field, diagnostic_field in continuous_fields:
            if not _continuous_matches(
                legacy.get(legacy_field),
                diagnostic.get(diagnostic_field),
            ):
                raise M3MetricDiagnosticContractError(
                    f'legacy mismatch at case={identity!r}, '
                    f'field={legacy_field}: legacy={legacy.get(legacy_field)!r}, '
                    f'diagnostic={diagnostic.get(diagnostic_field)!r}, '
                    f'abs_tol={LEGACY_CONTINUOUS_ABS_TOLERANCE}, '
                    f'rel_tol={LEGACY_CONTINUOUS_REL_TOLERANCE}.'
                )
    return {
        'cross_check_pass': True,
        'case_count': expected_count,
        'identity_fields': list(REQUIRED_CASE_IDENTITY_FIELDS),
        'continuous_abs_tolerance': LEGACY_CONTINUOUS_ABS_TOLERANCE,
        'continuous_rel_tolerance': LEGACY_CONTINUOUS_REL_TOLERANCE,
    }


def load_jsonl(path) -> list:
    """Read strict JSONL without blank lines, duplicate keys, NaN, or Inf."""
    path = Path(path)
    if not path.is_file():
        raise M3MetricDiagnosticContractError(f'JSONL file is missing: {path}.')
    rows = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise M3MetricDiagnosticContractError(
                        f'JSONL contains blank line {line_number}: {path}.'
                    )
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_fields,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f'non-finite JSON constant {value}')
                        ),
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise M3MetricDiagnosticContractError(
                        f'invalid JSONL at {path}:{line_number}: {error}'
                    ) from error
                if not isinstance(row, Mapping):
                    raise M3MetricDiagnosticContractError(
                        f'JSONL row must be an object at {path}:{line_number}.'
                    )
                rows.append(row)
    except M3MetricDiagnosticContractError:
        raise
    except (OSError, UnicodeError) as error:
        raise M3MetricDiagnosticContractError(
            f'cannot read JSONL {path}: {error}'
        ) from error
    return rows


def validate_legacy_result_tree(
    legacy_results_root,
    expected_cases_by_fold: Mapping[str, Sequence[Mapping]],
    *,
    diagnostic_protocol: Optional[Mapping] = None,
) -> Mapping[str, object]:
    """Validate the authoritative ordered union of the five Fold JSONL files.

    The root ``cases.jsonl`` is non-authoritative provenance.  It may be absent,
    the complete union, or the exact result of one legitimate single-Fold
    invocation.  Any other root artifact fails closed.
    """
    root = Path(legacy_results_root)
    if not root.is_dir():
        raise M3MetricDiagnosticContractError(
            f'legacy result root does not exist: {root}.'
        )
    if diagnostic_protocol is None:
        diagnostic_contract = get_diagnostic_protocol_contract(
            DIAGNOSTIC_PROTOCOL_VERSION
        )
        expected_case_count = EXPECTED_TEST_CASE_COUNT
        expected_counts_by_fold = EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD
    else:
        diagnostic_contract = get_diagnostic_protocol_contract(
            diagnostic_protocol
        )
        expected_case_count = diagnostic_contract['expected_test_case_count']
        expected_counts_by_fold = {
            fold_id: len(rows)
            for fold_id, rows in expected_cases_by_fold.items()
        }
    expected_folds = tuple(EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD)
    if set(expected_cases_by_fold) != set(expected_folds):
        raise M3MetricDiagnosticContractError(
            'expected legacy manifest mapping must contain Fold1 through Fold5.'
        )
    per_fold = {}
    expected_all = []
    fold_all = []
    for fold_id in expected_folds:
        expected = list(expected_cases_by_fold[fold_id])
        required_count = expected_counts_by_fold[fold_id]
        if len(expected) != required_count:
            raise M3MetricDiagnosticContractError(
                f'{fold_id} formal manifest count mismatch: '
                f'expected={required_count}, actual={len(expected)}.'
            )
        rows = load_jsonl(root / fold_id / 'cases.jsonl')
        validate_case_identity_set(
            rows,
            expected,
            expected_count=required_count,
            label=f'legacy {fold_id}',
        )
        per_fold[fold_id] = rows
        expected_all.extend(expected)
        fold_all.extend(rows)

    manifest_union_count = sum(expected_counts_by_fold.values())
    if manifest_union_count != expected_case_count:
        raise M3MetricDiagnosticContractError(
            'diagnostic protocol/manifest case count mismatch: '
            f'protocol={expected_case_count}, manifests={manifest_union_count}.'
        )

    validate_case_identity_set(
        fold_all,
        expected_all,
        expected_count=expected_case_count,
        label='legacy per-fold union',
    )
    # This map call is intentionally separate from the set comparison above:
    # it makes the no-duplicate union contract explicit and fail closed.
    union_by_identity = _unique_identity_map(
        fold_all,
        'legacy per-fold union',
    )
    if len(union_by_identity) != expected_case_count:
        raise M3MetricDiagnosticContractError(
            'legacy per-fold union must contain exactly '
            f'{expected_case_count} unique identities.'
        )

    root_path = root / 'cases.jsonl'
    root_rows = None
    root_status = 'absent'
    root_detected_fold = None
    if root_path.exists():
        root_rows = load_jsonl(root_path)
        root_count = len(root_rows)
        if root_count == expected_case_count:
            validate_case_identity_set(
                root_rows,
                fold_all,
                expected_count=expected_case_count,
                label='legacy root complete union',
            )
            root_status = diagnostic_contract['legacy_complete_union_status']
        else:
            matching_folds = []
            for fold_id in expected_folds:
                required_count = expected_counts_by_fold[fold_id]
                if root_count != required_count:
                    continue
                try:
                    validate_case_identity_set(
                        root_rows,
                        per_fold[fold_id],
                        expected_count=required_count,
                        label=f'legacy root candidate {fold_id}',
                    )
                except M3MetricDiagnosticContractError:
                    continue
                matching_folds.append(fold_id)
            if len(matching_folds) != 1:
                raise M3MetricDiagnosticContractError(
                    'legacy root cases.jsonl is neither the complete '
                    f'{expected_case_count}-case '
                    'per-fold union nor one exact formal single-Fold invocation '
                    f'artifact; root_case_count={root_count}.'
                )
            root_status = 'single_fold_invocation_artifact'
            root_detected_fold = matching_folds[0]
    root_count = len(root_rows) if root_rows is not None else 0
    return {
        # The ordered Fold1..Fold5 concatenation is the only authoritative
        # global source used by subsequent 825-case metric cross-checks.
        'authoritative_union': fold_all,
        'per_fold': per_fold,
        'per_fold_counts': {
            fold_id: len(per_fold[fold_id]) for fold_id in expected_folds
        },
        'per_fold_union_count': len(fold_all),
        'per_fold_union_identity_audit': 'PASS',
        'root': root_rows,
        'legacy_root_cases_status': root_status,
        'legacy_root_cases_count': root_count,
        'legacy_root_cases_detected_fold': root_detected_fold,
    }


__all__ = [
    'CATASTROPHIC_ROTATION_THRESHOLD_DEG',
    'CLEAN10_DIAGNOSTIC_PROTOCOL_HASH',
    'CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION',
    'CLEAN10_EXPECTED_DEFECT_CONDITION_COUNT',
    'CLEAN10_EXPECTED_DEFECT_INSTANCE_COUNT',
    'CLEAN10_EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD',
    'CLEAN10_EXPECTED_PAT6_CASE_COUNT',
    'CLEAN10_EXPECTED_PATIENT_COUNT',
    'CLEAN10_EXPECTED_TEST_CASE_COUNT',
    'CONFIDENCE_FIELDS',
    'DIAGNOSTIC_PROTOCOL_HASH',
    'DIAGNOSTIC_PROTOCOL_VERSION',
    'DIAGNOSTIC_TRE_FIELDS',
    'EXPECTED_DEFECT_CONDITION_COUNT',
    'EXPECTED_DEFECT_INSTANCE_COUNT',
    'EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD',
    'EXPECTED_PAT6_CASE_COUNT',
    'EXPECTED_PATIENT_COUNT',
    'EXPECTED_TEST_CASE_COUNT',
    'LEGACY_CONTINUOUS_ABS_TOLERANCE',
    'LEGACY_CONTINUOUS_REL_TOLERANCE',
    'M3MetricDiagnosticContractError',
    'POINT_TRE_PERCENTILE',
    'POINT_TRE_PERCENTILE_METHOD',
    'REQUIRED_CASE_IDENTITY_FIELDS',
    'aggregate_case_group',
    'aggregate_diagnostic_cases',
    'build_pat6_forensic',
    'case_identity',
    'compute_diagnostic_protocol_hash',
    'compute_tre_diagnostics',
    'cross_check_legacy_cases',
    'legacy_parameter_rte_mm',
    'load_diagnostic_protocol',
    'load_jsonl',
    'make_rigid_transform',
    'metric_statistics',
    'rotation_error_degrees',
    'safe_correlation',
    'shift_transform_origin',
    'summarize_confidence',
    'transform_points',
    'get_diagnostic_protocol_contract',
    'validate_case_identity_set',
    'validate_diagnostic_protocol',
    'validate_legacy_result_tree',
    'validate_source_protocols',
    'SUPPORTED_DIAGNOSTIC_PROTOCOL_VERSIONS',
]
