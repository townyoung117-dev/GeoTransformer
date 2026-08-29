"""TRE observation separated from strict replay reproducibility for M3 clean10.

The frozen formal evaluation remains authoritative.  A valid replay transform
may contribute TRE observations even when its replayed formal metrics do not
match the frozen values at the strict v2 tolerances.
"""

import hashlib
import hmac
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from m3_metric_diagnostic import (
    CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
    CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION,
    DIAGNOSTIC_TRE_FIELDS,
    LEGACY_CONTINUOUS_ABS_TOLERANCE,
    LEGACY_CONTINUOUS_REL_TOLERANCE,
    M3MetricDiagnosticContractError,
    REQUIRED_CASE_IDENTITY_FIELDS,
    case_identity,
    metric_statistics,
)


OBSERVATION_PROTOCOL_VERSION = 'm3_metric_observation_clean10_v3'
# Filled with the canonical hash of the frozen JSON protocol below.
OBSERVATION_PROTOCOL_HASH = (
    '86eef924d1c697fbcee6e226e7b4c6041063f64b5da70f884630817b0ae02881'
)
SOURCE_TRAINING_PROTOCOL_VERSION = 'm3_6b_5fold_clean10_v2'
SOURCE_TRAINING_PROTOCOL_HASH = (
    '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'
)
SOURCE_EVALUATION_PROTOCOL_VERSION = 'm3_defect_eval_clean10_v2'
SOURCE_EVALUATION_PROTOCOL_HASH = (
    'cfd519b923be3b623cffec8c8f5830cb5b1160859461180eddeb7134a0adb6b0'
)
SOURCE_REFERENCE_POINTS = 'augmented_defective_point_physical_coordinates'
EXPECTED_PATIENT_COUNT = 10
EXPECTED_DEFECT_CONDITION_COUNT = 5
EXPECTED_DEFECT_INSTANCE_COUNT = 50
EXPECTED_TEST_CASE_COUNT = 750
EXPECTED_PAT6_CASE_COUNT = 0
EXPECTED_CASE_COUNTS_BY_FOLD = {
    'Fold1': 150,
    'Fold2': 150,
    'Fold3': 150,
    'Fold4': 150,
    'Fold5': 150,
}
STRICT_REPRODUCIBILITY_ABS_TOLERANCE = LEGACY_CONTINUOUS_ABS_TOLERANCE
STRICT_REPRODUCIBILITY_REL_TOLERANCE = LEGACY_CONTINUOUS_REL_TOLERANCE

REPRODUCIBILITY_DISCRETE_FIELDS = (
    ('solver_success', 'solver_success'),
    ('solver_status', 'solver_status'),
    ('registration_recall_hit', 'registration_recall_hit'),
    ('num_correspondences', 'num_correspondences'),
    ('num_inliers', 'num_inliers'),
)
REPRODUCIBILITY_CONTINUOUS_FIELDS = (
    ('rre_deg', 'rre_deg'),
    ('rte_mm', 'legacy_parameter_rte_mm'),
    ('inlier_ratio', 'inlier_ratio'),
)
REPRODUCIBILITY_COMPARED_FIELDS = tuple(
    formal_field
    for formal_field, _ in (
        *REPRODUCIBILITY_DISCRETE_FIELDS,
        *REPRODUCIBILITY_CONTINUOUS_FIELDS,
    )
)

_EXPECTED_OBSERVATION_PROTOCOL = {
    'observation_protocol_version': OBSERVATION_PROTOCOL_VERSION,
    'source_training_protocol_version': SOURCE_TRAINING_PROTOCOL_VERSION,
    'source_training_protocol_hash': SOURCE_TRAINING_PROTOCOL_HASH,
    'source_defect_evaluation_protocol_version': (
        SOURCE_EVALUATION_PROTOCOL_VERSION
    ),
    'source_defect_evaluation_protocol_hash': SOURCE_EVALUATION_PROTOCOL_HASH,
    'source_diagnostic_protocol_version': CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION,
    'source_diagnostic_protocol_hash': CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
    'source_reference_points': SOURCE_REFERENCE_POINTS,
    'centroid_tre_definition': (
        'norm((R_pred*c+t_pred)-(R_gt*c+t_gt)), '
        'c=mean(source_reference_points)'
    ),
    'point_tre_definition': (
        'd_i=norm((R_pred*x_i+t_pred)-(R_gt*x_i+t_gt)); '
        'report mean,median,rmse,p95,max over all source_reference_points'
    ),
    'legacy_parameter_rte_definition': 'norm(t_pred-t_gt)',
    'point_tre_percentile': 95,
    'point_tre_percentile_method': 'linear',
    'solver_failure_policy': 'null_tre_observation_metrics',
    'formal_metric_policy': 'frozen_formal_metrics_are_authoritative',
    'replay_policy': 'replay_is_observation_not_formal_metric_replacement',
    'strict_reproducibility_abs_tolerance': (
        STRICT_REPRODUCIBILITY_ABS_TOLERANCE
    ),
    'strict_reproducibility_rel_tolerance': (
        STRICT_REPRODUCIBILITY_REL_TOLERANCE
    ),
    'replay_mismatch_policy': (
        'record_failure_but_preserve_valid_tre_observation'
    ),
    'required_case_identity_fields': list(REQUIRED_CASE_IDENTITY_FIELDS),
    'expected_patient_count': EXPECTED_PATIENT_COUNT,
    'expected_defect_condition_count': EXPECTED_DEFECT_CONDITION_COUNT,
    'expected_defect_instance_count': EXPECTED_DEFECT_INSTANCE_COUNT,
    'expected_test_case_count': EXPECTED_TEST_CASE_COUNT,
    'expected_pat6_case_count': EXPECTED_PAT6_CASE_COUNT,
    'expected_case_counts_by_fold': EXPECTED_CASE_COUNTS_BY_FOLD,
    'observation_replaces_frozen_formal_metrics': False,
    'observation_changes_model_predictions': False,
}
_OBSERVATION_PROTOCOL_FIELDS = set(_EXPECTED_OBSERVATION_PROTOCOL) | {
    'observation_protocol_hash'
}


class M3MetricObservationContractError(M3MetricDiagnosticContractError):
    """Raised when an M3 metric observation contract is violated."""


def _reject_duplicate_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise M3MetricObservationContractError(
                f'duplicate JSON object field: {key!r}.'
            )
        result[key] = value
    return result


def compute_observation_protocol_hash(protocol: Mapping) -> str:
    """Hash every observation protocol field except its stored hash."""
    if not isinstance(protocol, Mapping):
        raise M3MetricObservationContractError(
            'observation protocol must be a mapping.'
        )
    payload = dict(protocol)
    payload.pop('observation_protocol_hash', None)
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M3MetricObservationContractError(
            'observation protocol must be canonical-JSON serializable.'
        ) from error
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_observation_protocol(protocol: Mapping) -> Mapping:
    """Fail closed on any deviation from the frozen v3 protocol."""
    if not isinstance(protocol, Mapping):
        raise M3MetricObservationContractError(
            'observation protocol must be a JSON object.'
        )
    actual_fields = set(protocol)
    missing = sorted(_OBSERVATION_PROTOCOL_FIELDS.difference(actual_fields))
    unexpected = sorted(actual_fields.difference(_OBSERVATION_PROTOCOL_FIELDS))
    if missing or unexpected:
        raise M3MetricObservationContractError(
            'observation protocol fields mismatch; '
            f'missing={missing}, unexpected={unexpected}.'
        )
    stored_hash = protocol['observation_protocol_hash']
    if (
        not isinstance(stored_hash, str)
        or len(stored_hash) != 64
        or any(character not in '0123456789abcdef' for character in stored_hash)
    ):
        raise M3MetricObservationContractError(
            'observation_protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    computed_hash = compute_observation_protocol_hash(protocol)
    if not hmac.compare_digest(stored_hash, computed_hash):
        raise M3MetricObservationContractError(
            'observation_protocol_hash mismatch: '
            f'stored={stored_hash}, computed={computed_hash}.'
        )
    if not hmac.compare_digest(stored_hash, OBSERVATION_PROTOCOL_HASH):
        raise M3MetricObservationContractError(
            'observation_protocol_hash is not the frozen hash for '
            f'{OBSERVATION_PROTOCOL_VERSION}: stored={stored_hash}, '
            f'expected={OBSERVATION_PROTOCOL_HASH}.'
        )
    for field, expected in _EXPECTED_OBSERVATION_PROTOCOL.items():
        actual = protocol[field]
        if actual != expected or type(actual) is not type(expected):
            raise M3MetricObservationContractError(
                f'{field} must be exactly {expected!r} for '
                f'{OBSERVATION_PROTOCOL_VERSION}.'
            )
    return protocol


def load_observation_protocol(path) -> Mapping:
    path = Path(path)
    if not path.is_file():
        raise M3MetricObservationContractError(
            f'observation protocol does not exist: {path}.'
        )
    try:
        with path.open('r', encoding='utf-8') as handle:
            protocol = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_fields,
            )
    except M3MetricObservationContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3MetricObservationContractError(
            f'cannot load observation protocol {path}: {error}'
        ) from error
    return validate_observation_protocol(protocol)


def validate_source_protocols(
    observation_protocol: Mapping,
    training_protocol: Mapping,
    evaluation_protocol: Mapping,
) -> None:
    """Bind v3 to the exact canonical clean10 training/evaluation pair."""
    validate_observation_protocol(observation_protocol)
    try:
        from defect_evaluation import (
            validate_defect_evaluation_protocol,
            validate_protocol_pair,
        )
        from training_protocol import validate_training_protocol

        validate_training_protocol(training_protocol)
        validate_defect_evaluation_protocol(evaluation_protocol)
        validate_protocol_pair(training_protocol, evaluation_protocol)
    except Exception as error:
        if isinstance(error, M3MetricObservationContractError):
            raise
        raise M3MetricObservationContractError(
            f'source protocol canonical validation failed: {error}'
        ) from error
    comparisons = (
        (
            training_protocol.get('protocol_version'),
            observation_protocol['source_training_protocol_version'],
            'source training protocol version',
        ),
        (
            training_protocol.get('protocol_hash'),
            observation_protocol['source_training_protocol_hash'],
            'source training protocol hash',
        ),
        (
            evaluation_protocol.get('evaluation_protocol_version'),
            observation_protocol[
                'source_defect_evaluation_protocol_version'
            ],
            'source evaluation protocol version',
        ),
        (
            evaluation_protocol.get('evaluation_protocol_hash'),
            observation_protocol['source_defect_evaluation_protocol_hash'],
            'source evaluation protocol hash',
        ),
    )
    for actual, expected, label in comparisons:
        if actual != expected or type(actual) is not type(expected):
            raise M3MetricObservationContractError(
                f'{label} mismatch: expected={expected!r}, actual={actual!r}.'
            )


def _finite_number(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise M3MetricObservationContractError(f'{name} must be a finite number.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3MetricObservationContractError(
            f'{name} must be a finite number.'
        ) from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise M3MetricObservationContractError(
            f'{name} must be a finite non-negative number.'
        )
    return result


def _require_protocol_stamp(row: Mapping, field: str, expected) -> None:
    actual = row.get(field)
    if actual != expected or type(actual) is not type(expected):
        raise M3MetricObservationContractError(
            f'{field} mismatch: expected={expected!r}, actual={actual!r}.'
        )


def _validate_formal_case(formal_case: Mapping, protocol: Mapping) -> None:
    case_identity(formal_case)
    for field, expected in (
        ('protocol_version', protocol['source_training_protocol_version']),
        ('protocol_hash', protocol['source_training_protocol_hash']),
        (
            'evaluation_protocol_version',
            protocol['source_defect_evaluation_protocol_version'],
        ),
        (
            'evaluation_protocol_hash',
            protocol['source_defect_evaluation_protocol_hash'],
        ),
    ):
        _require_protocol_stamp(formal_case, field, expected)
    success = formal_case.get('solver_success')
    recall = formal_case.get('registration_recall_hit')
    if not isinstance(success, (bool, np.bool_)):
        raise M3MetricObservationContractError(
            'frozen formal solver_success must be boolean.'
        )
    if not isinstance(recall, (bool, np.bool_)):
        raise M3MetricObservationContractError(
            'frozen formal registration_recall_hit must be boolean.'
        )
    if not bool(success) and bool(recall):
        raise M3MetricObservationContractError(
            'frozen formal solver failure cannot be a recall hit.'
        )
    for field in ('rre_deg', 'rte_mm'):
        value = formal_case.get(field)
        if bool(success):
            _finite_number(value, f'frozen formal {field}', nonnegative=True)
        elif value is not None:
            raise M3MetricObservationContractError(
                f'frozen formal solver failure must keep {field} null.'
            )
    ratio = _finite_number(
        formal_case.get('inlier_ratio'),
        'frozen formal inlier_ratio',
        nonnegative=True,
    )
    if ratio > 1.0:
        raise M3MetricObservationContractError(
            'frozen formal inlier_ratio must be in [0,1].'
        )


def _validate_replay_case(replay_case: Mapping, protocol: Mapping) -> None:
    case_identity(replay_case)
    for field, expected in (
        (
            'observation_protocol_version',
            protocol['observation_protocol_version'],
        ),
        ('observation_protocol_hash', protocol['observation_protocol_hash']),
        (
            'source_training_protocol_version',
            protocol['source_training_protocol_version'],
        ),
        (
            'source_training_protocol_hash',
            protocol['source_training_protocol_hash'],
        ),
        (
            'source_defect_evaluation_protocol_version',
            protocol['source_defect_evaluation_protocol_version'],
        ),
        (
            'source_defect_evaluation_protocol_hash',
            protocol['source_defect_evaluation_protocol_hash'],
        ),
        ('source_reference_points', protocol['source_reference_points']),
    ):
        _require_protocol_stamp(replay_case, field, expected)
    success = replay_case.get('solver_success')
    recall = replay_case.get('registration_recall_hit')
    if not isinstance(success, (bool, np.bool_)):
        raise M3MetricObservationContractError(
            'replay solver_success must be boolean.'
        )
    if not isinstance(recall, (bool, np.bool_)):
        raise M3MetricObservationContractError(
            'replay registration_recall_hit must be boolean.'
        )
    if not bool(success) and bool(recall):
        raise M3MetricObservationContractError(
            'replay solver failure cannot be a recall hit.'
        )
    for field in ('rre_deg', 'legacy_parameter_rte_mm', *DIAGNOSTIC_TRE_FIELDS):
        value = replay_case.get(field)
        if bool(success):
            _finite_number(value, f'replay {field}', nonnegative=True)
        elif value is not None:
            raise M3MetricObservationContractError(
                f'replay solver failure must keep {field} null.'
            )
    ratio = _finite_number(
        replay_case.get('inlier_ratio'),
        'replay inlier_ratio',
        nonnegative=True,
    )
    if ratio > 1.0:
        raise M3MetricObservationContractError(
            'replay inlier_ratio must be in [0,1].'
        )


def _continuous_match(formal_value, replay_value) -> tuple:
    if formal_value is None or replay_value is None:
        return (
            formal_value is None and replay_value is None,
            None,
            None,
        )
    formal = _finite_number(formal_value, 'formal reproducibility value')
    replay = _finite_number(replay_value, 'replay reproducibility value')
    absolute = abs(formal - replay)
    scale = max(abs(formal), abs(replay))
    relative = absolute / scale if scale else 0.0
    matches = math.isclose(
        formal,
        replay,
        rel_tol=STRICT_REPRODUCIBILITY_REL_TOLERANCE,
        abs_tol=STRICT_REPRODUCIBILITY_ABS_TOLERANCE,
    )
    return matches, float(absolute), float(relative)


def compare_replay_reproducibility(
    formal_case: Mapping,
    replay_case: Mapping,
) -> dict:
    """Compare formal/replay metrics without treating mismatch as structural."""
    if case_identity(formal_case) != case_identity(replay_case):
        raise M3MetricObservationContractError(
            'frozen formal/replay case identity mismatch.'
        )
    mismatched = []
    absolute_differences = []
    relative_differences = []
    for formal_field, replay_field in REPRODUCIBILITY_DISCRETE_FIELDS:
        formal_value = formal_case.get(formal_field)
        replay_value = replay_case.get(replay_field)
        if formal_value != replay_value or type(formal_value) is not type(
            replay_value
        ):
            mismatched.append(formal_field)
    for formal_field, replay_field in REPRODUCIBILITY_CONTINUOUS_FIELDS:
        matches, absolute, relative = _continuous_match(
            formal_case.get(formal_field),
            replay_case.get(replay_field),
        )
        if absolute is not None:
            absolute_differences.append(absolute)
            relative_differences.append(relative)
        if not matches:
            mismatched.append(formal_field)
    return {
        'status': 'PASS' if not mismatched else 'FAIL',
        'compared_fields': list(REPRODUCIBILITY_COMPARED_FIELDS),
        'abs_tolerance': STRICT_REPRODUCIBILITY_ABS_TOLERANCE,
        'rel_tolerance': STRICT_REPRODUCIBILITY_REL_TOLERANCE,
        'mismatched_fields': mismatched,
        'max_absolute_difference': (
            float(max(absolute_differences)) if absolute_differences else 0.0
        ),
        'max_relative_difference': (
            float(max(relative_differences)) if relative_differences else 0.0
        ),
    }


def build_observation_case(
    formal_case: Mapping,
    replay_case: Mapping,
    observation_protocol: Mapping,
) -> dict:
    """Build one v3 row while preserving the complete frozen formal row."""
    validate_observation_protocol(observation_protocol)
    _validate_formal_case(formal_case, observation_protocol)
    _validate_replay_case(replay_case, observation_protocol)
    identity = case_identity(formal_case)
    if identity != case_identity(replay_case):
        raise M3MetricObservationContractError(
            'frozen formal/replay case identity mismatch.'
        )
    for field in ('case_key', 'variant_index'):
        if field in formal_case and field in replay_case:
            if formal_case[field] != replay_case[field] or type(
                formal_case[field]
            ) is not type(replay_case[field]):
                raise M3MetricObservationContractError(
                    f'frozen formal/replay {field} mismatch.'
                )
    reproducibility = compare_replay_reproducibility(
        formal_case,
        replay_case,
    )
    replay_observation = {
        'replay_solver_success': bool(replay_case['solver_success']),
        'replay_solver_status': replay_case.get('solver_status'),
        'replay_registration_recall_hit': bool(
            replay_case['registration_recall_hit']
        ),
        'replay_rre_deg': replay_case.get('rre_deg'),
        'replay_legacy_parameter_rte_mm': replay_case.get(
            'legacy_parameter_rte_mm'
        ),
        'replay_inlier_ratio': replay_case.get('inlier_ratio'),
        'replay_num_correspondences': replay_case.get('num_correspondences'),
        'replay_num_inliers': replay_case.get('num_inliers'),
        **{field: replay_case.get(field) for field in DIAGNOSTIC_TRE_FIELDS},
    }
    for field in (
        'confidence_mean',
        'confidence_median',
        'confidence_p05',
        'confidence_p95',
        'confidence_min',
        'confidence_max',
        'source_centroid_norm_mm',
        'origin_rotation_lever_proxy_mm',
        'diagnostic_inference_runtime_ms',
    ):
        if field in replay_case:
            replay_observation[field] = replay_case[field]
    return {
        'observation_protocol_version': observation_protocol[
            'observation_protocol_version'
        ],
        'observation_protocol_hash': observation_protocol[
            'observation_protocol_hash'
        ],
        'source_training_protocol_version': observation_protocol[
            'source_training_protocol_version'
        ],
        'source_training_protocol_hash': observation_protocol[
            'source_training_protocol_hash'
        ],
        'source_defect_evaluation_protocol_version': observation_protocol[
            'source_defect_evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': observation_protocol[
            'source_defect_evaluation_protocol_hash'
        ],
        'source_reference_points': observation_protocol[
            'source_reference_points'
        ],
        'case_key': replay_case.get('case_key'),
        'fold_id': identity[0],
        'subject_id': identity[1],
        'defect_id': identity[2],
        'severity': identity[3],
        'variant_id': identity[4],
        'variant_index': replay_case.get('variant_index'),
        'perturbation_seed': identity[5],
        'checkpoint_path': replay_case.get('checkpoint_path'),
        'source_artifact_kind': 'v3_tre_observation',
        'frozen_formal': dict(formal_case),
        'replay_observation': replay_observation,
        'replay_reproducibility': reproducibility,
    }


def validate_observation_case(
    row: Mapping,
    observation_protocol: Mapping,
) -> None:
    """Validate a serialized v3/normalized observation row."""
    validate_observation_protocol(observation_protocol)
    identity = case_identity(row)
    for field, expected in (
        (
            'observation_protocol_version',
            observation_protocol['observation_protocol_version'],
        ),
        (
            'observation_protocol_hash',
            observation_protocol['observation_protocol_hash'],
        ),
        (
            'source_training_protocol_version',
            observation_protocol['source_training_protocol_version'],
        ),
        (
            'source_training_protocol_hash',
            observation_protocol['source_training_protocol_hash'],
        ),
        (
            'source_defect_evaluation_protocol_version',
            observation_protocol['source_defect_evaluation_protocol_version'],
        ),
        (
            'source_defect_evaluation_protocol_hash',
            observation_protocol['source_defect_evaluation_protocol_hash'],
        ),
        (
            'source_reference_points',
            observation_protocol['source_reference_points'],
        ),
    ):
        _require_protocol_stamp(row, field, expected)
    formal = row.get('frozen_formal')
    replay = row.get('replay_observation')
    reproducibility = row.get('replay_reproducibility')
    if not isinstance(formal, Mapping) or not isinstance(replay, Mapping):
        raise M3MetricObservationContractError(
            'observation row requires frozen_formal and replay_observation.'
        )
    _validate_formal_case(formal, observation_protocol)
    if identity != case_identity(formal):
        raise M3MetricObservationContractError(
            'observation/frozen formal identity mismatch.'
        )
    if not isinstance(reproducibility, Mapping):
        raise M3MetricObservationContractError(
            'observation row requires replay_reproducibility.'
        )
    status = reproducibility.get('status')
    mismatched = reproducibility.get('mismatched_fields')
    if status not in ('PASS', 'FAIL') or not isinstance(mismatched, list):
        raise M3MetricObservationContractError(
            'replay_reproducibility status/mismatched_fields are invalid.'
        )
    if (status == 'PASS') != (len(mismatched) == 0):
        raise M3MetricObservationContractError(
            'replay_reproducibility status contradicts mismatched_fields.'
        )
    if reproducibility.get('compared_fields') != list(
        REPRODUCIBILITY_COMPARED_FIELDS
    ):
        raise M3MetricObservationContractError(
            'replay_reproducibility compared_fields mismatch.'
        )
    if reproducibility.get('abs_tolerance') != (
        STRICT_REPRODUCIBILITY_ABS_TOLERANCE
    ) or reproducibility.get('rel_tolerance') != (
        STRICT_REPRODUCIBILITY_REL_TOLERANCE
    ):
        raise M3MetricObservationContractError(
            'replay_reproducibility tolerances must remain strict v2 values.'
        )
    replay_case = {
        **{
            field: row[field]
            for field in REQUIRED_CASE_IDENTITY_FIELDS
        },
        'observation_protocol_version': row['observation_protocol_version'],
        'observation_protocol_hash': row['observation_protocol_hash'],
        'source_training_protocol_version': row[
            'source_training_protocol_version'
        ],
        'source_training_protocol_hash': row['source_training_protocol_hash'],
        'source_defect_evaluation_protocol_version': row[
            'source_defect_evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': row[
            'source_defect_evaluation_protocol_hash'
        ],
        'source_reference_points': row['source_reference_points'],
        'solver_success': replay.get('replay_solver_success'),
        'solver_status': replay.get('replay_solver_status'),
        'registration_recall_hit': replay.get(
            'replay_registration_recall_hit'
        ),
        'rre_deg': replay.get('replay_rre_deg'),
        'legacy_parameter_rte_mm': replay.get(
            'replay_legacy_parameter_rte_mm'
        ),
        'inlier_ratio': replay.get('replay_inlier_ratio'),
        'num_correspondences': replay.get('replay_num_correspondences'),
        'num_inliers': replay.get('replay_num_inliers'),
        **{field: replay.get(field) for field in DIAGNOSTIC_TRE_FIELDS},
    }
    _validate_replay_case(replay_case, observation_protocol)
    expected_reproducibility = compare_replay_reproducibility(
        formal,
        replay_case,
    )
    if dict(reproducibility) != expected_reproducibility:
        raise M3MetricObservationContractError(
            'serialized replay_reproducibility does not match the preserved '
            'formal/replay values.'
        )


def _formal_summary(rows: Sequence[Mapping]) -> dict:
    formal = [row['frozen_formal'] for row in rows]
    total = len(formal)
    successes = sum(bool(row['solver_success']) for row in formal)
    hits = sum(bool(row['registration_recall_hit']) for row in formal)
    return {
        'case_count': total,
        'solver_success_count': successes,
        'solver_failure_count': total - successes,
        'registration_recall': float(hits / total) if total else 0.0,
        'rre_deg': metric_statistics(row.get('rre_deg') for row in formal),
        'rte_mm': metric_statistics(row.get('rte_mm') for row in formal),
        'inlier_ratio': metric_statistics(
            row.get('inlier_ratio') for row in formal
        ),
        'num_correspondences': metric_statistics(
            row.get('num_correspondences') for row in formal
        ),
        'num_inliers': metric_statistics(row.get('num_inliers') for row in formal),
        'authoritative_source': 'frozen_formal',
    }


def _tre_summary(rows: Sequence[Mapping]) -> dict:
    replay = [row['replay_observation'] for row in rows]
    return {
        'case_count': len(replay),
        'valid_tre_observation_count': sum(
            bool(row['replay_solver_success']) for row in replay
        ),
        'includes_reproducibility_failures': True,
        **{
            field: metric_statistics(row.get(field) for row in replay)
            for field in DIAGNOSTIC_TRE_FIELDS
        },
    }


def aggregate_observation_cases(
    rows: Sequence[Mapping],
    observation_protocol: Mapping,
    *,
    expected_case_count: Optional[int] = None,
) -> dict:
    """Aggregate formal metrics and TRE observations on separate paths."""
    cases = list(rows)
    if expected_case_count is not None and len(cases) != expected_case_count:
        raise M3MetricObservationContractError(
            'observation case count mismatch: '
            f'expected={expected_case_count}, actual={len(cases)}.'
        )
    identities = set()
    for row in cases:
        validate_observation_case(row, observation_protocol)
        identity = case_identity(row)
        if identity in identities:
            raise M3MetricObservationContractError(
                f'duplicate observation case identity: {identity!r}.'
            )
        identities.add(identity)
    pass_count = sum(
        row['replay_reproducibility']['status'] == 'PASS' for row in cases
    )
    fail_count = len(cases) - pass_count
    by_fold = defaultdict(list)
    for row in cases:
        by_fold[row['fold_id']].append(row)
    return {
        'observation_protocol_version': observation_protocol[
            'observation_protocol_version'
        ],
        'observation_protocol_hash': observation_protocol[
            'observation_protocol_hash'
        ],
        'case_count': len(cases),
        'identity_unique_count': len(identities),
        'reproducibility_pass_count': pass_count,
        'reproducibility_fail_count': fail_count,
        'reproducibility_pass_rate': (
            float(pass_count / len(cases)) if cases else 0.0
        ),
        'formal_metrics': _formal_summary(cases),
        'tre_observations': _tre_summary(cases),
        'per_fold': {
            fold_id: {
                'case_count': len(by_fold[fold_id]),
                'reproducibility_pass_count': sum(
                    row['replay_reproducibility']['status'] == 'PASS'
                    for row in by_fold[fold_id]
                ),
                'reproducibility_fail_count': sum(
                    row['replay_reproducibility']['status'] == 'FAIL'
                    for row in by_fold[fold_id]
                ),
                'formal_metrics': _formal_summary(by_fold[fold_id]),
                'tre_observations': _tre_summary(by_fold[fold_id]),
            }
            for fold_id in sorted(by_fold)
        },
        'TRE_OBSERVATION_COMPLETE': True,
        'STRICT_REPLAY_REPRODUCIBILITY_COMPLETE': True,
        'STRICT_REPLAY_REPRODUCIBILITY_ALL_PASS': fail_count == 0,
    }


def _formal_from_v2(row: Mapping, protocol: Mapping) -> dict:
    return {
        'protocol_version': protocol['source_training_protocol_version'],
        'protocol_hash': protocol['source_training_protocol_hash'],
        'evaluation_protocol_version': protocol[
            'source_defect_evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': protocol[
            'source_defect_evaluation_protocol_hash'
        ],
        'case_key': row.get('case_key'),
        'fold_id': row['fold_id'],
        'subject_id': row['subject_id'],
        'defect_id': row['defect_id'],
        'severity': row['severity'],
        'variant_id': row['variant_id'],
        'variant_index': row.get('variant_index'),
        'perturbation_seed': row['perturbation_seed'],
        'checkpoint_path': row.get('checkpoint_path'),
        'solver_success': row.get('solver_success'),
        'solver_status': row.get('solver_status'),
        'registration_recall_hit': row.get('registration_recall_hit'),
        'rre_deg': row.get('rre_deg'),
        'rte_mm': row.get('legacy_parameter_rte_mm'),
        'inlier_ratio': row.get('inlier_ratio'),
        'num_correspondences': row.get('num_correspondences'),
        'num_inliers': row.get('num_inliers'),
        'formal_value_provenance': (
            'strict_v2_replay_value_verified_against_frozen_formal'
        ),
    }


def normalize_strict_v2_case(
    row: Mapping,
    observation_protocol: Mapping,
) -> dict:
    """Normalize an official strict-PASS v2 diagnostic row for mixed output."""
    validate_observation_protocol(observation_protocol)
    case_identity(row)
    for field, expected in (
        ('diagnostic_protocol_version', CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION),
        ('diagnostic_protocol_hash', CLEAN10_DIAGNOSTIC_PROTOCOL_HASH),
        (
            'source_training_protocol_version',
            observation_protocol['source_training_protocol_version'],
        ),
        (
            'source_training_protocol_hash',
            observation_protocol['source_training_protocol_hash'],
        ),
        (
            'source_defect_evaluation_protocol_version',
            observation_protocol['source_defect_evaluation_protocol_version'],
        ),
        (
            'source_defect_evaluation_protocol_hash',
            observation_protocol['source_defect_evaluation_protocol_hash'],
        ),
        ('source_reference_points', observation_protocol['source_reference_points']),
    ):
        _require_protocol_stamp(row, field, expected)
    formal = _formal_from_v2(row, observation_protocol)
    replay_for_builder = dict(row)
    replay_for_builder.update(
        {
            'observation_protocol_version': observation_protocol[
                'observation_protocol_version'
            ],
            'observation_protocol_hash': observation_protocol[
                'observation_protocol_hash'
            ],
        }
    )
    normalized = build_observation_case(
        formal,
        replay_for_builder,
        observation_protocol,
    )
    normalized['source_artifact_kind'] = 'strict_v2_replay_pass'
    if normalized['replay_reproducibility']['status'] != 'PASS':
        raise M3MetricObservationContractError(
            'strict v2 source row did not normalize to reproducibility PASS.'
        )
    return normalized


def _unique_identity_map(rows: Sequence[Mapping], label: str) -> dict:
    result = {}
    for row in rows:
        identity = case_identity(row)
        if identity in result:
            raise M3MetricObservationContractError(
                f'duplicate {label} case identity: {identity!r}.'
            )
        result[identity] = row
    return result


def aggregate_mixed_clean10_750(
    v2_cases_by_fold: Mapping[str, Sequence[Mapping]],
    v3_fold5_cases: Sequence[Mapping],
    observation_protocol: Mapping,
    *,
    expected_cases_by_fold: Optional[Mapping[str, Sequence[Mapping]]] = None,
) -> dict:
    """Build a validated 600 strict-v2 + 150 observation-v3 CPU aggregate."""
    validate_observation_protocol(observation_protocol)
    expected_v2_folds = ('Fold1', 'Fold2', 'Fold3', 'Fold4')
    if tuple(v2_cases_by_fold) != expected_v2_folds:
        raise M3MetricObservationContractError(
            'v2 sources must contain ordered Fold1 through Fold4 exactly.'
        )
    normalized = []
    for fold_id in expected_v2_folds:
        rows = list(v2_cases_by_fold[fold_id])
        if len(rows) != EXPECTED_CASE_COUNTS_BY_FOLD[fold_id]:
            raise M3MetricObservationContractError(
                f'{fold_id} v2 case count mismatch: expected=150, '
                f'actual={len(rows)}.'
            )
        if any(row.get('fold_id') != fold_id for row in rows):
            raise M3MetricObservationContractError(
                f'{fold_id} v2 source contains a different fold identity.'
            )
        normalized.extend(
            normalize_strict_v2_case(row, observation_protocol) for row in rows
        )
    fold5 = list(v3_fold5_cases)
    if len(fold5) != EXPECTED_CASE_COUNTS_BY_FOLD['Fold5']:
        raise M3MetricObservationContractError(
            'Fold5 v3 case count mismatch: '
            f'expected=150, actual={len(fold5)}.'
        )
    for row in fold5:
        validate_observation_case(row, observation_protocol)
        if row.get('fold_id') != 'Fold5':
            raise M3MetricObservationContractError(
                'Fold5 v3 source contains a different fold identity.'
            )
    normalized.extend(fold5)
    identities = _unique_identity_map(normalized, 'mixed aggregate')
    if len(identities) != EXPECTED_TEST_CASE_COUNT:
        raise M3MetricObservationContractError(
            'mixed aggregate must contain exactly 750 unique identities.'
        )
    pat6_count = sum(row['subject_id'] == 'Pat6' for row in normalized)
    if pat6_count != EXPECTED_PAT6_CASE_COUNT:
        raise M3MetricObservationContractError(
            f'mixed aggregate Pat6 count mismatch: expected=0, actual={pat6_count}.'
        )
    if expected_cases_by_fold is not None:
        if tuple(expected_cases_by_fold) != tuple(EXPECTED_CASE_COUNTS_BY_FOLD):
            raise M3MetricObservationContractError(
                'expected manifest mapping must contain ordered Fold1-Fold5.'
            )
        expected = [
            row
            for fold_id in EXPECTED_CASE_COUNTS_BY_FOLD
            for row in expected_cases_by_fold[fold_id]
        ]
        expected_map = _unique_identity_map(expected, 'expected manifest')
        if len(expected_map) != EXPECTED_TEST_CASE_COUNT:
            raise M3MetricObservationContractError(
                'expected manifest union must contain 750 unique identities.'
            )
        if set(identities) != set(expected_map):
            missing = sorted(set(expected_map).difference(identities))
            unexpected = sorted(set(identities).difference(expected_map))
            raise M3MetricObservationContractError(
                'mixed aggregate identity mismatch; first_missing='
                f'{missing[0] if missing else None!r}, first_unexpected='
                f'{unexpected[0] if unexpected else None!r}.'
            )
    summary = aggregate_observation_cases(
        normalized,
        observation_protocol,
        expected_case_count=EXPECTED_TEST_CASE_COUNT,
    )
    reproducibility_summary = {
        'case_count': summary['case_count'],
        'reproducibility_pass_count': summary['reproducibility_pass_count'],
        'reproducibility_fail_count': summary['reproducibility_fail_count'],
        'reproducibility_pass_rate': summary['reproducibility_pass_rate'],
        'strict_abs_tolerance': STRICT_REPRODUCIBILITY_ABS_TOLERANCE,
        'strict_rel_tolerance': STRICT_REPRODUCIBILITY_REL_TOLERANCE,
        'TRE_OBSERVATION_COMPLETE': summary['TRE_OBSERVATION_COMPLETE'],
        'STRICT_REPLAY_REPRODUCIBILITY_COMPLETE': summary[
            'STRICT_REPLAY_REPRODUCIBILITY_COMPLETE'
        ],
        'STRICT_REPLAY_REPRODUCIBILITY_ALL_PASS': summary[
            'STRICT_REPLAY_REPRODUCIBILITY_ALL_PASS'
        ],
        'mismatched_cases': [
            {
                'case_key': row.get('case_key'),
                'fold_id': row['fold_id'],
                'subject_id': row['subject_id'],
                'defect_id': row['defect_id'],
                'severity': row['severity'],
                'variant_id': row['variant_id'],
                'perturbation_seed': row['perturbation_seed'],
                **row['replay_reproducibility'],
            }
            for row in normalized
            if row['replay_reproducibility']['status'] == 'FAIL'
        ],
    }
    provenance = {
        'observation_protocol_version': observation_protocol[
            'observation_protocol_version'
        ],
        'observation_protocol_hash': observation_protocol[
            'observation_protocol_hash'
        ],
        'source_training_protocol_version': observation_protocol[
            'source_training_protocol_version'
        ],
        'source_training_protocol_hash': observation_protocol[
            'source_training_protocol_hash'
        ],
        'source_defect_evaluation_protocol_version': observation_protocol[
            'source_defect_evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': observation_protocol[
            'source_defect_evaluation_protocol_hash'
        ],
        'source_reference_points': observation_protocol[
            'source_reference_points'
        ],
        'centroid_tre_definition': observation_protocol[
            'centroid_tre_definition'
        ],
        'point_tre_definition': observation_protocol['point_tre_definition'],
        'per_fold': {
            **{
                fold_id: {
                    'source': 'strict v2 replay PASS',
                    'case_count': EXPECTED_CASE_COUNTS_BY_FOLD[fold_id],
                }
                for fold_id in expected_v2_folds
            },
            'Fold5': {
                'source': 'v3 TRE observation',
                'case_count': EXPECTED_CASE_COUNTS_BY_FOLD['Fold5'],
                'strict_replay_reproducibility_may_contain_FAIL': True,
            },
        },
        'TRE_OBSERVATION_COMPLETE': True,
        'STRICT_REPLAY_REPRODUCIBILITY_COMPLETE': True,
        'STRICT_REPLAY_REPRODUCIBILITY_ALL_PASS': summary[
            'STRICT_REPLAY_REPRODUCIBILITY_ALL_PASS'
        ],
        'model_loaded': False,
        'gpu_used': False,
    }
    return {
        'cases': normalized,
        'summary': summary,
        'reproducibility_summary': reproducibility_summary,
        'provenance': provenance,
    }


__all__ = [
    'EXPECTED_CASE_COUNTS_BY_FOLD',
    'EXPECTED_DEFECT_CONDITION_COUNT',
    'EXPECTED_DEFECT_INSTANCE_COUNT',
    'EXPECTED_PAT6_CASE_COUNT',
    'EXPECTED_PATIENT_COUNT',
    'EXPECTED_TEST_CASE_COUNT',
    'M3MetricObservationContractError',
    'OBSERVATION_PROTOCOL_HASH',
    'OBSERVATION_PROTOCOL_VERSION',
    'REPRODUCIBILITY_COMPARED_FIELDS',
    'SOURCE_EVALUATION_PROTOCOL_HASH',
    'SOURCE_EVALUATION_PROTOCOL_VERSION',
    'SOURCE_REFERENCE_POINTS',
    'SOURCE_TRAINING_PROTOCOL_HASH',
    'SOURCE_TRAINING_PROTOCOL_VERSION',
    'STRICT_REPRODUCIBILITY_ABS_TOLERANCE',
    'STRICT_REPRODUCIBILITY_REL_TOLERANCE',
    'aggregate_mixed_clean10_750',
    'aggregate_observation_cases',
    'build_observation_case',
    'compare_replay_reproducibility',
    'compute_observation_protocol_hash',
    'load_observation_protocol',
    'normalize_strict_v2_case',
    'validate_observation_case',
    'validate_observation_protocol',
    'validate_source_protocols',
]
