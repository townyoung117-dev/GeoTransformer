"""Fail-closed validation-only aggregation for frozen M4-2D selection.

The normal execution path reads exactly one allowlisted
``validation/validation_cases.jsonl`` file for each of the frozen four
lambda values and five folds.  It never opens training logs, checkpoints,
completion markers, or any evaluation artifact outside that exact grid.
"""

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


AGGREGATOR_VERSION = 'm4_osseous_strength_selection_aggregator_v1'
FROZEN_PROTOCOL_VERSION = 'm4_osseous_strength_selection_clean10_v1'
FROZEN_PROTOCOL_SHA256 = (
    'a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38'
)
_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = (
    _MODULE_DIR / 'protocols' / 'm4_osseous_strength_selection_clean10_v1.json'
)
SUMMARY_BASENAME = 'validation_selection_summary.json'

_FROZEN_LAMBDAS = (0.25, 0.5, 1.0, 2.0)
_FROZEN_LAMBDA_DIRECTORIES = {
    0.25: 'lambda_0p25',
    0.5: 'lambda_0p50',
    1.0: 'lambda_1p00',
    2.0: 'lambda_2p00',
}
_FROZEN_FOLDS = ('Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5')
_FROZEN_FOLD_DIRECTORIES = {
    'Fold1': 'fold1',
    'Fold2': 'fold2',
    'Fold3': 'fold3',
    'Fold4': 'fold4',
    'Fold5': 'fold5',
}
_FROZEN_VAL_SUBJECTS = {
    'Fold1': ('Pat7', 'Pat4'),
    'Fold2': ('Pat11', 'Pat3'),
    'Fold3': ('Pat8', 'Pat9'),
    'Fold4': ('Pat1', 'Pat2'),
    'Fold5': ('Pat12', 'Pat5'),
}
_FROZEN_READY_SUBJECTS = (
    'Pat1', 'Pat2', 'Pat3', 'Pat4', 'Pat5',
    'Pat7', 'Pat8', 'Pat9', 'Pat11', 'Pat12',
)
_FROZEN_DEFECTS = (
    'defect_001_left_maxilla_cheek_small',
    'defect_001_left_maxilla_cheek_medium',
    'defect_001_left_maxilla_cheek_large',
    'defect_001_right_maxilla_cheek_small',
    'defect_001_right_maxilla_cheek_medium',
)
_FROZEN_SEVERITIES = ('mild', 'moderate', 'hard')
_CASE_FIELDS = (
    'artifact_scope',
    'protocol_version',
    'protocol_sha256',
    'lambda_oss',
    'fold_id',
    'subject_id',
    'defect_id',
    'severity',
    'variant_id',
    'perturbation_seed',
    'solver_success',
    'solver_status',
    'registration_recall_hit',
    'centroid_tre_mm',
    'point_tre_mean_mm',
    'rre_deg',
)
_IDENTITY_FIELDS = (
    'lambda_oss',
    'fold_id',
    'subject_id',
    'defect_id',
    'severity',
    'variant_id',
    'perturbation_seed',
)
_METRICS = ('centroid_tre_mm', 'point_tre_mean_mm', 'rre_deg')
_PATIENT_METRIC_FIELDS = {
    'centroid_tre_mm': 'median_centroid_tre_mm',
    'point_tre_mean_mm': 'median_point_tre_mean_mm',
    'rre_deg': 'median_rre_deg',
}


class M4OsseousStrengthAggregationError(RuntimeError):
    """Raised whenever validation-only selection cannot be proven safe."""


def _reject_duplicate_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise M4OsseousStrengthAggregationError(
                f'duplicate JSON object field: {key!r}.'
            )
        result[key] = value
    return result


def _reject_nonstandard_json_number(token):
    raise M4OsseousStrengthAggregationError(
        f'non-standard JSON number is forbidden: {token!r}.'
    )


def _canonical_json(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M4OsseousStrengthAggregationError(
            'value is not canonical-JSON serializable.'
        ) from error


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _load_json_object(path, name):
    path = Path(path)
    if _is_linklike(path):
        raise M4OsseousStrengthAggregationError(
            f'{name} must not be a symbolic link or reparse point: {path}.'
        )
    if not path.is_file():
        raise M4OsseousStrengthAggregationError(f'{name} is missing: {path}.')
    try:
        with path.open('r', encoding='utf-8') as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_fields,
                parse_constant=_reject_nonstandard_json_number,
            )
    except M4OsseousStrengthAggregationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M4OsseousStrengthAggregationError(
            f'cannot load {name} {path}: {error}'
        ) from error
    if not isinstance(value, Mapping):
        raise M4OsseousStrengthAggregationError(f'{name} must be a JSON object.')
    return value


def _protocol_value(protocol, *keys):
    value = protocol
    traversed = []
    for key in keys:
        traversed.append(str(key))
        if not isinstance(value, Mapping) or key not in value:
            raise M4OsseousStrengthAggregationError(
                f'frozen protocol is missing {".".join(traversed)}.'
            )
        value = value[key]
    return value


def _require_equal(actual, expected, name):
    if actual != expected or type(actual) is not type(expected):
        raise M4OsseousStrengthAggregationError(
            f'{name} differs from the frozen contract; '
            f'expected={expected!r}, actual={actual!r}.'
        )


def _is_linklike(path):
    path = Path(path)
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), 'st_file_attributes', 0)
    except OSError:
        return False
    reparse_flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    return bool(attributes & reparse_flag)


def _verify_protocol_sidecar(protocol_path, digest):
    sidecar = Path(protocol_path).with_suffix('.sha256')
    if _is_linklike(sidecar) or not sidecar.is_file():
        raise M4OsseousStrengthAggregationError(
            f'frozen protocol SHA sidecar is missing or link-like: {sidecar}.'
        )
    try:
        text = sidecar.read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError) as error:
        raise M4OsseousStrengthAggregationError(
            f'cannot read frozen protocol SHA sidecar {sidecar}: {error}'
        ) from error
    expected = (
        f'{digest}  {Path(protocol_path).name} '
        '(canonical JSON excluding protocol_sha256)'
    )
    if text != expected:
        raise M4OsseousStrengthAggregationError(
            'frozen protocol SHA sidecar does not exactly bind the manifest.'
        )


def validate_frozen_protocol(protocol, *, protocol_path=None):
    """Validate the canonical post-freeze SHA and aggregator-facing semantics."""
    if not isinstance(protocol, Mapping):
        raise M4OsseousStrengthAggregationError(
            'frozen selection protocol must be a JSON object.'
        )
    _require_equal(
        protocol.get('protocol_version'),
        FROZEN_PROTOCOL_VERSION,
        'protocol_version',
    )
    stored_digest = protocol.get('protocol_sha256')
    if not isinstance(stored_digest, str):
        raise M4OsseousStrengthAggregationError(
            'protocol_sha256 must be a lowercase SHA-256 string.'
        )
    hash_input = dict(protocol)
    hash_input.pop('protocol_sha256', None)
    computed_digest = _canonical_sha256(hash_input)
    if not hmac.compare_digest(stored_digest, computed_digest):
        raise M4OsseousStrengthAggregationError(
            'frozen protocol stored and computed SHA-256 values differ.'
        )
    if not hmac.compare_digest(stored_digest, FROZEN_PROTOCOL_SHA256):
        raise M4OsseousStrengthAggregationError(
            'protocol does not match the embedded post-freeze SHA trust anchor.'
        )
    if protocol_path is not None:
        _verify_protocol_sidecar(protocol_path, stored_digest)

    _require_equal(
        protocol.get('protocol_status'),
        'FROZEN_BEFORE_GPU_SELECTION',
        'protocol_status',
    )
    _require_equal(protocol.get('lambda_oss_grid'), list(_FROZEN_LAMBDAS), 'lambda grid')
    _require_equal(
        _protocol_value(protocol, 'clean10', 'ready_subject_ids'),
        list(_FROZEN_READY_SUBJECTS),
        'clean10 ready subjects',
    )
    _require_equal(
        _protocol_value(protocol, 'clean10', 'excluded_subject_ids'),
        ['Pat6'],
        'clean10 excluded subjects',
    )
    for fold_id in _FROZEN_FOLDS:
        _require_equal(
            _protocol_value(protocol, 'clean10', 'folds', fold_id, 'val_subject_ids'),
            list(_FROZEN_VAL_SUBJECTS[fold_id]),
            f'{fold_id} validation subjects',
        )
    _require_equal(protocol.get('defect_conditions'), list(_FROZEN_DEFECTS), 'defects')
    perturbations = _protocol_value(protocol, 'validation_protocol', 'perturbations')
    if not isinstance(perturbations, list) or len(perturbations) != 3:
        raise M4OsseousStrengthAggregationError(
            'frozen validation perturbations must contain three entries.'
        )
    for item, severity in zip(perturbations, _FROZEN_SEVERITIES):
        _require_equal(item.get('purpose'), 'val', f'{severity} purpose')
        _require_equal(item.get('severity'), severity, f'{severity} name')
        _require_equal(item.get('variant_id'), 0, f'{severity} variant')
    expected_counts = {
        'expected_cases_per_patient': 15,
        'expected_patients_per_fold': 2,
        'expected_cases_per_fold': 30,
        'expected_unique_patients_per_lambda': 10,
        'expected_cases_per_lambda': 150,
    }
    for field, expected in expected_counts.items():
        _require_equal(
            _protocol_value(protocol, 'validation_protocol', field),
            expected,
            f'validation_protocol.{field}',
        )
    _require_equal(
        _protocol_value(protocol, 'validation_protocol', 'case_identity'),
        list(_IDENTITY_FIELDS),
        'case identity',
    )
    _require_equal(
        _protocol_value(protocol, 'validation_only_artifact', 'required_case_fields'),
        list(_CASE_FIELDS),
        'validation case allowlist',
    )
    _require_equal(
        _protocol_value(protocol, 'validation_only_artifact', 'basename'),
        'validation_cases.jsonl',
        'validation artifact basename',
    )
    _require_equal(
        _protocol_value(protocol, 'validation_only_artifact', 'scope_field', 'required_value'),
        'validation_only',
        'validation artifact scope',
    )
    _require_equal(
        _protocol_value(protocol, 'validation_only_artifact', 'forbidden_result_key_token'),
        'test',
        'result-key firewall token',
    )
    _require_equal(
        _protocol_value(protocol, 'no_test_firewall', 'aggregator_input_policy'),
        'only allowlisted validation-only artifacts; reject result keys or values that claim test scope',
        'aggregator firewall policy',
    )
    _require_equal(
        _protocol_value(protocol, 'output_layout', 'root'),
        'checkpoints/m4_osseous_strength_selection_v1',
        'frozen output root',
    )
    for lambda_oss, directory in _FROZEN_LAMBDA_DIRECTORIES.items():
        actual = None
        for key, value in _protocol_value(protocol, 'output_layout', 'lambda_directories').items():
            if isinstance(key, str) and float(key) == lambda_oss:
                actual = value
        _require_equal(actual, directory, f'lambda {lambda_oss} directory')
    _require_equal(
        _protocol_value(protocol, 'output_layout', 'fold_directories'),
        [_FROZEN_FOLD_DIRECTORIES[fold] for fold in _FROZEN_FOLDS],
        'fold directories',
    )
    _require_equal(
        _protocol_value(protocol, 'output_layout', 'per_fold', 'validation_directory'),
        'validation',
        'validation directory',
    )
    _require_equal(
        _protocol_value(
            protocol,
            'source_protocols',
            'clean10_training',
            'canonical_sha256',
        ),
        '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c',
        'seed source protocol SHA',
    )
    _require_equal(
        _protocol_value(protocol, 'training_settings', 'perturbation_seed_scheme_version'),
        'm3_6b_seed_v1',
        'seed scheme',
    )
    _require_equal(
        _protocol_value(protocol, 'training_settings', 'perturbation_root_seed'),
        20260815,
        'perturbation root seed',
    )
    _require_equal(
        _protocol_value(protocol, 'selection_rule', 'primary', 'practical_tie_threshold_mm'),
        0.25,
        'primary tie threshold',
    )
    _require_equal(
        _protocol_value(protocol, 'selection_rule', 'secondary', 'practical_tie_threshold_mm'),
        0.25,
        'secondary tie threshold',
    )
    _require_equal(
        _protocol_value(protocol, 'selection_rule', 'tertiary', 'practical_tie_threshold_deg'),
        0.25,
        'tertiary tie threshold',
    )
    _require_equal(
        _protocol_value(protocol, 'selection_rule', 'final_output_key'),
        'SELECTED_LAMBDA_OSS',
        'selection output key',
    )
    return protocol


def load_frozen_protocol(path=DEFAULT_PROTOCOL_PATH):
    path = Path(path)
    protocol = _load_json_object(path, 'frozen M4 selection protocol')
    return validate_frozen_protocol(protocol, protocol_path=path)


def _derive_expected_seed_from_validated(
    protocol,
    fold_id,
    subject_id,
    severity,
    variant_id=0,
):
    payload = {
        'scheme_version': _protocol_value(
            protocol, 'training_settings', 'perturbation_seed_scheme_version'
        ),
        'protocol_hash': _protocol_value(
            protocol,
            'source_protocols',
            'clean10_training',
            'canonical_sha256',
        ),
        'fold_id': fold_id,
        'root_seed': _protocol_value(
            protocol, 'training_settings', 'perturbation_root_seed'
        ),
        'purpose': 'val',
        'epoch': None,
        'subject_id': subject_id,
        'severity': severity,
        'variant_id': variant_id,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


def derive_expected_seed(protocol, fold_id, subject_id, severity, variant_id=0):
    """Reproduce the frozen m3_6b_seed_v1 uint64 derivation locally."""
    protocol = validate_frozen_protocol(protocol)
    return _derive_expected_seed_from_validated(
        protocol,
        fold_id,
        subject_id,
        severity,
        variant_id,
    )


def build_expected_case_identities(protocol, fold_id, lambda_oss):
    """Return the exact 30 frozen identities for one lambda/fold pair."""
    protocol = validate_frozen_protocol(protocol)
    if lambda_oss not in _FROZEN_LAMBDAS or isinstance(lambda_oss, bool):
        raise M4OsseousStrengthAggregationError(
            f'lambda_oss is outside the frozen grid: {lambda_oss!r}.'
        )
    if fold_id not in _FROZEN_FOLDS:
        raise M4OsseousStrengthAggregationError(f'unknown fold: {fold_id!r}.')
    rows = []
    for subject_id in _FROZEN_VAL_SUBJECTS[fold_id]:
        if subject_id == 'Pat6':
            raise M4OsseousStrengthAggregationError('Pat6 is forbidden.')
        for defect_id in _FROZEN_DEFECTS:
            for severity in _FROZEN_SEVERITIES:
                rows.append(
                    {
                        'lambda_oss': float(lambda_oss),
                        'fold_id': fold_id,
                        'subject_id': subject_id,
                        'defect_id': defect_id,
                        'severity': severity,
                        'variant_id': 0,
                        'perturbation_seed': _derive_expected_seed_from_validated(
                            protocol, fold_id, subject_id, severity, 0
                        ),
                    }
                )
    if len(rows) != 30:
        raise M4OsseousStrengthAggregationError(
            f'{fold_id} expected identity count is not 30.'
        )
    return tuple(rows)


def _identity_key(row):
    return _canonical_json([row[field] for field in _IDENTITY_FIELDS])


def _reject_result_firewall_tokens(value, path='row'):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise M4OsseousStrengthAggregationError(
                    f'{path} contains a non-string JSON key.'
                )
            if 'test' in key.casefold():
                raise M4OsseousStrengthAggregationError(
                    f'{path} contains forbidden result key token {key!r}.'
                )
            _reject_result_firewall_tokens(item, f'{path}.{key}')
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_result_firewall_tokens(item, f'{path}[{index}]')
    elif isinstance(value, str):
        tokens = {
            token
            for token in re.split(r'[^a-z0-9]+', value.casefold().strip())
            if token
        }
        if tokens.intersection({'test', 'testing'}):
            raise M4OsseousStrengthAggregationError(
                f'{path} claims forbidden evaluation scope.'
            )


def _require_json_number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M4OsseousStrengthAggregationError(f'{name} must be a JSON number.')
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise M4OsseousStrengthAggregationError(
            f'{name} must be a finite JSON number.'
        ) from error
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise M4OsseousStrengthAggregationError(
            f'{name} must be a finite{ " non-negative" if nonnegative else ""} number.'
        )
    return number


def validate_validation_cases(rows, protocol, fold_id, lambda_oss):
    """Validate one exact allowlisted 30-row validation artifact."""
    protocol = validate_frozen_protocol(protocol)
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence):
        raise M4OsseousStrengthAggregationError(
            'validation cases must be a sequence of JSON objects.'
        )
    expected_rows = build_expected_case_identities(protocol, fold_id, lambda_oss)
    expected = {_identity_key(row): row for row in expected_rows}
    if len(rows) != len(expected):
        raise M4OsseousStrengthAggregationError(
            f'{fold_id}/lambda={lambda_oss} must contain exactly 30 cases.'
        )
    seen = set()
    for index, row in enumerate(rows):
        name = f'{fold_id}/lambda={lambda_oss} row {index}'
        if not isinstance(row, Mapping):
            raise M4OsseousStrengthAggregationError(f'{name} must be an object.')
        _reject_result_firewall_tokens(row, name)
        actual_fields = set(row)
        expected_fields = set(_CASE_FIELDS)
        if actual_fields != expected_fields:
            raise M4OsseousStrengthAggregationError(
                f'{name} fields mismatch; missing='
                f'{sorted(expected_fields - actual_fields)}, unexpected='
                f'{sorted(actual_fields - expected_fields)}.'
            )
        _require_equal(row['artifact_scope'], 'validation_only', f'{name} scope')
        _require_equal(
            row['protocol_version'], FROZEN_PROTOCOL_VERSION, f'{name} protocol version'
        )
        _require_equal(
            row['protocol_sha256'], FROZEN_PROTOCOL_SHA256, f'{name} protocol SHA'
        )
        if row['subject_id'] == 'Pat6':
            raise M4OsseousStrengthAggregationError(f'{name} contains excluded Pat6.')
        key = _identity_key(row)
        if key in seen:
            raise M4OsseousStrengthAggregationError(f'{name} duplicates an identity.')
        expected_row = expected.get(key)
        if expected_row is None:
            raise M4OsseousStrengthAggregationError(
                f'{name} has an identity outside the frozen validation manifest.'
            )
        for field in _IDENTITY_FIELDS:
            _require_equal(row[field], expected_row[field], f'{name}.{field}')
        seen.add(key)
        success = row['solver_success']
        if not isinstance(success, bool):
            raise M4OsseousStrengthAggregationError(
                f'{name}.solver_success must be boolean.'
            )
        status = row['solver_status']
        if not isinstance(status, str) or not status.strip():
            raise M4OsseousStrengthAggregationError(
                f'{name}.solver_status must be a non-empty string.'
            )
        if not isinstance(row['registration_recall_hit'], bool):
            raise M4OsseousStrengthAggregationError(
                f'{name}.registration_recall_hit must be boolean.'
            )
        if success:
            if status != 'success':
                raise M4OsseousStrengthAggregationError(
                    f'{name} success row must use solver_status="success".'
                )
            for metric in _METRICS:
                _require_json_number(row[metric], f'{name}.{metric}', nonnegative=True)
        else:
            if status == 'success':
                raise M4OsseousStrengthAggregationError(
                    f'{name} failure row cannot use solver_status="success".'
                )
            if row['registration_recall_hit'] is not False:
                raise M4OsseousStrengthAggregationError(
                    f'{name} failure row cannot be a recall hit.'
                )
            if any(row[metric] is not None for metric in _METRICS):
                raise M4OsseousStrengthAggregationError(
                    f'{name} failure metrics must all be null.'
                )
    if seen != set(expected):
        raise M4OsseousStrengthAggregationError(
            f'{fold_id}/lambda={lambda_oss} is missing frozen identities.'
        )
    return tuple(rows)


def _load_jsonl(path, name):
    path = Path(path)
    if _is_linklike(path):
        raise M4OsseousStrengthAggregationError(
            f'{name} must not be a symbolic link or reparse point: {path}.'
        )
    if not path.is_file():
        raise M4OsseousStrengthAggregationError(f'{name} is missing: {path}.')
    rows = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise M4OsseousStrengthAggregationError(
                        f'{name} contains a blank line at {line_number}.'
                    )
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_fields,
                        parse_constant=_reject_nonstandard_json_number,
                    )
                except M4OsseousStrengthAggregationError:
                    raise
                except json.JSONDecodeError as error:
                    raise M4OsseousStrengthAggregationError(
                        f'{name} has invalid JSON at line {line_number}: {error}'
                    ) from error
                if not isinstance(value, Mapping):
                    raise M4OsseousStrengthAggregationError(
                        f'{name} line {line_number} must be an object.'
                    )
                rows.append(value)
    except M4OsseousStrengthAggregationError:
        raise
    except (OSError, UnicodeError) as error:
        raise M4OsseousStrengthAggregationError(
            f'cannot read {name} {path}: {error}'
        ) from error
    return rows


def collapse_patient_rows(rows, protocol, fold_id, lambda_oss):
    """Collapse 30 validated cases to the two frozen patient units."""
    protocol = validate_frozen_protocol(protocol)
    rows = validate_validation_cases(rows, protocol, fold_id, lambda_oss)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['subject_id']].append(row)
    patient_rows = []
    for subject_id in _FROZEN_VAL_SUBJECTS[fold_id]:
        cases = grouped[subject_id]
        if len(cases) != 15:
            raise M4OsseousStrengthAggregationError(
                f'{fold_id}/{subject_id} must contain exactly 15 cases.'
            )
        successes = [row for row in cases if row['solver_success']]
        recall_hits = sum(row['registration_recall_hit'] for row in cases)
        patient = {
            'lambda_oss': float(lambda_oss),
            'fold_id': fold_id,
            'subject_id': subject_id,
            'case_count': len(cases),
            'solver_success_count': len(successes),
            'solver_failure_count': len(cases) - len(successes),
            'solver_success_rate': len(successes) / len(cases),
            'registration_recall_hits': recall_hits,
            'registration_recall_rate': recall_hits / len(cases),
            'candidate_complete': bool(successes),
        }
        for metric, patient_field in _PATIENT_METRIC_FIELDS.items():
            patient[patient_field] = (
                float(statistics.median(row[metric] for row in successes))
                if successes
                else None
            )
        patient_rows.append(patient)
    if set(grouped) != set(_FROZEN_VAL_SUBJECTS[fold_id]):
        raise M4OsseousStrengthAggregationError(
            f'{fold_id} patient identities differ from the frozen mapping.'
        )
    return tuple(patient_rows)


def _p95_linear(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise M4OsseousStrengthAggregationError('cannot summarize an empty metric.')
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _metric_statistics(patient_rows, patient_field, subject_order):
    if any(row[patient_field] is None for row in patient_rows):
        return None
    values = [float(row[patient_field]) for row in patient_rows]
    best = min(
        patient_rows,
        key=lambda row: (float(row[patient_field]), subject_order[row['subject_id']]),
    )
    worst = min(
        patient_rows,
        key=lambda row: (-float(row[patient_field]), subject_order[row['subject_id']]),
    )
    return {
        'mean': sum(values) / len(values),
        'median': float(statistics.median(values)),
        'p95_linear': _p95_linear(values),
        'best_patient': {
            'subject_id': best['subject_id'],
            'value': float(best[patient_field]),
        },
        'worst_patient': {
            'subject_id': worst['subject_id'],
            'value': float(worst[patient_field]),
        },
    }


def _summarize_patients(patient_rows, protocol):
    order = {
        subject_id: index
        for index, subject_id in enumerate(
            _protocol_value(protocol, 'clean10', 'ready_subject_ids')
        )
    }
    return {
        metric: _metric_statistics(patient_rows, patient_field, order)
        for metric, patient_field in _PATIENT_METRIC_FIELDS.items()
    }


def _descriptive_counts(patient_rows):
    case_count = sum(row['case_count'] for row in patient_rows)
    success_count = sum(row['solver_success_count'] for row in patient_rows)
    failure_count = sum(row['solver_failure_count'] for row in patient_rows)
    recall_hits = sum(row['registration_recall_hits'] for row in patient_rows)
    if case_count <= 0 or success_count + failure_count != case_count:
        raise M4OsseousStrengthAggregationError(
            'descriptive patient counts are internally inconsistent.'
        )
    return {
        'case_count': case_count,
        'solver_success_count': success_count,
        'solver_failure_count': failure_count,
        'solver_success_rate': success_count / case_count,
        'registration_recall_hits': recall_hits,
        'registration_recall_rate': recall_hits / case_count,
    }


def _build_lambda_summary(protocol, lambda_oss, fold_rows):
    patient_rows = []
    fold_summaries = []
    for fold_id in _FROZEN_FOLDS:
        patients = list(
            collapse_patient_rows(fold_rows[fold_id], protocol, fold_id, lambda_oss)
        )
        patient_rows.extend(patients)
        fold_summaries.append(
            {
                'lambda_oss': float(lambda_oss),
                'fold_id': fold_id,
                'patient_count': len(patients),
                'candidate_complete': all(row['candidate_complete'] for row in patients),
                **_descriptive_counts(patients),
                'metrics': _summarize_patients(patients, protocol),
            }
        )
    subjects = [row['subject_id'] for row in patient_rows]
    if len(patient_rows) != 10 or len(set(subjects)) != 10:
        raise M4OsseousStrengthAggregationError(
            f'lambda={lambda_oss} must collapse to ten unique patients.'
        )
    if set(subjects) != set(_FROZEN_READY_SUBJECTS):
        raise M4OsseousStrengthAggregationError(
            f'lambda={lambda_oss} patient set differs from clean10.'
        )
    complete = all(row['candidate_complete'] for row in patient_rows)
    metrics = _summarize_patients(patient_rows, protocol)
    selection_metrics = {
        metric: (metrics[metric]['median'] if metrics[metric] is not None else None)
        for metric in _METRICS
    }
    return {
        'lambda_oss': float(lambda_oss),
        'unique_patient_count': len(patient_rows),
        'candidate_complete': complete,
        **_descriptive_counts(patient_rows),
        'incomplete_patient_ids': [
            row['subject_id'] for row in patient_rows if not row['candidate_complete']
        ],
        'patient_rows': patient_rows,
        'fold_summaries': fold_summaries,
        'metrics': metrics,
        'selection_metrics': selection_metrics,
    }


def apply_selection_rule(lambda_summaries, protocol):
    """Apply the three anchored-to-stage-min filters, then smallest lambda."""
    protocol = validate_frozen_protocol(protocol)
    if (
        isinstance(lambda_summaries, (str, bytes, Mapping))
        or not isinstance(lambda_summaries, Sequence)
        or len(lambda_summaries) != 4
    ):
        raise M4OsseousStrengthAggregationError(
            'selection requires exactly four lambda summaries.'
        )
    for summary in lambda_summaries:
        if not isinstance(summary, Mapping):
            raise M4OsseousStrengthAggregationError(
                'each lambda summary must be a mapping.'
            )
    actual_lambdas = [summary.get('lambda_oss') for summary in lambda_summaries]
    if (
        any(type(value) is not float for value in actual_lambdas)
        or set(actual_lambdas) != set(_FROZEN_LAMBDAS)
    ):
        raise M4OsseousStrengthAggregationError(
            'selection summaries must contain each frozen lambda exactly once.'
        )
    lambda_summaries = sorted(
        lambda_summaries, key=lambda summary: summary['lambda_oss']
    )
    for summary in lambda_summaries:
        if not isinstance(summary.get('candidate_complete'), bool):
            raise M4OsseousStrengthAggregationError(
                'candidate_complete must be boolean.'
            )
        metrics = summary.get('selection_metrics')
        if not isinstance(metrics, Mapping):
            raise M4OsseousStrengthAggregationError(
                'selection_metrics must be a mapping.'
            )
        if summary['candidate_complete']:
            for metric in _METRICS:
                _require_json_number(
                    metrics.get(metric),
                    f'lambda={summary["lambda_oss"]} selection metric {metric}',
                    nonnegative=True,
                )
    if any(not summary['candidate_complete'] for summary in lambda_summaries):
        for summary in lambda_summaries:
            summary['practical_tie_status'] = {
                'eligibility': (
                    'eligible_candidate' if summary['candidate_complete'] else 'incomplete_candidate'
                ),
                'primary': 'not_applied',
                'secondary': 'not_applied',
                'tertiary': 'not_applied',
                'final_survivor': False,
                'selected': False,
            }
        return {
            'eligible': False,
            'reason': 'one_or_more_zero_success_patients',
            'trace': None,
            'selected_lambda_oss': None,
        }

    primary_metric = 'centroid_tre_mm'
    secondary_metric = 'point_tre_mean_mm'
    tertiary_metric = 'rre_deg'
    primary_min = min(
        summary['selection_metrics'][primary_metric] for summary in lambda_summaries
    )
    primary_limit = primary_min + _protocol_value(
        protocol, 'selection_rule', 'primary', 'practical_tie_threshold_mm'
    )
    primary = [
        summary
        for summary in lambda_summaries
        if summary['selection_metrics'][primary_metric] <= primary_limit
    ]
    secondary_min = min(
        summary['selection_metrics'][secondary_metric] for summary in primary
    )
    secondary_limit = secondary_min + _protocol_value(
        protocol, 'selection_rule', 'secondary', 'practical_tie_threshold_mm'
    )
    secondary = [
        summary
        for summary in primary
        if summary['selection_metrics'][secondary_metric] <= secondary_limit
    ]
    tertiary_min = min(
        summary['selection_metrics'][tertiary_metric] for summary in secondary
    )
    tertiary_limit = tertiary_min + _protocol_value(
        protocol, 'selection_rule', 'tertiary', 'practical_tie_threshold_deg'
    )
    tertiary = [
        summary
        for summary in secondary
        if summary['selection_metrics'][tertiary_metric] <= tertiary_limit
    ]
    selected = min(summary['lambda_oss'] for summary in tertiary)
    primary_values = {summary['lambda_oss'] for summary in primary}
    secondary_values = {summary['lambda_oss'] for summary in secondary}
    tertiary_values = {summary['lambda_oss'] for summary in tertiary}
    for summary in lambda_summaries:
        value = summary['lambda_oss']
        primary_delta = summary['selection_metrics'][primary_metric] - primary_min
        secondary_delta = (
            summary['selection_metrics'][secondary_metric] - secondary_min
            if value in primary_values
            else None
        )
        tertiary_delta = (
            summary['selection_metrics'][tertiary_metric] - tertiary_min
            if value in secondary_values
            else None
        )
        summary['practical_tie_status'] = {
            'eligibility': 'eligible_candidate',
            'primary': (
                'within_stage_min_threshold' if value in primary_values else 'outside_stage_min_threshold'
            ),
            'secondary': (
                'within_stage_min_threshold'
                if value in secondary_values
                else ('outside_stage_min_threshold' if value in primary_values else 'not_reached')
            ),
            'tertiary': (
                'within_stage_min_threshold'
                if value in tertiary_values
                else ('outside_stage_min_threshold' if value in secondary_values else 'not_reached')
            ),
            'primary_delta_from_stage_min': primary_delta,
            'secondary_delta_from_stage_min': secondary_delta,
            'tertiary_delta_from_stage_min': tertiary_delta,
            'final_survivor': value in tertiary_values,
            'selected': value == selected,
        }
    return {
        'eligible': True,
        'reason': 'complete_validation_grid',
        'trace': {
            'primary_threshold_mm': _protocol_value(
                protocol,
                'selection_rule',
                'primary',
                'practical_tie_threshold_mm',
            ),
            'primary_stage_min': primary_min,
            'primary_stage_limit': primary_limit,
            'primary_survivors': [summary['lambda_oss'] for summary in primary],
            'secondary_threshold_mm': _protocol_value(
                protocol,
                'selection_rule',
                'secondary',
                'practical_tie_threshold_mm',
            ),
            'secondary_stage_min': secondary_min,
            'secondary_stage_limit': secondary_limit,
            'secondary_survivors': [summary['lambda_oss'] for summary in secondary],
            'tertiary_threshold_deg': _protocol_value(
                protocol,
                'selection_rule',
                'tertiary',
                'practical_tie_threshold_deg',
            ),
            'tertiary_stage_min': tertiary_min,
            'tertiary_stage_limit': tertiary_limit,
            'tertiary_survivors': [summary['lambda_oss'] for summary in tertiary],
            'final_policy': 'smallest_lambda_oss',
        },
        'selected_lambda_oss': selected,
    }


def aggregate_grid_rows(grid_rows, protocol):
    """Aggregate an already-loaded exact 4x5 mapping of validation rows."""
    protocol = validate_frozen_protocol(protocol)
    if not isinstance(grid_rows, Mapping):
        raise M4OsseousStrengthAggregationError('grid_rows must be a mapping.')
    expected_keys = {
        (float(lambda_oss), fold_id)
        for lambda_oss in _FROZEN_LAMBDAS
        for fold_id in _FROZEN_FOLDS
    }
    actual_keys = set(grid_rows)
    if actual_keys != expected_keys:
        raise M4OsseousStrengthAggregationError(
            f'grid must contain exactly 4x5 entries; missing='
            f'{sorted(expected_keys - actual_keys)}, unexpected='
            f'{sorted(actual_keys - expected_keys)}.'
        )
    lambda_summaries = []
    for lambda_oss in _FROZEN_LAMBDAS:
        fold_rows = {
            fold_id: grid_rows[(float(lambda_oss), fold_id)]
            for fold_id in _FROZEN_FOLDS
        }
        lambda_summaries.append(
            _build_lambda_summary(protocol, float(lambda_oss), fold_rows)
        )
    decision = apply_selection_rule(lambda_summaries, protocol)
    result = {
        'artifact_scope': 'validation_only',
        'aggregator_version': AGGREGATOR_VERSION,
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'grid_shape': {'lambda_count': 4, 'fold_count': 5},
        'case_count': sum(summary['case_count'] for summary in lambda_summaries),
        'selection_eligible': decision['eligible'],
        'selection_reason': decision['reason'],
        'selection_trace': decision['trace'],
        'lambda_summaries': lambda_summaries,
    }
    if decision['eligible']:
        result['SELECTED_LAMBDA_OSS'] = decision['selected_lambda_oss']
    _canonical_json(result)
    return result


def frozen_output_root(protocol):
    protocol = validate_frozen_protocol(protocol)
    raw = _protocol_value(protocol, 'output_layout', 'root')
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or '..' in Path(raw).parts:
        raise M4OsseousStrengthAggregationError(
            'frozen output root must be a repository-relative path without traversal.'
        )
    return Path(os.path.abspath(_REPO_ROOT / Path(raw)))


def _assert_regular_directory(path, name):
    if _is_linklike(path) or not Path(path).is_dir():
        raise M4OsseousStrengthAggregationError(
            f'{name} must be an existing non-link directory: {path}.'
        )


def _assert_no_linklike_components(path, anchor, name):
    path = Path(os.path.abspath(Path(path)))
    anchor = Path(os.path.abspath(Path(anchor)))
    try:
        relative = path.relative_to(anchor)
    except ValueError as error:
        raise M4OsseousStrengthAggregationError(
            f'{name} escapes its trusted anchor: {path}.'
        ) from error
    current = anchor
    if _is_linklike(current):
        raise M4OsseousStrengthAggregationError(
            f'{name} trusted anchor is link-like: {current}.'
        )
    for component in relative.parts:
        current = current / component
        if _is_linklike(current):
            raise M4OsseousStrengthAggregationError(
                f'{name} contains a link-like path component: {current}.'
            )


def _assert_contained_regular_file(path, root, name):
    path = Path(path)
    if _is_linklike(path):
        raise M4OsseousStrengthAggregationError(
            f'{name} must not be a symbolic link or reparse point: {path}.'
        )
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M4OsseousStrengthAggregationError(f'{name} is missing: {path}.') from error
    if not stat.S_ISREG(mode):
        raise M4OsseousStrengthAggregationError(
            f'{name} must be a regular file: {path}.'
        )
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise M4OsseousStrengthAggregationError(
            f'{name} escapes the frozen output root: {path}.'
        ) from error


def _validate_result_root(protocol, result_root):
    expected = frozen_output_root(protocol)
    actual = Path(os.path.abspath(Path(result_root)))
    if os.path.normcase(str(actual)) != os.path.normcase(str(expected)):
        raise M4OsseousStrengthAggregationError(
            f'only the frozen output root is accepted: expected={expected}, actual={actual}.'
        )
    repository = Path(os.path.abspath(_REPO_ROOT))
    _assert_no_linklike_components(expected, repository, 'frozen output root')
    _assert_regular_directory(expected, 'frozen output root')
    try:
        expected.resolve(strict=True).relative_to(repository.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise M4OsseousStrengthAggregationError(
            'frozen output root escapes the repository.'
        ) from error
    return expected


def load_validation_grid(protocol, result_root):
    """Read exactly the 20 allowlisted validation JSONL artifacts."""
    protocol = validate_frozen_protocol(protocol)
    root = _validate_result_root(protocol, result_root)
    validation_directory = _protocol_value(
        protocol, 'output_layout', 'per_fold', 'validation_directory'
    )
    artifact_basename = _protocol_value(
        protocol, 'validation_only_artifact', 'basename'
    )
    grid = {}
    for lambda_oss in _FROZEN_LAMBDAS:
        lambda_path = root / _FROZEN_LAMBDA_DIRECTORIES[lambda_oss]
        _assert_regular_directory(lambda_path, f'lambda={lambda_oss} directory')
        for fold_id in _FROZEN_FOLDS:
            fold_path = lambda_path / _FROZEN_FOLD_DIRECTORIES[fold_id]
            validation_path = fold_path / validation_directory
            artifact = validation_path / artifact_basename
            _assert_regular_directory(fold_path, f'{fold_id}/lambda={lambda_oss} directory')
            _assert_regular_directory(
                validation_path, f'{fold_id}/lambda={lambda_oss} validation directory'
            )
            name = f'{fold_id}/lambda={lambda_oss} validation artifact'
            _assert_contained_regular_file(artifact, root, name)
            rows = _load_jsonl(artifact, name)
            grid[(float(lambda_oss), fold_id)] = validate_validation_cases(
                rows, protocol, fold_id, float(lambda_oss)
            )
    if len(grid) != 20:
        raise M4OsseousStrengthAggregationError(
            'validation grid did not produce exactly 20 allowlisted artifacts.'
        )
    return grid


def aggregate_validation_root(result_root, *, protocol_path=DEFAULT_PROTOCOL_PATH):
    protocol = load_frozen_protocol(protocol_path)
    grid = load_validation_grid(protocol, result_root)
    return aggregate_grid_rows(grid, protocol)


def run_contract_audit(*, protocol_path=DEFAULT_PROTOCOL_PATH):
    """CPU-only audit; reads protocol/sidecar, but no result root and writes nothing."""
    protocol = load_frozen_protocol(protocol_path)
    identity_count = 0
    seeds = set()
    for lambda_oss in _FROZEN_LAMBDAS:
        for fold_id in _FROZEN_FOLDS:
            identities = build_expected_case_identities(protocol, fold_id, lambda_oss)
            identity_count += len(identities)
            seeds.update(row['perturbation_seed'] for row in identities)
    return {
        'artifact_scope': 'validation_only',
        'aggregator_version': AGGREGATOR_VERSION,
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'contract_audit': 'PASS',
        'result_root_read': False,
        'files_written': False,
        'lambda_fold_pairs': 20,
        'expected_case_identities': identity_count,
        'unique_subject_severity_seeds': len(seeds),
    }


def _atomic_write_summary(path, result, root):
    path = Path(os.path.abspath(Path(path)))
    expected = root / SUMMARY_BASENAME
    if os.path.normcase(str(path)) != os.path.normcase(str(expected)):
        raise M4OsseousStrengthAggregationError(
            f'output JSON is isolated to the frozen summary path: {expected}.'
        )
    if _is_linklike(path):
        raise M4OsseousStrengthAggregationError(
            f'output JSON must not be a symbolic link or reparse point: {path}.'
        )
    if path.exists():
        raise M4OsseousStrengthAggregationError(
            f'output JSON already exists; refusing overwrite: {path}.'
        )
    payload = _canonical_json(result) + '\n'
    descriptor = None
    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix='.m4_validation_summary.', suffix='.tmp', dir=root
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise M4OsseousStrengthAggregationError(
            f'cannot atomically write validation summary {path}: {error}'
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Frozen M4 osseous-strength validation-only aggregator.'
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--contract-audit', action='store_true')
    modes.add_argument('--aggregate', action='store_true')
    parser.add_argument('--result-root', type=Path)
    parser.add_argument('--output-json', type=Path)
    return parser


def run(args):
    if args.contract_audit:
        if args.result_root is not None or args.output_json is not None:
            raise M4OsseousStrengthAggregationError(
                '--contract-audit forbids result-root and output-json.'
            )
        return run_contract_audit()
    if args.result_root is None:
        raise M4OsseousStrengthAggregationError('--aggregate requires --result-root.')
    protocol = load_frozen_protocol()
    root = _validate_result_root(protocol, args.result_root)
    grid = load_validation_grid(protocol, root)
    result = aggregate_grid_rows(grid, protocol)
    if args.output_json is not None:
        _atomic_write_summary(args.output_json, result, root)
    return result


def main(argv=None):
    result = run(build_argument_parser().parse_args(argv))
    print(_canonical_json(result))


if __name__ == '__main__':
    main()


__all__ = [
    'AGGREGATOR_VERSION',
    'DEFAULT_PROTOCOL_PATH',
    'FROZEN_PROTOCOL_SHA256',
    'FROZEN_PROTOCOL_VERSION',
    'M4OsseousStrengthAggregationError',
    'SUMMARY_BASENAME',
    'aggregate_grid_rows',
    'aggregate_validation_root',
    'apply_selection_rule',
    'build_argument_parser',
    'build_expected_case_identities',
    'collapse_patient_rows',
    'derive_expected_seed',
    'frozen_output_root',
    'load_frozen_protocol',
    'load_validation_grid',
    'run',
    'run_contract_audit',
    'validate_frozen_protocol',
    'validate_validation_cases',
]
