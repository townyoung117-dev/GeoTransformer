"""Validation-only producer for frozen M4 osseous-strength selection.

This entry deliberately has no test-evaluation mode.  Execution loads only the
validation indices of one clean10 Fold, observes the existing M4 osseous
inference path, and emits the exact allowlisted ``validation_cases.jsonl``
artifact.  Contract-audit and completion verification are CPU-only and never
construct a dataset or model.
"""

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


PRODUCER_VERSION = 'm4_osseous_strength_validation_v1'
COMPLETION_MARKER_VERSION = 'm4_osseous_strength_selection_completion_v1'
CHECKPOINT_RECORD_VERSION = 'm4_osseous_strength_checkpoint_record_v1'
FROZEN_PROTOCOL_VERSION = 'm4_osseous_strength_selection_clean10_v1'
FROZEN_PROTOCOL_SHA256 = (
    'a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38'
)
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parent
    / 'protocols'
    / 'm4_osseous_strength_selection_clean10_v1.json'
)
_REPO_ROOT = Path(__file__).resolve().parents[2]

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
_CHECKPOINT_RECORD_FIELDS = frozenset(
    {
        'artifact_scope',
        'record_version',
        'protocol_version',
        'protocol_sha256',
        'lambda_oss',
        'fold_id',
        'best_epoch',
        'best_val_objective',
        'checkpoint_path',
        'checkpoint_sha256',
        'training_config_sha256',
    }
)
_MARKER_FIELDS = frozenset(
    {
        'artifact_scope',
        'marker_version',
        'complete',
        'producer_version',
        'protocol_version',
        'protocol_sha256',
        'lambda_oss',
        'fold_id',
        'validation_subject_ids',
        'validation_case_count',
        'validation_cases_path',
        'validation_cases_sha256',
        'checkpoint_record_path',
        'checkpoint_record_sha256',
        'checkpoint_sha256',
    }
)


class M4OsseousStrengthValidationContractError(RuntimeError):
    """Raised when validation-only selection cannot be proven safe."""


def _reject_duplicate_json_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise M4OsseousStrengthValidationContractError(
                f'duplicate JSON object field: {key!r}.'
            )
        output[key] = value
    return output


def _load_json(path, name):
    path = Path(path)
    if not path.is_file():
        raise M4OsseousStrengthValidationContractError(
            f'{name} does not exist: {path}.'
        )
    try:
        with path.open('r', encoding='utf-8') as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_json_fields)
    except M4OsseousStrengthValidationContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M4OsseousStrengthValidationContractError(
            f'cannot load {name} {path}: {error}'
        ) from error
    if not isinstance(value, Mapping):
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a JSON object.'
        )
    return value


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
        raise M4OsseousStrengthValidationContractError(
            'value is not canonical-JSON serializable.'
        ) from error


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open('rb') as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise M4OsseousStrengthValidationContractError(
            f'cannot hash file {path}: {error}'
        ) from error
    return digest.hexdigest()


def _require_exact_fields(value, expected, name):
    if not isinstance(value, Mapping):
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a JSON object.'
        )
    actual = set(value)
    missing = sorted(set(expected).difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing or unexpected:
        raise M4OsseousStrengthValidationContractError(
            f'{name} fields mismatch; missing={missing}, unexpected={unexpected}.'
        )
    return value


def _finite_number(value, name, *, nonnegative=False):
    if isinstance(value, bool):
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a finite number.'
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a finite number.'
        ) from error
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a finite{ " non-negative" if nonnegative else ""} number.'
        )
    return number


def _finite_json_number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a JSON number.'
        )
    return _finite_number(value, name, nonnegative=nonnegative)


def _require_equal(actual, expected, name):
    if actual != expected or type(actual) is not type(expected):
        raise M4OsseousStrengthValidationContractError(
            f'{name} does not match the frozen protocol; '
            f'expected={expected!r}, actual={actual!r}.'
        )


def _selection_protocol_hash(protocol):
    hash_input = dict(protocol)
    hash_input.pop('protocol_sha256', None)
    return _canonical_sha256(hash_input)


def _verify_protocol_sidecar(protocol_path, protocol_sha256):
    sidecar = Path(protocol_path).with_suffix('.sha256')
    if not sidecar.is_file():
        raise M4OsseousStrengthValidationContractError(
            f'frozen protocol SHA sidecar is missing: {sidecar}.'
        )
    try:
        text = sidecar.read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError) as error:
        raise M4OsseousStrengthValidationContractError(
            f'cannot read frozen protocol SHA sidecar {sidecar}: {error}'
        ) from error
    expected = (
        f'{protocol_sha256}  {Path(protocol_path).name} '
        '(canonical JSON excluding protocol_sha256)'
    )
    if text != expected:
        raise M4OsseousStrengthValidationContractError(
            'frozen protocol SHA sidecar does not exactly bind the canonical manifest.'
        )


def _load_source_protocols(protocol):
    from defect_evaluation import load_defect_evaluation_protocol
    from m3_metric_diagnostic import load_diagnostic_protocol

    source_specs = protocol['source_protocols']

    def source_path(key):
        raw = source_specs[key]['path']
        if not isinstance(raw, str) or not raw:
            raise M4OsseousStrengthValidationContractError(
                f'source_protocols.{key}.path must be a non-empty relative path.'
            )
        path = Path(raw)
        if path.is_absolute() or '..' in path.parts:
            raise M4OsseousStrengthValidationContractError(
                f'source_protocols.{key}.path must remain inside the repository.'
            )
        return _REPO_ROOT / path

    training_path = source_path('clean10_training')
    training_raw = _load_json(training_path, 'frozen clean10 training protocol')
    training_spec = source_specs['clean10_training']
    training_hash_input = dict(training_raw)
    stored_training_hash = training_hash_input.pop('protocol_hash', None)
    computed_training_hash = _canonical_sha256(training_hash_input)
    if (
        training_raw.get('protocol_version') != training_spec['protocol_version']
        or stored_training_hash != training_spec['canonical_sha256']
        or computed_training_hash != training_spec['canonical_sha256']
    ):
        raise M4OsseousStrengthValidationContractError(
            'clean10 source version or canonical SHA-256 differs from frozen protocol.'
        )
    safe_folds = {}
    for fold_id, frozen_fold in protocol['clean10']['folds'].items():
        source_fold = training_raw.get('folds', {}).get(fold_id)
        if not isinstance(source_fold, Mapping):
            raise M4OsseousStrengthValidationContractError(
                f'clean10 source is missing {fold_id}.'
            )
        safe_fold = {}
        for split_field in ('train_subject_ids', 'val_subject_ids'):
            source_subjects = source_fold.get(split_field)
            if source_subjects != frozen_fold[split_field]:
                raise M4OsseousStrengthValidationContractError(
                    f'{fold_id}.{split_field} differs from frozen clean10.'
                )
            safe_fold[split_field] = list(source_subjects)
        safe_folds[fold_id] = safe_fold
    training = {
        'protocol_version': training_raw.get('protocol_version'),
        'protocol_hash': stored_training_hash,
        'seed_scheme_version': training_raw.get('seed_scheme_version'),
        'perturbation_root_seed': training_raw.get('perturbation_root_seed'),
        'ready_subject_ids': list(training_raw.get('ready_subject_ids', ())),
        'folds': safe_folds,
        'validation_perturbations': [
            dict(spec) for spec in training_raw.get('validation_perturbations', ())
        ],
    }
    evaluation = load_defect_evaluation_protocol(
        source_path('formal_training_settings')
    )
    diagnostic = load_diagnostic_protocol(source_path('origin_invariant_metrics'))
    resolved = {
        'clean10_training': (
            training,
            training['protocol_version'],
            training['protocol_hash'],
        ),
        'formal_training_settings': (
            evaluation,
            evaluation['evaluation_protocol_version'],
            evaluation['evaluation_protocol_hash'],
        ),
        'origin_invariant_metrics': (
            diagnostic,
            diagnostic['diagnostic_protocol_version'],
            diagnostic['diagnostic_protocol_hash'],
        ),
    }
    for key, (_, version, digest) in resolved.items():
        spec = source_specs[key]
        if spec['protocol_version'] != version:
            raise M4OsseousStrengthValidationContractError(
                f'source protocol version mismatch for {key}.'
            )
        if not hmac.compare_digest(spec['canonical_sha256'], digest):
            raise M4OsseousStrengthValidationContractError(
                f'source protocol hash mismatch for {key}.'
            )
    return training, evaluation, diagnostic


def validate_selection_protocol(protocol, *, protocol_path=None):
    """Validate the frozen selection manifest and its immutable SHA trust anchor."""
    if not isinstance(protocol, Mapping):
        raise M4OsseousStrengthValidationContractError(
            'selection protocol must be a JSON object.'
        )
    if protocol.get('protocol_version') != FROZEN_PROTOCOL_VERSION:
        raise M4OsseousStrengthValidationContractError(
            'selection protocol version is not the frozen M4-2D version.'
        )
    stored_hash = protocol.get('protocol_sha256')
    if not isinstance(stored_hash, str):
        raise M4OsseousStrengthValidationContractError(
            'protocol_sha256 must be a lowercase SHA-256 digest.'
        )
    computed_hash = _selection_protocol_hash(protocol)
    if not hmac.compare_digest(stored_hash, computed_hash):
        raise M4OsseousStrengthValidationContractError(
            f'protocol SHA mismatch: stored={stored_hash}, computed={computed_hash}.'
        )
    if not hmac.compare_digest(stored_hash, FROZEN_PROTOCOL_SHA256):
        raise M4OsseousStrengthValidationContractError(
            'selection protocol does not match the post-freeze SHA trust anchor.'
        )
    if protocol_path is not None:
        _verify_protocol_sidecar(protocol_path, stored_hash)

    _require_equal(protocol.get('protocol_status'), 'FROZEN_BEFORE_GPU_SELECTION', 'protocol_status')
    _require_equal(protocol.get('lambda_oss_grid'), [0.25, 0.5, 1.0, 2.0], 'lambda_oss_grid')
    _require_equal(protocol['clean10']['excluded_subject_ids'], ['Pat6'], 'excluded_subject_ids')
    _require_equal(protocol['defect_conditions'], [
        'defect_001_left_maxilla_cheek_small',
        'defect_001_left_maxilla_cheek_medium',
        'defect_001_left_maxilla_cheek_large',
        'defect_001_right_maxilla_cheek_small',
        'defect_001_right_maxilla_cheek_medium',
    ], 'defect_conditions')
    settings = protocol['training_settings']
    frozen_settings = {
        'epochs': 20,
        'seed': 20260815,
        'batch_size': 1,
        'precision': 'fp32',
        'temperature': 0.1,
        'sinkhorn_iterations': 20,
        'alpha_init': 1.0,
    }
    for field, expected in frozen_settings.items():
        _require_equal(settings.get(field), expected, f'training_settings.{field}')
    _require_equal(settings['optimizer']['name'], 'AdamW', 'optimizer.name')
    _require_equal(settings['optimizer']['learning_rate'], 0.0003, 'optimizer.learning_rate')
    _require_equal(settings['optimizer']['weight_decay'], 0.0001, 'optimizer.weight_decay')
    _require_equal(settings['scheduler']['name'], 'none', 'scheduler.name')
    _require_equal(settings['m4_hard']['enable_m4_defect_mapping'], True, 'm4 hard mapping')
    _require_equal(settings['m4_hard']['hard_constraint_active'], True, 'm4 hard active')
    _require_equal(settings['m4_soft']['enable_m4_soft_modulation'], True, 'm4 soft enabled')
    _require_equal(settings['m4_soft']['sigma_mm'], 60.0, 'm4 soft sigma')
    _require_equal(settings['m4_soft']['strength'], 2.0, 'm4 soft strength')
    _require_equal(settings['osseous_score']['center_hu'], 300.0, 'osseous center')
    _require_equal(settings['osseous_score']['tau_hu'], 100.0, 'osseous tau')
    _require_equal(settings['osseous_score']['radius_mm'], 20.0, 'osseous radius')
    _require_equal(settings['only_variable_across_candidates'], 'lambda_oss', 'candidate variable')
    _require_equal(protocol['checkpoint_selection']['required_basename'], 'best_val_loss.pt', 'checkpoint basename')
    _require_equal(protocol['checkpoint_selection']['scope'], 'validation_only', 'checkpoint scope')
    _require_equal(protocol['checkpoint_selection']['direction'], 'minimize', 'checkpoint direction')
    _require_equal(protocol['checkpoint_selection']['comparison'], 'strict_less_than', 'checkpoint comparison')
    _require_equal(protocol['checkpoint_selection']['equal_value_policy'], 'keep_earliest_epoch', 'checkpoint tie policy')
    _require_equal(
        protocol['validation_only_artifact']['required_case_fields'],
        list(_CASE_FIELDS),
        'case allowlist',
    )
    _require_equal(protocol['validation_only_artifact']['unexpected_case_fields'], 'reject', 'unexpected case policy')
    _require_equal(protocol['validation_only_artifact']['duplicate_json_keys'], 'reject', 'duplicate key policy')
    _require_equal(protocol['no_test_firewall']['test_sample_loading'], 'forbidden', 'sample firewall')
    _require_equal(protocol['no_test_firewall']['test_evaluation'], 'forbidden', 'evaluation firewall')

    training, evaluation, diagnostic = _load_source_protocols(protocol)
    if tuple(protocol['clean10']['ready_subject_ids']) != tuple(training['ready_subject_ids']):
        raise M4OsseousStrengthValidationContractError(
            'selection ready subjects do not match the frozen clean10 source.'
        )
    return protocol, training, evaluation, diagnostic


def load_selection_protocol(path=DEFAULT_PROTOCOL_PATH):
    path = Path(path)
    protocol = _load_json(path, 'M4 osseous-strength selection protocol')
    return validate_selection_protocol(protocol, protocol_path=path)


def _resolve_lambda(protocol, value):
    number = _finite_number(value, 'lambda_oss', nonnegative=True)
    for candidate in protocol['lambda_oss_grid']:
        if number == float(candidate):
            return float(candidate)
    raise M4OsseousStrengthValidationContractError(
        f'lambda_oss={number!r} is outside the frozen grid.'
    )


def _resolve_fold(protocol, fold_id):
    if not isinstance(fold_id, str) or fold_id not in protocol['clean10']['folds']:
        raise M4OsseousStrengthValidationContractError(
            f'unknown fold_id {fold_id!r}.'
        )
    return protocol['clean10']['folds'][fold_id]


def _fold_paths(protocol, fold_id, lambda_oss, fold_dir):
    fold = _resolve_fold(protocol, fold_id)
    lambda_oss = _resolve_lambda(protocol, lambda_oss)
    directory_map = protocol['output_layout']['lambda_directories']
    lambda_directory = None
    for key, directory in directory_map.items():
        if float(key) == lambda_oss:
            lambda_directory = directory
            break
    if lambda_directory is None:
        raise M4OsseousStrengthValidationContractError(
            'frozen output layout has no directory for lambda_oss.'
        )
    if (
        not fold_id.startswith('Fold')
        or not fold_id[4:].isdigit()
        or f'Fold{int(fold_id[4:])}' != fold_id
    ):
        raise M4OsseousStrengthValidationContractError(
            'fold_id cannot be mapped to a frozen output directory.'
        )
    fold_position = int(fold_id[4:])
    fold_directories = protocol['output_layout']['fold_directories']
    if fold_position < 1 or fold_position > len(fold_directories):
        raise M4OsseousStrengthValidationContractError(
            'fold_id is outside the frozen output directory mapping.'
        )
    expected_fold_name = fold_directories[fold_position - 1]
    output_root = Path(
        os.path.abspath(_REPO_ROOT / protocol['output_layout']['root'])
    )
    lambda_path = output_root / lambda_directory
    expected = Path(os.path.abspath(lambda_path / expected_fold_name))
    actual = Path(os.path.abspath(Path(fold_dir)))
    if actual != expected:
        raise M4OsseousStrengthValidationContractError(
            f'fold directory violates frozen output isolation: expected={expected}, actual={actual}.'
        )
    per_fold = protocol['output_layout']['per_fold']
    paths = {
        'fold': fold,
        'lambda_oss': lambda_oss,
        'fold_dir': actual,
        'checkpoint': actual / per_fold['checkpoint_directory'] / protocol['checkpoint_selection']['required_basename'],
        'training_jsonl': actual / per_fold['training_jsonl'],
        'validation_dir': actual / per_fold['validation_directory'],
        'validation_cases': actual / per_fold['validation_directory'] / protocol['validation_only_artifact']['basename'],
        'checkpoint_record': actual / per_fold['checkpoint_record'],
        'completion_marker': actual / per_fold['completion_marker'],
    }
    filesystem_paths = {
        'output root': output_root,
        'lambda directory': lambda_path,
        'fold directory': actual,
        'checkpoint directory': paths['checkpoint'].parent,
        'training JSONL': paths['training_jsonl'],
        'validation directory': paths['validation_dir'],
        'validation cases': paths['validation_cases'],
        'checkpoint record': paths['checkpoint_record'],
        'completion marker': paths['completion_marker'],
    }
    for name, path in filesystem_paths.items():
        if path.is_symlink():
            raise M4OsseousStrengthValidationContractError(
                f'{name} must not be a symbolic link: {path}.'
            )
    return paths


def build_validation_manifest(protocol, training_protocol, fold_id, lambda_oss):
    """Build the 30 expected validation identities without reading a dataset."""
    from perturbation import derive_perturbation_seed

    fold = _resolve_fold(protocol, fold_id)
    lambda_oss = _resolve_lambda(protocol, lambda_oss)
    specs = tuple(
        dict(spec) for spec in training_protocol['validation_perturbations']
    )
    frozen_specs = protocol['validation_protocol']['perturbations']
    if len(specs) != len(frozen_specs):
        raise M4OsseousStrengthValidationContractError(
            'validation perturbation count differs from the frozen selection protocol.'
        )
    for source, frozen in zip(specs, frozen_specs):
        for field in ('purpose', 'severity', 'max_rotation_deg', 'max_translation_mm'):
            if source[field] != frozen[field]:
                raise M4OsseousStrengthValidationContractError(
                    f'validation perturbation {field} differs from the source protocol.'
                )
        if source['variant_count'] != 1 or frozen['variant_id'] != 0:
            raise M4OsseousStrengthValidationContractError(
                'validation perturbations must contain exactly variant_id=0.'
            )
    rows = []
    for subject_id in fold['val_subject_ids']:
        for defect_id in protocol['defect_conditions']:
            for spec in specs:
                seed = derive_perturbation_seed(
                    scheme_version=training_protocol['seed_scheme_version'],
                    protocol_hash=training_protocol['protocol_hash'],
                    fold_id=fold_id,
                    root_seed=training_protocol['perturbation_root_seed'],
                    purpose='val',
                    epoch=None,
                    subject_id=subject_id,
                    severity=spec['severity'],
                    variant_id=0,
                )
                rows.append(
                    {
                        'lambda_oss': lambda_oss,
                        'fold_id': fold_id,
                        'subject_id': subject_id,
                        'defect_id': defect_id,
                        'severity': spec['severity'],
                        'variant_id': 0,
                        'perturbation_seed': seed,
                        'max_rotation_deg': float(spec['max_rotation_deg']),
                        'max_translation_mm': float(spec['max_translation_mm']),
                    }
                )
    expected = protocol['validation_protocol']['expected_cases_per_fold']
    if len(rows) != expected:
        raise M4OsseousStrengthValidationContractError(
            f'{fold_id} validation manifest must contain exactly {expected} cases.'
        )
    identities = [
        tuple(row[field] for field in protocol['validation_protocol']['case_identity'])
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise M4OsseousStrengthValidationContractError(
            'validation manifest contains duplicate case identities.'
        )
    return tuple(rows)


def _load_jsonl(path, name):
    path = Path(path)
    if not path.is_file():
        raise M4OsseousStrengthValidationContractError(
            f'{name} does not exist: {path}.'
        )
    rows = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise M4OsseousStrengthValidationContractError(
                        f'{name} contains a blank line at {line_number}.'
                    )
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_fields,
                    )
                except json.JSONDecodeError as error:
                    raise M4OsseousStrengthValidationContractError(
                        f'{name} has invalid JSON at line {line_number}: {error}'
                    ) from error
                if not isinstance(row, Mapping):
                    raise M4OsseousStrengthValidationContractError(
                        f'{name} line {line_number} must be a JSON object.'
                    )
                rows.append(row)
    except M4OsseousStrengthValidationContractError:
        raise
    except (OSError, UnicodeError) as error:
        raise M4OsseousStrengthValidationContractError(
            f'cannot read {name} {path}: {error}'
        ) from error
    return rows


def _reject_result_firewall_tokens(value, path='row'):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise M4OsseousStrengthValidationContractError(
                    f'{path} contains a non-string JSON key.'
                )
            if 'test' in key.casefold():
                raise M4OsseousStrengthValidationContractError(
                    f'{path} contains forbidden result key token in {key!r}.'
                )
            _reject_result_firewall_tokens(item, f'{path}.{key}')
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_result_firewall_tokens(item, f'{path}[{index}]')
    elif isinstance(value, str):
        scope_tokens = {
            token
            for token in re.split(r'[^a-z0-9]+', value.casefold().strip())
            if token
        }
        if scope_tokens.intersection({'test', 'testing'}):
            raise M4OsseousStrengthValidationContractError(
                f'{path} claims a forbidden evaluation scope.'
            )


def validate_validation_cases(
    rows,
    protocol,
    training_protocol,
    fold_id,
    lambda_oss,
):
    """Validate the strict allowlist and complete validation identity set."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise M4OsseousStrengthValidationContractError(
            'validation cases must be a sequence of JSON objects.'
        )
    expected = build_validation_manifest(
        protocol,
        training_protocol,
        fold_id,
        lambda_oss,
    )
    identity_fields = tuple(protocol['validation_protocol']['case_identity'])
    expected_map = {
        _canonical_json([row[field] for field in identity_fields]): row
        for row in expected
    }
    seen = set()
    for position, row in enumerate(rows):
        _require_exact_fields(row, _CASE_FIELDS, f'validation row {position}')
        _reject_result_firewall_tokens(row, f'validation row {position}')
        _require_equal(row['artifact_scope'], 'validation_only', 'artifact_scope')
        _require_equal(row['protocol_version'], FROZEN_PROTOCOL_VERSION, 'protocol_version')
        _require_equal(row['protocol_sha256'], FROZEN_PROTOCOL_SHA256, 'protocol_sha256')
        identity = [row[field] for field in identity_fields]
        identity_key = _canonical_json(identity)
        if identity_key in seen:
            raise M4OsseousStrengthValidationContractError(
                f'duplicate validation case identity: {identity!r}.'
            )
        if identity_key not in expected_map:
            raise M4OsseousStrengthValidationContractError(
                f'unexpected validation case identity: {identity!r}.'
            )
        expected_row = expected_map[identity_key]
        for field in identity_fields:
            _require_equal(row[field], expected_row[field], f'case identity {field}')
        seen.add(identity_key)
        success = row['solver_success']
        if not isinstance(success, bool):
            raise M4OsseousStrengthValidationContractError(
                'solver_success must be boolean.'
            )
        if not isinstance(row['solver_status'], str) or not row['solver_status'].strip():
            raise M4OsseousStrengthValidationContractError(
                'solver_status must be a non-empty string.'
            )
        if not isinstance(row['registration_recall_hit'], bool):
            raise M4OsseousStrengthValidationContractError(
                'registration_recall_hit must be boolean.'
            )
        metrics = ('centroid_tre_mm', 'point_tre_mean_mm', 'rre_deg')
        if success:
            if row['solver_status'] != 'success':
                raise M4OsseousStrengthValidationContractError(
                    'a solver-success row must use solver_status="success".'
                )
            for metric in metrics:
                _finite_json_number(row[metric], metric, nonnegative=True)
        else:
            if row['registration_recall_hit'] is not False:
                raise M4OsseousStrengthValidationContractError(
                    'a solver-failure row cannot be a registration recall hit.'
                )
            if any(row[metric] is not None for metric in metrics):
                raise M4OsseousStrengthValidationContractError(
                    'solver-failure TRE/RRE metrics must remain null.'
                )
    if seen != set(expected_map):
        missing = sorted(set(expected_map).difference(seen))
        raise M4OsseousStrengthValidationContractError(
            f'validation artifact is incomplete; missing identities={missing!r}.'
        )
    return tuple(rows)


def _training_config_expected(protocol, training_protocol, fold_id, lambda_oss):
    from config import (
        GT_HIGH_CONFIDENCE_DISTANCE_MM,
        GT_PRIMARY_MAX_DISTANCE_MM,
    )
    from train_m3 import FORMAL_PROTOCOL_STATUS
    from train_m3_defect import (
        M4_OSSEOUS_HYPERPARAMETER_STATUS,
        M4_SOFT_HYPERPARAMETER_STATUS,
    )
    from training import TRAINING_DEFAULTS_STATUS

    settings = protocol['training_settings']
    return {
        'learning_rate': settings['optimizer']['learning_rate'],
        'weight_decay': settings['optimizer']['weight_decay'],
        'batch_size': settings['batch_size'],
        'precision': settings['precision'],
        'seed': settings['seed'],
        'temperature': settings['temperature'],
        'sinkhorn_iterations': settings['sinkhorn_iterations'],
        'alpha_init': settings['alpha_init'],
        'primary_max_distance_mm': float(GT_PRIMARY_MAX_DISTANCE_MM),
        'high_confidence_distance_mm': float(GT_HIGH_CONFIDENCE_DISTANCE_MM),
        'hyperparameter_status': TRAINING_DEFAULTS_STATUS,
        'split_status': FORMAL_PROTOCOL_STATUS,
        'enable_m4_defect_mapping': True,
        'm4_hard_constraint_active': True,
        'm4_soft_modulation_enabled': True,
        'm4_soft_modulation_active': True,
        'm4_soft_sigma_mm': settings['m4_soft']['sigma_mm'],
        'm4_soft_strength': settings['m4_soft']['strength'],
        'm4_soft_hyperparameter_status': M4_SOFT_HYPERPARAMETER_STATUS,
        'm4_osseous_prior_enabled': True,
        'm4_osseous_prior_active': True,
        'm4_osseous_strength': lambda_oss,
        'osseous_center_hu': settings['osseous_score']['center_hu'],
        'osseous_tau_hu': settings['osseous_score']['tau_hu'],
        'osseous_radius_mm': settings['osseous_score']['radius_mm'],
        'm4_osseous_hyperparameter_status': M4_OSSEOUS_HYPERPARAMETER_STATUS,
    }


def _audit_defect_provenance_without_test(
    value,
    protocol,
    training_protocol,
    fold_id,
    name,
):
    if not isinstance(value, Mapping):
        raise M4OsseousStrengthValidationContractError(
            f'{name} must be a mapping.'
        )
    fold = training_protocol['folds'].get(fold_id)
    if not isinstance(fold, Mapping):
        raise M4OsseousStrengthValidationContractError(
            f'{name} refers to an unknown fold.'
        )

    def records(partition, subject_ids):
        return [
            {
                'subject_id': subject_id,
                'defect_id': defect_id,
                'fold_id': fold_id,
                'partition': partition,
            }
            for subject_id in subject_ids
            for defect_id in protocol['defect_conditions']
        ]

    expected = {
        'adapter': 'pointct_defect_training_v1',
        'fold_id': fold_id,
        'defect_ids': list(protocol['defect_conditions']),
        'train_instances': records('train', fold['train_subject_ids']),
        'val_instances': records('val', fold['val_subject_ids']),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise M4OsseousStrengthValidationContractError(
                f'{name}.{field} differs from frozen train/validation provenance.'
            )
    return expected


def _audit_training_log(path, protocol, training_protocol, fold_id, lambda_oss):
    rows = _load_jsonl(path, 'training JSONL')
    expected_count = protocol['training_settings']['epochs']
    if len(rows) != expected_count:
        raise M4OsseousStrengthValidationContractError(
            f'training JSONL must contain exactly {expected_count} epoch records.'
        )
    expected_config = _training_config_expected(
        protocol,
        training_protocol,
        fold_id,
        lambda_oss,
    )
    by_epoch = {}
    running_best = math.inf
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise M4OsseousStrengthValidationContractError(
                f'training JSONL row {position} must be an object.'
            )
        epoch = row.get('epoch')
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise M4OsseousStrengthValidationContractError(
                f'training JSONL row {position} has an invalid epoch.'
            )
        if epoch in by_epoch:
            raise M4OsseousStrengthValidationContractError(
                f'training JSONL contains duplicate epoch {epoch}.'
            )
        if epoch != position:
            raise M4OsseousStrengthValidationContractError(
                'training JSONL epochs must appear in exact zero-based order.'
            )
        for field, expected in (
            ('formal_protocol', True),
            ('protocol_version', training_protocol['protocol_version']),
            ('protocol_hash', training_protocol['protocol_hash']),
            ('fold_id', fold_id),
            ('num_val_cases', protocol['validation_protocol']['expected_cases_per_fold']),
            ('num_val_subjects', protocol['validation_protocol']['expected_patients_per_fold']),
            ('num_val_perturbations', protocol['validation_protocol']['expected_cases_per_fold']),
            (
                'learning_rate',
                protocol['training_settings']['optimizer']['learning_rate'],
            ),
        ):
            if row.get(field) != expected:
                raise M4OsseousStrengthValidationContractError(
                    f'training JSONL epoch {epoch} has mismatched {field}.'
                )
        for field in (
            'enable_m4_defect_mapping',
            'm4_hard_constraint_active',
            'm4_soft_modulation_enabled',
            'm4_soft_modulation_active',
            'm4_soft_sigma_mm',
            'm4_soft_strength',
            'm4_osseous_prior_enabled',
            'm4_osseous_prior_active',
            'm4_osseous_strength',
            'osseous_center_hu',
            'osseous_tau_hu',
            'osseous_radius_mm',
        ):
            if row.get(field) != expected_config[field]:
                raise M4OsseousStrengthValidationContractError(
                    f'training JSONL epoch {epoch} has mismatched {field}.'
                )
        _audit_defect_provenance_without_test(
            row.get('defect_training_provenance'),
            protocol,
            training_protocol,
            fold_id,
            f'training JSONL epoch {epoch} defect provenance',
        )
        objective = _finite_json_number(
            row.get('val_mean_loss'),
            f'epoch {epoch} val_mean_loss',
        )
        improved = objective < running_best
        running_best = min(running_best, objective)
        if row.get('best_checkpoint_updated') is not improved:
            raise M4OsseousStrengthValidationContractError(
                f'training JSONL epoch {epoch} violates strict-less checkpoint selection.'
            )
        recorded_best = _finite_json_number(
            row.get('best_val_loss'),
            f'epoch {epoch} best_val_loss',
        )
        if recorded_best != running_best:
            raise M4OsseousStrengthValidationContractError(
                f'training JSONL epoch {epoch} has an inconsistent running best.'
            )
        by_epoch[epoch] = objective
    if set(by_epoch) != set(range(expected_count)):
        raise M4OsseousStrengthValidationContractError(
            'training JSONL epochs are not the exact frozen zero-based budget.'
        )
    best_epoch = min(by_epoch, key=lambda epoch: (by_epoch[epoch], epoch))
    return best_epoch, by_epoch[best_epoch]


def audit_checkpoint(
    checkpoint_path,
    training_jsonl_path,
    protocol,
    training_protocol,
    fold_id,
    lambda_oss,
):
    """Load one checkpoint on CPU and validate all available frozen provenance."""
    from training import CHECKPOINT_VERSION, _load_checkpoint_file
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != protocol['checkpoint_selection']['required_basename']:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint basename is not the frozen best-validation basename.'
        )
    if not checkpoint_path.is_file():
        raise M4OsseousStrengthValidationContractError(
            f'checkpoint does not exist: {checkpoint_path}.'
        )
    try:
        payload = _load_checkpoint_file(checkpoint_path, map_location='cpu')
    except Exception as error:
        if isinstance(error, M4OsseousStrengthValidationContractError):
            raise
        raise M4OsseousStrengthValidationContractError(
            f'checkpoint structural validation failed: {error}'
        ) from error
    if not isinstance(payload, Mapping):
        raise M4OsseousStrengthValidationContractError(
            'checkpoint payload must be a mapping.'
        )
    required_fields = {
        'checkpoint_version',
        'epoch',
        'global_step',
        'point_encoder_state_dict',
        'ct_encoder_state_dict',
        'matcher_state_dict',
        'optimizer_state_dict',
        'train_subject_ids',
        'val_subject_ids',
        'seed',
        'training_config',
        'best_val_loss',
        'formal_protocol',
        'protocol_version',
        'protocol_hash',
        'fold_id',
        'perturbation_root_seed',
        'perturbation_seed_scheme_version',
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise M4OsseousStrengthValidationContractError(
            f'checkpoint is malformed; missing fields={missing}.'
        )
    if payload.get('checkpoint_version') != CHECKPOINT_VERSION:
        raise M4OsseousStrengthValidationContractError(
            'selection requires the formal checkpoint version.'
        )
    if payload.get('formal_protocol') is not True:
        raise M4OsseousStrengthValidationContractError(
            'selection requires a formal training checkpoint.'
        )
    for field in ('epoch', 'global_step', 'seed'):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise M4OsseousStrengthValidationContractError(
                f'checkpoint {field} must be a non-negative integer.'
            )
    for field in (
        'point_encoder_state_dict',
        'ct_encoder_state_dict',
        'matcher_state_dict',
        'optimizer_state_dict',
        'training_config',
    ):
        if not isinstance(payload.get(field), Mapping):
            raise M4OsseousStrengthValidationContractError(
                f'checkpoint {field} must be a mapping.'
            )
    fold = training_protocol['folds'].get(fold_id)
    if not isinstance(fold, Mapping):
        raise M4OsseousStrengthValidationContractError(
            'checkpoint refers to an unknown safe train/validation fold.'
        )
    expected_metadata = {
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'fold_id': fold_id,
        'train_subject_ids': list(fold['train_subject_ids']),
        'val_subject_ids': list(fold['val_subject_ids']),
        'seed': protocol['training_settings']['seed'],
        'perturbation_root_seed': protocol['training_settings']['perturbation_root_seed'],
        'perturbation_seed_scheme_version': protocol['training_settings']['perturbation_seed_scheme_version'],
    }
    for field, expected in expected_metadata.items():
        if payload.get(field) != expected:
            raise M4OsseousStrengthValidationContractError(
                f'checkpoint {field} does not match the frozen Fold/config contract.'
            )
    expected_config = _training_config_expected(
        protocol,
        training_protocol,
        fold_id,
        lambda_oss,
    )
    training_config = payload.get('training_config')
    if not isinstance(training_config, Mapping):
        raise M4OsseousStrengthValidationContractError(
            'checkpoint training_config must be a mapping.'
        )
    expected_training_config_fields = set(expected_config) | {
        'defect_training_provenance'
    }
    actual_training_config_fields = set(training_config)
    if actual_training_config_fields != expected_training_config_fields:
        missing = sorted(
            expected_training_config_fields.difference(actual_training_config_fields)
        )
        unexpected = sorted(
            actual_training_config_fields.difference(expected_training_config_fields)
        )
        raise M4OsseousStrengthValidationContractError(
            'checkpoint training_config fields mismatch; '
            f'missing={missing}, unexpected={unexpected}.'
        )
    for field, expected in expected_config.items():
        if training_config.get(field) != expected:
            raise M4OsseousStrengthValidationContractError(
                f'checkpoint training_config.{field} differs from the frozen selection config.'
            )
    safe_defect_provenance = _audit_defect_provenance_without_test(
        training_config.get('defect_training_provenance'),
        protocol,
        training_protocol,
        fold_id,
        'checkpoint training_config.defect_training_provenance',
    )
    safe_training_config = {
        **expected_config,
        'defect_training_provenance': safe_defect_provenance,
    }
    optimizer_state = payload.get('optimizer_state_dict')
    parameter_groups = optimizer_state.get('param_groups') if isinstance(optimizer_state, Mapping) else None
    if not isinstance(parameter_groups, Sequence) or not parameter_groups:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint optimizer state has no parameter groups.'
        )
    for group in parameter_groups:
        if group.get('lr') != protocol['training_settings']['optimizer']['learning_rate']:
            raise M4OsseousStrengthValidationContractError(
                'checkpoint optimizer learning rate differs from frozen constant scheduling.'
            )
        if group.get('weight_decay') != protocol['training_settings']['optimizer']['weight_decay']:
            raise M4OsseousStrengthValidationContractError(
                'checkpoint optimizer weight decay differs from frozen AdamW config.'
            )
    best_epoch, best_objective = _audit_training_log(
        training_jsonl_path,
        protocol,
        training_protocol,
        fold_id,
        lambda_oss,
    )
    if payload.get('epoch') != best_epoch:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint epoch does not match the validation-only strict best epoch.'
        )
    checkpoint_objective = _finite_json_number(
        payload.get('best_val_loss'),
        'checkpoint best_val_loss',
    )
    if checkpoint_objective != best_objective:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint objective does not match the training JSONL validation minimum.'
        )
    return payload, {
        'artifact_scope': 'validation_only',
        'record_version': CHECKPOINT_RECORD_VERSION,
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'lambda_oss': float(lambda_oss),
        'fold_id': fold_id,
        'best_epoch': best_epoch,
        'best_val_objective': best_objective,
        'checkpoint_path': str(checkpoint_path.resolve(strict=True)),
        'checkpoint_sha256': _file_sha256(checkpoint_path),
        'training_config_sha256': _canonical_sha256(safe_training_config),
    }


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='\n',
            prefix=f'.{path.name}.',
            suffix='.tmp',
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception as error:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if isinstance(error, M4OsseousStrengthValidationContractError):
            raise
        raise M4OsseousStrengthValidationContractError(
            f'cannot atomically write {path}: {error}'
        ) from error


def _write_json(path, value):
    _atomic_write_text(path, _canonical_json(value) + '\n')


def _write_jsonl(path, rows):
    _atomic_write_text(path, ''.join(_canonical_json(row) + '\n' for row in rows))


def _evaluate_validation_case(
    raw_sample,
    manifest_case,
    protocol,
    training_protocol,
    evaluation_protocol,
    point_encoder,
    ct_encoder,
    matcher,
    device,
):
    import numpy as np

    from evaluate_m4_osseous_prior import run_m4_osseous_prior_inference
    from evaluation import (
        correspondence_inlier_metrics,
        evaluate_registration_result,
    )
    from m3_metric_diagnostic import (
        compute_tre_diagnostics,
        make_rigid_transform,
        rotation_error_degrees,
    )
    from perturbation import augment_point_sample

    augmented = augment_point_sample(
        raw_sample,
        seed=manifest_case['perturbation_seed'],
        max_rotation_deg=manifest_case['max_rotation_deg'],
        max_translation_mm=manifest_case['max_translation_mm'],
        scheme_version=training_protocol['seed_scheme_version'],
    )
    effective_gt = np.asarray(augmented['gt_transform'], dtype=np.float64)
    settings = protocol['training_settings']
    inference = run_m4_osseous_prior_inference(
        sample=augmented,
        point_encoder=point_encoder,
        ct_encoder=ct_encoder,
        matcher=matcher,
        evaluation_protocol=evaluation_protocol,
        device=device,
        sigma_mm=settings['m4_soft']['sigma_mm'],
        soft_strength=settings['m4_soft']['strength'],
        osseous_strength=manifest_case['lambda_oss'],
    )
    filter_output = inference['filter_output']
    registration_output = inference['registration_output']
    inlier_metrics = correspondence_inlier_metrics(
        inference['point_physical'],
        inference['ct_physical'],
        filter_output['point_indices'],
        filter_output['ct_indices'],
        effective_gt,
        evaluation_protocol['correspondence_inlier_threshold_mm'],
    )
    metrics = evaluate_registration_result(
        registration_output,
        effective_gt,
        evaluation_protocol['registration_rre_threshold_deg'],
        evaluation_protocol['registration_rte_threshold_mm'],
    )
    if (
        inlier_metrics['num_correspondences'] == 0
        and bool(metrics['solver_success'])
    ):
        raise M4OsseousStrengthValidationContractError(
            'zero-correspondence registration must fail closed.'
        )
    declared_success = registration_output.get('success')
    if not isinstance(declared_success, (bool, np.bool_)):
        raise M4OsseousStrengthValidationContractError(
            'registration_output.success must be boolean.'
        )
    if bool(declared_success) != bool(metrics['solver_success']):
        raise M4OsseousStrengthValidationContractError(
            'M4 inference and frozen metric evaluator disagree on solver success.'
        )
    predicted = None
    if bool(declared_success):
        predicted = make_rigid_transform(
            registration_output.get('rotation'),
            registration_output.get('translation'),
            'predicted_transform',
        )
    tre = compute_tre_diagnostics(
        inference['point_physical'],
        predicted,
        effective_gt,
        solver_success=bool(declared_success),
    )
    if bool(declared_success):
        diagnostic_rre = rotation_error_degrees(
            predicted[:3, :3],
            effective_gt[:3, :3],
        )
        if not math.isclose(
            diagnostic_rre,
            float(metrics['rre_deg']),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise M4OsseousStrengthValidationContractError(
                'origin-invariant diagnostic and frozen evaluator disagree on RRE.'
            )
    return {
        'artifact_scope': 'validation_only',
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'lambda_oss': manifest_case['lambda_oss'],
        'fold_id': manifest_case['fold_id'],
        'subject_id': manifest_case['subject_id'],
        'defect_id': manifest_case['defect_id'],
        'severity': manifest_case['severity'],
        'variant_id': manifest_case['variant_id'],
        'perturbation_seed': manifest_case['perturbation_seed'],
        'solver_success': bool(metrics['solver_success']),
        'solver_status': metrics['solver_status'],
        'registration_recall_hit': bool(metrics['registration_recall_hit']),
        'centroid_tre_mm': tre['centroid_tre_mm'],
        'point_tre_mean_mm': tre['point_tre_mean_mm'],
        'rre_deg': metrics['rre_deg'],
    }


def _ensure_execution_targets_unused(paths):
    marker = paths['completion_marker']
    if marker.exists():
        return False
    occupied = [
        path
        for path in (paths['validation_cases'], paths['checkpoint_record'])
        if path.exists()
    ]
    validation_dir = paths['validation_dir']
    if validation_dir.exists():
        occupied.extend(child for child in validation_dir.iterdir() if child.exists())
    if occupied:
        raise M4OsseousStrengthValidationContractError(
            'incomplete validation outputs already exist; refusing to overwrite: '
            f'{sorted(str(path) for path in set(occupied))}.'
        )
    return True


def run_execute_validation(
    *,
    protocol_path,
    fold_id,
    lambda_oss,
    fold_dir,
    data_root,
    checkpoint_path=None,
    device_name='cuda',
):
    """Execute exactly one Fold/lambda validation grid and mark it complete."""
    protocol, training_protocol, evaluation_protocol, _ = load_selection_protocol(
        protocol_path
    )
    paths = _fold_paths(protocol, fold_id, lambda_oss, fold_dir)
    if checkpoint_path is not None and Path(checkpoint_path).resolve() != paths['checkpoint']:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint path must be the frozen per-Fold best checkpoint path.'
        )
    if not _ensure_execution_targets_unused(paths):
        result = verify_completion(
            protocol_path=protocol_path,
            fold_id=fold_id,
            lambda_oss=lambda_oss,
            fold_dir=fold_dir,
            checkpoint_path=checkpoint_path,
        )
        return {**result, 'skipped_existing_complete': True}
    payload, checkpoint_record = audit_checkpoint(
        paths['checkpoint'],
        paths['training_jsonl'],
        protocol,
        training_protocol,
        fold_id,
        paths['lambda_oss'],
    )

    import torch

    from dataset import create_dataset
    from evaluate_m3 import _build_models_from_checkpoint
    from train_m3_defect import _validate_raw_defect_identity

    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise M4OsseousStrengthValidationContractError(
            f'invalid execution device {device_name!r}.'
        ) from error
    if device.type not in {'cpu', 'cuda'}:
        raise M4OsseousStrengthValidationContractError(
            'validation execution supports only CPU or CUDA.'
        )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise M4OsseousStrengthValidationContractError(
            'CUDA validation was requested but CUDA is unavailable.'
        )
    models = _build_models_from_checkpoint(
        payload,
        {'training_config': payload['training_config']},
        device,
    )
    validation_identities = tuple(
        (subject_id, defect_id)
        for subject_id in paths['fold']['val_subject_ids']
        for defect_id in protocol['defect_conditions']
    )
    dataset = create_dataset(data_root, defect_variants=validation_identities)
    records = getattr(dataset, 'records', None)
    if not isinstance(records, list) or len(records) != len(validation_identities):
        raise M4OsseousStrengthValidationContractError(
            'validation-only dataset must expose exactly the ten allowlisted records.'
        )
    raw_by_identity = {}
    expected_identity_set = set(validation_identities)
    for dataset_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise M4OsseousStrengthValidationContractError(
                'validation-only dataset records must be mappings.'
            )
        identity = (record.get('subject_id'), record.get('defect_id'))
        if identity not in expected_identity_set or identity in raw_by_identity:
            raise M4OsseousStrengthValidationContractError(
                'validation-only dataset contains an unexpected or duplicate identity.'
            )
        raw_sample = dataset[dataset_index]
        actual_identity = _validate_raw_defect_identity(raw_sample, record)
        if tuple(actual_identity) != tuple(identity):
            raise M4OsseousStrengthValidationContractError(
                'loaded validation sample identity differs from its allowlisted record.'
            )
        raw_by_identity[tuple(identity)] = raw_sample
    if set(raw_by_identity) != expected_identity_set:
        raise M4OsseousStrengthValidationContractError(
            'validation-only dataset is missing an allowlisted identity.'
        )
    manifest = build_validation_manifest(
        protocol,
        training_protocol,
        fold_id,
        paths['lambda_oss'],
    )
    rows = []
    for manifest_case in manifest:
        identity = (manifest_case['subject_id'], manifest_case['defect_id'])
        if identity not in raw_by_identity:
            raise M4OsseousStrengthValidationContractError(
                'validation manifest requested an identity outside val indices.'
            )
        rows.append(
            _evaluate_validation_case(
                raw_by_identity[identity],
                manifest_case,
                protocol,
                training_protocol,
                evaluation_protocol,
                *models,
                device,
            )
        )
    validate_validation_cases(
        rows,
        protocol,
        training_protocol,
        fold_id,
        paths['lambda_oss'],
    )
    _write_jsonl(paths['validation_cases'], rows)
    _write_json(paths['checkpoint_record'], checkpoint_record)
    marker = {
        'artifact_scope': 'validation_only',
        'marker_version': COMPLETION_MARKER_VERSION,
        'complete': True,
        'producer_version': PRODUCER_VERSION,
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'lambda_oss': paths['lambda_oss'],
        'fold_id': fold_id,
        'validation_subject_ids': list(paths['fold']['val_subject_ids']),
        'validation_case_count': len(rows),
        'validation_cases_path': 'validation/validation_cases.jsonl',
        'validation_cases_sha256': _file_sha256(paths['validation_cases']),
        'checkpoint_record_path': 'checkpoint_record.json',
        'checkpoint_record_sha256': _file_sha256(paths['checkpoint_record']),
        'checkpoint_sha256': checkpoint_record['checkpoint_sha256'],
    }
    _reject_result_firewall_tokens(marker, 'completion marker')
    _write_json(paths['completion_marker'], marker)
    return {
        'artifact_scope': 'validation_only',
        'complete': True,
        'fold_id': fold_id,
        'lambda_oss': paths['lambda_oss'],
        'validation_case_count': len(rows),
        'completion_marker': str(paths['completion_marker']),
        'skipped_existing_complete': False,
    }


def verify_completion(
    *,
    protocol_path,
    fold_id,
    lambda_oss,
    fold_dir,
    checkpoint_path=None,
):
    """CPU-only verification of a completed Fold/lambda selection artifact."""
    protocol, training_protocol, _, _ = load_selection_protocol(protocol_path)
    paths = _fold_paths(protocol, fold_id, lambda_oss, fold_dir)
    if checkpoint_path is not None and Path(checkpoint_path).resolve() != paths['checkpoint']:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint path must be the frozen per-Fold best checkpoint path.'
        )
    _, expected_record = audit_checkpoint(
        paths['checkpoint'],
        paths['training_jsonl'],
        protocol,
        training_protocol,
        fold_id,
        paths['lambda_oss'],
    )
    record = _load_json(paths['checkpoint_record'], 'checkpoint record')
    _require_exact_fields(record, _CHECKPOINT_RECORD_FIELDS, 'checkpoint record')
    _reject_result_firewall_tokens(record, 'checkpoint record')
    if record != expected_record:
        raise M4OsseousStrengthValidationContractError(
            'checkpoint record does not match CPU-verified checkpoint provenance.'
        )
    rows = _load_jsonl(paths['validation_cases'], 'validation cases')
    validate_validation_cases(
        rows,
        protocol,
        training_protocol,
        fold_id,
        paths['lambda_oss'],
    )
    marker = _load_json(paths['completion_marker'], 'completion marker')
    _require_exact_fields(marker, _MARKER_FIELDS, 'completion marker')
    _reject_result_firewall_tokens(marker, 'completion marker')
    expected_marker = {
        'artifact_scope': 'validation_only',
        'marker_version': COMPLETION_MARKER_VERSION,
        'complete': True,
        'producer_version': PRODUCER_VERSION,
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'lambda_oss': paths['lambda_oss'],
        'fold_id': fold_id,
        'validation_subject_ids': list(paths['fold']['val_subject_ids']),
        'validation_case_count': protocol['validation_protocol']['expected_cases_per_fold'],
        'validation_cases_path': 'validation/validation_cases.jsonl',
        'validation_cases_sha256': _file_sha256(paths['validation_cases']),
        'checkpoint_record_path': 'checkpoint_record.json',
        'checkpoint_record_sha256': _file_sha256(paths['checkpoint_record']),
        'checkpoint_sha256': expected_record['checkpoint_sha256'],
    }
    if marker != expected_marker:
        raise M4OsseousStrengthValidationContractError(
            'completion marker does not match the verified artifact digests/provenance.'
        )
    return {
        'artifact_scope': 'validation_only',
        'complete': True,
        'fold_id': fold_id,
        'lambda_oss': paths['lambda_oss'],
        'validation_case_count': len(rows),
        'completion_marker': str(paths['completion_marker']),
    }


def run_contract_audit(
    *,
    protocol_path,
    fold_id,
    lambda_oss,
    fold_dir,
    checkpoint_path=None,
):
    """CPU-only audit; no dataset, model construction, or accelerator access."""
    protocol, training_protocol, _, _ = load_selection_protocol(protocol_path)
    paths = _fold_paths(protocol, fold_id, lambda_oss, fold_dir)
    build_validation_manifest(
        protocol,
        training_protocol,
        fold_id,
        paths['lambda_oss'],
    )
    checkpoint_audit = None
    if checkpoint_path is not None:
        if Path(checkpoint_path).resolve() != paths['checkpoint']:
            raise M4OsseousStrengthValidationContractError(
                'checkpoint path must be the frozen per-Fold best checkpoint path.'
            )
        _, checkpoint_audit = audit_checkpoint(
            paths['checkpoint'],
            paths['training_jsonl'],
            protocol,
            training_protocol,
            fold_id,
            paths['lambda_oss'],
        )
    return {
        'artifact_scope': 'validation_only',
        'contract_audit': 'PASS',
        'protocol_version': FROZEN_PROTOCOL_VERSION,
        'protocol_sha256': FROZEN_PROTOCOL_SHA256,
        'fold_id': fold_id,
        'lambda_oss': paths['lambda_oss'],
        'expected_validation_case_count': protocol['validation_protocol']['expected_cases_per_fold'],
        'expected_validation_subject_ids': list(paths['fold']['val_subject_ids']),
        'checkpoint_audited': checkpoint_audit is not None,
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Frozen M4 osseous-strength validation-only producer.'
    )
    parser.add_argument('--protocol-manifest', type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument('--fold-id', required=True)
    parser.add_argument('--lambda-oss', type=float, required=True)
    parser.add_argument('--fold-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', default='cuda')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--contract-audit', action='store_true')
    modes.add_argument('--execute-validation', action='store_true')
    modes.add_argument('--verify-completion', action='store_true')
    return parser


def run(args):
    common = {
        'protocol_path': args.protocol_manifest,
        'fold_id': args.fold_id,
        'lambda_oss': args.lambda_oss,
        'fold_dir': args.fold_dir,
        'checkpoint_path': args.checkpoint,
    }
    if args.contract_audit:
        if args.data_root is not None:
            raise M4OsseousStrengthValidationContractError(
                '--contract-audit forbids --data-root.'
            )
        return run_contract_audit(**common)
    if args.verify_completion:
        if args.data_root is not None:
            raise M4OsseousStrengthValidationContractError(
                '--verify-completion forbids --data-root.'
            )
        return verify_completion(**common)
    if args.data_root is None:
        raise M4OsseousStrengthValidationContractError(
            '--execute-validation requires --data-root.'
        )
    return run_execute_validation(
        **common,
        data_root=args.data_root,
        device_name=args.device,
    )


def main(argv=None):
    result = run(build_argument_parser().parse_args(argv))
    print(_canonical_json(result))


if __name__ == '__main__':
    main()


__all__ = [
    'CHECKPOINT_RECORD_VERSION',
    'COMPLETION_MARKER_VERSION',
    'DEFAULT_PROTOCOL_PATH',
    'FROZEN_PROTOCOL_SHA256',
    'FROZEN_PROTOCOL_VERSION',
    'M4OsseousStrengthValidationContractError',
    'PRODUCER_VERSION',
    'audit_checkpoint',
    'build_argument_parser',
    'build_validation_manifest',
    'load_selection_protocol',
    'run',
    'run_contract_audit',
    'run_execute_validation',
    'validate_selection_protocol',
    'validate_validation_cases',
    'verify_completion',
]
