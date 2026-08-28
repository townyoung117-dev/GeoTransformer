"""Versioned M3-6B-2 five-fold training protocol contracts."""

import hashlib
import hmac
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_VERSION = 'm3_6b_5fold_v1'
PROTOCOL_HASH = 'c0c635c1cb897ce33d15ba3ed58abdb12a8c5a8feb6a82fe51910146e8a99fac'
CLEAN10_PROTOCOL_VERSION = 'm3_6b_5fold_clean10_v2'
CLEAN10_PROTOCOL_HASH = '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'
SEED_SCHEME_VERSION = 'm3_6b_seed_v1'
PERTURBATION_ROOT_SEED = 20260815
COORDINATE_SYSTEM = 'LPS'
PHYSICAL_UNIT = 'mm'
TRANSFORM_DIRECTION = 'Point Cloud -> CT'

READY_SUBJECT_IDS = (
    'Pat1',
    'Pat2',
    'Pat3',
    'Pat4',
    'Pat5',
    'Pat6',
    'Pat7',
    'Pat8',
    'Pat9',
    'Pat11',
    'Pat12',
)
SUBJECT_GROUPS = {
    'G1': ('Pat12', 'Pat6', 'Pat5'),
    'G2': ('Pat7', 'Pat4'),
    'G3': ('Pat11', 'Pat3'),
    'G4': ('Pat8', 'Pat9'),
    'G5': ('Pat1', 'Pat2'),
}
FOLD_SUBJECTS = {
    'Fold1': {
        'test_subject_ids': SUBJECT_GROUPS['G1'],
        'val_subject_ids': SUBJECT_GROUPS['G2'],
        'train_subject_ids': SUBJECT_GROUPS['G3'] + SUBJECT_GROUPS['G4'] + SUBJECT_GROUPS['G5'],
    },
    'Fold2': {
        'test_subject_ids': SUBJECT_GROUPS['G2'],
        'val_subject_ids': SUBJECT_GROUPS['G3'],
        'train_subject_ids': SUBJECT_GROUPS['G1'] + SUBJECT_GROUPS['G4'] + SUBJECT_GROUPS['G5'],
    },
    'Fold3': {
        'test_subject_ids': SUBJECT_GROUPS['G3'],
        'val_subject_ids': SUBJECT_GROUPS['G4'],
        'train_subject_ids': SUBJECT_GROUPS['G1'] + SUBJECT_GROUPS['G2'] + SUBJECT_GROUPS['G5'],
    },
    'Fold4': {
        'test_subject_ids': SUBJECT_GROUPS['G4'],
        'val_subject_ids': SUBJECT_GROUPS['G5'],
        'train_subject_ids': SUBJECT_GROUPS['G1'] + SUBJECT_GROUPS['G2'] + SUBJECT_GROUPS['G3'],
    },
    'Fold5': {
        'test_subject_ids': SUBJECT_GROUPS['G5'],
        'val_subject_ids': SUBJECT_GROUPS['G1'],
        'train_subject_ids': SUBJECT_GROUPS['G2'] + SUBJECT_GROUPS['G3'] + SUBJECT_GROUPS['G4'],
    },
}

CLEAN10_READY_SUBJECT_IDS = (
    'Pat1',
    'Pat2',
    'Pat3',
    'Pat4',
    'Pat5',
    'Pat7',
    'Pat8',
    'Pat9',
    'Pat11',
    'Pat12',
)
CLEAN10_SUBJECT_GROUPS = {
    'G1': ('Pat12', 'Pat5'),
    'G2': ('Pat7', 'Pat4'),
    'G3': ('Pat11', 'Pat3'),
    'G4': ('Pat8', 'Pat9'),
    'G5': ('Pat1', 'Pat2'),
}
CLEAN10_FOLD_SUBJECTS = {
    'Fold1': {
        'test_subject_ids': CLEAN10_SUBJECT_GROUPS['G1'],
        'val_subject_ids': CLEAN10_SUBJECT_GROUPS['G2'],
        'train_subject_ids': (
            CLEAN10_SUBJECT_GROUPS['G3']
            + CLEAN10_SUBJECT_GROUPS['G4']
            + CLEAN10_SUBJECT_GROUPS['G5']
        ),
    },
    'Fold2': {
        'test_subject_ids': CLEAN10_SUBJECT_GROUPS['G2'],
        'val_subject_ids': CLEAN10_SUBJECT_GROUPS['G3'],
        'train_subject_ids': (
            CLEAN10_SUBJECT_GROUPS['G1']
            + CLEAN10_SUBJECT_GROUPS['G4']
            + CLEAN10_SUBJECT_GROUPS['G5']
        ),
    },
    'Fold3': {
        'test_subject_ids': CLEAN10_SUBJECT_GROUPS['G3'],
        'val_subject_ids': CLEAN10_SUBJECT_GROUPS['G4'],
        'train_subject_ids': (
            CLEAN10_SUBJECT_GROUPS['G1']
            + CLEAN10_SUBJECT_GROUPS['G2']
            + CLEAN10_SUBJECT_GROUPS['G5']
        ),
    },
    'Fold4': {
        'test_subject_ids': CLEAN10_SUBJECT_GROUPS['G4'],
        'val_subject_ids': CLEAN10_SUBJECT_GROUPS['G5'],
        'train_subject_ids': (
            CLEAN10_SUBJECT_GROUPS['G1']
            + CLEAN10_SUBJECT_GROUPS['G2']
            + CLEAN10_SUBJECT_GROUPS['G3']
        ),
    },
    'Fold5': {
        'test_subject_ids': CLEAN10_SUBJECT_GROUPS['G5'],
        'val_subject_ids': CLEAN10_SUBJECT_GROUPS['G1'],
        'train_subject_ids': (
            CLEAN10_SUBJECT_GROUPS['G2']
            + CLEAN10_SUBJECT_GROUPS['G3']
            + CLEAN10_SUBJECT_GROUPS['G4']
        ),
    },
}

# The original public constants above intentionally remain v1 aliases.  Runtime
# validation and adapters select one of these exact contracts by manifest version.
_PROTOCOL_CONTRACTS = {
    PROTOCOL_VERSION: {
        'protocol_version': PROTOCOL_VERSION,
        'protocol_hash': PROTOCOL_HASH,
        'ready_subject_ids': READY_SUBJECT_IDS,
        'groups': SUBJECT_GROUPS,
        'folds': FOLD_SUBJECTS,
        'excluded_subject_ids': (),
    },
    CLEAN10_PROTOCOL_VERSION: {
        'protocol_version': CLEAN10_PROTOCOL_VERSION,
        'protocol_hash': CLEAN10_PROTOCOL_HASH,
        'ready_subject_ids': CLEAN10_READY_SUBJECT_IDS,
        'groups': CLEAN10_SUBJECT_GROUPS,
        'folds': CLEAN10_FOLD_SUBJECTS,
        'excluded_subject_ids': ('Pat6',),
    },
}
SUPPORTED_PROTOCOL_VERSIONS = tuple(_PROTOCOL_CONTRACTS)

_TOP_LEVEL_FIELDS = {
    'protocol_version',
    'protocol_hash',
    'seed_scheme_version',
    'perturbation_root_seed',
    'coordinate_system',
    'physical_unit',
    'transform_direction',
    'ready_subject_ids',
    'groups',
    'folds',
    'train_perturbation',
    'validation_perturbations',
    'test_perturbations',
}
_FOLD_FIELDS = {'train_subject_ids', 'val_subject_ids', 'test_subject_ids'}
_PERTURBATION_FIELDS = {
    'purpose',
    'severity',
    'max_rotation_deg',
    'max_translation_mm',
    'variant_count',
    'epoch_policy',
}
_EXPECTED_VALIDATION_BOUNDS = {
    'mild': (5.0, 5.0),
    'moderate': (10.0, 10.0),
    'hard': (20.0, 20.0),
}


class M3TrainingProtocolError(ValueError):
    pass


def _contract_for_version(protocol_version: str) -> Mapping:
    if not isinstance(protocol_version, str) or protocol_version not in _PROTOCOL_CONTRACTS:
        raise M3TrainingProtocolError(
            f'unsupported protocol_version {protocol_version!r}; '
            f'expected one of {list(SUPPORTED_PROTOCOL_VERSIONS)}.'
        )
    return _PROTOCOL_CONTRACTS[protocol_version]


def get_training_protocol_contract(protocol_or_version) -> dict:
    """Return a defensive copy of one supported versioned training contract."""
    if isinstance(protocol_or_version, Mapping):
        validate_training_protocol(protocol_or_version)
        protocol_version = protocol_or_version['protocol_version']
    else:
        protocol_version = protocol_or_version
    contract = _contract_for_version(protocol_version)
    return {
        'protocol_version': contract['protocol_version'],
        'protocol_hash': contract['protocol_hash'],
        'ready_subject_ids': tuple(contract['ready_subject_ids']),
        'groups': {
            group_id: tuple(subject_ids)
            for group_id, subject_ids in contract['groups'].items()
        },
        'folds': {
            fold_id: {
                split_name: tuple(subject_ids)
                for split_name, subject_ids in fold.items()
            }
            for fold_id, fold in contract['folds'].items()
        },
        'excluded_subject_ids': tuple(contract['excluded_subject_ids']),
        'patient_count': len(contract['ready_subject_ids']),
    }


def _require_exact_fields(value, expected, name: str):
    if not isinstance(value, Mapping):
        raise M3TrainingProtocolError(f'{name} must be a JSON object.')
    actual = set(value)
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing or unexpected:
        raise M3TrainingProtocolError(
            f'{name} fields mismatch; missing={missing}, unexpected={unexpected}.'
        )
    return value


def _require_subject_ids(value, name: str, *, allow_empty: bool = False) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M3TrainingProtocolError(f'{name} must be a JSON subject array.')
    normalized = []
    for subject_id in value:
        if not isinstance(subject_id, str) or not subject_id or subject_id != subject_id.strip():
            raise M3TrainingProtocolError(f'{name} contains an invalid subject ID.')
        normalized.append(subject_id)
    if not normalized and not allow_empty:
        raise M3TrainingProtocolError(f'{name} must not be empty.')
    duplicates = sorted(subject for subject, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise M3TrainingProtocolError(f'{name} contains duplicate subjects: {duplicates}.')
    return tuple(normalized)


def _require_finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool):
        raise M3TrainingProtocolError(f'{name} must be a finite non-negative number.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3TrainingProtocolError(
            f'{name} must be a finite non-negative number.'
        ) from error
    if not math.isfinite(number) or number < 0.0:
        raise M3TrainingProtocolError(f'{name} must be a finite non-negative number.')
    return number


def _require_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M3TrainingProtocolError(f'{name} must be a positive integer.')
    return value


def _canonical_protocol_json(protocol) -> str:
    if not isinstance(protocol, Mapping):
        raise M3TrainingProtocolError('protocol must be a mapping.')
    hash_input = dict(protocol)
    hash_input.pop('protocol_hash', None)
    try:
        return json.dumps(
            hash_input,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M3TrainingProtocolError('protocol must be canonical-JSON serializable.') from error


def compute_protocol_hash(protocol) -> str:
    """Hash the full manifest object after excluding only ``protocol_hash``."""
    canonical_json = _canonical_protocol_json(protocol)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def _validate_perturbation_spec(
    spec,
    *,
    name: str,
    purpose: str,
    severity: str,
    bounds,
    variant_count: int,
    epoch_policy: str,
):
    _require_exact_fields(spec, _PERTURBATION_FIELDS, name)
    if spec['purpose'] != purpose:
        raise M3TrainingProtocolError(f'{name} purpose must be {purpose!r}.')
    if spec['severity'] != severity:
        raise M3TrainingProtocolError(f'{name} severity must be {severity!r}.')
    rotation = _require_finite_nonnegative(spec['max_rotation_deg'], f'{name}.max_rotation_deg')
    translation = _require_finite_nonnegative(
        spec['max_translation_mm'],
        f'{name}.max_translation_mm',
    )
    if (rotation, translation) != bounds:
        raise M3TrainingProtocolError(f'{name} perturbation bounds do not match the formal protocol.')
    if _require_positive_integer(spec['variant_count'], f'{name}.variant_count') != variant_count:
        raise M3TrainingProtocolError(f'{name} variant_count must be {variant_count}.')
    if spec['epoch_policy'] != epoch_policy:
        raise M3TrainingProtocolError(f'{name} epoch_policy must be {epoch_policy!r}.')


def _validate_subject_contract(protocol, contract):
    ready = _require_subject_ids(protocol['ready_subject_ids'], 'ready_subject_ids')
    expected_ready = tuple(contract['ready_subject_ids'])
    protocol_version = contract['protocol_version']
    if ready != expected_ready:
        raise M3TrainingProtocolError(
            f'ready_subject_ids do not match {protocol_version}.'
        )
    if 'Pat10' in ready:
        raise M3TrainingProtocolError('Pat10 is not ready and must not enter the protocol.')
    forbidden = sorted(set(ready).intersection(contract['excluded_subject_ids']))
    if forbidden:
        raise M3TrainingProtocolError(
            f'{protocol_version} contains excluded subjects: {forbidden}.'
        )

    groups = protocol['groups']
    expected_groups = contract['groups']
    _require_exact_fields(groups, set(expected_groups), 'groups')
    group_subjects = []
    for group_id, expected_subjects in expected_groups.items():
        subjects = _require_subject_ids(groups[group_id], f'groups.{group_id}')
        if subjects != expected_subjects:
            raise M3TrainingProtocolError(f'groups.{group_id} does not match the formal protocol.')
        group_subjects.extend(subjects)
    if Counter(group_subjects) != Counter(ready):
        raise M3TrainingProtocolError('group subject union must equal ready_subject_ids exactly once.')

    folds = protocol['folds']
    expected_folds = contract['folds']
    _require_exact_fields(folds, set(expected_folds), 'folds')
    training_counts = Counter()
    validation_counts = Counter()
    test_counts = Counter()
    ready_set = set(ready)
    for fold_id, expected_fold in expected_folds.items():
        fold = folds[fold_id]
        _require_exact_fields(fold, _FOLD_FIELDS, f'folds.{fold_id}')
        resolved = {}
        for split_name in ('train_subject_ids', 'val_subject_ids', 'test_subject_ids'):
            subjects = _require_subject_ids(fold[split_name], f'folds.{fold_id}.{split_name}')
            if subjects != expected_fold[split_name]:
                raise M3TrainingProtocolError(
                    f'folds.{fold_id}.{split_name} does not match the formal protocol.'
                )
            unknown = sorted(set(subjects).difference(ready_set))
            if unknown:
                raise M3TrainingProtocolError(
                    f'folds.{fold_id}.{split_name} contains unknown subjects: {unknown}.'
                )
            resolved[split_name] = subjects
        train_set = set(resolved['train_subject_ids'])
        val_set = set(resolved['val_subject_ids'])
        test_set = set(resolved['test_subject_ids'])
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise M3TrainingProtocolError(f'folds.{fold_id} contains split leakage.')
        if train_set | val_set | test_set != ready_set:
            raise M3TrainingProtocolError(
                f'folds.{fold_id} split union must equal ready_subject_ids.'
            )
        training_counts.update(resolved['train_subject_ids'])
        validation_counts.update(resolved['val_subject_ids'])
        test_counts.update(resolved['test_subject_ids'])
    expected_counts = Counter({subject_id: 1 for subject_id in ready})
    if validation_counts != expected_counts:
        raise M3TrainingProtocolError('every ready subject must validate exactly once across folds.')
    if test_counts != expected_counts:
        raise M3TrainingProtocolError('every ready subject must test exactly once across folds.')
    expected_training_counts = Counter({subject_id: 3 for subject_id in ready})
    if training_counts != expected_training_counts:
        raise M3TrainingProtocolError('every ready subject must train exactly three times across folds.')


def _validate_perturbation_contract(protocol):
    _validate_perturbation_spec(
        protocol['train_perturbation'],
        name='train_perturbation',
        purpose='train',
        severity='train',
        bounds=(20.0, 20.0),
        variant_count=1,
        epoch_policy='zero_based_epoch',
    )
    validation = protocol['validation_perturbations']
    test = protocol['test_perturbations']
    for value, name in ((validation, 'validation_perturbations'), (test, 'test_perturbations')):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
            raise M3TrainingProtocolError(f'{name} must contain exactly three severity specs.')
    for index, (severity, bounds) in enumerate(_EXPECTED_VALIDATION_BOUNDS.items()):
        _validate_perturbation_spec(
            validation[index],
            name=f'validation_perturbations[{index}]',
            purpose='val',
            severity=severity,
            bounds=bounds,
            variant_count=1,
            epoch_policy='fixed_none',
        )
        _validate_perturbation_spec(
            test[index],
            name=f'test_perturbations[{index}]',
            purpose='test',
            severity=severity,
            bounds=bounds,
            variant_count=5,
            epoch_policy='fixed_none',
        )


def validate_training_protocol(protocol, *, verify_hash: bool = True):
    """Fail closed on any deviation from a supported M3-6B-2 protocol."""
    _require_exact_fields(protocol, _TOP_LEVEL_FIELDS, 'protocol')
    contract = _contract_for_version(protocol['protocol_version'])
    protocol_hash = protocol['protocol_hash']
    if (
        not isinstance(protocol_hash, str)
        or len(protocol_hash) != 64
        or any(character not in '0123456789abcdef' for character in protocol_hash)
    ):
        raise M3TrainingProtocolError('protocol_hash must be a lowercase SHA-256 hex digest.')
    if verify_hash:
        computed_hash = compute_protocol_hash(protocol)
        if not hmac.compare_digest(protocol_hash, computed_hash):
            raise M3TrainingProtocolError(
                f'protocol_hash mismatch: stored={protocol_hash}, computed={computed_hash}.'
            )
    if not hmac.compare_digest(protocol_hash, contract['protocol_hash']):
        raise M3TrainingProtocolError(
            f'protocol_hash is not the frozen hash for {contract["protocol_version"]}: '
            f'stored={protocol_hash}, expected={contract["protocol_hash"]}.'
        )
    expected_scalars = {
        'protocol_version': contract['protocol_version'],
        'seed_scheme_version': SEED_SCHEME_VERSION,
        'perturbation_root_seed': PERTURBATION_ROOT_SEED,
        'coordinate_system': COORDINATE_SYSTEM,
        'physical_unit': PHYSICAL_UNIT,
        'transform_direction': TRANSFORM_DIRECTION,
    }
    for field, expected in expected_scalars.items():
        if protocol[field] != expected or type(protocol[field]) is not type(expected):
            raise M3TrainingProtocolError(
                f'{field} must be exactly {expected!r} for {contract["protocol_version"]}.'
            )
    _validate_subject_contract(protocol, contract)
    _validate_perturbation_contract(protocol)
    return protocol


def _reject_duplicate_json_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise M3TrainingProtocolError(f'duplicate JSON object field: {key!r}.')
        output[key] = value
    return output


def load_training_protocol(path):
    path = Path(path)
    if not path.is_file():
        raise M3TrainingProtocolError(f'protocol manifest does not exist: {path}.')
    try:
        with path.open('r', encoding='utf-8') as handle:
            protocol = json.load(handle, object_pairs_hook=_reject_duplicate_json_fields)
    except M3TrainingProtocolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3TrainingProtocolError(f'cannot load protocol manifest {path}: {error}') from error
    return validate_training_protocol(protocol)


def resolve_fold(protocol, fold_id):
    validate_training_protocol(protocol)
    contract = _contract_for_version(protocol['protocol_version'])
    fold_subjects = contract['folds']
    if not isinstance(fold_id, str) or fold_id not in fold_subjects:
        raise M3TrainingProtocolError(
            f'unknown fold_id {fold_id!r}; expected one of {list(fold_subjects)}.'
        )
    fold = protocol['folds'][fold_id]
    return {
        'fold_id': fold_id,
        'train_subject_ids': tuple(fold['train_subject_ids']),
        'val_subject_ids': tuple(fold['val_subject_ids']),
        'test_subject_ids': tuple(fold['test_subject_ids']),
    }


def train_perturbation_spec(protocol):
    validate_training_protocol(protocol)
    return dict(protocol['train_perturbation'])


def validation_perturbation_specs(protocol):
    validate_training_protocol(protocol)
    return tuple(dict(spec) for spec in protocol['validation_perturbations'])


def test_perturbation_specs(protocol):
    validate_training_protocol(protocol)
    return tuple(dict(spec) for spec in protocol['test_perturbations'])


def validate_dataset_ready_subjects(protocol, actual_subject_ids):
    """Require one ready dataset record for every formal subject and no others."""
    validate_training_protocol(protocol)
    actual = _require_subject_ids(actual_subject_ids, 'dataset ready subject IDs')
    expected = tuple(protocol['ready_subject_ids'])
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual))
        unexpected = sorted(set(actual).difference(expected))
        raise M3TrainingProtocolError(
            f'dataset ready subjects mismatch; missing={missing}, unexpected={unexpected}.'
        )
    return expected


__all__ = [
    'CLEAN10_FOLD_SUBJECTS',
    'CLEAN10_PROTOCOL_HASH',
    'CLEAN10_PROTOCOL_VERSION',
    'CLEAN10_READY_SUBJECT_IDS',
    'CLEAN10_SUBJECT_GROUPS',
    'COORDINATE_SYSTEM',
    'FOLD_SUBJECTS',
    'M3TrainingProtocolError',
    'PERTURBATION_ROOT_SEED',
    'PHYSICAL_UNIT',
    'PROTOCOL_HASH',
    'PROTOCOL_VERSION',
    'READY_SUBJECT_IDS',
    'SEED_SCHEME_VERSION',
    'SUBJECT_GROUPS',
    'SUPPORTED_PROTOCOL_VERSIONS',
    'TRANSFORM_DIRECTION',
    'compute_protocol_hash',
    'get_training_protocol_contract',
    'load_training_protocol',
    'resolve_fold',
    'test_perturbation_specs',
    'train_perturbation_spec',
    'validate_dataset_ready_subjects',
    'validate_training_protocol',
    'validation_perturbation_specs',
]
