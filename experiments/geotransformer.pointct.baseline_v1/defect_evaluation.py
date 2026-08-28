"""Fail-closed M3 defect evaluation contracts and structural adapters.

The frozen M3 perturbation primitives and evaluation metrics remain the source
of truth.  This module only expands a patient-level test split over the five
formal defect conditions and validates defect-training checkpoint provenance.
"""

import hashlib
import hmac
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from defect_training import (
    DEFECT_CONDITION_COUNT,
    DEFECT_IDS,
    EXPECTED_DEFECT_INSTANCE_COUNT,
    READY_PATIENT_COUNT,
    build_defect_variants,
    get_defect_training_contract,
)
from evaluation import M3EvaluationContractError, aggregate_case_metrics
from perturbation import derive_perturbation_seed, sample_rigid_perturbation
from training_protocol import (
    resolve_fold,
    test_perturbation_specs,
    validate_training_protocol,
)


DEFECT_EVALUATION_PROTOCOL_VERSION = 'm3_defect_eval_v1'
DEFECT_EVALUATION_PROTOCOL_HASH = (
    '66b93d340a79947295ba1b75738291001192b9848f3cc48a8144492915fd49fe'
)
CLEAN10_DEFECT_EVALUATION_PROTOCOL_VERSION = 'm3_defect_eval_clean10_v2'
CLEAN10_DEFECT_EVALUATION_PROTOCOL_HASH = (
    'cfd519b923be3b623cffec8c8f5830cb5b1160859461180eddeb7134a0adb6b0'
)
DEFECT_TEST_MANIFEST_VERSION = 'm3_defect_test_manifest_v1'
CLEAN10_DEFECT_TEST_MANIFEST_VERSION = 'm3_defect_test_manifest_clean10_v2'
REQUIRED_TRAINING_PROTOCOL_VERSION = 'm3_6b_5fold_v1'
REQUIRED_TRAINING_PROTOCOL_HASH = (
    'c0c635c1cb897ce33d15ba3ed58abdb12a8c5a8feb6a82fe51910146e8a99fac'
)
SOURCE_EVALUATION_PROTOCOL_VERSION = 'm3_7_eval_v1'
SOURCE_EVALUATION_PROTOCOL_HASH = (
    '5c1125745765664acbdf5883d9a3e328465836b33ac32898b72e9b417b03eda6'
)
REQUIRED_CHECKPOINT_BASENAME = 'best_val_loss.pt'
REQUIRED_DEFECT_TRAINING_ADAPTER = 'pointct_defect_training_v1'
REQUIRED_EPOCH_COUNT = 20
CASES_PER_DEFECT_INSTANCE = 15
EXPECTED_TEST_CASE_COUNT = EXPECTED_DEFECT_INSTANCE_COUNT * CASES_PER_DEFECT_INSTANCE

REQUIRED_FORMAL_TRAINING = {
    'seed': 20260815,
    'learning_rate': 0.0003,
    'weight_decay': 0.0001,
    'batch_size': 1,
    'precision': 'fp32',
    'temperature': 0.10,
    'sinkhorn_iterations': 20,
    'alpha_init': 1.0,
    'max_epochs': REQUIRED_EPOCH_COUNT,
}
_CHECKPOINT_TRAINING_CONFIG_FIELDS = (
    'learning_rate',
    'weight_decay',
    'batch_size',
    'precision',
    'seed',
    'temperature',
    'sinkhorn_iterations',
    'alpha_init',
)
_EVALUATION_PROTOCOL_FIELDS = {
    'evaluation_protocol_version',
    'evaluation_protocol_hash',
    'source_evaluation_protocol_version',
    'source_evaluation_protocol_hash',
    'required_training_protocol_version',
    'required_training_protocol_hash',
    'test_perturbation_source',
    'perturbation_seed_source',
    'registration_rre_threshold_deg',
    'registration_rte_threshold_mm',
    'correspondence_inlier_threshold_mm',
    'matching_filter_min_confidence',
    'required_checkpoint_basename',
    'required_formal_training',
    'required_defect_training_adapter',
    'required_defect_ids',
    'required_enable_m4_defect_mapping',
    'required_epoch_count',
    'solver_failure_metric_policy',
    'zero_correspondence_policy',
    'runtime_scope',
}
_EXPECTED_EVALUATION_PROTOCOL = {
    'evaluation_protocol_version': DEFECT_EVALUATION_PROTOCOL_VERSION,
    'source_evaluation_protocol_version': SOURCE_EVALUATION_PROTOCOL_VERSION,
    'source_evaluation_protocol_hash': SOURCE_EVALUATION_PROTOCOL_HASH,
    'required_training_protocol_version': REQUIRED_TRAINING_PROTOCOL_VERSION,
    'required_training_protocol_hash': REQUIRED_TRAINING_PROTOCOL_HASH,
    'test_perturbation_source': 'm3_6b_5fold_v1.test_perturbations',
    'perturbation_seed_source': 'm3_6b_seed_v1.subject_based_without_defect_id',
    'registration_rre_threshold_deg': 5.0,
    'registration_rte_threshold_mm': 10.0,
    'correspondence_inlier_threshold_mm': 15.0,
    'matching_filter_min_confidence': None,
    'required_checkpoint_basename': REQUIRED_CHECKPOINT_BASENAME,
    'required_formal_training': REQUIRED_FORMAL_TRAINING,
    'required_defect_training_adapter': REQUIRED_DEFECT_TRAINING_ADAPTER,
    'required_defect_ids': list(DEFECT_IDS),
    'required_enable_m4_defect_mapping': False,
    'required_epoch_count': REQUIRED_EPOCH_COUNT,
    'solver_failure_metric_policy': 'null_rre_rte',
    'zero_correspondence_policy': (
        'zero_inliers_zero_ratio_registration_fail_closed'
    ),
    'runtime_scope': (
        'model_forward+matching+mutual_filtering+weighted_registration'
    ),
}
_CLEAN10_EXPECTED_EVALUATION_PROTOCOL = {
    **_EXPECTED_EVALUATION_PROTOCOL,
    'evaluation_protocol_version': CLEAN10_DEFECT_EVALUATION_PROTOCOL_VERSION,
    'required_training_protocol_version': 'm3_6b_5fold_clean10_v2',
    'required_training_protocol_hash': (
        '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'
    ),
    'test_perturbation_source': 'm3_6b_5fold_clean10_v2.test_perturbations',
}
_EVALUATION_PROTOCOL_CONTRACTS = {
    DEFECT_EVALUATION_PROTOCOL_VERSION: {
        'evaluation_protocol_hash': DEFECT_EVALUATION_PROTOCOL_HASH,
        'test_manifest_version': DEFECT_TEST_MANIFEST_VERSION,
        'expected': _EXPECTED_EVALUATION_PROTOCOL,
    },
    CLEAN10_DEFECT_EVALUATION_PROTOCOL_VERSION: {
        'evaluation_protocol_hash': CLEAN10_DEFECT_EVALUATION_PROTOCOL_HASH,
        'test_manifest_version': CLEAN10_DEFECT_TEST_MANIFEST_VERSION,
        'expected': _CLEAN10_EXPECTED_EVALUATION_PROTOCOL,
    },
}
_BEST_VAL_LOSS_ABS_TOLERANCE = 1e-12


class DefectEvaluationContractError(M3EvaluationContractError):
    """Raised when the formal M3 defect evaluation contract is violated."""


def _evaluation_contract_for_version(evaluation_protocol_version: str) -> Mapping:
    try:
        return _EVALUATION_PROTOCOL_CONTRACTS[evaluation_protocol_version]
    except (KeyError, TypeError) as error:
        raise DefectEvaluationContractError(
            f'unsupported evaluation_protocol_version '
            f'{evaluation_protocol_version!r}; expected one of '
            f'{list(_EVALUATION_PROTOCOL_CONTRACTS)}.'
        ) from error


def _reject_duplicate_json_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise DefectEvaluationContractError(
                f'duplicate JSON object field: {key!r}.'
            )
        output[key] = value
    return output


def _canonical_protocol_json(protocol: Mapping) -> str:
    if not isinstance(protocol, Mapping):
        raise DefectEvaluationContractError(
            'defect evaluation protocol must be a mapping.'
        )
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
        raise DefectEvaluationContractError(
            'defect evaluation protocol must be canonical-JSON serializable.'
        ) from error


def compute_defect_evaluation_protocol_hash(protocol: Mapping) -> str:
    """Hash every field except ``evaluation_protocol_hash``."""
    canonical_json = _canonical_protocol_json(protocol)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def validate_defect_evaluation_protocol(protocol: Mapping) -> Mapping:
    """Fail closed on any deviation from a versioned evaluation manifest."""
    if not isinstance(protocol, Mapping):
        raise DefectEvaluationContractError(
            'defect evaluation protocol must be a JSON object.'
        )
    actual_fields = set(protocol)
    missing = sorted(_EVALUATION_PROTOCOL_FIELDS.difference(actual_fields))
    unexpected = sorted(actual_fields.difference(_EVALUATION_PROTOCOL_FIELDS))
    if missing or unexpected:
        raise DefectEvaluationContractError(
            'defect evaluation protocol fields mismatch; '
            f'missing={missing}, unexpected={unexpected}.'
        )
    contract = _evaluation_contract_for_version(
        protocol['evaluation_protocol_version']
    )
    stored_hash = protocol['evaluation_protocol_hash']
    if (
        not isinstance(stored_hash, str)
        or len(stored_hash) != 64
        or any(character not in '0123456789abcdef' for character in stored_hash)
    ):
        raise DefectEvaluationContractError(
            'evaluation_protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    computed_hash = compute_defect_evaluation_protocol_hash(protocol)
    if not hmac.compare_digest(stored_hash, computed_hash):
        raise DefectEvaluationContractError(
            'evaluation_protocol_hash mismatch: '
            f'stored={stored_hash}, computed={computed_hash}.'
        )
    if not hmac.compare_digest(
        stored_hash,
        contract['evaluation_protocol_hash'],
    ):
        raise DefectEvaluationContractError(
            'evaluation_protocol_hash is not the frozen hash for '
            f'{protocol["evaluation_protocol_version"]}: stored={stored_hash}, '
            f'expected={contract["evaluation_protocol_hash"]}.'
        )
    for field, expected in contract['expected'].items():
        actual = protocol[field]
        if actual != expected or type(actual) is not type(expected):
            raise DefectEvaluationContractError(
                f'{field} must be exactly {expected!r} for '
                f'{protocol["evaluation_protocol_version"]}.'
            )
    return protocol


def load_defect_evaluation_protocol(path) -> Mapping:
    """Load the frozen defect evaluation manifest with duplicate-key rejection."""
    path = Path(path)
    if not path.is_file():
        raise DefectEvaluationContractError(
            f'defect evaluation protocol does not exist: {path}.'
        )
    try:
        with path.open('r', encoding='utf-8') as handle:
            protocol = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_fields,
            )
    except DefectEvaluationContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DefectEvaluationContractError(
            f'cannot load defect evaluation protocol {path}: {error}'
        ) from error
    return validate_defect_evaluation_protocol(protocol)


def validate_protocol_pair(training_protocol, evaluation_protocol) -> None:
    """Validate the exact training/evaluation protocol pairing."""
    try:
        validate_training_protocol(training_protocol)
    except Exception as error:
        raise DefectEvaluationContractError(
            f'training protocol validation failed: {error}'
        ) from error
    validate_defect_evaluation_protocol(evaluation_protocol)
    if (
        training_protocol['protocol_version']
        != evaluation_protocol['required_training_protocol_version']
    ):
        raise DefectEvaluationContractError(
            'defect evaluation training protocol version mismatch.'
        )
    if (
        training_protocol['protocol_hash']
        != evaluation_protocol['required_training_protocol_hash']
    ):
        raise DefectEvaluationContractError(
            'defect evaluation training protocol hash mismatch.'
        )


def _generate_defect_test_manifest(
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> dict:
    validate_protocol_pair(training_protocol, evaluation_protocol)
    evaluation_contract = _evaluation_contract_for_version(
        evaluation_protocol['evaluation_protocol_version']
    )
    fold = resolve_fold(training_protocol, fold_id)
    specs = test_perturbation_specs(training_protocol)
    cases = []
    seen_case_keys = set()

    for subject_id in fold['test_subject_ids']:
        for defect_id in DEFECT_IDS:
            for spec in specs:
                for variant_index in range(spec['variant_count']):
                    perturbation_identifier = (
                        f'{fold_id}|{subject_id}|{spec["severity"]}|{variant_index}'
                    )
                    case_key = f'{fold_id}|{subject_id}|{defect_id}|{spec["severity"]}|{variant_index}'
                    if case_key in seen_case_keys:
                        raise DefectEvaluationContractError(
                            f'duplicate defect test case key generated: {case_key}.'
                        )
                    seed_arguments = {
                        'scheme_version': training_protocol['seed_scheme_version'],
                        'protocol_hash': training_protocol['protocol_hash'],
                        'fold_id': fold_id,
                        'root_seed': training_protocol['perturbation_root_seed'],
                        'purpose': spec['purpose'],
                        'epoch': None,
                        'subject_id': subject_id,
                        'severity': spec['severity'],
                        'variant_id': variant_index,
                    }
                    # Deliberately do not add defect_id.  This is the exact frozen
                    # subject-based M3 test seed payload.
                    seed = derive_perturbation_seed(**seed_arguments)
                    if seed != derive_perturbation_seed(**seed_arguments):
                        raise DefectEvaluationContractError(
                            f'perturbation seed is non-deterministic for {case_key}.'
                        )
                    sampled = sample_rigid_perturbation(
                        seed,
                        spec['max_rotation_deg'],
                        spec['max_translation_mm'],
                    )
                    repeated = sample_rigid_perturbation(
                        seed,
                        spec['max_rotation_deg'],
                        spec['max_translation_mm'],
                    )
                    if (
                        sampled['angle_deg'] != repeated['angle_deg']
                        or not np.array_equal(
                            sampled['translation_mm'], repeated['translation_mm']
                        )
                        or not np.array_equal(sampled['R_aug'], repeated['R_aug'])
                    ):
                        raise DefectEvaluationContractError(
                            f'perturbation sample is non-deterministic for {case_key}.'
                        )
                    nonidentity = bool(
                        abs(sampled['angle_deg']) > 0.0
                        or np.any(sampled['translation_mm'] != 0.0)
                    )
                    if not nonidentity:
                        raise DefectEvaluationContractError(
                            f'formal test perturbation is identity for {case_key}.'
                        )
                    seen_case_keys.add(case_key)
                    cases.append(
                        {
                            'case_key': case_key,
                            'perturbation_identifier': perturbation_identifier,
                            'fold_id': fold_id,
                            'subject_id': subject_id,
                            'defect_id': defect_id,
                            'purpose': spec['purpose'],
                            'epoch': None,
                            'severity': spec['severity'],
                            'variant_id': variant_index,
                            'variant_index': variant_index,
                            'perturbation_seed': seed,
                            'max_rotation_deg': float(spec['max_rotation_deg']),
                            'max_translation_mm': float(
                                spec['max_translation_mm']
                            ),
                            'perturbation_angle_deg': float(sampled['angle_deg']),
                            'perturbation_translation_mm': sampled[
                                'translation_mm'
                            ].astype(float).tolist(),
                            'perturbation_nonidentity': True,
                        }
                    )

    expected_instances = len(fold['test_subject_ids']) * DEFECT_CONDITION_COUNT
    expected_cases = expected_instances * CASES_PER_DEFECT_INSTANCE
    if len(cases) != expected_cases:
        raise DefectEvaluationContractError(
            f'{fold_id} must contain exactly {expected_cases} defect test cases.'
        )
    instance_counts = Counter(
        (case['subject_id'], case['defect_id']) for case in cases
    )
    if set(instance_counts.values()) != {CASES_PER_DEFECT_INSTANCE}:
        raise DefectEvaluationContractError(
            'every defect test instance must contain exactly 15 cases.'
        )
    severity_counts = Counter(
        (case['subject_id'], case['defect_id'], case['severity'])
        for case in cases
    )
    for identity in instance_counts:
        for spec in specs:
            if severity_counts[identity + (spec['severity'],)] != spec['variant_count']:
                raise DefectEvaluationContractError(
                    f'defect test severity count mismatch for {identity!r}.'
                )

    return {
        'test_manifest_version': evaluation_contract['test_manifest_version'],
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol[
            'evaluation_protocol_hash'
        ],
        'fold_id': fold_id,
        'test_subject_ids': list(fold['test_subject_ids']),
        'defect_ids': list(DEFECT_IDS),
        'seed_scheme_version': training_protocol['seed_scheme_version'],
        'perturbation_root_seed': training_protocol['perturbation_root_seed'],
        'perturbation_seed_includes_defect_id': False,
        'cases_per_defect_instance': CASES_PER_DEFECT_INSTANCE,
        'total_test_instances': expected_instances,
        'total_cases': len(cases),
        'cases': cases,
    }


def build_defect_test_manifest(
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> dict:
    """Build twice and compare one formal defect Fold test manifest."""
    first = _generate_defect_test_manifest(
        training_protocol,
        evaluation_protocol,
        fold_id,
    )
    second = _generate_defect_test_manifest(
        training_protocol,
        evaluation_protocol,
        fold_id,
    )
    if first != second:
        raise DefectEvaluationContractError(
            'defect test manifest generation is non-deterministic.'
        )
    return first


def validate_expanded_partitions(
    fold_id: str,
    ready_subject_ids: Sequence[str],
    train_instance_ids: Sequence[tuple],
    val_instance_ids: Sequence[tuple],
    test_instance_ids: Sequence[tuple],
) -> None:
    """Validate patient and instance isolation for one expanded defect Fold."""
    expected_instances = set(build_defect_variants(tuple(ready_subject_ids)))
    partitions = (
        tuple(train_instance_ids),
        tuple(val_instance_ids),
        tuple(test_instance_ids),
    )
    partition_sets = tuple(set(partition) for partition in partitions)
    if any(len(partition) != len(partition_set) for partition, partition_set in zip(partitions, partition_sets)):
        raise DefectEvaluationContractError(
            f'{fold_id} contains a duplicate subject-defect instance.'
        )
    for partition in partitions:
        for identity in partition:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or identity not in expected_instances
            ):
                raise DefectEvaluationContractError(
                    f'{fold_id} contains an unknown subject-defect instance {identity!r}.'
                )
    patient_sets = tuple(
        {subject_id for subject_id, _ in partition}
        for partition in partitions
    )
    if (
        patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        raise DefectEvaluationContractError(
            f'{fold_id} contains patient-level leakage.'
        )
    if (
        partition_sets[0] & partition_sets[1]
        or partition_sets[0] & partition_sets[2]
        or partition_sets[1] & partition_sets[2]
    ):
        raise DefectEvaluationContractError(
            f'{fold_id} contains instance-level leakage.'
        )
    if set().union(*partition_sets) != expected_instances:
        raise DefectEvaluationContractError(
            f'{fold_id} does not cover all {len(expected_instances)} '
            'formal defect instances.'
        )
    for partition in partitions:
        by_subject = defaultdict(set)
        for subject_id, defect_id in partition:
            by_subject[subject_id].add(defect_id)
        if any(defects != set(DEFECT_IDS) for defects in by_subject.values()):
            raise DefectEvaluationContractError(
                f'{fold_id} splits the five defect conditions of a patient.'
            )


def run_protocol_audit(training_protocol, evaluation_protocol) -> dict:
    """Audit all five Fold expansions without reading data or checkpoints."""
    validate_protocol_pair(training_protocol, evaluation_protocol)
    defect_contract = get_defect_training_contract(training_protocol)
    expected_instance_count = defect_contract['expected_defect_instance_count']
    expected_test_case_count = expected_instance_count * CASES_PER_DEFECT_INSTANCE
    fold_results = {}
    global_instances = []
    global_case_keys = []
    test_patient_counts = Counter()
    subject_to_test_fold = {}

    for fold_id in defect_contract['fold_ids']:
        fold = resolve_fold(training_protocol, fold_id)

        def expand(field):
            return tuple(
                (subject_id, defect_id)
                for subject_id in fold[field]
                for defect_id in DEFECT_IDS
            )

        train_instances = expand('train_subject_ids')
        val_instances = expand('val_subject_ids')
        test_instances = expand('test_subject_ids')
        validate_expanded_partitions(
            fold_id,
            tuple(training_protocol['ready_subject_ids']),
            train_instances,
            val_instances,
            test_instances,
        )
        manifest = build_defect_test_manifest(
            training_protocol,
            evaluation_protocol,
            fold_id,
        )
        for subject_id in fold['test_subject_ids']:
            if subject_id in subject_to_test_fold:
                raise DefectEvaluationContractError(
                    f'patient {subject_id!r} tests in more than one Fold.'
                )
            subject_to_test_fold[subject_id] = fold_id
        test_patient_counts.update(fold['test_subject_ids'])
        global_instances.extend(test_instances)
        global_case_keys.extend(case['case_key'] for case in manifest['cases'])
        fold_results[fold_id] = {
            'test_patient_count': len(fold['test_subject_ids']),
            'test_instance_count': len(test_instances),
            'test_case_count': manifest['total_cases'],
        }

    expected_once = Counter(
        {subject_id: 1 for subject_id in training_protocol['ready_subject_ids']}
    )
    if test_patient_counts != expected_once:
        raise DefectEvaluationContractError(
            'every ready patient must test exactly once across all Folds.'
        )
    if (
        len(global_instances) != expected_instance_count
        or len(set(global_instances)) != len(global_instances)
    ):
        raise DefectEvaluationContractError(
            'global test subject-defect pairs are not exactly '
            f'{expected_instance_count} unique instances.'
        )
    if (
        len(global_case_keys) != expected_test_case_count
        or len(set(global_case_keys)) != len(global_case_keys)
    ):
        raise DefectEvaluationContractError(
            'global defect test cases are not exactly '
            f'{expected_test_case_count} unique cases.'
        )
    return {
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol[
            'evaluation_protocol_hash'
        ],
        'ready_patient_count': defect_contract['ready_patient_count'],
        'defect_condition_count': DEFECT_CONDITION_COUNT,
        'expected_test_instance_count': len(global_instances),
        'expected_test_case_count': len(global_case_keys),
        'patient_level_leakage_pass': True,
        'instance_level_leakage_pass': True,
        'pair_uniqueness_pass': True,
        'perturbation_count_pass': True,
        'frozen_subject_based_perturbation_pass': True,
        'folds': fold_results,
    }


def expected_defect_training_provenance(training_protocol, fold_id: str) -> dict:
    """Build the exact defect provenance written by ``train_m3_defect.py``."""
    fold = resolve_fold(training_protocol, fold_id)

    def records(partition, subject_ids):
        return [
            {
                'subject_id': subject_id,
                'defect_id': defect_id,
                'fold_id': fold_id,
                'partition': partition,
            }
            for subject_id in subject_ids
            for defect_id in DEFECT_IDS
        ]

    return {
        'adapter': REQUIRED_DEFECT_TRAINING_ADAPTER,
        'fold_id': fold_id,
        'defect_ids': list(DEFECT_IDS),
        'train_instances': records('train', fold['train_subject_ids']),
        'val_instances': records('val', fold['val_subject_ids']),
        'test_instances': records('test', fold['test_subject_ids']),
    }


def _require_finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise DefectEvaluationContractError(f'{name} must be a finite float.')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DefectEvaluationContractError(
            f'{name} must be a finite float.'
        ) from error
    if not math.isfinite(result):
        raise DefectEvaluationContractError(f'{name} must be a finite float.')
    return result


def validate_defect_checkpoint_metadata(
    checkpoint_path,
    payload,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> dict:
    """Validate defect checkpoint metadata without restoring model state."""
    validate_protocol_pair(training_protocol, evaluation_protocol)
    checkpoint_path = Path(checkpoint_path)
    required_basename = evaluation_protocol['required_checkpoint_basename']
    if checkpoint_path.name != required_basename:
        raise DefectEvaluationContractError(
            'defect evaluation requires a checkpoint named exactly '
            f'{required_basename}; got {checkpoint_path.name!r}.'
        )
    if not isinstance(payload, Mapping):
        raise DefectEvaluationContractError('checkpoint payload must be a mapping.')
    required_metadata = {
        'formal_protocol',
        'protocol_version',
        'protocol_hash',
        'fold_id',
        'train_subject_ids',
        'val_subject_ids',
        'test_subject_ids',
        'perturbation_root_seed',
        'perturbation_seed_scheme_version',
        'epoch',
        'best_val_loss',
        'seed',
        'training_config',
    }
    missing = sorted(required_metadata.difference(payload))
    if missing:
        raise DefectEvaluationContractError(
            f'formal defect checkpoint metadata is missing fields: {missing}.'
        )
    if payload['formal_protocol'] is not True:
        raise DefectEvaluationContractError(
            'checkpoint formal_protocol must be true.'
        )
    fold = resolve_fold(training_protocol, fold_id)
    comparisons = (
        ('protocol_version', training_protocol['protocol_version']),
        ('protocol_hash', training_protocol['protocol_hash']),
        ('fold_id', fold_id),
        ('perturbation_root_seed', training_protocol['perturbation_root_seed']),
        (
            'perturbation_seed_scheme_version',
            training_protocol['seed_scheme_version'],
        ),
    )
    for field, expected in comparisons:
        actual = payload.get(field)
        if actual != expected or type(actual) is not type(expected):
            raise DefectEvaluationContractError(
                f'checkpoint {field} does not match the formal defect Fold.'
            )
    for field, expected in (
        ('train_subject_ids', fold['train_subject_ids']),
        ('val_subject_ids', fold['val_subject_ids']),
        ('test_subject_ids', fold['test_subject_ids']),
    ):
        if tuple(payload.get(field, ())) != tuple(expected):
            raise DefectEvaluationContractError(
                f'checkpoint {field} does not match {fold_id}.'
            )
    epoch = payload['epoch']
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise DefectEvaluationContractError(
            'checkpoint epoch must be a non-negative integer.'
        )
    best_val_loss = _require_finite_float(
        payload['best_val_loss'], 'checkpoint best_val_loss'
    )
    required_seed = evaluation_protocol['required_formal_training']['seed']
    if payload['seed'] != required_seed or type(payload['seed']) is not type(required_seed):
        raise DefectEvaluationContractError(
            'checkpoint seed does not match formal defect training.'
        )

    training_config = payload['training_config']
    if not isinstance(training_config, Mapping):
        raise DefectEvaluationContractError(
            'checkpoint training_config must be a mapping.'
        )
    missing_training = [
        field
        for field in _CHECKPOINT_TRAINING_CONFIG_FIELDS
        if field not in training_config
    ]
    if missing_training:
        raise DefectEvaluationContractError(
            'checkpoint training_config is missing formal fields: '
            f'{missing_training}.'
        )
    if training_config['seed'] != payload['seed'] or type(training_config['seed']) is not type(payload['seed']):
        raise DefectEvaluationContractError(
            'checkpoint training_config.seed does not match checkpoint seed.'
        )
    for field in _CHECKPOINT_TRAINING_CONFIG_FIELDS:
        expected = evaluation_protocol['required_formal_training'][field]
        actual = training_config[field]
        if actual != expected or type(actual) is not type(expected):
            raise DefectEvaluationContractError(
                f'checkpoint training_config.{field} does not match formal '
                f'defect training: expected={expected!r}, actual={actual!r}.'
            )
    if 'enable_m4_defect_mapping' not in training_config:
        raise DefectEvaluationContractError(
            'checkpoint is missing enable_m4_defect_mapping provenance; '
            'old complete-subject checkpoints are not accepted.'
        )
    if training_config['enable_m4_defect_mapping'] is not False:
        raise DefectEvaluationContractError(
            'M3 defect baseline evaluation requires enable_m4_defect_mapping=false.'
        )
    actual_provenance = training_config.get('defect_training_provenance')
    expected_provenance = expected_defect_training_provenance(
        training_protocol,
        fold_id,
    )
    if actual_provenance != expected_provenance:
        raise DefectEvaluationContractError(
            'checkpoint defect_training_provenance does not match the formal '
            f'{get_defect_training_contract(training_protocol)["expected_defect_instance_count"]}'
            '-instance defect adapter; old complete-subject checkpoints are not accepted.'
        )
    return {
        'fold_id': fold_id,
        'protocol_version': payload['protocol_version'],
        'protocol_hash': payload['protocol_hash'],
        'test_subject_ids': tuple(payload['test_subject_ids']),
        'epoch': int(epoch),
        'best_val_loss': best_val_loss,
        'seed': int(payload['seed']),
        'training_config': dict(training_config),
        'defect_training_provenance': expected_provenance,
        'enable_m4_defect_mapping': False,
    }


def load_jsonl_records(path) -> list:
    """Load a JSONL file with duplicate-field and blank-line rejection."""
    path = Path(path)
    if not path.is_file():
        raise DefectEvaluationContractError(
            f'JSONL checkpoint provenance is missing: {path}; '
            'best-checkpoint cross-check cannot be performed.'
        )
    records = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DefectEvaluationContractError(
                        f'JSONL contains a blank line at {line_number}: {path}.'
                    )
                try:
                    record = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_fields,
                    )
                except json.JSONDecodeError as error:
                    raise DefectEvaluationContractError(
                        f'invalid JSONL record at line {line_number}: {error}'
                    ) from error
                records.append(record)
    except DefectEvaluationContractError:
        raise
    except (OSError, UnicodeError) as error:
        raise DefectEvaluationContractError(
            f'cannot read JSONL checkpoint provenance {path}: {error}'
        ) from error
    return records


def validate_jsonl_best_checkpoint(
    records: Sequence[Mapping],
    checkpoint_metadata: Mapping,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> dict:
    """Cross-check all 20 JSONL epochs against the actual best checkpoint."""
    validate_protocol_pair(training_protocol, evaluation_protocol)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise DefectEvaluationContractError('JSONL records must be a sequence.')
    required_count = evaluation_protocol['required_epoch_count']
    if len(records) != required_count:
        raise DefectEvaluationContractError(
            f'{fold_id} JSONL must contain exactly {required_count} epoch records; '
            f'got {len(records)}.'
        )
    expected_provenance = expected_defect_training_provenance(
        training_protocol,
        fold_id,
    )
    by_epoch = {}
    running_best = math.inf
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL record {position} must be a mapping.'
            )
        epoch = record.get('epoch')
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL record {position} has an invalid epoch.'
            )
        if epoch in by_epoch:
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL contains duplicate epoch {epoch}.'
            )
        if record.get('formal_protocol') is not True:
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL epoch {epoch} is not formal protocol output.'
            )
        for field, expected in (
            ('protocol_version', training_protocol['protocol_version']),
            ('protocol_hash', training_protocol['protocol_hash']),
            ('fold_id', fold_id),
        ):
            if record.get(field) != expected:
                raise DefectEvaluationContractError(
                    f'{fold_id} JSONL epoch {epoch} has mismatched {field}.'
                )
        if tuple(record.get('test_subject_ids', ())) != tuple(
            resolve_fold(training_protocol, fold_id)['test_subject_ids']
        ):
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL epoch {epoch} has mismatched test_subject_ids.'
            )
        if record.get('enable_m4_defect_mapping') is not False:
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL epoch {epoch} must record '
                'enable_m4_defect_mapping=false.'
            )
        if record.get('defect_training_provenance') != expected_provenance:
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL epoch {epoch} has invalid defect provenance.'
            )
        val_loss = _require_finite_float(
            record.get('val_mean_loss'),
            f'{fold_id} JSONL epoch {epoch} val_mean_loss',
        )
        expected_updated = val_loss < running_best
        running_best = min(running_best, val_loss)
        recorded_best = _require_finite_float(
            record.get('best_val_loss'),
            f'{fold_id} JSONL epoch {epoch} best_val_loss',
        )
        if not math.isclose(
            recorded_best,
            running_best,
            rel_tol=0.0,
            abs_tol=_BEST_VAL_LOSS_ABS_TOLERANCE,
        ):
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL epoch {epoch} running best_val_loss is inconsistent.'
            )
        if record.get('best_checkpoint_updated') is not expected_updated:
            raise DefectEvaluationContractError(
                f'{fold_id} JSONL epoch {epoch} best_checkpoint_updated is inconsistent.'
            )
        by_epoch[epoch] = {'record': record, 'val_mean_loss': val_loss}

    expected_epochs = set(range(required_count))
    if set(by_epoch) != expected_epochs:
        missing = sorted(expected_epochs.difference(by_epoch))
        unexpected = sorted(set(by_epoch).difference(expected_epochs))
        raise DefectEvaluationContractError(
            f'{fold_id} JSONL epochs mismatch; missing={missing}, unexpected={unexpected}.'
        )
    best_epoch = min(
        by_epoch,
        key=lambda epoch: (by_epoch[epoch]['val_mean_loss'], epoch),
    )
    best_val_loss = by_epoch[best_epoch]['val_mean_loss']
    if checkpoint_metadata.get('epoch') != best_epoch:
        raise DefectEvaluationContractError(
            f'{fold_id} JSONL best epoch does not match checkpoint epoch: '
            f'jsonl={best_epoch}, checkpoint={checkpoint_metadata.get("epoch")!r}.'
        )
    checkpoint_loss = _require_finite_float(
        checkpoint_metadata.get('best_val_loss'),
        'checkpoint best_val_loss',
    )
    if not math.isclose(
        checkpoint_loss,
        best_val_loss,
        rel_tol=0.0,
        abs_tol=_BEST_VAL_LOSS_ABS_TOLERANCE,
    ):
        raise DefectEvaluationContractError(
            f'{fold_id} JSONL minimum val_mean_loss does not match checkpoint '
            f'best_val_loss: jsonl={best_val_loss!r}, checkpoint={checkpoint_loss!r}.'
        )
    return {
        'jsonl_epoch_count': len(records),
        'jsonl_best_epoch': best_epoch,
        'jsonl_min_val_mean_loss': best_val_loss,
        'checkpoint_epoch': checkpoint_metadata['epoch'],
        'checkpoint_best_val_loss': checkpoint_loss,
        'best_checkpoint_cross_check_pass': True,
    }


def resolve_fold_artifacts(
    checkpoint_root,
    fold_id: str,
    evaluation_protocol,
    *,
    json_log_root=None,
) -> tuple:
    """Resolve the fixed checkpoint and JSONL locations for one Fold."""
    validate_defect_evaluation_protocol(evaluation_protocol)
    if fold_id not in ('Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'):
        raise DefectEvaluationContractError(f'unknown fold_id {fold_id!r}.')
    checkpoint_root = Path(checkpoint_root)
    log_root = checkpoint_root if json_log_root is None else Path(json_log_root)
    checkpoint_path = (
        checkpoint_root
        / fold_id
        / evaluation_protocol['required_checkpoint_basename']
    )
    jsonl_path = log_root / f'{fold_id}.jsonl'
    return checkpoint_path, jsonl_path


def read_and_validate_defect_checkpoint(
    checkpoint_path,
    jsonl_path,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> tuple:
    """Load a checkpoint on CPU, then validate metadata and JSONL provenance."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise DefectEvaluationContractError(
            f'defect evaluation checkpoint does not exist: {checkpoint_path}.'
        )
    try:
        from evaluate_m3 import _load_checkpoint_for_execution

        payload, checkpoint_contract = _load_checkpoint_for_execution(
            checkpoint_path
        )
    except Exception as error:
        if isinstance(error, DefectEvaluationContractError):
            raise
        raise DefectEvaluationContractError(
            f'checkpoint structural validation failed: {error}'
        ) from error
    if checkpoint_contract.get('formal_protocol') is not True:
        raise DefectEvaluationContractError(
            'checkpoint must be a formal protocol checkpoint.'
        )
    metadata = validate_defect_checkpoint_metadata(
        checkpoint_path,
        payload,
        training_protocol,
        evaluation_protocol,
        fold_id,
    )
    records = load_jsonl_records(jsonl_path)
    jsonl_validation = validate_jsonl_best_checkpoint(
        records,
        metadata,
        training_protocol,
        evaluation_protocol,
        fold_id,
    )
    metadata['jsonl_path'] = str(Path(jsonl_path).resolve())
    metadata['jsonl_validation'] = jsonl_validation
    return payload, metadata


def _aggregate_by_field(cases: Sequence[Mapping], field: str) -> dict:
    groups = defaultdict(list)
    for case in cases:
        value = case.get(field)
        if not isinstance(value, str) or not value:
            raise DefectEvaluationContractError(
                f'every case requires a non-empty {field}.'
            )
        groups[value].append(case)
    return {
        key: aggregate_case_metrics(groups[key])
        for key in sorted(groups)
    }


def aggregate_defect_case_metrics(cases: Sequence[Mapping]) -> dict:
    """Reuse frozen M3 aggregation and add per-defect/per-Fold groupings."""
    cases = list(cases)
    summary = aggregate_case_metrics(cases)
    summary['per_defect_condition'] = _aggregate_by_field(cases, 'defect_id')
    summary['per_fold'] = _aggregate_by_field(cases, 'fold_id')
    return summary


__all__ = [
    'CASES_PER_DEFECT_INSTANCE',
    'CLEAN10_DEFECT_EVALUATION_PROTOCOL_HASH',
    'CLEAN10_DEFECT_EVALUATION_PROTOCOL_VERSION',
    'CLEAN10_DEFECT_TEST_MANIFEST_VERSION',
    'DEFECT_EVALUATION_PROTOCOL_HASH',
    'DEFECT_EVALUATION_PROTOCOL_VERSION',
    'DEFECT_TEST_MANIFEST_VERSION',
    'DefectEvaluationContractError',
    'EXPECTED_TEST_CASE_COUNT',
    'REQUIRED_EPOCH_COUNT',
    'REQUIRED_FORMAL_TRAINING',
    'aggregate_defect_case_metrics',
    'build_defect_test_manifest',
    'compute_defect_evaluation_protocol_hash',
    'expected_defect_training_provenance',
    'load_defect_evaluation_protocol',
    'load_jsonl_records',
    'read_and_validate_defect_checkpoint',
    'resolve_fold_artifacts',
    'run_protocol_audit',
    'validate_defect_checkpoint_metadata',
    'validate_defect_evaluation_protocol',
    'validate_expanded_partitions',
    'validate_jsonl_best_checkpoint',
    'validate_protocol_pair',
]
