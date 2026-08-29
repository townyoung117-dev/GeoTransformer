"""Contracts and CPU aggregation for clean10 complete-vs-defect pairing."""

import hashlib
import hmac
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from defect_training import DEFECT_IDS
from m3_metric_diagnostic import DIAGNOSTIC_TRE_FIELDS, metric_statistics
from m3_metric_observation import (
    OBSERVATION_PROTOCOL_HASH,
    OBSERVATION_PROTOCOL_VERSION,
    validate_observation_case,
    validate_observation_protocol,
)


PAIRED_PROTOCOL_VERSION = 'm3_complete_vs_defect_paired_clean10_v1'
PAIRED_PROTOCOL_HASH = (
    '8849d7346913502a1322a2aa1f940993665bae67ad26154f846b75c944c3b5c7'
)
SOURCE_CODE_COMMIT = 'e38f56be2227a9ebf89d0c7f755a3ad03cdc3970'
SOURCE_TRAINING_PROTOCOL_VERSION = 'm3_6b_5fold_clean10_v2'
SOURCE_TRAINING_PROTOCOL_HASH = (
    '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'
)
SOURCE_EVALUATION_PROTOCOL_VERSION = 'm3_defect_eval_clean10_v2'
SOURCE_EVALUATION_PROTOCOL_HASH = (
    'cfd519b923be3b623cffec8c8f5830cb5b1160859461180eddeb7134a0adb6b0'
)
PATIENT_COUNT = 10
COMPLETE_INSTANCE_COUNT = 10
COMPLETE_INFERENCE_CASE_COUNT = 150
DEFECT_CONDITION_COUNT = 5
PAIRED_COMPARISON_COUNT = 750
COMPLETE_CASES_PER_FOLD = 30
PAIRED_CASES_PER_FOLD = 150
EXPECTED_PAT6_COUNT = 0
EXPECTED_PAT10_COUNT = 0
ALLOWED_SUBJECT_IDS = (
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
FOLD_IDS = ('Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5')

_EXPECTED_PAIRED_PROTOCOL = {
    'paired_protocol_version': PAIRED_PROTOCOL_VERSION,
    'source_code_commit': SOURCE_CODE_COMMIT,
    'source_training_protocol_version': SOURCE_TRAINING_PROTOCOL_VERSION,
    'source_training_protocol_hash': SOURCE_TRAINING_PROTOCOL_HASH,
    'source_defect_evaluation_protocol_version': (
        SOURCE_EVALUATION_PROTOCOL_VERSION
    ),
    'source_defect_evaluation_protocol_hash': SOURCE_EVALUATION_PROTOCOL_HASH,
    'source_tre_observation_protocol_version': OBSERVATION_PROTOCOL_VERSION,
    'source_tre_observation_protocol_hash': OBSERVATION_PROTOCOL_HASH,
    'complete_dataset_factory': 'create_dataset(data_root)_without_defect_variants',
    'complete_input_contract': 'complete_point_cloud_plus_complete_ct',
    'source_reference_points': (
        'corresponding_augmented_input_point_physical_coordinates'
    ),
    'complete_source_reference_points': (
        'augmented_complete_point_physical_coordinates'
    ),
    'defect_source_reference_points': (
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
    'point_tre_percentile': 95,
    'point_tre_percentile_method': 'linear',
    'tre_fields': list(DIAGNOSTIC_TRE_FIELDS),
    'delta_definition': 'defect_minus_complete',
    'patient_cluster_aggregation': (
        'mean_delta_over_5_defects_x_15_perturbations_per_patient'
    ),
    'statistical_policy': (
        'descriptive_case_and_patient_level_statistics_only_no_independent_t_test'
    ),
    'interpretation_policy': 'numeric_outputs_only_no_automatic_causal_claim',
    'defect_bundle_cases_filename': 'cases_750.jsonl',
    'patient_count': PATIENT_COUNT,
    'complete_instance_count': COMPLETE_INSTANCE_COUNT,
    'complete_inference_case_count': COMPLETE_INFERENCE_CASE_COUNT,
    'defect_condition_count': DEFECT_CONDITION_COUNT,
    'paired_comparison_count': PAIRED_COMPARISON_COUNT,
    'complete_cases_per_fold': COMPLETE_CASES_PER_FOLD,
    'paired_cases_per_fold': PAIRED_CASES_PER_FOLD,
    'expected_pat6_count': EXPECTED_PAT6_COUNT,
    'expected_pat10_count': EXPECTED_PAT10_COUNT,
    'allowed_subject_ids': list(ALLOWED_SUBJECT_IDS),
    'defect_ids': list(DEFECT_IDS),
    'training_performed': False,
    'model_or_matcher_changed': False,
    'formal_evaluation_changed': False,
}
_PAIRED_PROTOCOL_FIELDS = set(_EXPECTED_PAIRED_PROTOCOL) | {
    'paired_protocol_hash'
}


class M3CompleteVsDefectPairedContractError(RuntimeError):
    """Raised when paired sensitivity structure or provenance is invalid."""


def _reject_duplicate_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise M3CompleteVsDefectPairedContractError(
                f'duplicate JSON object field: {key!r}.'
            )
        result[key] = value
    return result


def compute_paired_protocol_hash(protocol: Mapping) -> str:
    if not isinstance(protocol, Mapping):
        raise M3CompleteVsDefectPairedContractError(
            'paired protocol must be a mapping.'
        )
    payload = dict(protocol)
    payload.pop('paired_protocol_hash', None)
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M3CompleteVsDefectPairedContractError(
            'paired protocol must be canonical-JSON serializable.'
        ) from error
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_paired_protocol(protocol: Mapping) -> Mapping:
    if not isinstance(protocol, Mapping):
        raise M3CompleteVsDefectPairedContractError(
            'paired protocol must be a JSON object.'
        )
    actual_fields = set(protocol)
    missing = sorted(_PAIRED_PROTOCOL_FIELDS.difference(actual_fields))
    unexpected = sorted(actual_fields.difference(_PAIRED_PROTOCOL_FIELDS))
    if missing or unexpected:
        raise M3CompleteVsDefectPairedContractError(
            'paired protocol fields mismatch; '
            f'missing={missing}, unexpected={unexpected}.'
        )
    stored_hash = protocol['paired_protocol_hash']
    if (
        not isinstance(stored_hash, str)
        or len(stored_hash) != 64
        or any(character not in '0123456789abcdef' for character in stored_hash)
    ):
        raise M3CompleteVsDefectPairedContractError(
            'paired_protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    computed = compute_paired_protocol_hash(protocol)
    if not hmac.compare_digest(stored_hash, computed):
        raise M3CompleteVsDefectPairedContractError(
            'paired_protocol_hash mismatch: '
            f'stored={stored_hash}, computed={computed}.'
        )
    if not hmac.compare_digest(stored_hash, PAIRED_PROTOCOL_HASH):
        raise M3CompleteVsDefectPairedContractError(
            'paired_protocol_hash is not the frozen hash for '
            f'{PAIRED_PROTOCOL_VERSION}: stored={stored_hash}, '
            f'expected={PAIRED_PROTOCOL_HASH}.'
        )
    for field, expected in _EXPECTED_PAIRED_PROTOCOL.items():
        actual = protocol[field]
        if actual != expected or type(actual) is not type(expected):
            raise M3CompleteVsDefectPairedContractError(
                f'{field} must be exactly {expected!r} for '
                f'{PAIRED_PROTOCOL_VERSION}.'
            )
    return protocol


def load_paired_protocol(path) -> Mapping:
    path = Path(path)
    if not path.is_file():
        raise M3CompleteVsDefectPairedContractError(
            f'paired protocol does not exist: {path}.'
        )
    try:
        with path.open('r', encoding='utf-8') as handle:
            protocol = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_fields,
            )
    except M3CompleteVsDefectPairedContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3CompleteVsDefectPairedContractError(
            f'cannot load paired protocol {path}: {error}'
        ) from error
    return validate_paired_protocol(protocol)


def validate_source_protocols(
    paired_protocol: Mapping,
    training_protocol: Mapping,
    evaluation_protocol: Mapping,
    observation_protocol: Mapping,
) -> None:
    """Bind the experiment to the exact frozen clean10 sources."""
    validate_paired_protocol(paired_protocol)
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
        if isinstance(error, M3CompleteVsDefectPairedContractError):
            raise
        raise M3CompleteVsDefectPairedContractError(
            f'source protocol canonical validation failed: {error}'
        ) from error
    comparisons = (
        (
            training_protocol.get('protocol_version'),
            paired_protocol['source_training_protocol_version'],
            'source training protocol version',
        ),
        (
            training_protocol.get('protocol_hash'),
            paired_protocol['source_training_protocol_hash'],
            'source training protocol hash',
        ),
        (
            evaluation_protocol.get('evaluation_protocol_version'),
            paired_protocol['source_defect_evaluation_protocol_version'],
            'source evaluation protocol version',
        ),
        (
            evaluation_protocol.get('evaluation_protocol_hash'),
            paired_protocol['source_defect_evaluation_protocol_hash'],
            'source evaluation protocol hash',
        ),
        (
            observation_protocol.get('observation_protocol_version'),
            paired_protocol['source_tre_observation_protocol_version'],
            'source observation protocol version',
        ),
        (
            observation_protocol.get('observation_protocol_hash'),
            paired_protocol['source_tre_observation_protocol_hash'],
            'source observation protocol hash',
        ),
    )
    for actual, expected, label in comparisons:
        if actual != expected or type(actual) is not type(expected):
            raise M3CompleteVsDefectPairedContractError(
                f'{label} mismatch: expected={expected!r}, actual={actual!r}.'
            )
    if observation_protocol['centroid_tre_definition'] != paired_protocol[
        'centroid_tre_definition'
    ] or observation_protocol['point_tre_definition'] != paired_protocol[
        'point_tre_definition'
    ]:
        raise M3CompleteVsDefectPairedContractError(
            'paired TRE definitions do not match frozen v3 observation.'
        )
    if observation_protocol['source_reference_points'] != paired_protocol[
        'defect_source_reference_points'
    ]:
        raise M3CompleteVsDefectPairedContractError(
            'paired defect source_reference_points do not match v3.'
        )


def _complete_dataset_selection(
    dataset,
    training_protocol: Mapping,
) -> tuple:
    """Validate raw complete records and resolve the protocol-selected indices."""
    try:
        from training_protocol import (
            get_training_protocol_contract,
            validate_training_protocol,
        )

        validate_training_protocol(training_protocol)
        training_contract = get_training_protocol_contract(training_protocol)
    except Exception as error:
        raise M3CompleteVsDefectPairedContractError(
            f'clean10 training protocol validation failed: {error}'
        ) from error
    if tuple(training_protocol['ready_subject_ids']) != ALLOWED_SUBJECT_IDS:
        raise M3CompleteVsDefectPairedContractError(
            'clean10 allowed subject contract mismatch.'
        )
    if getattr(dataset, 'defect_enabled', None) is not False:
        raise M3CompleteVsDefectPairedContractError(
            'complete dataset must be created without defect_variants.'
        )
    records = getattr(dataset, 'records', None)
    if not isinstance(records, list):
        raise M3CompleteVsDefectPairedContractError(
            'complete dataset must expose manifest records as a list.'
        )
    selected_subject_ids = tuple(training_protocol['ready_subject_ids'])
    excluded_subject_ids = tuple(training_contract['excluded_subject_ids'])
    if len(selected_subject_ids) != COMPLETE_INSTANCE_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'selected clean10 complete instance count mismatch: '
            f'expected={COMPLETE_INSTANCE_COUNT}, '
            f'actual={len(selected_subject_ids)}.'
        )
    if set(selected_subject_ids).intersection(excluded_subject_ids):
        raise M3CompleteVsDefectPairedContractError(
            'training protocol selected/excluded subject sets overlap.'
        )
    raw_subject_ids = []
    raw_subject_to_index = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise M3CompleteVsDefectPairedContractError(
                'complete dataset records must be mappings.'
            )
        subject_id = record.get('subject_id')
        if not isinstance(subject_id, str) or not subject_id:
            raise M3CompleteVsDefectPairedContractError(
                'complete dataset record requires subject_id.'
            )
        if 'defect_id' in record:
            raise M3CompleteVsDefectPairedContractError(
                'complete dataset record must not contain defect_id.'
            )
        for field in ('pointcloud_path', 'ct_path'):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise M3CompleteVsDefectPairedContractError(
                    f'complete dataset record requires non-empty {field}.'
                )
        if subject_id in raw_subject_to_index:
            raise M3CompleteVsDefectPairedContractError(
                'complete dataset contains duplicate patients: '
                f'{[subject_id]}.'
            )
        raw_subject_to_index[subject_id] = index
        raw_subject_ids.append(subject_id)
    selected_set = set(selected_subject_ids)
    raw_set = set(raw_subject_ids)
    missing = sorted(selected_set.difference(raw_set))
    if missing:
        raise M3CompleteVsDefectPairedContractError(
            'raw complete dataset is missing selected clean10 patients: '
            f'{missing}.'
        )
    raw_extra = raw_set.difference(selected_set)
    unknown_extra = sorted(raw_extra.difference(excluded_subject_ids))
    if unknown_extra:
        raise M3CompleteVsDefectPairedContractError(
            'raw complete dataset contains ready patients not selected or '
            f'explicitly excluded by the training protocol: {unknown_extra}.'
        )
    selected_subject_to_index = {
        subject_id: raw_subject_to_index[subject_id]
        for subject_id in selected_subject_ids
    }
    if set(selected_subject_to_index) != selected_set:
        raise M3CompleteVsDefectPairedContractError(
            'selected complete subject mapping keys mismatch clean10 protocol.'
        )
    if (
        len(selected_subject_to_index) != COMPLETE_INSTANCE_COUNT
        or len(set(selected_subject_to_index.values())) != COMPLETE_INSTANCE_COUNT
    ):
        raise M3CompleteVsDefectPairedContractError(
            'selected complete subject mapping must contain 10 unique indices.'
        )
    return (
        records,
        raw_subject_ids,
        selected_subject_ids,
        excluded_subject_ids,
        selected_subject_to_index,
    )


def build_clean10_complete_subject_index(
    dataset,
    training_protocol: Mapping,
) -> dict:
    """Map exactly the frozen clean10 subjects to unique raw dataset indices."""
    selection = _complete_dataset_selection(dataset, training_protocol)
    return dict(selection[4])


def audit_complete_dataset(dataset, training_protocol: Mapping) -> dict:
    """Report raw historical records separately from clean10 selection."""
    (
        records,
        raw_subject_ids,
        selected_subject_ids,
        excluded_subject_ids,
        selected_subject_to_index,
    ) = _complete_dataset_selection(dataset, training_protocol)
    raw_excluded_present = [
        subject_id
        for subject_id in excluded_subject_ids
        if subject_id in set(raw_subject_ids)
    ]
    pat6_raw_count = raw_subject_ids.count('Pat6')
    pat6_selected_count = selected_subject_ids.count('Pat6')
    pat10_selected_count = selected_subject_ids.count('Pat10')
    if pat6_selected_count or pat10_selected_count:
        raise M3CompleteVsDefectPairedContractError(
            'selected clean10 subjects contain excluded Pat6 or Pat10.'
        )
    return {
        'dataset_mode': 'complete_without_defect_variants',
        'selection_source': 'frozen_clean10_training_protocol',
        'raw_ready_patient_count': len(records),
        'raw_ready_subject_ids': list(raw_subject_ids),
        'selected_patient_count': len(selected_subject_ids),
        'selected_complete_instance_count': len(selected_subject_to_index),
        'selected_subject_ids': list(selected_subject_ids),
        'raw_excluded_subject_ids_present': raw_excluded_present,
        'pat6_raw_count': pat6_raw_count,
        'pat6_selected_count': pat6_selected_count,
        'pat10_selected_count': pat10_selected_count,
        # Backward-compatible aliases now explicitly describe selection, not
        # the historical raw ready-record count.
        'patient_count': len(selected_subject_ids),
        'complete_instance_count': len(selected_subject_to_index),
        'subject_ids': list(selected_subject_ids),
        'instances_per_patient': {
            subject_id: 1 for subject_id in selected_subject_ids
        },
        'pat6_count': pat6_selected_count,
        'pat10_count': pat10_selected_count,
        'defect_masks_consumed': False,
    }


def _complete_identity(row: Mapping) -> tuple:
    if not isinstance(row, Mapping):
        raise M3CompleteVsDefectPairedContractError(
            'complete case must be a mapping.'
        )
    values = []
    for field in ('fold_id', 'subject_id', 'severity'):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise M3CompleteVsDefectPairedContractError(
                f'complete identity field {field} must be non-empty.'
            )
        values.append(value)
    for field in ('variant_id', 'perturbation_seed'):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise M3CompleteVsDefectPairedContractError(
                f'complete identity field {field} must be an integer.'
            )
        values.append(int(value))
    return tuple(values)


def _pair_base_identity(row: Mapping) -> tuple:
    complete = _complete_identity(row)
    return complete[:4]


def _paired_identity(row: Mapping) -> tuple:
    complete = _complete_identity(row)
    defect_id = row.get('defect_id')
    if not isinstance(defect_id, str) or not defect_id:
        raise M3CompleteVsDefectPairedContractError(
            'paired identity defect_id must be non-empty.'
        )
    return complete[:2] + (defect_id,) + complete[2:]


def _require_unique(rows: Sequence[Mapping], identity_fn, label: str) -> dict:
    result = {}
    for row in rows:
        identity = identity_fn(row)
        if identity in result:
            raise M3CompleteVsDefectPairedContractError(
                f'duplicate {label} identity: {identity!r}.'
            )
        result[identity] = row
    return result


def build_complete_manifest(
    defect_manifests: Mapping[str, Mapping],
    paired_protocol: Mapping,
) -> dict:
    """Collapse five defects sharing one perturbation into 150 complete cases."""
    validate_paired_protocol(paired_protocol)
    if tuple(defect_manifests) != FOLD_IDS:
        raise M3CompleteVsDefectPairedContractError(
            'defect manifests must contain ordered Fold1-Fold5.'
        )
    all_complete = []
    per_fold = {}
    for fold_id in FOLD_IDS:
        manifest = defect_manifests[fold_id]
        rows = list(manifest.get('cases', ()))
        if len(rows) != PAIRED_CASES_PER_FOLD:
            raise M3CompleteVsDefectPairedContractError(
                f'{fold_id} defect manifest must contain 150 cases.'
            )
        groups = defaultdict(list)
        for row in rows:
            if row.get('fold_id') != fold_id:
                raise M3CompleteVsDefectPairedContractError(
                    f'{fold_id} defect manifest contains a different Fold.'
                )
            groups[
                (
                    row.get('subject_id'),
                    row.get('severity'),
                    row.get('variant_id'),
                )
            ].append(row)
        complete_rows = []
        for key in sorted(groups, key=lambda value: (value[0], value[1], value[2])):
            defect_rows = groups[key]
            defect_ids = [row.get('defect_id') for row in defect_rows]
            if len(defect_rows) != DEFECT_CONDITION_COUNT or set(defect_ids) != set(
                DEFECT_IDS
            ):
                raise M3CompleteVsDefectPairedContractError(
                    f'{fold_id}/{key} must contain exactly the five frozen defects.'
                )
            seeds = {row.get('perturbation_seed') for row in defect_rows}
            if len(seeds) != 1:
                raise M3CompleteVsDefectPairedContractError(
                    f'{fold_id}/{key} paired defect seeds are inconsistent.'
                )
            representative = defect_rows[0]
            for field in (
                'purpose',
                'epoch',
                'variant_index',
                'max_rotation_deg',
                'max_translation_mm',
                'perturbation_angle_deg',
                'perturbation_translation_mm',
                'perturbation_nonidentity',
            ):
                if any(row.get(field) != representative.get(field) for row in defect_rows):
                    raise M3CompleteVsDefectPairedContractError(
                        f'{fold_id}/{key} paired perturbation field {field} differs.'
                    )
            complete_case_key = (
                f'{fold_id}|{representative["subject_id"]}|complete|'
                f'{representative["severity"]}|{representative["variant_id"]}'
            )
            complete_rows.append(
                {
                    'complete_case_key': complete_case_key,
                    'perturbation_identifier': representative.get(
                        'perturbation_identifier'
                    ),
                    'fold_id': fold_id,
                    'subject_id': representative['subject_id'],
                    'input_kind': 'complete',
                    'purpose': representative.get('purpose'),
                    'epoch': representative.get('epoch'),
                    'severity': representative['severity'],
                    'variant_id': representative['variant_id'],
                    'variant_index': representative.get('variant_index'),
                    'perturbation_seed': representative['perturbation_seed'],
                    'max_rotation_deg': representative['max_rotation_deg'],
                    'max_translation_mm': representative[
                        'max_translation_mm'
                    ],
                    'perturbation_angle_deg': representative.get(
                        'perturbation_angle_deg'
                    ),
                    'perturbation_translation_mm': representative.get(
                        'perturbation_translation_mm'
                    ),
                    'perturbation_nonidentity': representative.get(
                        'perturbation_nonidentity'
                    ),
                    'paired_defect_ids': list(DEFECT_IDS),
                }
            )
        if len(complete_rows) != COMPLETE_CASES_PER_FOLD:
            raise M3CompleteVsDefectPairedContractError(
                f'{fold_id} complete case count mismatch: '
                f'expected=30, actual={len(complete_rows)}.'
            )
        _require_unique(complete_rows, _complete_identity, f'{fold_id} complete')
        per_fold[fold_id] = complete_rows
        all_complete.extend(complete_rows)
    _require_unique(all_complete, _complete_identity, 'complete manifest union')
    subjects = {row['subject_id'] for row in all_complete}
    if len(all_complete) != COMPLETE_INFERENCE_CASE_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'complete manifest union must contain 150 cases.'
        )
    if subjects != set(ALLOWED_SUBJECT_IDS):
        raise M3CompleteVsDefectPairedContractError(
            'complete manifest patient set mismatch.'
        )
    if any(row['subject_id'] in ('Pat6', 'Pat10') for row in all_complete):
        raise M3CompleteVsDefectPairedContractError(
            'complete manifest contains excluded Pat6 or Pat10.'
        )
    return {
        'paired_protocol_version': paired_protocol['paired_protocol_version'],
        'paired_protocol_hash': paired_protocol['paired_protocol_hash'],
        'complete_inference_case_count': len(all_complete),
        'complete_cases_per_fold': COMPLETE_CASES_PER_FOLD,
        'per_fold_case_counts': {
            fold_id: len(per_fold[fold_id]) for fold_id in FOLD_IDS
        },
        'cases': all_complete,
        'per_fold': per_fold,
    }


def _finite_number(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise M3CompleteVsDefectPairedContractError(
            f'{name} must be a finite number.'
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3CompleteVsDefectPairedContractError(
            f'{name} must be a finite number.'
        ) from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise M3CompleteVsDefectPairedContractError(
            f'{name} must be a finite number.'
        )
    return result


def validate_complete_case(row: Mapping, paired_protocol: Mapping) -> None:
    validate_paired_protocol(paired_protocol)
    _complete_identity(row)
    for field, expected in (
        ('paired_protocol_version', paired_protocol['paired_protocol_version']),
        ('paired_protocol_hash', paired_protocol['paired_protocol_hash']),
        (
            'source_training_protocol_version',
            paired_protocol['source_training_protocol_version'],
        ),
        (
            'source_training_protocol_hash',
            paired_protocol['source_training_protocol_hash'],
        ),
        (
            'source_defect_evaluation_protocol_version',
            paired_protocol['source_defect_evaluation_protocol_version'],
        ),
        (
            'source_defect_evaluation_protocol_hash',
            paired_protocol['source_defect_evaluation_protocol_hash'],
        ),
        (
            'source_reference_points',
            paired_protocol['complete_source_reference_points'],
        ),
        ('input_kind', 'complete'),
    ):
        actual = row.get(field)
        if actual != expected or type(actual) is not type(expected):
            raise M3CompleteVsDefectPairedContractError(
                f'complete case {field} mismatch: '
                f'expected={expected!r}, actual={actual!r}.'
            )
    checkpoint = row.get('checkpoint_path')
    if not isinstance(checkpoint, str) or not checkpoint:
        raise M3CompleteVsDefectPairedContractError(
            'complete case requires checkpoint_path.'
        )
    success = row.get('solver_success')
    if not isinstance(success, bool):
        raise M3CompleteVsDefectPairedContractError(
            'complete case solver_success must be boolean.'
        )
    for field in DIAGNOSTIC_TRE_FIELDS:
        value = row.get(field)
        if success:
            _finite_number(value, f'complete {field}', nonnegative=True)
        elif value is not None:
            raise M3CompleteVsDefectPairedContractError(
                f'complete solver failure must keep {field} null.'
            )


def _descriptive_statistics(values: Sequence, *, total_count: int) -> dict:
    finite = []
    for value in values:
        if value is not None:
            finite.append(_finite_number(value, 'descriptive statistic value'))
    array = np.asarray(finite, dtype=np.float64)
    return {
        'total_count': total_count,
        'sample_count': len(finite),
        'mean': float(np.mean(array, dtype=np.float64)) if finite else None,
        'median': float(np.median(array)) if finite else None,
        'min': float(np.min(array)) if finite else None,
        'max': float(np.max(array)) if finite else None,
    }


def aggregate_complete_cases(
    rows: Sequence[Mapping],
    paired_protocol: Mapping,
    *,
    expected_count: Optional[int] = None,
) -> dict:
    cases = list(rows)
    if expected_count is not None and len(cases) != expected_count:
        raise M3CompleteVsDefectPairedContractError(
            'complete case count mismatch: '
            f'expected={expected_count}, actual={len(cases)}.'
        )
    for row in cases:
        validate_complete_case(row, paired_protocol)
    identities = _require_unique(cases, _complete_identity, 'complete result')
    by_fold = Counter(row['fold_id'] for row in cases)
    return {
        'case_count': len(cases),
        'identity_unique_count': len(identities),
        'solver_success_count': sum(row['solver_success'] for row in cases),
        'solver_failure_count': sum(not row['solver_success'] for row in cases),
        'per_fold_case_counts': {
            fold_id: by_fold[fold_id] for fold_id in FOLD_IDS
        },
        'tre_metrics': {
            field: metric_statistics(row.get(field) for row in cases)
            for field in DIAGNOSTIC_TRE_FIELDS
        },
    }


def _metric_payload(row: Mapping, source: str) -> dict:
    if source == 'complete':
        return {field: row.get(field) for field in DIAGNOSTIC_TRE_FIELDS}
    replay = row['replay_observation']
    return {field: replay.get(field) for field in DIAGNOSTIC_TRE_FIELDS}


def _group_delta_summary(rows: Sequence[Mapping]) -> dict:
    cases = list(rows)
    return {
        'case_count': len(cases),
        'complete_unique_case_count': len(
            {
                (
                    row['fold_id'],
                    row['subject_id'],
                    row['severity'],
                    row['variant_id'],
                    row['perturbation_seed'],
                )
                for row in cases
            }
        ),
        'delta_metrics': {
            f'delta_{field}': _descriptive_statistics(
                [row.get(f'delta_{field}') for row in cases],
                total_count=len(cases),
            )
            for field in DIAGNOSTIC_TRE_FIELDS
        },
    }


def _aggregate_groups(rows: Sequence[Mapping], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise M3CompleteVsDefectPairedContractError(
                f'paired row requires non-empty {field}.'
            )
        groups[value].append(row)
    return {
        key: _group_delta_summary(groups[key])
        for key in sorted(groups)
    }


def build_paired_comparison(
    complete_rows: Sequence[Mapping],
    defect_rows: Sequence[Mapping],
    paired_protocol: Mapping,
    observation_protocol: Mapping,
    *,
    expected_complete_manifest: Optional[Sequence[Mapping]] = None,
) -> dict:
    """Create 750 paired rows and case-/patient-level descriptive summaries."""
    validate_paired_protocol(paired_protocol)
    validate_observation_protocol(observation_protocol)
    complete = list(complete_rows)
    defects = list(defect_rows)
    if len(complete) != COMPLETE_INFERENCE_CASE_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'complete input must contain exactly 150 cases.'
        )
    if len(defects) != PAIRED_COMPARISON_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'defect input must contain exactly 750 cases.'
        )
    for row in complete:
        validate_complete_case(row, paired_protocol)
    complete_by_identity = _require_unique(
        complete,
        _complete_identity,
        'complete result',
    )
    if expected_complete_manifest is not None:
        expected_map = _require_unique(
            list(expected_complete_manifest),
            _complete_identity,
            'expected complete manifest',
        )
        if set(complete_by_identity) != set(expected_map):
            missing = sorted(set(expected_map).difference(complete_by_identity))
            unexpected = sorted(set(complete_by_identity).difference(expected_map))
            raise M3CompleteVsDefectPairedContractError(
                'complete result identity mismatch; first_missing='
                f'{missing[0] if missing else None!r}, first_unexpected='
                f'{unexpected[0] if unexpected else None!r}.'
            )
    defect_by_identity = {}
    defect_groups = defaultdict(list)
    for row in defects:
        validate_observation_case(row, observation_protocol)
        identity = _paired_identity(row)
        if identity in defect_by_identity:
            raise M3CompleteVsDefectPairedContractError(
                f'duplicate defect pair identity: {identity!r}.'
            )
        defect_by_identity[identity] = row
        defect_groups[_pair_base_identity(row)].append(row)
    if len(defect_by_identity) != PAIRED_COMPARISON_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'defect input must contain 750 unique pair identities.'
        )
    paired_rows = []
    for base_identity, complete_row in sorted(
        ((_pair_base_identity(row), row) for row in complete),
        key=lambda item: item[0],
    ):
        defect_group = defect_groups.get(base_identity, [])
        defect_ids = [row.get('defect_id') for row in defect_group]
        if len(defect_group) != DEFECT_CONDITION_COUNT or set(defect_ids) != set(
            DEFECT_IDS
        ):
            raise M3CompleteVsDefectPairedContractError(
                f'complete case {base_identity!r} must pair with five defects.'
            )
        seeds = {int(row['perturbation_seed']) for row in defect_group}
        if len(seeds) != 1 or seeds != {int(complete_row['perturbation_seed'])}:
            raise M3CompleteVsDefectPairedContractError(
                f'complete/defect seed mismatch for {base_identity!r}.'
            )
        complete_checkpoint = complete_row['checkpoint_path']
        complete_metrics = _metric_payload(complete_row, 'complete')
        for defect_row in sorted(
            defect_group,
            key=lambda row: row['defect_id'],
        ):
            defect_checkpoint = defect_row.get('checkpoint_path')
            if (
                not isinstance(defect_checkpoint, str)
                or not defect_checkpoint
                or defect_checkpoint != complete_checkpoint
            ):
                raise M3CompleteVsDefectPairedContractError(
                    f'complete/defect checkpoint mismatch for '
                    f'{_paired_identity(defect_row)!r}.'
                )
            defect_metrics = _metric_payload(defect_row, 'defect')
            deltas = {}
            for field in DIAGNOSTIC_TRE_FIELDS:
                complete_value = complete_metrics[field]
                defect_value = defect_metrics[field]
                deltas[f'delta_{field}'] = (
                    None
                    if complete_value is None or defect_value is None
                    else float(defect_value) - float(complete_value)
                )
            paired_rows.append(
                {
                    'paired_protocol_version': paired_protocol[
                        'paired_protocol_version'
                    ],
                    'paired_protocol_hash': paired_protocol[
                        'paired_protocol_hash'
                    ],
                    'fold_id': complete_row['fold_id'],
                    'subject_id': complete_row['subject_id'],
                    'defect_id': defect_row['defect_id'],
                    'severity': complete_row['severity'],
                    'variant_id': complete_row['variant_id'],
                    'perturbation_seed': complete_row['perturbation_seed'],
                    'checkpoint_path': complete_checkpoint,
                    'complete': complete_metrics,
                    'defect': defect_metrics,
                    **deltas,
                }
            )
    paired_map = _require_unique(paired_rows, _paired_identity, 'paired result')
    if len(paired_rows) != PAIRED_COMPARISON_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'paired output must contain exactly 750 rows.'
        )
    if any(row['subject_id'] in ('Pat6', 'Pat10') for row in paired_rows):
        raise M3CompleteVsDefectPairedContractError(
            'paired output contains excluded Pat6 or Pat10.'
        )
    fold_counts = Counter(row['fold_id'] for row in paired_rows)
    if any(fold_counts[fold_id] != PAIRED_CASES_PER_FOLD for fold_id in FOLD_IDS):
        raise M3CompleteVsDefectPairedContractError(
            f'paired per-Fold count mismatch: {dict(fold_counts)}.'
        )
    per_subject = _aggregate_groups(paired_rows, 'subject_id')
    if len(per_subject) != PATIENT_COUNT:
        raise M3CompleteVsDefectPairedContractError(
            'patient-level aggregate must contain exactly 10 patients.'
        )
    if any(value['case_count'] != 75 for value in per_subject.values()):
        raise M3CompleteVsDefectPairedContractError(
            'each patient aggregate must contain 75 paired cases.'
        )
    patient_level_metrics = {}
    for field in DIAGNOSTIC_TRE_FIELDS:
        delta_field = f'delta_{field}'
        patient_means = {
            subject_id: per_subject[subject_id]['delta_metrics'][delta_field][
                'mean'
            ]
            for subject_id in sorted(per_subject)
        }
        patient_level_metrics[delta_field] = {
            **_descriptive_statistics(
                list(patient_means.values()),
                total_count=PATIENT_COUNT,
            ),
            'patient_means': patient_means,
            'positive_patient_count': sum(
                value is not None and value > 0.0
                for value in patient_means.values()
            ),
        }
    complete_summary = {
        'case_count': len(complete),
        'tre_metrics': {
            field: _descriptive_statistics(
                [row.get(field) for row in complete],
                total_count=len(complete),
            )
            for field in DIAGNOSTIC_TRE_FIELDS
        },
    }
    defect_summary = {
        'case_count': len(defects),
        'tre_metrics': {
            field: _descriptive_statistics(
                [row['replay_observation'].get(field) for row in defects],
                total_count=len(defects),
            )
            for field in DIAGNOSTIC_TRE_FIELDS
        },
    }
    case_level = _group_delta_summary(paired_rows)
    point_delta = case_level['delta_metrics']['delta_point_tre_mean_mm']
    summary = {
        'paired_protocol_version': paired_protocol['paired_protocol_version'],
        'paired_protocol_hash': paired_protocol['paired_protocol_hash'],
        'complete': complete_summary,
        'defect': defect_summary,
        'paired_case_level': case_level,
        'patient_level': {
            'patient_count': len(per_subject),
            'independent_patient_count': len(per_subject),
            'delta_metrics': patient_level_metrics,
        },
        'complete_point_tre_mean_mm': complete_summary['tre_metrics'][
            'point_tre_mean_mm'
        ]['mean'],
        'complete_centroid_tre_mean_mm': complete_summary['tre_metrics'][
            'centroid_tre_mm'
        ]['mean'],
        'defect_point_tre_mean_mm': defect_summary['tre_metrics'][
            'point_tre_mean_mm'
        ]['mean'],
        'defect_centroid_tre_mean_mm': defect_summary['tre_metrics'][
            'centroid_tre_mm'
        ]['mean'],
        'mean_delta_point_tre_mm': point_delta['mean'],
        'median_delta_point_tre_mm': point_delta['median'],
        'patients_with_positive_mean_delta_point_tre': (
            patient_level_metrics['delta_point_tre_mean_mm'][
                'positive_patient_count'
            ]
        ),
        'statistical_policy': paired_protocol['statistical_policy'],
        'interpretation_policy': paired_protocol['interpretation_policy'],
        'independent_sample_t_test_performed': False,
    }
    provenance = {
        'paired_protocol_version': paired_protocol['paired_protocol_version'],
        'paired_protocol_hash': paired_protocol['paired_protocol_hash'],
        'source_code_commit': paired_protocol['source_code_commit'],
        'source_training_protocol_version': paired_protocol[
            'source_training_protocol_version'
        ],
        'source_training_protocol_hash': paired_protocol[
            'source_training_protocol_hash'
        ],
        'source_defect_evaluation_protocol_version': paired_protocol[
            'source_defect_evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': paired_protocol[
            'source_defect_evaluation_protocol_hash'
        ],
        'source_tre_observation_protocol_version': paired_protocol[
            'source_tre_observation_protocol_version'
        ],
        'source_tre_observation_protocol_hash': paired_protocol[
            'source_tre_observation_protocol_hash'
        ],
        'delta_definition': paired_protocol['delta_definition'],
        'model_loaded': False,
        'gpu_used': False,
        'training_performed': False,
        'automatic_causal_interpretation': False,
    }
    return {
        'paired_rows': paired_rows,
        'summary': summary,
        'per_subject': per_subject,
        'per_defect': _aggregate_groups(paired_rows, 'defect_id'),
        'per_severity': _aggregate_groups(paired_rows, 'severity'),
        'per_fold': _aggregate_groups(paired_rows, 'fold_id'),
        'provenance': provenance,
        'paired_identity_unique_count': len(paired_map),
    }


__all__ = [
    'ALLOWED_SUBJECT_IDS',
    'COMPLETE_CASES_PER_FOLD',
    'COMPLETE_INFERENCE_CASE_COUNT',
    'COMPLETE_INSTANCE_COUNT',
    'DEFECT_CONDITION_COUNT',
    'EXPECTED_PAT10_COUNT',
    'EXPECTED_PAT6_COUNT',
    'FOLD_IDS',
    'M3CompleteVsDefectPairedContractError',
    'PAIRED_CASES_PER_FOLD',
    'PAIRED_COMPARISON_COUNT',
    'PAIRED_PROTOCOL_HASH',
    'PAIRED_PROTOCOL_VERSION',
    'PATIENT_COUNT',
    'aggregate_complete_cases',
    'audit_complete_dataset',
    'build_clean10_complete_subject_index',
    'build_complete_manifest',
    'build_paired_comparison',
    'compute_paired_protocol_hash',
    'load_paired_protocol',
    'validate_complete_case',
    'validate_paired_protocol',
    'validate_source_protocols',
]
