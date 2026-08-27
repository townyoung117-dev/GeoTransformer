"""Formal 55-instance defect dataset and patient-level split adapter.

This module deliberately leaves the complete-subject M3 baseline contract
untouched.  Patients remain the split unit while ``(subject_id, defect_id)``
pairs are the dataset and training-instance unit.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from training_protocol import (
    FOLD_SUBJECTS,
    M3TrainingProtocolError,
    READY_SUBJECT_IDS,
    resolve_fold,
    validate_training_protocol,
)


DEFECT_IDS = (
    'defect_001_left_maxilla_cheek_small',
    'defect_001_left_maxilla_cheek_medium',
    'defect_001_left_maxilla_cheek_large',
    'defect_001_right_maxilla_cheek_small',
    'defect_001_right_maxilla_cheek_medium',
)
READY_PATIENT_COUNT = len(READY_SUBJECT_IDS)
DEFECT_CONDITION_COUNT = len(DEFECT_IDS)
EXPECTED_DEFECT_INSTANCE_COUNT = READY_PATIENT_COUNT * DEFECT_CONDITION_COUNT


class DefectTrainingContractError(RuntimeError):
    """Raised when the formal defect-training contract is not exact."""


@dataclass(frozen=True)
class FormalDefectSplit:
    """A formal patient split expanded to deterministic defect instances."""

    fold_id: str
    train_subject_ids: tuple
    val_subject_ids: tuple
    test_subject_ids: tuple
    train_indices: tuple
    val_indices: tuple
    test_indices: tuple
    train_instance_ids: tuple
    val_instance_ids: tuple
    test_instance_ids: tuple

    @property
    def all_instance_ids(self):
        return self.train_instance_ids + self.val_instance_ids + self.test_instance_ids


def _validate_protocol(protocol):
    try:
        validate_training_protocol(protocol)
    except M3TrainingProtocolError as error:
        raise DefectTrainingContractError(
            f'formal defect protocol validation failed: {error}'
        ) from error
    return protocol


def build_defect_variants(ready_subject_ids: Sequence[str]) -> tuple:
    """Expand the frozen formal patients to 55 ordered explicit defect pairs."""
    if isinstance(ready_subject_ids, (str, bytes)) or not isinstance(
        ready_subject_ids, Sequence
    ):
        raise DefectTrainingContractError('ready_subject_ids must be an explicit sequence.')
    subject_ids = tuple(ready_subject_ids)
    if subject_ids != READY_SUBJECT_IDS:
        missing = sorted(set(READY_SUBJECT_IDS).difference(subject_ids))
        unexpected = sorted(set(subject_ids).difference(READY_SUBJECT_IDS))
        raise DefectTrainingContractError(
            'defect ready patients must exactly match the frozen formal protocol; '
            f'missing={missing}, unexpected={unexpected}.'
        )
    if len(set(subject_ids)) != READY_PATIENT_COUNT:
        raise DefectTrainingContractError('defect ready patients must be unique.')
    if 'Pat10' in subject_ids:
        raise DefectTrainingContractError('Pat10 must not enter formal defect training.')

    variants = tuple(
        (subject_id, defect_id)
        for subject_id in subject_ids
        for defect_id in DEFECT_IDS
    )
    if len(variants) != EXPECTED_DEFECT_INSTANCE_COUNT or len(set(variants)) != len(variants):
        raise DefectTrainingContractError('formal defect pair expansion is not exactly 55 unique pairs.')
    patient_counts = Counter(subject_id for subject_id, _ in variants)
    defect_counts = Counter(defect_id for _, defect_id in variants)
    if set(patient_counts.values()) != {DEFECT_CONDITION_COUNT}:
        raise DefectTrainingContractError('every ready patient must expand to exactly five defects.')
    if set(defect_counts.values()) != {READY_PATIENT_COUNT}:
        raise DefectTrainingContractError('every formal defect must cover exactly 11 patients.')
    return variants


def create_formal_defect_dataset(data_root, protocol, *, dataset_factory=None):
    """Construct the existing PointCTDataset through its explicit defect selector."""
    _validate_protocol(protocol)
    if dataset_factory is None:
        from dataset import create_dataset as dataset_factory

    variants = build_defect_variants(tuple(protocol['ready_subject_ids']))
    dataset = dataset_factory(data_root, defect_variants=variants)
    validate_formal_defect_dataset(dataset, protocol)
    return dataset


def _validated_identity_to_index(dataset, protocol):
    _validate_protocol(protocol)
    records = getattr(dataset, 'records', None)
    if not isinstance(records, list):
        raise DefectTrainingContractError('defect dataset must expose records as a list.')

    expected_subjects = tuple(protocol['ready_subject_ids'])
    expected_subject_set = set(expected_subjects)
    expected_defect_set = set(DEFECT_IDS)
    identity_to_index = {}
    subject_to_indices = {subject_id: [] for subject_id in expected_subjects}
    per_subject_defects = {subject_id: Counter() for subject_id in expected_subjects}
    per_defect_subjects = {defect_id: Counter() for defect_id in DEFECT_IDS}

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DefectTrainingContractError('every defect dataset record must be a mapping.')
        subject_id = record.get('subject_id')
        defect_id = record.get('defect_id')
        if not isinstance(subject_id, str) or not subject_id or subject_id != subject_id.strip():
            raise DefectTrainingContractError('every defect record requires an exact subject_id.')
        if subject_id not in expected_subject_set:
            raise DefectTrainingContractError(
                f'defect dataset contains unknown patient {subject_id!r}.'
            )
        if not isinstance(defect_id, str) or not defect_id or defect_id != defect_id.strip():
            raise DefectTrainingContractError('every defect record requires an exact defect_id.')
        if defect_id not in expected_defect_set:
            raise DefectTrainingContractError(
                f'defect dataset contains unknown defect_id {defect_id!r}.'
            )

        identity = (subject_id, defect_id)
        if identity in identity_to_index:
            raise DefectTrainingContractError(
                f'defect dataset contains duplicate instance {identity!r}.'
            )
        identity_to_index[identity] = index
        subject_to_indices[subject_id].append(index)
        per_subject_defects[subject_id][defect_id] += 1
        per_defect_subjects[defect_id][subject_id] += 1

    actual_subjects = {subject_id for subject_id, _ in identity_to_index}
    if actual_subjects != expected_subject_set:
        missing = sorted(expected_subject_set.difference(actual_subjects))
        unexpected = sorted(actual_subjects.difference(expected_subject_set))
        raise DefectTrainingContractError(
            f'defect dataset ready patients mismatch; missing={missing}, unexpected={unexpected}.'
        )
    for subject_id in expected_subjects:
        expected_counts = Counter({defect_id: 1 for defect_id in DEFECT_IDS})
        if per_subject_defects[subject_id] != expected_counts:
            raise DefectTrainingContractError(
                f'patient {subject_id!r} must contain every formal defect exactly once.'
            )
        if len(subject_to_indices[subject_id]) != DEFECT_CONDITION_COUNT:
            raise DefectTrainingContractError(
                f'patient {subject_id!r} must map to exactly five dataset indices.'
            )
    for defect_id in DEFECT_IDS:
        expected_counts = Counter({subject_id: 1 for subject_id in expected_subjects})
        if per_defect_subjects[defect_id] != expected_counts:
            raise DefectTrainingContractError(
                f'defect condition {defect_id!r} must contain all 11 patients exactly once.'
            )
    if len(identity_to_index) != EXPECTED_DEFECT_INSTANCE_COUNT:
        raise DefectTrainingContractError(
            f'defect dataset must contain exactly {EXPECTED_DEFECT_INSTANCE_COUNT} instances.'
        )
    return identity_to_index, subject_to_indices


def validate_formal_defect_dataset(dataset, protocol):
    """Validate the exact 11-patient by 5-defect dataset contract."""
    _validated_identity_to_index(dataset, protocol)
    if len(dataset) != EXPECTED_DEFECT_INSTANCE_COUNT:
        raise DefectTrainingContractError(
            f'defect dataset len must be {EXPECTED_DEFECT_INSTANCE_COUNT}; got {len(dataset)}.'
        )
    return dataset


def build_formal_defect_split(dataset, protocol, fold_id) -> FormalDefectSplit:
    """Expand one frozen patient-level fold over a validated multi-record dataset."""
    identity_to_index, _ = _validated_identity_to_index(dataset, protocol)
    try:
        fold = resolve_fold(protocol, fold_id)
    except M3TrainingProtocolError as error:
        raise DefectTrainingContractError(f'cannot resolve formal defect fold: {error}') from error

    train_subjects = tuple(fold['train_subject_ids'])
    val_subjects = tuple(fold['val_subject_ids'])
    test_subjects = tuple(fold['test_subject_ids'])
    patient_partitions = tuple(map(set, (train_subjects, val_subjects, test_subjects)))
    if (
        patient_partitions[0] & patient_partitions[1]
        or patient_partitions[0] & patient_partitions[2]
        or patient_partitions[1] & patient_partitions[2]
    ):
        raise DefectTrainingContractError(f'{fold_id} contains patient-level leakage.')
    if set().union(*patient_partitions) != set(protocol['ready_subject_ids']):
        raise DefectTrainingContractError(f'{fold_id} patient union is not the 11 ready patients.')

    def expand(subject_ids):
        instance_ids = tuple(
            (subject_id, defect_id)
            for subject_id in subject_ids
            for defect_id in DEFECT_IDS
        )
        indices = tuple(identity_to_index[identity] for identity in instance_ids)
        return indices, instance_ids

    train_indices, train_instances = expand(train_subjects)
    val_indices, val_instances = expand(val_subjects)
    test_indices, test_instances = expand(test_subjects)
    instance_partitions = tuple(map(set, (train_instances, val_instances, test_instances)))
    if (
        instance_partitions[0] & instance_partitions[1]
        or instance_partitions[0] & instance_partitions[2]
        or instance_partitions[1] & instance_partitions[2]
    ):
        raise DefectTrainingContractError(f'{fold_id} contains instance-level leakage.')
    expected_instances = set(build_defect_variants(tuple(protocol['ready_subject_ids'])))
    if set().union(*instance_partitions) != expected_instances:
        raise DefectTrainingContractError(f'{fold_id} does not cover all 55 defect instances.')
    index_partitions = tuple(map(set, (train_indices, val_indices, test_indices)))
    if (
        index_partitions[0] & index_partitions[1]
        or index_partitions[0] & index_partitions[2]
        or index_partitions[1] & index_partitions[2]
        or set().union(*index_partitions) != set(range(EXPECTED_DEFECT_INSTANCE_COUNT))
    ):
        raise DefectTrainingContractError(f'{fold_id} dataset-index partition is not exact.')

    return FormalDefectSplit(
        fold_id=fold['fold_id'],
        train_subject_ids=train_subjects,
        val_subject_ids=val_subjects,
        test_subject_ids=test_subjects,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_instance_ids=train_instances,
        val_instance_ids=val_instances,
        test_instance_ids=test_instances,
    )


def build_split_provenance(split: FormalDefectSplit) -> dict:
    """Return JSON-safe patient/defect/fold provenance for logs and checkpoints."""
    if not isinstance(split, FormalDefectSplit):
        raise DefectTrainingContractError('split provenance requires a FormalDefectSplit.')

    def records(partition, instance_ids):
        return [
            {
                'subject_id': subject_id,
                'defect_id': defect_id,
                'fold_id': split.fold_id,
                'partition': partition,
            }
            for subject_id, defect_id in instance_ids
        ]

    return {
        'adapter': 'pointct_defect_training_v1',
        'fold_id': split.fold_id,
        'defect_ids': list(DEFECT_IDS),
        'train_instances': records('train', split.train_instance_ids),
        'val_instances': records('val', split.val_instance_ids),
        'test_instances': records('test', split.test_instance_ids),
    }


def expected_fold_instance_counts(protocol) -> dict:
    """Derive train/val/test instance counts from the frozen patient protocol."""
    _validate_protocol(protocol)
    counts = {}
    for fold_id in FOLD_SUBJECTS:
        fold = resolve_fold(protocol, fold_id)
        counts[fold_id] = tuple(
            len(fold[field]) * DEFECT_CONDITION_COUNT
            for field in ('train_subject_ids', 'val_subject_ids', 'test_subject_ids')
        )
    return counts


__all__ = [
    'DEFECT_CONDITION_COUNT',
    'DEFECT_IDS',
    'DefectTrainingContractError',
    'EXPECTED_DEFECT_INSTANCE_COUNT',
    'FormalDefectSplit',
    'READY_PATIENT_COUNT',
    'build_defect_variants',
    'build_formal_defect_split',
    'build_split_provenance',
    'create_formal_defect_dataset',
    'expected_fold_instance_counts',
    'validate_formal_defect_dataset',
]
