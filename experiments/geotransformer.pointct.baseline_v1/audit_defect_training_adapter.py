"""CPU/static audit for versioned formal defect-training adapters."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defect_training import (
    DEFECT_CONDITION_COUNT,
    DEFECT_IDS,
    DefectTrainingContractError,
    build_defect_variants,
    build_formal_defect_split,
    create_formal_defect_dataset,
    expected_fold_instance_counts,
    get_defect_training_contract,
)
from geotransformer.datasets.registration.pointct.dataset import DEFECT_ARTIFACT_FILENAMES
from training_protocol import load_training_protocol


EXPECTED_FOLD_INSTANCE_COUNTS = {
    'Fold1': (30, 10, 15),
    'Fold2': (35, 10, 10),
    'Fold3': (35, 10, 10),
    'Fold4': (35, 10, 10),
    'Fold5': (30, 15, 10),
}
CLEAN10_EXPECTED_FOLD_INSTANCE_COUNTS = {
    fold_id: (30, 10, 10)
    for fold_id in ('Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5')
}
EXPECTED_FOLD_INSTANCE_COUNTS_BY_PROTOCOL = {
    'm3_6b_5fold_v1': EXPECTED_FOLD_INSTANCE_COUNTS,
    'm3_6b_5fold_clean10_v2': CLEAN10_EXPECTED_FOLD_INSTANCE_COUNTS,
}


class DefectTrainingAuditError(RuntimeError):
    pass


class _StaticDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)


def run_static_protocol_audit(protocol):
    defect_contract = get_defect_training_contract(protocol)
    variants = build_defect_variants(tuple(protocol['ready_subject_ids']))
    records = [
        {'subject_id': subject_id, 'defect_id': defect_id}
        for subject_id, defect_id in variants
    ]
    dataset = _StaticDataset(records)
    patient_counts = Counter(subject_id for subject_id, _ in variants)
    defect_counts = Counter(defect_id for _, defect_id in variants)
    fold_counts = expected_fold_instance_counts(protocol)
    expected_fold_counts = EXPECTED_FOLD_INSTANCE_COUNTS_BY_PROTOCOL[
        protocol['protocol_version']
    ]
    if fold_counts != expected_fold_counts:
        raise DefectTrainingAuditError(
            f'fold instance regression mismatch: {fold_counts!r}.'
        )

    validation_patient_counts = Counter()
    test_patient_counts = Counter()
    fold_splits = {}
    for fold_id in protocol['folds']:
        split = build_formal_defect_split(dataset, protocol, fold_id)
        validation_patient_counts.update(split.val_subject_ids)
        test_patient_counts.update(split.test_subject_ids)
        partition_sets = tuple(
            map(
                set,
                (
                    split.train_instance_ids,
                    split.val_instance_ids,
                    split.test_instance_ids,
                ),
            )
        )
        if (
            partition_sets[0] & partition_sets[1]
            or partition_sets[0] & partition_sets[2]
            or partition_sets[1] & partition_sets[2]
        ):
            raise DefectTrainingAuditError(f'{fold_id} has instance leakage.')
        fold_splits[fold_id] = split

    expected_once = Counter({subject_id: 1 for subject_id in protocol['ready_subject_ids']})
    if validation_patient_counts != expected_once or test_patient_counts != expected_once:
        raise DefectTrainingAuditError(
            'every patient must appear in validation once and test once across five folds.'
        )
    return {
        'protocol_version': protocol['protocol_version'],
        'ready_patient_count': len(patient_counts),
        'defect_condition_count': len(defect_counts),
        'expected_instance_count': len(variants),
        'pair_uniqueness_pass': len(set(variants)) == len(variants),
        'per_patient_pass': set(patient_counts.values()) == {DEFECT_CONDITION_COUNT},
        'per_condition_pass': set(defect_counts.values())
        == {defect_contract['ready_patient_count']},
        'patient_leakage_pass': True,
        'instance_leakage_pass': True,
        'fold_counts': fold_counts,
        'fold_splits': fold_splits,
    }


def _load_manifest(path):
    try:
        with path.open('r', encoding='utf-8') as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DefectTrainingAuditError(f'cannot read dataset manifest {path}: {error}') from error
    if not isinstance(value, dict) or not isinstance(value.get('subjects'), list):
        raise DefectTrainingAuditError('dataset manifest must contain a subjects list.')
    return value


def detect_complete_real_defect_data(data_root, protocol):
    """Detect all selected core artifact sets without reading medical arrays."""
    defect_contract = get_defect_training_contract(protocol)
    data_root = Path(data_root).expanduser().resolve()
    manifest_path = data_root / 'dataset_manifest.json'
    if not manifest_path.is_file():
        return False, f'dataset manifest is absent: {manifest_path}'
    manifest = _load_manifest(manifest_path)
    ready_by_subject = {}
    for record in manifest['subjects']:
        if not isinstance(record, dict):
            raise DefectTrainingAuditError('dataset manifest contains a non-object subject record.')
        if record.get('ready_for_baseline') is not True:
            continue
        subject_id = record.get('subject_id')
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise DefectTrainingAuditError('ready manifest record has an invalid subject_id.')
        ready_by_subject.setdefault(subject_id, []).append(record)

    expected_subjects = set(protocol['ready_subject_ids'])
    actual_subjects = set(ready_by_subject)
    allowed_source_only_subjects = set(defect_contract['excluded_subject_ids'])
    unexpected = sorted(
        actual_subjects.difference(expected_subjects).difference(
            allowed_source_only_subjects
        )
    )
    if unexpected:
        raise DefectTrainingAuditError(
            f'ready manifest contains forbidden/unknown patients: {unexpected}.'
        )
    missing_subjects = sorted(expected_subjects.difference(actual_subjects))
    if missing_subjects:
        return False, f'ready manifest is missing patients: {missing_subjects}'
    duplicate_subjects = sorted(
        subject_id for subject_id, records in ready_by_subject.items() if len(records) != 1
    )
    if duplicate_subjects:
        raise DefectTrainingAuditError(
            f'ready manifest must map each patient to one parent record: {duplicate_subjects}.'
        )

    missing_artifacts = []
    for subject_id, defect_id in build_defect_variants(tuple(protocol['ready_subject_ids'])):
        pair_metadata_path = ready_by_subject[subject_id][0].get('pair_metadata_path')
        if not isinstance(pair_metadata_path, str) or not pair_metadata_path.strip():
            raise DefectTrainingAuditError(
                f'patient {subject_id!r} has invalid pair_metadata_path.'
            )
        parent_metadata = (data_root / pair_metadata_path).resolve()
        try:
            parent_metadata.relative_to(data_root)
        except ValueError as error:
            raise DefectTrainingAuditError(
                f'patient {subject_id!r} pair_metadata_path escapes data root.'
            ) from error
        variant_dir = parent_metadata.parent / 'defects' / defect_id
        for filename in DEFECT_ARTIFACT_FILENAMES.values():
            artifact = variant_dir / filename
            if not artifact.is_file():
                missing_artifacts.append(artifact)
    if missing_artifacts:
        required_artifact_count = (
            defect_contract['expected_defect_instance_count']
            * len(DEFECT_ARTIFACT_FILENAMES)
        )
        return (
            False,
            f'{len(missing_artifacts)} of {required_artifact_count} required '
            'defect artifact files are absent',
        )
    return (
        True,
        f'all {defect_contract["expected_defect_instance_count"]} '
        'defect core artifact sets are present',
    )


def run_real_data_audit(data_root, protocol):
    defect_contract = get_defect_training_contract(protocol)
    dataset = create_formal_defect_dataset(data_root, protocol)
    if len(dataset) != defect_contract['expected_defect_instance_count']:
        raise DefectTrainingContractError(
            'real defect dataset does not contain exactly '
            f'{defect_contract["expected_defect_instance_count"]} records.'
        )
    splits = {
        fold_id: build_formal_defect_split(dataset, protocol, fold_id)
        for fold_id in protocol['folds']
    }
    return {'dataset': dataset, 'splits': splits}


def audit_defect_training_adapter(data_root, protocol_manifest):
    protocol = load_training_protocol(protocol_manifest)
    static = run_static_protocol_audit(protocol)
    complete, reason = detect_complete_real_defect_data(data_root, protocol)
    real = run_real_data_audit(data_root, protocol) if complete else None
    return {
        'static': static,
        'real_data_audit_run': real is not None,
        'real_data_reason': reason,
        'real': real,
    }


def _pass(value):
    return 'PASS' if value else 'FAIL'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Audit a versioned PointCT patient-level defect adapter on CPU.'
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol-manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_defect_training_adapter(args.data_root, args.protocol_manifest)
    static = result['static']
    print(f'READY_PATIENT_COUNT={static["ready_patient_count"]}')
    print(f'DEFECT_CONDITION_COUNT={static["defect_condition_count"]}')
    print(f'EXPECTED_INSTANCE_COUNT={static["expected_instance_count"]}')
    print(f'PAIR_UNIQUENESS={_pass(static["pair_uniqueness_pass"])}')
    print(f'PER_PATIENT_5_CONDITIONS={_pass(static["per_patient_pass"])}')
    if static['protocol_version'] == 'm3_6b_5fold_v1':
        print(f'PER_CONDITION_11_PATIENTS={_pass(static["per_condition_pass"])}')
    else:
        print(
            f'PER_CONDITION_{static["ready_patient_count"]}_PATIENTS='
            f'{_pass(static["per_condition_pass"])}'
        )
    print(f'PATIENT_LEVEL_LEAKAGE={_pass(static["patient_leakage_pass"])}')
    print(f'INSTANCE_LEVEL_LEAKAGE={_pass(static["instance_leakage_pass"])}')
    for fold_id, counts in static['fold_counts'].items():
        print(f'{fold_id.upper()}_TRAIN_VAL_TEST_INSTANCE_COUNT={counts[0]}/{counts[1]}/{counts[2]}')
    instance_count = static['expected_instance_count']
    print(
        f'REAL_{instance_count}_DATA_AUDIT_RUN='
        f'{str(result["real_data_audit_run"]).lower()}'
    )
    print(f'REAL_{instance_count}_DATA_AUDIT_REASON={result["real_data_reason"]}')


if __name__ == '__main__':
    main()
