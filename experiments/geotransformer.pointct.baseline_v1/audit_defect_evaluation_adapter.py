"""Protocol, dataset, and checkpoint audit for M3 defect evaluation."""

import argparse
import json
from pathlib import Path

from defect_evaluation import (
    load_defect_evaluation_protocol,
    read_and_validate_defect_checkpoint,
    resolve_fold_artifacts,
    run_protocol_audit,
)
from defect_training import (
    build_formal_defect_split,
    create_formal_defect_dataset,
)
from training_protocol import FOLD_SUBJECTS, load_training_protocol


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
)
DEFAULT_EVALUATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_v1.json'
)


def run_real_dataset_audit(data_root, training_protocol) -> dict:
    """Validate the real 55 records and every patient-level Fold expansion."""
    dataset = create_formal_defect_dataset(data_root, training_protocol)
    folds = {}
    for fold_id in FOLD_SUBJECTS:
        split = build_formal_defect_split(dataset, training_protocol, fold_id)
        folds[fold_id] = {
            'train_instances': len(split.train_instance_ids),
            'val_instances': len(split.val_instance_ids),
            'test_instances': len(split.test_instance_ids),
        }
    return {'dataset_instance_count': len(dataset), 'folds': folds}


def run_real_checkpoint_audit(
    checkpoint_root,
    json_log_root,
    training_protocol,
    evaluation_protocol,
) -> dict:
    """Validate all five checkpoint payloads and all five 20-row JSONL logs."""
    folds = {}
    for fold_id in FOLD_SUBJECTS:
        checkpoint_path, jsonl_path = resolve_fold_artifacts(
            checkpoint_root,
            fold_id,
            evaluation_protocol,
            json_log_root=json_log_root,
        )
        _, metadata = read_and_validate_defect_checkpoint(
            checkpoint_path,
            jsonl_path,
            training_protocol,
            evaluation_protocol,
            fold_id,
        )
        folds[fold_id] = {
            'checkpoint_path': str(checkpoint_path.resolve()),
            'jsonl_path': str(jsonl_path.resolve()),
            'checkpoint_epoch': metadata['epoch'],
            'checkpoint_best_val_loss': metadata['best_val_loss'],
            **metadata['jsonl_validation'],
        }
    return {'folds': folds}


def run_audit(
    training_protocol,
    evaluation_protocol,
    *,
    data_root=None,
    checkpoint_root=None,
    json_log_root=None,
) -> dict:
    """Always run structural audit; run real audits only when paths are explicit."""
    protocol_audit = run_protocol_audit(
        training_protocol,
        evaluation_protocol,
    )
    result = {
        'protocol_audit_run': True,
        'real_55_data_audit_run': data_root is not None,
        'real_checkpoint_audit_run': checkpoint_root is not None,
        'protocol_audit': protocol_audit,
        'real_data_audit': None,
        'real_checkpoint_audit': None,
    }
    if data_root is not None:
        result['real_data_audit'] = run_real_dataset_audit(
            data_root,
            training_protocol,
        )
    if checkpoint_root is not None:
        result['real_checkpoint_audit'] = run_real_checkpoint_audit(
            checkpoint_root,
            json_log_root,
            training_protocol,
            evaluation_protocol,
        )
    return result


def _bool(value) -> str:
    return 'true' if value else 'false'


def print_audit_report(result) -> None:
    audit = result['protocol_audit']
    print(f'PROTOCOL_AUDIT_RUN={_bool(result["protocol_audit_run"])}')
    print(
        'REAL_55_DATA_AUDIT_RUN='
        f'{_bool(result["real_55_data_audit_run"])}'
    )
    print(
        'REAL_CHECKPOINT_AUDIT_RUN='
        f'{_bool(result["real_checkpoint_audit_run"])}'
    )
    print(f'PROTOCOL_VERSION={audit["protocol_version"]}')
    print(f'PROTOCOL_HASH={audit["protocol_hash"]}')
    print(f'READY_PATIENT_COUNT={audit["ready_patient_count"]}')
    print(f'DEFECT_CONDITION_COUNT={audit["defect_condition_count"]}')
    print(
        'EXPECTED_TEST_INSTANCE_COUNT='
        f'{audit["expected_test_instance_count"]}'
    )
    print(f'EXPECTED_TEST_CASE_COUNT={audit["expected_test_case_count"]}')
    for fold_id in FOLD_SUBJECTS:
        fold = audit['folds'][fold_id]
        print(f'{fold_id}_TEST_INSTANCES={fold["test_instance_count"]}')
        print(f'{fold_id}_TEST_CASES={fold["test_case_count"]}')
    print('PATIENT_LEVEL_LEAKAGE=PASS')
    print('INSTANCE_LEVEL_LEAKAGE=PASS')
    print('PAIR_UNIQUENESS=PASS')
    print('PERTURBATION_COUNT=PASS')
    print('FROZEN_TEST_PERTURBATION=PASS')


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Audit the M3 defect evaluation adapter.'
    )
    parser.add_argument(
        '--protocol-manifest',
        type=Path,
        default=DEFAULT_TRAINING_PROTOCOL,
    )
    parser.add_argument(
        '--evaluation-protocol',
        type=Path,
        default=DEFAULT_EVALUATION_PROTOCOL,
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        help='Optional explicit real 55-instance dataset root.',
    )
    parser.add_argument(
        '--checkpoint-root',
        type=Path,
        help='Optional explicit root containing FoldX/best_val_loss.pt.',
    )
    parser.add_argument(
        '--json-log-root',
        type=Path,
        help='Optional root containing FoldX.jsonl; defaults to checkpoint root.',
    )
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    training_protocol = load_training_protocol(args.protocol_manifest)
    evaluation_protocol = load_defect_evaluation_protocol(
        args.evaluation_protocol
    )
    result = run_audit(
        training_protocol,
        evaluation_protocol,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        json_log_root=args.json_log_root,
    )
    print_audit_report(result)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()


__all__ = [
    'DEFAULT_EVALUATION_PROTOCOL',
    'DEFAULT_TRAINING_PROTOCOL',
    'build_argument_parser',
    'main',
    'print_audit_report',
    'run_audit',
    'run_real_checkpoint_audit',
    'run_real_dataset_audit',
]
