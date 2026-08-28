"""Formal M3 defect baseline evaluation CLI.

Real inference requires the explicit ``--execute-test`` mode.  The default CLI
does not silently evaluate either one Fold or all five Folds.
"""

import argparse
import json
from pathlib import Path
from typing import Mapping

import evaluate_m3 as complete_evaluation
from config import DEFAULT_DATA_ROOT
from defect_evaluation import (
    DefectEvaluationContractError,
    aggregate_defect_case_metrics,
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
    read_and_validate_defect_checkpoint,
    resolve_fold_artifacts,
    run_protocol_audit,
    validate_protocol_pair,
)
from defect_training import (
    build_formal_defect_split,
    create_formal_defect_dataset,
)
from training_protocol import FOLD_SUBJECTS, load_training_protocol


DEFAULT_EVALUATION_PROTOCOL = (
    Path(__file__).resolve().parent / 'protocols' / 'm3_defect_eval_v1.json'
)


def _selected_fold_ids(args) -> tuple:
    all_folds = bool(getattr(args, 'all_folds', False))
    fold_id = getattr(args, 'fold_id', None)
    if all_folds == (fold_id is not None):
        raise DefectEvaluationContractError(
            'choose exactly one Fold selector: --fold-id or --all-folds.'
        )
    if all_folds:
        return tuple(FOLD_SUBJECTS)
    if fold_id not in FOLD_SUBJECTS:
        raise DefectEvaluationContractError(
            f'unknown fold_id {fold_id!r}; expected one of {list(FOLD_SUBJECTS)}.'
        )
    return (fold_id,)


def _validate_raw_defect_identity(raw_sample, manifest_case) -> None:
    if not isinstance(raw_sample, Mapping):
        raise DefectEvaluationContractError('dataset sample must be a mapping.')
    for field in ('subject_id', 'defect_id'):
        if raw_sample.get(field) != manifest_case[field]:
            raise DefectEvaluationContractError(
                f'dataset sample {field} does not match the defect test manifest: '
                f'expected={manifest_case[field]!r}, actual={raw_sample.get(field)!r}.'
            )


def _evaluate_defect_test_case(
    raw_sample,
    manifest_case,
    checkpoint_path,
    training_protocol,
    evaluation_protocol,
    point_encoder,
    ct_encoder,
    matcher,
    device,
) -> dict:
    """Add defect identity around the unchanged formal M3 case evaluator."""
    _validate_raw_defect_identity(raw_sample, manifest_case)
    result = complete_evaluation._evaluate_test_case(
        raw_sample,
        manifest_case,
        checkpoint_path,
        training_protocol,
        evaluation_protocol,
        point_encoder,
        ct_encoder,
        matcher,
        device,
    )
    result.update(
        {
            'case_key': manifest_case['case_key'],
            'defect_id': manifest_case['defect_id'],
            'variant_index': manifest_case['variant_index'],
            'perturbation_identifier': manifest_case[
                'perturbation_identifier'
            ],
        }
    )
    return result


def _create_defect_dataset_for_execution(data_root, training_protocol):
    """Create the real 55-instance dataset only in explicit execution mode."""
    return create_formal_defect_dataset(data_root, training_protocol)


def run_manifest_only(
    training_protocol,
    evaluation_protocol,
    fold_ids,
    output_dir,
) -> dict:
    """Generate defect manifests without data, checkpoints, models, or GPU."""
    output_dir = Path(output_dir)
    audit = run_protocol_audit(training_protocol, evaluation_protocol)
    selected = {}
    for fold_id in fold_ids:
        manifest = build_defect_test_manifest(
            training_protocol,
            evaluation_protocol,
            fold_id,
        )
        manifest_path = output_dir / fold_id / 'test_manifest.json'
        complete_evaluation._write_json(manifest_path, manifest)
        selected[fold_id] = {
            'test_subject_ids': manifest['test_subject_ids'],
            'test_instances': manifest['total_test_instances'],
            'test_cases': manifest['total_cases'],
            'test_manifest_path': str(manifest_path.resolve()),
        }
    result = {
        'mode': 'manifest-only',
        'selected_folds': list(fold_ids),
        'folds': selected,
        'protocol_audit': audit,
    }
    complete_evaluation._write_json(output_dir / 'audit_summary.json', result)
    return result


def run_execute_fold(
    *,
    data_root,
    checkpoint_path,
    jsonl_path,
    output_dir,
    device_name: str,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> tuple:
    """Validate provenance first, then execute one formal defect Fold."""
    payload, checkpoint_metadata = read_and_validate_defect_checkpoint(
        checkpoint_path,
        jsonl_path,
        training_protocol,
        evaluation_protocol,
        fold_id,
    )

    import torch

    from train_m3 import resolve_device

    device = resolve_device(device_name)
    point_encoder, ct_encoder, matcher = (
        complete_evaluation._build_models_from_checkpoint(
            payload,
            checkpoint_metadata,
            device,
        )
    )
    dataset = _create_defect_dataset_for_execution(data_root, training_protocol)
    split = build_formal_defect_split(
        dataset,
        training_protocol,
        fold_id,
    )
    manifest = build_defect_test_manifest(
        training_protocol,
        evaluation_protocol,
        fold_id,
    )
    identity_to_index = dict(zip(split.test_instance_ids, split.test_indices))
    if len(identity_to_index) != len(split.test_instance_ids):
        raise DefectEvaluationContractError(
            f'{fold_id} test instance-to-index mapping is not unique.'
        )

    cases = []
    with torch.no_grad():
        for manifest_case in manifest['cases']:
            identity = (
                manifest_case['subject_id'],
                manifest_case['defect_id'],
            )
            try:
                dataset_index = identity_to_index[identity]
            except KeyError as error:
                raise DefectEvaluationContractError(
                    f'{fold_id} manifest instance is absent from the test split: '
                    f'{identity!r}.'
                ) from error
            raw_sample = dataset[dataset_index]
            cases.append(
                _evaluate_defect_test_case(
                    raw_sample,
                    manifest_case,
                    checkpoint_path,
                    training_protocol,
                    evaluation_protocol,
                    point_encoder,
                    ct_encoder,
                    matcher,
                    device,
                )
            )

    output_dir = Path(output_dir)
    summary = {
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol[
            'evaluation_protocol_hash'
        ],
        'fold_id': fold_id,
        'test_subject_ids': list(split.test_subject_ids),
        'test_defect_instance_count': len(split.test_instance_ids),
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        'checkpoint_epoch': checkpoint_metadata['epoch'],
        'checkpoint_best_val_loss': checkpoint_metadata['best_val_loss'],
        'jsonl_path': checkpoint_metadata['jsonl_path'],
        'jsonl_validation': checkpoint_metadata['jsonl_validation'],
        **aggregate_defect_case_metrics(cases),
    }
    complete_evaluation._write_json(
        output_dir / 'test_manifest.json',
        manifest,
    )
    complete_evaluation._write_jsonl(output_dir / 'cases.jsonl', cases)
    complete_evaluation._write_json(output_dir / 'summary.json', summary)
    return summary, cases


def run_execute_test(
    *,
    data_root,
    checkpoint_root,
    json_log_root,
    output_dir,
    device_name,
    training_protocol,
    evaluation_protocol,
    fold_ids,
) -> dict:
    """Execute one explicitly selected Fold or all explicitly selected Folds."""
    output_dir = Path(output_dir)
    all_cases = []
    per_fold = {}
    for fold_id in fold_ids:
        checkpoint_path, jsonl_path = resolve_fold_artifacts(
            checkpoint_root,
            fold_id,
            evaluation_protocol,
            json_log_root=json_log_root,
        )
        summary, cases = run_execute_fold(
            data_root=data_root,
            checkpoint_path=checkpoint_path,
            jsonl_path=jsonl_path,
            output_dir=output_dir / fold_id,
            device_name=device_name,
            training_protocol=training_protocol,
            evaluation_protocol=evaluation_protocol,
            fold_id=fold_id,
        )
        per_fold[fold_id] = summary
        all_cases.extend(cases)

    summary = {
        'mode': 'execute-test',
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol[
            'evaluation_protocol_hash'
        ],
        'selected_folds': list(fold_ids),
        'fold_summaries': per_fold,
        **aggregate_defect_case_metrics(all_cases),
    }
    complete_evaluation._write_jsonl(output_dir / 'cases.jsonl', all_cases)
    complete_evaluation._write_json(output_dir / 'summary.json', summary)
    return summary


def run_evaluation(args) -> dict:
    """Resolve an explicit safe mode and run the selected Fold set."""
    manifest_only = bool(getattr(args, 'manifest_only', False))
    execute_test = bool(getattr(args, 'execute_test', False))
    if manifest_only == execute_test:
        raise DefectEvaluationContractError(
            'explicitly choose exactly one mode: --manifest-only or '
            '--execute-test. Real evaluation is disabled by default.'
        )
    fold_ids = _selected_fold_ids(args)
    training_protocol = load_training_protocol(args.protocol_manifest)
    evaluation_protocol = load_defect_evaluation_protocol(
        args.evaluation_protocol
    )
    validate_protocol_pair(training_protocol, evaluation_protocol)
    if manifest_only:
        return run_manifest_only(
            training_protocol,
            evaluation_protocol,
            fold_ids,
            args.output_dir,
        )
    checkpoint_root = getattr(args, 'checkpoint_root', None)
    if checkpoint_root is None:
        raise DefectEvaluationContractError(
            '--execute-test requires --checkpoint-root.'
        )
    return run_execute_test(
        data_root=args.data_root,
        checkpoint_root=checkpoint_root,
        json_log_root=getattr(args, 'json_log_root', None),
        output_dir=args.output_dir,
        device_name=args.device,
        training_protocol=training_protocol,
        evaluation_protocol=evaluation_protocol,
        fold_ids=fold_ids,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Formal patient-level M3 defect baseline evaluation.'
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--protocol-manifest', type=Path, required=True)
    parser.add_argument(
        '--evaluation-protocol',
        type=Path,
        default=DEFAULT_EVALUATION_PROTOCOL,
    )
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument(
        '--json-log-root',
        type=Path,
        help='Defaults to checkpoint-root; expects Fold1.jsonl ... Fold5.jsonl.',
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    folds = parser.add_mutually_exclusive_group(required=True)
    folds.add_argument('--fold-id', choices=tuple(FOLD_SUBJECTS))
    folds.add_argument('--all-folds', action='store_true')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--manifest-only', action='store_true')
    modes.add_argument('--execute-test', action='store_true')
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    result = run_evaluation(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()


__all__ = [
    'DEFAULT_EVALUATION_PROTOCOL',
    'build_argument_parser',
    'main',
    'run_evaluation',
    'run_execute_fold',
    'run_execute_test',
    'run_manifest_only',
]
