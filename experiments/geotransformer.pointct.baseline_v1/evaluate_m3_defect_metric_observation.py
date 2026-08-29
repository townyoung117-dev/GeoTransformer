"""M3 clean10 v3 TRE observation and CPU-only mixed aggregation CLI.

Only ``--execute-observation`` loads models or resolves a device.  The
``--aggregate-750`` path reads existing JSONL artifacts and never performs
inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import evaluate_m3 as complete_evaluation
import evaluate_m3_defect_metric_diagnostic as diagnostic_cli
from config import DEFAULT_DATA_ROOT
from defect_evaluation import (
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
    resolve_fold_artifacts,
    run_protocol_audit,
)
from m3_metric_diagnostic import (
    CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION,
    load_diagnostic_protocol,
    load_jsonl,
    validate_case_identity_set,
    validate_legacy_result_tree,
)
from m3_metric_observation import (
    EXPECTED_CASE_COUNTS_BY_FOLD,
    EXPECTED_DEFECT_CONDITION_COUNT,
    EXPECTED_DEFECT_INSTANCE_COUNT,
    EXPECTED_PAT6_CASE_COUNT,
    EXPECTED_PATIENT_COUNT,
    EXPECTED_TEST_CASE_COUNT,
    M3MetricObservationContractError,
    aggregate_mixed_clean10_750,
    aggregate_observation_cases,
    build_observation_case,
    load_observation_protocol,
    validate_source_protocols,
)
from training_protocol import load_training_protocol


EXPERIMENT_DIR = Path(__file__).resolve().parent
CLEAN10_TRAINING_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_clean10_v2.json'
)
CLEAN10_EVALUATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_clean10_v2.json'
)
CLEAN10_DIAGNOSTIC_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_diagnostic_clean10_v2.json'
)
OBSERVATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_observation_clean10_v3.json'
)
DEFAULT_OUTPUT_ROOT_NAME = 'defect_m3_clean10_metric_observation_v3'
DEFAULT_AGGREGATE_OUTPUT_ROOT_NAME = (
    'defect_m3_clean10_metric_observation_750_v3'
)


def _load_protocol_bundle(args) -> tuple:
    training = load_training_protocol(args.protocol_manifest)
    evaluation = load_defect_evaluation_protocol(args.evaluation_protocol)
    observation = load_observation_protocol(args.observation_protocol)
    validate_source_protocols(observation, training, evaluation)
    return training, evaluation, observation


def _all_manifests(training, evaluation) -> dict:
    return {
        fold_id: build_defect_test_manifest(training, evaluation, fold_id)
        for fold_id in training['folds']
    }


def _validate_manifest_contract(
    training: Mapping,
    evaluation: Mapping,
    manifests: Mapping[str, Mapping],
) -> dict:
    """Require the exact dynamic 10/50/750 clean10 manifest contract."""
    audit = run_protocol_audit(training, evaluation)
    folds = tuple(training['folds'])
    if tuple(manifests) != tuple(EXPECTED_CASE_COUNTS_BY_FOLD):
        raise M3MetricObservationContractError(
            'observation manifests must contain ordered Fold1-Fold5.'
        )
    all_cases = []
    for fold_id in folds:
        rows = list(manifests[fold_id]['cases'])
        expected_count = EXPECTED_CASE_COUNTS_BY_FOLD[fold_id]
        if len(rows) != expected_count or manifests[fold_id].get(
            'total_cases'
        ) != expected_count:
            raise M3MetricObservationContractError(
                f'{fold_id} manifest case count mismatch: expected=150, '
                f'actual={len(rows)}.'
            )
        all_cases.extend(rows)
    patients = {row['subject_id'] for row in all_cases}
    defects = {row['defect_id'] for row in all_cases}
    instances = {(row['subject_id'], row['defect_id']) for row in all_cases}
    identities = {
        (
            row['fold_id'],
            row['subject_id'],
            row['defect_id'],
            row['severity'],
            row['variant_id'],
            row['perturbation_seed'],
        )
        for row in all_cases
    }
    pat6_count = sum(row['subject_id'] == 'Pat6' for row in all_cases)
    comparisons = (
        (len(patients), EXPECTED_PATIENT_COUNT, 'patient count'),
        (
            len(defects),
            EXPECTED_DEFECT_CONDITION_COUNT,
            'defect condition count',
        ),
        (
            len(instances),
            EXPECTED_DEFECT_INSTANCE_COUNT,
            'defect instance count',
        ),
        (len(all_cases), EXPECTED_TEST_CASE_COUNT, 'test case count'),
        (len(identities), EXPECTED_TEST_CASE_COUNT, 'unique identity count'),
        (pat6_count, EXPECTED_PAT6_CASE_COUNT, 'Pat6 case count'),
        (
            audit['ready_patient_count'],
            EXPECTED_PATIENT_COUNT,
            'source audit patient count',
        ),
        (
            audit['expected_test_instance_count'],
            EXPECTED_DEFECT_INSTANCE_COUNT,
            'source audit instance count',
        ),
        (
            audit['expected_test_case_count'],
            EXPECTED_TEST_CASE_COUNT,
            'source audit case count',
        ),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise M3MetricObservationContractError(
                f'{label} mismatch: expected={expected}, actual={actual}.'
            )
    return {
        'patient_count': len(patients),
        'defect_condition_count': len(defects),
        'defect_instance_count': len(instances),
        'test_case_count': len(all_cases),
        'identity_unique_count': len(identities),
        'pat6_case_count': pat6_count,
        'per_fold_case_counts': dict(EXPECTED_CASE_COUNTS_BY_FOLD),
    }


def _selected_fold_ids(args, training) -> tuple:
    folds = tuple(training['folds'])
    all_folds = bool(getattr(args, 'all_folds', False))
    fold_id = getattr(args, 'fold_id', None)
    if all_folds == (fold_id is not None):
        raise M3MetricObservationContractError(
            'choose exactly one Fold selector: --fold-id or --all-folds.'
        )
    if all_folds:
        return folds
    if fold_id not in folds:
        raise M3MetricObservationContractError(
            f'unknown fold_id {fold_id!r}; expected one of {list(folds)}.'
        )
    return (fold_id,)


def _require_empty_output_root(output_root, *input_roots) -> Path:
    output = Path(output_root).resolve()
    for input_root in input_roots:
        if input_root is not None and output == Path(input_root).resolve():
            raise M3MetricObservationContractError(
                'output root must not equal an input artifact root.'
            )
    if output.exists():
        if not output.is_dir():
            raise M3MetricObservationContractError(
                f'output root exists and is not a directory: {output}.'
            )
        try:
            nonempty = next(output.iterdir(), None) is not None
        except OSError as error:
            raise M3MetricObservationContractError(
                f'cannot inspect output root {output}: {error}'
            ) from error
        if nonempty:
            raise M3MetricObservationContractError(
                f'output root already exists and is non-empty: {output}. '
                'Automatic overwrite is forbidden.'
            )
    return output


def _diagnostic_adapter_protocol(observation: Mapping) -> dict:
    """Adapt only the field names consumed by the frozen replay helper."""
    return {
        'diagnostic_protocol_version': observation[
            'observation_protocol_version'
        ],
        'diagnostic_protocol_hash': observation['observation_protocol_hash'],
        'source_reference_points': observation['source_reference_points'],
    }


def _stamp_observation_replay_rows(
    rows: Sequence[Mapping],
    observation: Mapping,
) -> list:
    stamped = []
    for source in rows:
        row = dict(source)
        row['observation_protocol_version'] = row.pop(
            'diagnostic_protocol_version'
        )
        row['observation_protocol_hash'] = row.pop(
            'diagnostic_protocol_hash'
        )
        stamped.append(row)
    return stamped


def _reproducibility_report(rows: Sequence[Mapping], summary: Mapping) -> dict:
    return {
        'case_count': summary['case_count'],
        'reproducibility_pass_count': summary['reproducibility_pass_count'],
        'reproducibility_fail_count': summary['reproducibility_fail_count'],
        'reproducibility_pass_rate': summary['reproducibility_pass_rate'],
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
            for row in rows
            if row['replay_reproducibility']['status'] == 'FAIL'
        ],
    }


def run_execute_observation(
    *,
    data_root,
    checkpoint_root,
    json_log_root,
    output_root,
    formal_results,
    device_name,
    fold_ids,
    training_protocol,
    evaluation_protocol,
    observation_protocol,
    manifests,
) -> dict:
    """Execute replay and record mismatches without discarding valid TRE."""
    all_rows = []
    fold_summaries = {}
    adapter = _diagnostic_adapter_protocol(observation_protocol)
    for fold_id in fold_ids:
        checkpoint_path, jsonl_path = resolve_fold_artifacts(
            checkpoint_root,
            fold_id,
            evaluation_protocol,
            json_log_root=json_log_root,
        )
        replay_rows, checkpoint_metadata = diagnostic_cli._execute_fold(
            data_root=data_root,
            checkpoint_path=checkpoint_path,
            jsonl_path=jsonl_path,
            device_name=device_name,
            training_protocol=training_protocol,
            evaluation_protocol=evaluation_protocol,
            diagnostic_protocol=adapter,
            manifest=manifests[fold_id],
        )
        replay_rows = _stamp_observation_replay_rows(
            replay_rows,
            observation_protocol,
        )
        formal_rows = formal_results['per_fold'][fold_id]
        validate_case_identity_set(
            replay_rows,
            formal_rows,
            expected_count=EXPECTED_CASE_COUNTS_BY_FOLD[fold_id],
            label=f'observation replay {fold_id}',
        )
        formal_by_identity = {
            (
                row['fold_id'],
                row['subject_id'],
                row['defect_id'],
                row['severity'],
                row['variant_id'],
                row['perturbation_seed'],
            ): row
            for row in formal_rows
        }
        observation_rows = []
        for replay in replay_rows:
            identity = (
                replay['fold_id'],
                replay['subject_id'],
                replay['defect_id'],
                replay['severity'],
                replay['variant_id'],
                replay['perturbation_seed'],
            )
            observation_rows.append(
                build_observation_case(
                    formal_by_identity[identity],
                    replay,
                    observation_protocol,
                )
            )
        expected_count = EXPECTED_CASE_COUNTS_BY_FOLD[fold_id]
        summary = aggregate_observation_cases(
            observation_rows,
            observation_protocol,
            expected_case_count=expected_count,
        )
        summary['fold_id'] = fold_id
        report = _reproducibility_report(observation_rows, summary)
        manifest = {
            'observation_protocol_version': observation_protocol[
                'observation_protocol_version'
            ],
            'observation_protocol_hash': observation_protocol[
                'observation_protocol_hash'
            ],
            'fold_id': fold_id,
            'case_count': len(observation_rows),
            'source_test_manifest': manifests[fold_id],
            'checkpoint_path': str(Path(checkpoint_path).resolve()),
            'checkpoint_epoch': checkpoint_metadata['epoch'],
            'checkpoint_best_val_loss': checkpoint_metadata['best_val_loss'],
            'formal_metric_policy': observation_protocol[
                'formal_metric_policy'
            ],
            'replay_policy': observation_protocol['replay_policy'],
        }
        fold_dir = Path(output_root) / fold_id
        complete_evaluation._write_jsonl(
            fold_dir / 'observation_cases.jsonl',
            observation_rows,
        )
        complete_evaluation._write_json(
            fold_dir / 'observation_summary.json',
            summary,
        )
        complete_evaluation._write_json(
            fold_dir / 'observation_manifest.json',
            manifest,
        )
        complete_evaluation._write_json(
            fold_dir / 'reproducibility_report.json',
            report,
        )
        all_rows.extend(observation_rows)
        fold_summaries[fold_id] = summary
    root_summary = aggregate_observation_cases(
        all_rows,
        observation_protocol,
        expected_case_count=sum(
            EXPECTED_CASE_COUNTS_BY_FOLD[fold_id] for fold_id in fold_ids
        ),
    )
    root_summary.update(
        {
            'mode': 'execute-observation',
            'selected_folds': list(fold_ids),
            'fold_summaries': fold_summaries,
            'model_loaded': True,
            'gpu_used': str(device_name).lower().startswith('cuda'),
        }
    )
    complete_evaluation._write_jsonl(
        Path(output_root) / 'observation_cases.jsonl',
        all_rows,
    )
    complete_evaluation._write_json(
        Path(output_root) / 'observation_summary.json',
        root_summary,
    )
    complete_evaluation._write_json(
        Path(output_root) / 'reproducibility_report.json',
        _reproducibility_report(all_rows, root_summary),
    )
    return root_summary


def _artifact_path(root, fold_id: str, filename: str) -> Path:
    root = Path(root)
    direct = root / filename
    if direct.is_file() and root.name == fold_id:
        return direct
    return root / fold_id / filename


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_sha256sums(output_root: Path, filenames: Sequence[str]) -> None:
    lines = [
        f'{_sha256_file(output_root / filename)}  {filename}'
        for filename in filenames
    ]
    path = output_root / 'SHA256SUMS'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def run_cpu_aggregate_750(
    *,
    v2_results_root,
    v3_results_root,
    output_root,
    training_protocol,
    evaluation_protocol,
    observation_protocol,
    manifests,
) -> dict:
    """Read 600 v2 + 150 v3 cases and write a CPU-only 750-case bundle."""
    validate_source_protocols(
        observation_protocol,
        training_protocol,
        evaluation_protocol,
    )
    _validate_manifest_contract(
        training_protocol,
        evaluation_protocol,
        manifests,
    )
    v2_cases = {
        fold_id: load_jsonl(
            _artifact_path(
                v2_results_root,
                fold_id,
                'diagnostic_cases.jsonl',
            )
        )
        for fold_id in ('Fold1', 'Fold2', 'Fold3', 'Fold4')
    }
    v3_cases = load_jsonl(
        _artifact_path(
            v3_results_root,
            'Fold5',
            'observation_cases.jsonl',
        )
    )
    result = aggregate_mixed_clean10_750(
        v2_cases,
        v3_cases,
        observation_protocol,
        expected_cases_by_fold={
            fold_id: manifests[fold_id]['cases']
            for fold_id in EXPECTED_CASE_COUNTS_BY_FOLD
        },
    )
    output = Path(output_root)
    filenames = (
        'cases_750.jsonl',
        'summary_750.json',
        'reproducibility_summary.json',
        'provenance.json',
    )
    complete_evaluation._write_jsonl(output / filenames[0], result['cases'])
    complete_evaluation._write_json(output / filenames[1], result['summary'])
    complete_evaluation._write_json(
        output / filenames[2],
        result['reproducibility_summary'],
    )
    provenance = dict(result['provenance'])
    provenance['inputs'] = {
        'v2_results_root': str(Path(v2_results_root).resolve()),
        'v3_results_root': str(Path(v3_results_root).resolve()),
    }
    complete_evaluation._write_json(output / filenames[3], provenance)
    _write_sha256sums(output, filenames)
    return {
        **result['summary'],
        'mode': 'aggregate-750',
        'output_root': str(output.resolve()),
        'model_loaded': False,
        'gpu_used': False,
    }


def run_evaluation(args) -> dict:
    execute = bool(getattr(args, 'execute_observation', False))
    aggregate = bool(getattr(args, 'aggregate_750', False))
    if execute == aggregate:
        raise M3MetricObservationContractError(
            'choose exactly one mode: --execute-observation or --aggregate-750.'
        )
    training, evaluation, observation = _load_protocol_bundle(args)
    manifests = _all_manifests(training, evaluation)
    _validate_manifest_contract(training, evaluation, manifests)
    if aggregate:
        v2_root = getattr(args, 'v2_results_root', None)
        v3_root = getattr(args, 'v3_results_root', None)
        if v2_root is None or v3_root is None:
            raise M3MetricObservationContractError(
                '--aggregate-750 requires --v2-results-root and '
                '--v3-results-root.'
            )
        output = _require_empty_output_root(
            args.output_root,
            v2_root,
            v3_root,
        )
        return run_cpu_aggregate_750(
            v2_results_root=v2_root,
            v3_results_root=v3_root,
            output_root=output,
            training_protocol=training,
            evaluation_protocol=evaluation,
            observation_protocol=observation,
            manifests=manifests,
        )
    formal_root = getattr(args, 'formal_results_root', None)
    checkpoint_root = getattr(args, 'checkpoint_root', None)
    if formal_root is None or checkpoint_root is None:
        raise M3MetricObservationContractError(
            '--execute-observation requires --formal-results-root and '
            '--checkpoint-root.'
        )
    output = _require_empty_output_root(args.output_root, formal_root)
    fold_ids = _selected_fold_ids(args, training)
    clean10_diagnostic = load_diagnostic_protocol(CLEAN10_DIAGNOSTIC_PROTOCOL)
    if clean10_diagnostic['diagnostic_protocol_version'] != (
        CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION
    ):
        raise M3MetricObservationContractError(
            'clean10 v2 diagnostic protocol binding is invalid.'
        )
    formal_results = validate_legacy_result_tree(
        formal_root,
        {
            fold_id: manifests[fold_id]['cases']
            for fold_id in training['folds']
        },
        diagnostic_protocol=clean10_diagnostic,
    )
    return run_execute_observation(
        data_root=args.data_root,
        checkpoint_root=checkpoint_root,
        json_log_root=getattr(args, 'json_log_root', None),
        output_root=output,
        formal_results=formal_results,
        device_name=args.device,
        fold_ids=fold_ids,
        training_protocol=training,
        evaluation_protocol=evaluation,
        observation_protocol=observation,
        manifests=manifests,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Observe M3 clean10 TRE independently from strict replay '
            'reproducibility, or aggregate 600 v2 + 150 v3 cases on CPU.'
        )
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        '--protocol-manifest',
        type=Path,
        default=CLEAN10_TRAINING_PROTOCOL,
    )
    parser.add_argument(
        '--evaluation-protocol',
        type=Path,
        default=CLEAN10_EVALUATION_PROTOCOL,
    )
    parser.add_argument(
        '--observation-protocol',
        type=Path,
        default=OBSERVATION_PROTOCOL,
    )
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--json-log-root', type=Path)
    parser.add_argument('--formal-results-root', type=Path)
    parser.add_argument('--v2-results-root', type=Path)
    parser.add_argument('--v3-results-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--device',
        default='cpu',
        help='Used only by --execute-observation.',
    )
    folds = parser.add_mutually_exclusive_group()
    folds.add_argument('--fold-id', choices=tuple(EXPECTED_CASE_COUNTS_BY_FOLD))
    folds.add_argument('--all-folds', action='store_true')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--execute-observation', action='store_true')
    modes.add_argument('--aggregate-750', action='store_true')
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    result = run_evaluation(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()


__all__ = [
    'CLEAN10_DIAGNOSTIC_PROTOCOL',
    'CLEAN10_EVALUATION_PROTOCOL',
    'CLEAN10_TRAINING_PROTOCOL',
    'DEFAULT_AGGREGATE_OUTPUT_ROOT_NAME',
    'DEFAULT_OUTPUT_ROOT_NAME',
    'OBSERVATION_PROTOCOL',
    'build_argument_parser',
    'main',
    'run_cpu_aggregate_750',
    'run_evaluation',
    'run_execute_observation',
]
