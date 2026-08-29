"""Independent CLI for origin-invariant M3 defect registration diagnostics.

Only ``--execute-diagnostic`` performs model inference.  ``--audit-only`` is a
CPU path for protocol, manifest, legacy-result, identity, and optional
checkpoint-provenance validation.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import evaluate_m3 as complete_evaluation
import evaluate_m3_defect as defect_cli
from config import DEFAULT_DATA_ROOT
from defect_evaluation import (
    DefectEvaluationContractError,
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
    read_and_validate_defect_checkpoint,
    resolve_fold_artifacts,
    run_protocol_audit,
    validate_protocol_pair,
)
from defect_training import build_formal_defect_split, create_formal_defect_dataset
from evaluation import correspondence_inlier_metrics, evaluate_registration_result
from m3_metric_diagnostic import (
    M3MetricDiagnosticContractError,
    aggregate_diagnostic_cases,
    compute_tre_diagnostics,
    cross_check_legacy_cases,
    get_diagnostic_protocol_contract,
    load_diagnostic_protocol,
    make_rigid_transform,
    rotation_error_degrees,
    summarize_confidence,
    validate_legacy_result_tree,
    validate_source_protocols,
)
from perturbation import augment_point_sample
from training_protocol import FOLD_SUBJECTS, load_training_protocol


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
)
DEFAULT_EVALUATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_v1.json'
)
DEFAULT_DIAGNOSTIC_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_diagnostic_v1.json'
)
CLEAN10_TRAINING_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_clean10_v2.json'
)
CLEAN10_EVALUATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_clean10_v2.json'
)
CLEAN10_DIAGNOSTIC_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_diagnostic_clean10_v2.json'
)
DEFAULT_OUTPUT_ROOT_NAME = 'defect_m3_metric_diagnostic_v1'
FROZEN_FORMAL_OUTPUT_ROOT_NAME = 'defect_m3_eval_formal'
CLEAN10_FROZEN_FORMAL_OUTPUT_ROOT_NAME = 'defect_m3_clean10_eval_formal'
DIAGNOSTIC_RRE_ABS_TOLERANCE = 1e-10


def _selected_fold_ids(args, training_protocol=None) -> tuple:
    fold_ids = tuple(
        FOLD_SUBJECTS
        if training_protocol is None
        else training_protocol['folds']
    )
    all_folds = bool(getattr(args, 'all_folds', False))
    fold_id = getattr(args, 'fold_id', None)
    if all_folds == (fold_id is not None):
        raise M3MetricDiagnosticContractError(
            'choose exactly one Fold selector: --fold-id or --all-folds.'
        )
    if all_folds:
        return fold_ids
    if fold_id not in fold_ids:
        raise M3MetricDiagnosticContractError(
            f'unknown fold_id {fold_id!r}; expected one of {list(fold_ids)}.'
        )
    return (fold_id,)


def _explicit_mode(args) -> str:
    audit_only = bool(getattr(args, 'audit_only', False))
    execute = bool(getattr(args, 'execute_diagnostic', False))
    if audit_only == execute:
        raise M3MetricDiagnosticContractError(
            'explicitly choose exactly one mode: --audit-only or '
            '--execute-diagnostic. GPU inference is disabled by default.'
        )
    return 'audit-only' if audit_only else 'execute-diagnostic'


def _require_safe_empty_output_root(output_root, legacy_results_root) -> Path:
    output = Path(output_root)
    legacy = Path(legacy_results_root)
    try:
        output_resolved = output.resolve()
        legacy_resolved = legacy.resolve()
    except OSError as error:
        raise M3MetricDiagnosticContractError(
            f'cannot resolve output/legacy roots: {error}'
        ) from error
    if output_resolved == legacy_resolved:
        raise M3MetricDiagnosticContractError(
            'diagnostic output root must not equal the frozen legacy result root.'
        )
    frozen_names = (
        FROZEN_FORMAL_OUTPUT_ROOT_NAME,
        CLEAN10_FROZEN_FORMAL_OUTPUT_ROOT_NAME,
    )
    if output_resolved.name in frozen_names:
        raise M3MetricDiagnosticContractError(
            f'refusing to write diagnostic output into '
            f'{output_resolved.name!r}.'
        )
    if output.exists():
        if not output.is_dir():
            raise M3MetricDiagnosticContractError(
                f'output root exists and is not a directory: {output}.'
            )
        try:
            nonempty = next(output.iterdir(), None) is not None
        except OSError as error:
            raise M3MetricDiagnosticContractError(
                f'cannot inspect output root {output}: {error}'
            ) from error
        if nonempty:
            raise M3MetricDiagnosticContractError(
                f'output root already exists and is non-empty: {output}. '
                'Automatic overwrite is forbidden.'
            )
    return output


def _load_protocol_bundle(args) -> tuple:
    training_protocol = load_training_protocol(args.protocol_manifest)
    evaluation_protocol = load_defect_evaluation_protocol(
        args.evaluation_protocol
    )
    diagnostic_protocol = load_diagnostic_protocol(args.diagnostic_protocol)
    try:
        validate_protocol_pair(training_protocol, evaluation_protocol)
    except DefectEvaluationContractError as error:
        raise M3MetricDiagnosticContractError(
            f'frozen source protocol validation failed: {error}'
        ) from error
    validate_source_protocols(
        diagnostic_protocol,
        training_protocol,
        evaluation_protocol,
    )
    return training_protocol, evaluation_protocol, diagnostic_protocol


def _all_manifests(training_protocol, evaluation_protocol) -> Mapping[str, Mapping]:
    return {
        fold_id: build_defect_test_manifest(
            training_protocol,
            evaluation_protocol,
            fold_id,
        )
        for fold_id in training_protocol['folds']
    }


def _validate_manifest_contract(
    training_protocol,
    diagnostic_protocol,
    manifests: Mapping[str, Mapping],
    protocol_audit: Mapping,
) -> dict:
    """Cross-check dynamic manifests against the selected diagnostic contract."""
    contract = get_diagnostic_protocol_contract(diagnostic_protocol)
    fold_ids = tuple(training_protocol['folds'])
    if tuple(manifests) != fold_ids:
        raise M3MetricDiagnosticContractError(
            'diagnostic manifest folds do not match the training protocol.'
        )
    all_cases = [
        case
        for fold_id in fold_ids
        for case in manifests[fold_id]['cases']
    ]
    patient_ids = {case['subject_id'] for case in all_cases}
    defect_ids = {case['defect_id'] for case in all_cases}
    instance_ids = {
        (case['subject_id'], case['defect_id']) for case in all_cases
    }
    pat6_case_count = sum(
        case['subject_id'] == 'Pat6' for case in all_cases
    )
    comparisons = (
        (
            len(patient_ids),
            contract['expected_patient_count'],
            'patient count',
        ),
        (
            len(defect_ids),
            contract['expected_defect_condition_count'],
            'defect condition count',
        ),
        (
            len(instance_ids),
            contract['expected_defect_instance_count'],
            'defect instance count',
        ),
        (
            len(all_cases),
            contract['expected_test_case_count'],
            'test case count',
        ),
        (
            protocol_audit['ready_patient_count'],
            contract['expected_patient_count'],
            'source audit patient count',
        ),
        (
            protocol_audit['expected_test_instance_count'],
            contract['expected_defect_instance_count'],
            'source audit defect instance count',
        ),
        (
            protocol_audit['expected_test_case_count'],
            contract['expected_test_case_count'],
            'source audit test case count',
        ),
        (
            pat6_case_count,
            contract['expected_pat6_case_count'],
            'Pat6 case count',
        ),
    )
    for actual, expected, name in comparisons:
        if actual != expected:
            raise M3MetricDiagnosticContractError(
                f'{name} mismatch: expected={expected}, actual={actual}.'
            )
    return {
        'patient_count': len(patient_ids),
        'defect_condition_count': len(defect_ids),
        'defect_instance_count': len(instance_ids),
        'test_case_count': len(all_cases),
        'pat6_case_count': pat6_case_count,
        'per_fold_case_counts': {
            fold_id: manifests[fold_id]['total_cases']
            for fold_id in fold_ids
        },
    }


def _validate_checkpoint_roots_cpu(
    checkpoint_root,
    json_log_root,
    fold_ids: Sequence[str],
    training_protocol,
    evaluation_protocol,
) -> dict:
    """Load and validate checkpoint metadata on CPU without creating models."""
    results = {}
    for fold_id in fold_ids:
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
        results[fold_id] = {
            'checkpoint_path': str(checkpoint_path.resolve()),
            'checkpoint_epoch': metadata['epoch'],
            'checkpoint_best_val_loss': metadata['best_val_loss'],
            'jsonl_path': metadata['jsonl_path'],
            'jsonl_validation': metadata['jsonl_validation'],
            'cpu_metadata_validation_pass': True,
        }
    return results


def _expected_pat6_count(manifests: Sequence[Mapping]) -> int:
    return sum(
        1
        for manifest in manifests
        for case in manifest['cases']
        if case['subject_id'] == 'Pat6'
    )


def run_audit_only(
    *,
    output_root,
    legacy_results_root,
    legacy_results,
    checkpoint_root,
    json_log_root,
    fold_ids,
    training_protocol,
    evaluation_protocol,
    diagnostic_protocol,
    manifests,
) -> dict:
    """Run the no-model, no-GPU diagnostic preflight."""
    protocol_audit = run_protocol_audit(
        training_protocol,
        evaluation_protocol,
    )
    manifest_audit = _validate_manifest_contract(
        training_protocol,
        diagnostic_protocol,
        manifests,
        protocol_audit,
    )
    diagnostic_contract = get_diagnostic_protocol_contract(
        diagnostic_protocol
    )
    legacy = legacy_results
    checkpoint_audit = None
    if checkpoint_root is not None:
        checkpoint_audit = _validate_checkpoint_roots_cpu(
            checkpoint_root,
            json_log_root,
            fold_ids,
            training_protocol,
            evaluation_protocol,
        )
    selected_case_count = sum(manifests[fold]['total_cases'] for fold in fold_ids)
    result = {
        'mode': 'audit-only',
        'diagnostic_protocol_version': diagnostic_protocol[
            'diagnostic_protocol_version'
        ],
        'diagnostic_protocol_hash': diagnostic_protocol[
            'diagnostic_protocol_hash'
        ],
        'selected_folds': list(fold_ids),
        'selected_expected_case_count': selected_case_count,
        'global_expected_case_count': diagnostic_contract[
            'expected_test_case_count'
        ],
        'LEGACY_PER_FOLD_COUNTS': dict(legacy['per_fold_counts']),
        'LEGACY_PER_FOLD_UNION_COUNT': legacy['per_fold_union_count'],
        'LEGACY_PER_FOLD_UNION_IDENTITY_AUDIT': legacy[
            'per_fold_union_identity_audit'
        ],
        'legacy_root_cases_status': legacy['legacy_root_cases_status'],
        'legacy_root_cases_count': legacy['legacy_root_cases_count'],
        'legacy_root_cases_detected_fold': legacy[
            'legacy_root_cases_detected_fold'
        ],
        'protocol_audit': protocol_audit,
        'legacy_result_audit': {
            'root': str(Path(legacy_results_root).resolve()),
            'legacy_case_authoritative_source': diagnostic_protocol[
                'legacy_case_authoritative_source'
            ],
            'per_fold_case_counts': dict(legacy['per_fold_counts']),
            'per_fold_union_count': legacy['per_fold_union_count'],
            'per_fold_union_identity_audit': legacy[
                'per_fold_union_identity_audit'
            ],
            'legacy_root_cases_status': legacy['legacy_root_cases_status'],
            'legacy_root_cases_count': legacy['legacy_root_cases_count'],
            'legacy_root_cases_detected_fold': legacy[
                'legacy_root_cases_detected_fold'
            ],
        },
        'checkpoint_audit': checkpoint_audit,
        'model_loaded': False,
        'gpu_used': False,
    }
    if not diagnostic_contract['pat6_forensic_required']:
        result.update(
            {
                'expected_patient_count': manifest_audit['patient_count'],
                'expected_defect_condition_count': manifest_audit[
                    'defect_condition_count'
                ],
                'expected_defect_instance_count': manifest_audit[
                    'defect_instance_count'
                ],
                'pat6_case_count': manifest_audit['pat6_case_count'],
                'per_fold_expected_case_counts': manifest_audit[
                    'per_fold_case_counts'
                ],
            }
        )
    complete_evaluation._write_json(
        Path(output_root) / 'diagnostic_manifest.json',
        result,
    )
    return result


def _validate_raw_identity(raw_sample, manifest_case) -> None:
    try:
        defect_cli._validate_raw_defect_identity(raw_sample, manifest_case)
    except DefectEvaluationContractError as error:
        raise M3MetricDiagnosticContractError(str(error)) from error


def _evaluate_diagnostic_case(
    raw_sample,
    manifest_case,
    checkpoint_path,
    training_protocol,
    evaluation_protocol,
    diagnostic_protocol,
    point_encoder,
    ct_encoder,
    matcher,
    device,
) -> dict:
    """Observe one unchanged frozen inference output and add diagnostics."""
    _validate_raw_identity(raw_sample, manifest_case)
    augmented = augment_point_sample(
        raw_sample,
        seed=manifest_case['perturbation_seed'],
        max_rotation_deg=manifest_case['max_rotation_deg'],
        max_translation_mm=manifest_case['max_translation_mm'],
        scheme_version=training_protocol['seed_scheme_version'],
    )
    effective_gt = np.asarray(augmented['gt_transform'], dtype=np.float64)
    inference = complete_evaluation._run_inference_case(
        augmented,
        point_encoder,
        ct_encoder,
        matcher,
        evaluation_protocol,
        device,
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
    registration_metrics = evaluate_registration_result(
        registration_output,
        effective_gt,
        evaluation_protocol['registration_rre_threshold_deg'],
        evaluation_protocol['registration_rte_threshold_mm'],
    )
    solver_declared_success = registration_output.get('success')
    if not isinstance(solver_declared_success, (bool, np.bool_)):
        raise M3MetricDiagnosticContractError(
            'registration_output.success must be boolean.'
        )
    if inlier_metrics['num_correspondences'] == 0 and bool(
        registration_metrics['solver_success']
    ):
        raise M3MetricDiagnosticContractError(
            'zero-correspondence registration must fail closed.'
        )
    if bool(solver_declared_success):
        predicted_transform = make_rigid_transform(
            registration_output.get('rotation'),
            registration_output.get('translation'),
            'predicted_transform',
        )
    else:
        predicted_transform = None
    tre = compute_tre_diagnostics(
        inference['point_physical'],
        predicted_transform,
        effective_gt,
        solver_success=bool(solver_declared_success),
    )
    if bool(solver_declared_success) != bool(
        registration_metrics['solver_success']
    ):
        raise M3MetricDiagnosticContractError(
            'frozen evaluator and diagnostic disagree on solver success; '
            'an invalid success transform is not accepted as a normal failure.'
        )
    if bool(solver_declared_success):
        diagnostic_rre = rotation_error_degrees(
            predicted_transform[:3, :3],
            effective_gt[:3, :3],
        )
        if not math.isclose(
            diagnostic_rre,
            float(registration_metrics['rre_deg']),
            rel_tol=0.0,
            abs_tol=DIAGNOSTIC_RRE_ABS_TOLERANCE,
        ):
            raise M3MetricDiagnosticContractError(
                'diagnostic RRE cross-check disagrees with the frozen evaluator.'
            )
        if not math.isclose(
            float(tre['legacy_parameter_rte_mm']),
            float(registration_metrics['rte_mm']),
            rel_tol=0.0,
            abs_tol=DIAGNOSTIC_RRE_ABS_TOLERANCE,
        ):
            raise M3MetricDiagnosticContractError(
                'legacy parameter RTE cross-check disagrees with the frozen evaluator.'
            )
    confidence = summarize_confidence(
        filter_output['confidence'],
        inlier_metrics['num_correspondences'],
    )
    return {
        'diagnostic_protocol_version': diagnostic_protocol[
            'diagnostic_protocol_version'
        ],
        'diagnostic_protocol_hash': diagnostic_protocol[
            'diagnostic_protocol_hash'
        ],
        'source_training_protocol_version': training_protocol[
            'protocol_version'
        ],
        'source_training_protocol_hash': training_protocol['protocol_hash'],
        'source_defect_evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': evaluation_protocol[
            'evaluation_protocol_hash'
        ],
        'source_reference_points': diagnostic_protocol[
            'source_reference_points'
        ],
        'case_key': manifest_case['case_key'],
        'fold_id': manifest_case['fold_id'],
        'subject_id': manifest_case['subject_id'],
        'defect_id': manifest_case['defect_id'],
        'severity': manifest_case['severity'],
        'variant_id': manifest_case['variant_id'],
        'variant_index': manifest_case['variant_index'],
        'perturbation_seed': manifest_case['perturbation_seed'],
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        **inlier_metrics,
        'solver_success': registration_metrics['solver_success'],
        'solver_status': registration_metrics['solver_status'],
        'rre_deg': registration_metrics['rre_deg'],
        'registration_recall_hit': registration_metrics[
            'registration_recall_hit'
        ],
        **tre,
        **confidence,
        'diagnostic_inference_runtime_ms': inference['inference_runtime_ms'],
    }


def _execute_fold(
    *,
    data_root,
    checkpoint_path,
    jsonl_path,
    device_name,
    training_protocol,
    evaluation_protocol,
    diagnostic_protocol,
    manifest,
) -> tuple:
    payload, checkpoint_metadata = read_and_validate_defect_checkpoint(
        checkpoint_path,
        jsonl_path,
        training_protocol,
        evaluation_protocol,
        manifest['fold_id'],
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
    dataset = create_formal_defect_dataset(data_root, training_protocol)
    split = build_formal_defect_split(
        dataset,
        training_protocol,
        manifest['fold_id'],
    )
    identity_to_index = dict(zip(split.test_instance_ids, split.test_indices))
    if len(identity_to_index) != len(split.test_instance_ids):
        raise M3MetricDiagnosticContractError(
            f'{manifest["fold_id"]} test instance mapping is not unique.'
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
                raise M3MetricDiagnosticContractError(
                    f'{manifest["fold_id"]} manifest instance is absent from '
                    f'the frozen test split: {identity!r}.'
                ) from error
            cases.append(
                _evaluate_diagnostic_case(
                    dataset[dataset_index],
                    manifest_case,
                    checkpoint_path,
                    training_protocol,
                    evaluation_protocol,
                    diagnostic_protocol,
                    point_encoder,
                    ct_encoder,
                    matcher,
                    device,
                )
            )
    return cases, checkpoint_metadata


def _summary_with_provenance(
    cases,
    manifests,
    diagnostic_protocol,
    legacy_cross_check,
) -> dict:
    expected_case_count = sum(manifest['total_cases'] for manifest in manifests)
    diagnostic_contract = get_diagnostic_protocol_contract(
        diagnostic_protocol
    )
    expected_pat6 = (
        _expected_pat6_count(manifests)
        if diagnostic_contract['pat6_forensic_required']
        else None
    )
    summary = aggregate_diagnostic_cases(
        cases,
        expected_case_count=expected_case_count,
        expected_pat6_case_count=expected_pat6,
        diagnostic_protocol=diagnostic_protocol,
    )
    return {
        'diagnostic_protocol_version': diagnostic_protocol[
            'diagnostic_protocol_version'
        ],
        'diagnostic_protocol_hash': diagnostic_protocol[
            'diagnostic_protocol_hash'
        ],
        'source_training_protocol_version': diagnostic_protocol[
            'source_training_protocol_version'
        ],
        'source_training_protocol_hash': diagnostic_protocol[
            'source_training_protocol_hash'
        ],
        'source_defect_evaluation_protocol_version': diagnostic_protocol[
            'source_defect_evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': diagnostic_protocol[
            'source_defect_evaluation_protocol_hash'
        ],
        'selected_folds': [manifest['fold_id'] for manifest in manifests],
        'legacy_cross_check': legacy_cross_check,
        **summary,
    }


def run_execute_diagnostic(
    *,
    data_root,
    checkpoint_root,
    json_log_root,
    output_root,
    legacy_results,
    device_name,
    fold_ids,
    training_protocol,
    evaluation_protocol,
    diagnostic_protocol,
    manifests,
) -> dict:
    """Execute unchanged inference and write independent diagnostic artifacts."""
    all_cases = []
    per_fold_cases = {}
    checkpoint_metadata = {}
    for fold_id in fold_ids:
        checkpoint_path, jsonl_path = resolve_fold_artifacts(
            checkpoint_root,
            fold_id,
            evaluation_protocol,
            json_log_root=json_log_root,
        )
        cases, metadata = _execute_fold(
            data_root=data_root,
            checkpoint_path=checkpoint_path,
            jsonl_path=jsonl_path,
            device_name=device_name,
            training_protocol=training_protocol,
            evaluation_protocol=evaluation_protocol,
            diagnostic_protocol=diagnostic_protocol,
            manifest=manifests[fold_id],
        )
        per_fold_cases[fold_id] = cases
        checkpoint_metadata[fold_id] = metadata
        all_cases.extend(cases)

    if tuple(fold_ids) == tuple(training_protocol['folds']):
        selected_legacy = list(legacy_results['authoritative_union'])
    else:
        selected_legacy = [
            row
            for fold_id in fold_ids
            for row in legacy_results['per_fold'][fold_id]
        ]
    selected_expected_count = sum(
        manifests[fold_id]['total_cases'] for fold_id in fold_ids
    )
    root_cross_check = cross_check_legacy_cases(
        selected_legacy,
        all_cases,
        expected_count=selected_expected_count,
    )

    fold_summaries = {}
    for fold_id in fold_ids:
        fold_cross_check = cross_check_legacy_cases(
            legacy_results['per_fold'][fold_id],
            per_fold_cases[fold_id],
            expected_count=manifests[fold_id]['total_cases'],
        )
        fold_summary = _summary_with_provenance(
            per_fold_cases[fold_id],
            [manifests[fold_id]],
            diagnostic_protocol,
            fold_cross_check,
        )
        fold_manifest = {
            'diagnostic_protocol_version': diagnostic_protocol[
                'diagnostic_protocol_version'
            ],
            'diagnostic_protocol_hash': diagnostic_protocol[
                'diagnostic_protocol_hash'
            ],
            'fold_id': fold_id,
            'case_count': len(per_fold_cases[fold_id]),
            'source_test_manifest': manifests[fold_id],
            'checkpoint_path': per_fold_cases[fold_id][0]['checkpoint_path'],
            'checkpoint_epoch': checkpoint_metadata[fold_id]['epoch'],
            'checkpoint_best_val_loss': checkpoint_metadata[fold_id][
                'best_val_loss'
            ],
            'legacy_cross_check': fold_cross_check,
        }
        fold_dir = Path(output_root) / fold_id
        complete_evaluation._write_jsonl(
            fold_dir / 'diagnostic_cases.jsonl',
            per_fold_cases[fold_id],
        )
        complete_evaluation._write_json(
            fold_dir / 'diagnostic_summary.json',
            fold_summary,
        )
        complete_evaluation._write_json(
            fold_dir / 'diagnostic_manifest.json',
            fold_manifest,
        )
        fold_summaries[fold_id] = fold_summary

    root_summary = _summary_with_provenance(
        all_cases,
        [manifests[fold_id] for fold_id in fold_ids],
        diagnostic_protocol,
        root_cross_check,
    )
    root_summary['mode'] = 'execute-diagnostic'
    root_summary['fold_summaries'] = fold_summaries
    complete_evaluation._write_jsonl(
        Path(output_root) / 'diagnostic_cases.jsonl',
        all_cases,
    )
    complete_evaluation._write_json(
        Path(output_root) / 'diagnostic_summary.json',
        root_summary,
    )
    if 'pat6_forensic' in root_summary:
        complete_evaluation._write_json(
            Path(output_root) / 'pat6_forensic.json',
            root_summary['pat6_forensic'],
        )
    return root_summary


def run_evaluation(args) -> dict:
    """Validate safe selectors, source contracts, and execute one explicit mode."""
    mode = _explicit_mode(args)
    legacy_results_root = getattr(args, 'legacy_results_root', None)
    if legacy_results_root is None:
        raise M3MetricDiagnosticContractError(
            '--legacy-results-root is required for formal identity and metric '
            'cross-checking.'
        )
    output_root = _require_safe_empty_output_root(
        args.output_root,
        legacy_results_root,
    )
    training_protocol, evaluation_protocol, diagnostic_protocol = (
        _load_protocol_bundle(args)
    )
    fold_ids = _selected_fold_ids(args, training_protocol)
    manifests = _all_manifests(training_protocol, evaluation_protocol)
    _validate_manifest_contract(
        training_protocol,
        diagnostic_protocol,
        manifests,
        run_protocol_audit(training_protocol, evaluation_protocol),
    )
    legacy_results = validate_legacy_result_tree(
        legacy_results_root,
        {
            fold_id: manifests[fold_id]['cases']
            for fold_id in training_protocol['folds']
        },
        diagnostic_protocol=diagnostic_protocol,
    )
    if mode == 'audit-only':
        return run_audit_only(
            output_root=output_root,
            legacy_results_root=legacy_results_root,
            legacy_results=legacy_results,
            checkpoint_root=getattr(args, 'checkpoint_root', None),
            json_log_root=getattr(args, 'json_log_root', None),
            fold_ids=fold_ids,
            training_protocol=training_protocol,
            evaluation_protocol=evaluation_protocol,
            diagnostic_protocol=diagnostic_protocol,
            manifests=manifests,
        )
    checkpoint_root = getattr(args, 'checkpoint_root', None)
    if checkpoint_root is None:
        raise M3MetricDiagnosticContractError(
            '--execute-diagnostic requires --checkpoint-root.'
        )
    return run_execute_diagnostic(
        data_root=args.data_root,
        checkpoint_root=checkpoint_root,
        json_log_root=getattr(args, 'json_log_root', None),
        output_root=output_root,
        legacy_results=legacy_results,
        device_name=args.device,
        fold_ids=fold_ids,
        training_protocol=training_protocol,
        evaluation_protocol=evaluation_protocol,
        diagnostic_protocol=diagnostic_protocol,
        manifests=manifests,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Independent origin-invariant diagnostics for the frozen M3 '
            'defect baseline.'
        )
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
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
        '--diagnostic-protocol',
        type=Path,
        default=DEFAULT_DIAGNOSTIC_PROTOCOL,
    )
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--json-log-root', type=Path)
    parser.add_argument('--legacy-results-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--device',
        default='cpu',
        help='Used only by --execute-diagnostic; audit-only never resolves a GPU.',
    )
    folds = parser.add_mutually_exclusive_group(required=True)
    folds.add_argument('--fold-id', choices=tuple(FOLD_SUBJECTS))
    folds.add_argument('--all-folds', action='store_true')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--audit-only', action='store_true')
    modes.add_argument('--execute-diagnostic', action='store_true')
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
    'DEFAULT_DIAGNOSTIC_PROTOCOL',
    'DEFAULT_EVALUATION_PROTOCOL',
    'DEFAULT_OUTPUT_ROOT_NAME',
    'DEFAULT_TRAINING_PROTOCOL',
    'build_argument_parser',
    'main',
    'run_audit_only',
    'run_evaluation',
    'run_execute_diagnostic',
]
