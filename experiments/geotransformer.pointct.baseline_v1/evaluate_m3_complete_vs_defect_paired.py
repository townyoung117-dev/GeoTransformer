"""Two-stage CLI for frozen M3 clean10 complete-vs-defect sensitivity.

``--execute-complete`` is the only mode that loads models.  The
``--build-paired-comparison`` mode is strictly CPU-only JSON aggregation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import evaluate_m3 as complete_evaluation
from config import DEFAULT_DATA_ROOT
from dataset import create_dataset
from defect_evaluation import (
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
    read_and_validate_defect_checkpoint,
    resolve_fold_artifacts,
)
from evaluation import correspondence_inlier_metrics, evaluate_registration_result
from m3_complete_vs_defect_paired import (
    COMPLETE_INFERENCE_CASE_COUNT,
    FOLD_IDS,
    M3CompleteVsDefectPairedContractError,
    aggregate_complete_cases,
    audit_complete_dataset,
    build_complete_manifest,
    build_paired_comparison,
    load_paired_protocol,
    validate_source_protocols,
)
from m3_metric_diagnostic import (
    compute_tre_diagnostics,
    load_jsonl,
    make_rigid_transform,
    rotation_error_degrees,
    summarize_confidence,
)
from m3_metric_observation import load_observation_protocol
from perturbation import augment_point_sample
from training_protocol import load_training_protocol


EXPERIMENT_DIR = Path(__file__).resolve().parent
CLEAN10_TRAINING_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_clean10_v2.json'
)
CLEAN10_EVALUATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_clean10_v2.json'
)
OBSERVATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_observation_clean10_v3.json'
)
PAIRED_PROTOCOL = (
    EXPERIMENT_DIR
    / 'protocols'
    / 'm3_complete_vs_defect_paired_clean10_v1.json'
)
DEFAULT_COMPLETE_OUTPUT_ROOT_NAME = 'm3_complete_clean10_tre_150_v1'
DEFAULT_PAIRED_OUTPUT_ROOT_NAME = 'm3_complete_vs_defect_paired_clean10_v1'
INTERNAL_RRE_ABS_TOLERANCE = 1e-10


def _load_protocol_bundle(args) -> tuple:
    training = load_training_protocol(args.protocol_manifest)
    evaluation = load_defect_evaluation_protocol(args.evaluation_protocol)
    observation = load_observation_protocol(args.observation_protocol)
    paired = load_paired_protocol(args.paired_protocol)
    validate_source_protocols(paired, training, evaluation, observation)
    return training, evaluation, observation, paired


def _defect_manifests(training, evaluation) -> dict:
    return {
        fold_id: build_defect_test_manifest(training, evaluation, fold_id)
        for fold_id in training['folds']
    }


def _require_empty_output_root(output_root, *input_paths) -> Path:
    output = Path(output_root).resolve()
    for input_path in input_paths:
        if input_path is None:
            continue
        candidate = Path(input_path).resolve()
        if candidate.is_file():
            candidate = candidate.parent
        if output == candidate:
            raise M3CompleteVsDefectPairedContractError(
                'output root must not equal an input artifact location.'
            )
    if output.exists():
        if not output.is_dir():
            raise M3CompleteVsDefectPairedContractError(
                f'output root exists and is not a directory: {output}.'
            )
        try:
            nonempty = next(output.iterdir(), None) is not None
        except OSError as error:
            raise M3CompleteVsDefectPairedContractError(
                f'cannot inspect output root {output}: {error}'
            ) from error
        if nonempty:
            raise M3CompleteVsDefectPairedContractError(
                f'output root already exists and is non-empty: {output}. '
                'Automatic overwrite is forbidden.'
            )
    return output


def _validate_complete_raw_sample(raw_sample: Mapping, manifest_case: Mapping) -> None:
    if not isinstance(raw_sample, Mapping):
        raise M3CompleteVsDefectPairedContractError(
            'complete dataset sample must be a mapping.'
        )
    if raw_sample.get('subject_id') != manifest_case['subject_id']:
        raise M3CompleteVsDefectPairedContractError(
            'complete dataset subject does not match manifest: '
            f'expected={manifest_case["subject_id"]!r}, '
            f'actual={raw_sample.get("subject_id")!r}.'
        )
    forbidden = ('defect_id', 'point_defect_mask', 'ct_defect_mask')
    present = [field for field in forbidden if field in raw_sample]
    if present:
        raise M3CompleteVsDefectPairedContractError(
            f'complete sample contains forbidden defect fields: {present}.'
        )
    points = np.asarray(raw_sample.get('point_xyz_phys'))
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or points.shape[0] == 0
        or not np.issubdtype(points.dtype, np.number)
        or not np.all(np.isfinite(points))
    ):
        raise M3CompleteVsDefectPairedContractError(
            'complete source points must be finite [N,3], N>0.'
        )
    ct = np.asarray(raw_sample.get('ct_volume'))
    if (
        ct.ndim != 3
        or any(size <= 0 for size in ct.shape)
        or not np.issubdtype(ct.dtype, np.number)
        or (np.issubdtype(ct.dtype, np.floating) and not np.all(np.isfinite(ct)))
    ):
        raise M3CompleteVsDefectPairedContractError(
            'complete CT must be a finite/non-empty numeric 3D volume.'
        )


def _evaluate_complete_case(
    raw_sample,
    manifest_case,
    checkpoint_path,
    training_protocol,
    evaluation_protocol,
    paired_protocol,
    point_encoder,
    ct_encoder,
    matcher,
    device,
) -> dict:
    """Run the unchanged frozen inference and compute v3-identical TRE."""
    _validate_complete_raw_sample(raw_sample, manifest_case)
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
        raise M3CompleteVsDefectPairedContractError(
            'registration_output.success must be boolean.'
        )
    if inlier_metrics['num_correspondences'] == 0 and bool(
        registration_metrics['solver_success']
    ):
        raise M3CompleteVsDefectPairedContractError(
            'zero-correspondence registration must fail closed.'
        )
    predicted_transform = None
    if bool(solver_declared_success):
        predicted_transform = make_rigid_transform(
            registration_output.get('rotation'),
            registration_output.get('translation'),
            'predicted_transform',
        )
    tre = compute_tre_diagnostics(
        inference['point_physical'],
        predicted_transform,
        effective_gt,
        solver_success=bool(solver_declared_success),
    )
    if bool(solver_declared_success) != bool(
        registration_metrics['solver_success']
    ):
        raise M3CompleteVsDefectPairedContractError(
            'frozen evaluator and complete TRE observer disagree on solver '
            'success; invalid transforms fail closed.'
        )
    if bool(solver_declared_success):
        observed_rre = rotation_error_degrees(
            predicted_transform[:3, :3],
            effective_gt[:3, :3],
        )
        if not math.isclose(
            observed_rre,
            float(registration_metrics['rre_deg']),
            rel_tol=0.0,
            abs_tol=INTERNAL_RRE_ABS_TOLERANCE,
        ):
            raise M3CompleteVsDefectPairedContractError(
                'complete RRE cross-check disagrees with frozen evaluator.'
            )
        if not math.isclose(
            float(tre['legacy_parameter_rte_mm']),
            float(registration_metrics['rte_mm']),
            rel_tol=0.0,
            abs_tol=INTERNAL_RRE_ABS_TOLERANCE,
        ):
            raise M3CompleteVsDefectPairedContractError(
                'complete RTE cross-check disagrees with frozen evaluator.'
            )
    confidence = summarize_confidence(
        filter_output['confidence'],
        inlier_metrics['num_correspondences'],
    )
    return {
        'paired_protocol_version': paired_protocol['paired_protocol_version'],
        'paired_protocol_hash': paired_protocol['paired_protocol_hash'],
        'source_training_protocol_version': training_protocol['protocol_version'],
        'source_training_protocol_hash': training_protocol['protocol_hash'],
        'source_defect_evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'source_defect_evaluation_protocol_hash': evaluation_protocol[
            'evaluation_protocol_hash'
        ],
        'source_reference_points': paired_protocol[
            'complete_source_reference_points'
        ],
        'complete_case_key': manifest_case['complete_case_key'],
        'fold_id': manifest_case['fold_id'],
        'subject_id': manifest_case['subject_id'],
        'input_kind': 'complete',
        'severity': manifest_case['severity'],
        'variant_id': manifest_case['variant_id'],
        'variant_index': manifest_case['variant_index'],
        'perturbation_seed': manifest_case['perturbation_seed'],
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        **inlier_metrics,
        'solver_success': registration_metrics['solver_success'],
        'solver_status': registration_metrics['solver_status'],
        'registration_recall_hit': registration_metrics[
            'registration_recall_hit'
        ],
        'complete_rre_deg': registration_metrics['rre_deg'],
        **tre,
        **confidence,
        'complete_inference_runtime_ms': inference['inference_runtime_ms'],
    }


def _execute_complete_fold(
    *,
    dataset,
    checkpoint_path,
    jsonl_path,
    device_name,
    training_protocol,
    evaluation_protocol,
    paired_protocol,
    complete_manifest_cases,
    fold_id,
) -> tuple:
    """Explicit model path; imports torch/models only after mode selection."""
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
    subject_to_index = {
        record['subject_id']: index for index, record in enumerate(dataset.records)
    }
    if len(subject_to_index) != len(dataset.records):
        raise M3CompleteVsDefectPairedContractError(
            'complete dataset subject mapping is not unique.'
        )
    cases = []
    with torch.no_grad():
        for manifest_case in complete_manifest_cases:
            subject_id = manifest_case['subject_id']
            try:
                dataset_index = subject_to_index[subject_id]
            except KeyError as error:
                raise M3CompleteVsDefectPairedContractError(
                    f'{fold_id} complete patient is absent: {subject_id!r}.'
                ) from error
            cases.append(
                _evaluate_complete_case(
                    dataset[dataset_index],
                    manifest_case,
                    checkpoint_path,
                    training_protocol,
                    evaluation_protocol,
                    paired_protocol,
                    point_encoder,
                    ct_encoder,
                    matcher,
                    device,
                )
            )
    return cases, checkpoint_metadata


def run_execute_complete(
    *,
    data_root,
    checkpoint_root,
    json_log_root,
    output_root,
    device_name,
    training_protocol,
    evaluation_protocol,
    paired_protocol,
    complete_manifest,
) -> dict:
    """Execute exactly 150 complete cases using the five frozen checkpoints."""
    # This exact call is the audited default complete-subject path.  No
    # defect_variants argument is supplied.
    dataset = create_dataset(data_root)
    dataset_audit = audit_complete_dataset(dataset, training_protocol)
    all_cases = []
    checkpoint_provenance = {}
    for fold_id in FOLD_IDS:
        checkpoint_path, jsonl_path = resolve_fold_artifacts(
            checkpoint_root,
            fold_id,
            evaluation_protocol,
            json_log_root=json_log_root,
        )
        cases, metadata = _execute_complete_fold(
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            jsonl_path=jsonl_path,
            device_name=device_name,
            training_protocol=training_protocol,
            evaluation_protocol=evaluation_protocol,
            paired_protocol=paired_protocol,
            complete_manifest_cases=complete_manifest['per_fold'][fold_id],
            fold_id=fold_id,
        )
        all_cases.extend(cases)
        checkpoint_provenance[fold_id] = {
            'checkpoint_path': str(Path(checkpoint_path).resolve()),
            'checkpoint_epoch': metadata['epoch'],
            'checkpoint_best_val_loss': metadata['best_val_loss'],
            'jsonl_path': metadata['jsonl_path'],
            'jsonl_validation': metadata['jsonl_validation'],
        }
    summary = aggregate_complete_cases(
        all_cases,
        paired_protocol,
        expected_count=COMPLETE_INFERENCE_CASE_COUNT,
    )
    if any(
        count != 30 for count in summary['per_fold_case_counts'].values()
    ):
        raise M3CompleteVsDefectPairedContractError(
            'complete execution must produce 30 cases per Fold.'
        )
    summary.update(
        {
            'mode': 'execute-complete',
            'complete_point_tre_mean_mm': summary['tre_metrics'][
                'point_tre_mean_mm'
            ]['mean'],
            'complete_centroid_tre_mean_mm': summary['tre_metrics'][
                'centroid_tre_mm'
            ]['mean'],
            'model_loaded': True,
            'gpu_used': str(device_name).lower().startswith('cuda'),
            'training_performed': False,
        }
    )
    manifest_output = {
        **complete_manifest,
        'complete_dataset_audit': dataset_audit,
    }
    provenance = {
        'paired_protocol_version': paired_protocol['paired_protocol_version'],
        'paired_protocol_hash': paired_protocol['paired_protocol_hash'],
        'source_code_commit': paired_protocol['source_code_commit'],
        'checkpoints': checkpoint_provenance,
        'same_frozen_pipeline_as_defect': True,
        'defect_masks_consumed': False,
        'training_performed': False,
        'model_or_matcher_changed': False,
    }
    output = Path(output_root)
    complete_evaluation._write_jsonl(
        output / 'complete_cases_150.jsonl',
        all_cases,
    )
    complete_evaluation._write_json(
        output / 'complete_summary.json',
        summary,
    )
    complete_evaluation._write_json(
        output / 'complete_manifest.json',
        manifest_output,
    )
    complete_evaluation._write_json(output / 'provenance.json', provenance)
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_sha256sums(output: Path, filenames: Sequence[str]) -> None:
    lines = [
        f'{_sha256_file(output / filename)}  {filename}' for filename in filenames
    ]
    (output / 'SHA256SUMS').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
        newline='\n',
    )


def run_build_paired_comparison(
    *,
    complete_cases_path,
    defect_cases_path,
    output_root,
    paired_protocol,
    observation_protocol,
    complete_manifest,
) -> dict:
    """CPU-only pairing path; loads no checkpoint, model, torch, or GPU."""
    complete_rows = load_jsonl(complete_cases_path)
    defect_rows = load_jsonl(defect_cases_path)
    result = build_paired_comparison(
        complete_rows,
        defect_rows,
        paired_protocol,
        observation_protocol,
        expected_complete_manifest=complete_manifest['cases'],
    )
    output = Path(output_root)
    filenames = (
        'paired_cases_750.jsonl',
        'paired_summary.json',
        'per_subject.json',
        'per_defect.json',
        'per_severity.json',
        'per_fold.json',
        'provenance.json',
    )
    complete_evaluation._write_jsonl(
        output / filenames[0],
        result['paired_rows'],
    )
    complete_evaluation._write_json(output / filenames[1], result['summary'])
    complete_evaluation._write_json(output / filenames[2], result['per_subject'])
    complete_evaluation._write_json(output / filenames[3], result['per_defect'])
    complete_evaluation._write_json(output / filenames[4], result['per_severity'])
    complete_evaluation._write_json(output / filenames[5], result['per_fold'])
    provenance = dict(result['provenance'])
    provenance['inputs'] = {
        'complete_cases_150': str(Path(complete_cases_path).resolve()),
        'defect_cases_750': str(Path(defect_cases_path).resolve()),
    }
    complete_evaluation._write_json(output / filenames[6], provenance)
    _write_sha256sums(output, filenames)
    return {
        **result['summary'],
        'mode': 'build-paired-comparison',
        'paired_identity_unique_count': result[
            'paired_identity_unique_count'
        ],
        'output_root': str(output.resolve()),
        'model_loaded': False,
        'gpu_used': False,
        'training_performed': False,
    }


def run_evaluation(args) -> dict:
    execute = bool(getattr(args, 'execute_complete', False))
    build = bool(getattr(args, 'build_paired_comparison', False))
    if execute == build:
        raise M3CompleteVsDefectPairedContractError(
            'choose exactly one mode: --execute-complete or '
            '--build-paired-comparison.'
        )
    training, evaluation, observation, paired = _load_protocol_bundle(args)
    defect_manifests = _defect_manifests(training, evaluation)
    complete_manifest = build_complete_manifest(defect_manifests, paired)
    if execute:
        checkpoint_root = getattr(args, 'checkpoint_root', None)
        if checkpoint_root is None:
            raise M3CompleteVsDefectPairedContractError(
                '--execute-complete requires --checkpoint-root.'
            )
        output = _require_empty_output_root(args.output_root)
        return run_execute_complete(
            data_root=args.data_root,
            checkpoint_root=checkpoint_root,
            json_log_root=getattr(args, 'json_log_root', None),
            output_root=output,
            device_name=args.device,
            training_protocol=training,
            evaluation_protocol=evaluation,
            paired_protocol=paired,
            complete_manifest=complete_manifest,
        )
    complete_cases_path = getattr(args, 'complete_cases', None)
    defect_bundle_root = getattr(args, 'defect_bundle_root', None)
    if complete_cases_path is None or defect_bundle_root is None:
        raise M3CompleteVsDefectPairedContractError(
            '--build-paired-comparison requires --complete-cases and '
            '--defect-bundle-root.'
        )
    defect_cases_path = Path(defect_bundle_root) / 'cases_750.jsonl'
    output = _require_empty_output_root(
        args.output_root,
        complete_cases_path,
        defect_bundle_root,
    )
    return run_build_paired_comparison(
        complete_cases_path=complete_cases_path,
        defect_cases_path=defect_cases_path,
        output_root=output,
        paired_protocol=paired,
        observation_protocol=observation,
        complete_manifest=complete_manifest,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Frozen M3 clean10 complete-vs-defect paired TRE sensitivity '
            'experiment.'
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
    parser.add_argument(
        '--paired-protocol',
        type=Path,
        default=PAIRED_PROTOCOL,
    )
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--json-log-root', type=Path)
    parser.add_argument('--complete-cases', type=Path)
    parser.add_argument('--defect-bundle-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--device',
        default='cpu',
        help='Used only by --execute-complete.',
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--execute-complete', action='store_true')
    modes.add_argument('--build-paired-comparison', action='store_true')
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    result = run_evaluation(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()


__all__ = [
    'CLEAN10_EVALUATION_PROTOCOL',
    'CLEAN10_TRAINING_PROTOCOL',
    'DEFAULT_COMPLETE_OUTPUT_ROOT_NAME',
    'DEFAULT_PAIRED_OUTPUT_ROOT_NAME',
    'OBSERVATION_PROTOCOL',
    'PAIRED_PROTOCOL',
    'build_argument_parser',
    'main',
    'run_build_paired_comparison',
    'run_evaluation',
    'run_execute_complete',
]
