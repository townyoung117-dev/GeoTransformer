"""Fail-closed M3-7 evaluation CLI for the frozen Point-CT Baseline V1."""

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

from config import DEFAULT_DATA_ROOT, make_cfg
from evaluation import (
    M3EvaluationContractError,
    aggregate_case_metrics,
    correspondence_inlier_metrics,
    evaluate_registration_result,
    load_evaluation_protocol,
    validate_evaluation_protocol,
)
from perturbation import (
    augment_point_sample,
    derive_perturbation_seed,
    sample_rigid_perturbation,
)
from training_protocol import (
    load_training_protocol,
    resolve_fold,
    test_perturbation_specs,
    validate_training_protocol,
)


TEST_MANIFEST_VERSION = 'm3_7_test_manifest_v1'
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
_BEST_VAL_LOSS_ABS_TOLERANCE = 1e-12


def _validate_protocol_pair(training_protocol, evaluation_protocol) -> None:
    validate_training_protocol(training_protocol)
    validate_evaluation_protocol(evaluation_protocol)
    required = evaluation_protocol['required_training_protocol_version']
    actual = training_protocol['protocol_version']
    if actual != required:
        raise M3EvaluationContractError(
            'evaluation protocol requires training protocol '
            f'{required!r}; got {actual!r}.'
        )


def _generate_test_manifest(training_protocol, evaluation_protocol, fold_id: str) -> dict:
    _validate_protocol_pair(training_protocol, evaluation_protocol)
    fold = resolve_fold(training_protocol, fold_id)
    specs = test_perturbation_specs(training_protocol)
    cases = []
    seen_keys = set()
    seen_seeds = set()

    for subject_id in fold['test_subject_ids']:
        for spec in specs:
            for variant_id in range(spec['variant_count']):
                case_key = f'{fold_id}|{subject_id}|{spec["severity"]}|{variant_id}'
                if case_key in seen_keys:
                    raise M3EvaluationContractError(
                        f'duplicate test case key generated: {case_key}.'
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
                    'variant_id': variant_id,
                }
                seed = derive_perturbation_seed(**seed_arguments)
                repeated_seed = derive_perturbation_seed(**seed_arguments)
                if seed != repeated_seed:
                    raise M3EvaluationContractError(
                        f'perturbation seed is non-deterministic for {case_key}.'
                    )
                if seed in seen_seeds:
                    raise M3EvaluationContractError(
                        f'duplicate perturbation seed generated for {case_key}.'
                    )
                sampled = sample_rigid_perturbation(
                    seed,
                    spec['max_rotation_deg'],
                    spec['max_translation_mm'],
                )
                repeated_sample = sample_rigid_perturbation(
                    seed,
                    spec['max_rotation_deg'],
                    spec['max_translation_mm'],
                )
                if (
                    sampled['angle_deg'] != repeated_sample['angle_deg']
                    or not np.array_equal(
                        sampled['translation_mm'],
                        repeated_sample['translation_mm'],
                    )
                    or not np.array_equal(sampled['R_aug'], repeated_sample['R_aug'])
                ):
                    raise M3EvaluationContractError(
                        f'perturbation sample is non-deterministic for {case_key}.'
                    )
                nonidentity = bool(
                    abs(sampled['angle_deg']) > 0.0
                    or np.any(sampled['translation_mm'] != 0.0)
                )
                if not nonidentity:
                    raise M3EvaluationContractError(
                        f'formal test perturbation is identity for {case_key}.'
                    )
                seen_keys.add(case_key)
                seen_seeds.add(seed)
                cases.append(
                    {
                        'case_key': case_key,
                        'fold_id': fold_id,
                        'subject_id': subject_id,
                        'purpose': spec['purpose'],
                        'epoch': None,
                        'severity': spec['severity'],
                        'variant_id': variant_id,
                        'perturbation_seed': seed,
                        'max_rotation_deg': float(spec['max_rotation_deg']),
                        'max_translation_mm': float(spec['max_translation_mm']),
                        'perturbation_angle_deg': float(sampled['angle_deg']),
                        'perturbation_translation_mm': (
                            sampled['translation_mm'].astype(float).tolist()
                        ),
                        'perturbation_nonidentity': True,
                    }
                )

    expected_per_subject = sum(spec['variant_count'] for spec in specs)
    subject_counts = Counter(case['subject_id'] for case in cases)
    expected_subject_counts = Counter(
        {subject_id: expected_per_subject for subject_id in fold['test_subject_ids']}
    )
    if subject_counts != expected_subject_counts:
        raise M3EvaluationContractError(
            'test manifest does not contain exactly 15 cases per test subject.'
        )
    severity_variant_counts = Counter(
        (case['subject_id'], case['severity']) for case in cases
    )
    for subject_id in fold['test_subject_ids']:
        for spec in specs:
            if severity_variant_counts[(subject_id, spec['severity'])] != spec['variant_count']:
                raise M3EvaluationContractError(
                    'test manifest severity/variant count mismatch for '
                    f'{subject_id}/{spec["severity"]}.'
                )

    return {
        'test_manifest_version': TEST_MANIFEST_VERSION,
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol['evaluation_protocol_hash'],
        'fold_id': fold_id,
        'test_subject_ids': list(fold['test_subject_ids']),
        'seed_scheme_version': training_protocol['seed_scheme_version'],
        'perturbation_root_seed': training_protocol['perturbation_root_seed'],
        'cases_per_subject': expected_per_subject,
        'total_cases': len(cases),
        'cases': cases,
    }


def build_test_manifest(training_protocol, evaluation_protocol, fold_id: str) -> dict:
    """Build and independently repeat the formal Fold test manifest.

    This function reads neither dataset subject contents nor a checkpoint.
    """
    first = _generate_test_manifest(training_protocol, evaluation_protocol, fold_id)
    second = _generate_test_manifest(training_protocol, evaluation_protocol, fold_id)
    if first != second:
        raise M3EvaluationContractError('test manifest generation is non-deterministic.')
    return first


def validate_evaluation_checkpoint_metadata(
    checkpoint_path,
    payload,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> dict:
    """Validate formal checkpoint metadata before any model state restoration."""
    _validate_protocol_pair(training_protocol, evaluation_protocol)
    checkpoint_path = Path(checkpoint_path)
    required_basename = evaluation_protocol['required_checkpoint_basename']
    if checkpoint_path.name != required_basename:
        raise M3EvaluationContractError(
            f'execute-test requires a checkpoint named exactly '
            f'{required_basename}; '
            f'got {checkpoint_path.name!r}. No fallback to last.pt is allowed.'
        )
    if not isinstance(payload, Mapping):
        raise M3EvaluationContractError('checkpoint payload must be a mapping.')
    if payload.get('formal_protocol') is not True:
        raise M3EvaluationContractError(
            'checkpoint formal_protocol must be true for execute-test.'
        )
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
        raise M3EvaluationContractError(
            f'formal checkpoint metadata is missing fields: {missing}.'
        )
    if not isinstance(payload['training_config'], Mapping):
        raise M3EvaluationContractError('checkpoint training_config must be a mapping.')

    fold = resolve_fold(training_protocol, fold_id)
    frozen_best = evaluation_protocol['formal_best_checkpoints'][fold_id]
    required_training = evaluation_protocol['required_formal_training']
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
        if payload.get(field) != expected:
            raise M3EvaluationContractError(
                f'checkpoint {field} does not match the current formal protocol.'
            )

    split_comparisons = (
        ('train_subject_ids', fold['train_subject_ids']),
        ('val_subject_ids', fold['val_subject_ids']),
        ('test_subject_ids', fold['test_subject_ids']),
    )
    for field, expected in split_comparisons:
        actual = tuple(payload.get(field, ()))
        if actual != tuple(expected):
            raise M3EvaluationContractError(
                f'checkpoint {field} does not match {fold_id}; checkpoint must not '
                'contain subjects from another Fold split.'
            )

    expected_epoch = frozen_best['epoch']
    if type(payload['epoch']) is not type(expected_epoch) or payload['epoch'] != expected_epoch:
        raise M3EvaluationContractError(
            f'checkpoint epoch does not match the frozen formal best for {fold_id}: '
            f'expected={expected_epoch}, actual={payload["epoch"]!r}.'
        )

    checkpoint_best_val_loss = payload['best_val_loss']
    if isinstance(checkpoint_best_val_loss, bool):
        raise M3EvaluationContractError(
            'checkpoint best_val_loss must be a finite float matching the frozen formal best.'
        )
    try:
        checkpoint_best_val_loss = float(checkpoint_best_val_loss)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3EvaluationContractError(
            'checkpoint best_val_loss must be a finite float matching the frozen formal best.'
        ) from error
    expected_best_val_loss = float(frozen_best['best_val_loss'])
    if not math.isfinite(checkpoint_best_val_loss) or not math.isclose(
        checkpoint_best_val_loss,
        expected_best_val_loss,
        rel_tol=0.0,
        abs_tol=_BEST_VAL_LOSS_ABS_TOLERANCE,
    ):
        raise M3EvaluationContractError(
            f'checkpoint best_val_loss does not match the frozen formal best for {fold_id}: '
            f'expected={expected_best_val_loss!r}, actual={checkpoint_best_val_loss!r}.'
        )

    expected_seed = required_training['seed']
    if type(payload['seed']) is not type(expected_seed) or payload['seed'] != expected_seed:
        raise M3EvaluationContractError(
            'checkpoint seed does not match required_formal_training.seed: '
            f'expected={expected_seed}, actual={payload["seed"]!r}.'
        )

    training_config = payload['training_config']
    missing_training_fields = [
        field for field in _CHECKPOINT_TRAINING_CONFIG_FIELDS if field not in training_config
    ]
    if missing_training_fields:
        raise M3EvaluationContractError(
            'checkpoint training_config is missing formal provenance fields: '
            f'{missing_training_fields}.'
        )
    if (
        type(training_config['seed']) is not type(payload['seed'])
        or training_config['seed'] != payload['seed']
    ):
        raise M3EvaluationContractError(
            'checkpoint training_config.seed does not match checkpoint payload seed.'
        )
    for field in _CHECKPOINT_TRAINING_CONFIG_FIELDS:
        expected = required_training[field]
        actual = training_config[field]
        if type(actual) is not type(expected) or actual != expected:
            raise M3EvaluationContractError(
                f'checkpoint training_config.{field} does not match '
                f'required_formal_training.{field}: '
                f'expected={expected!r}, actual={actual!r}.'
            )
    # required_formal_training.max_epochs is provenance-only: existing formal
    # checkpoints do not contain a max_epochs field, so it is not validated here.
    return {
        'fold_id': fold_id,
        'protocol_version': payload['protocol_version'],
        'protocol_hash': payload['protocol_hash'],
        'test_subject_ids': tuple(payload['test_subject_ids']),
        'perturbation_root_seed': int(payload['perturbation_root_seed']),
        'perturbation_seed_scheme_version': payload[
            'perturbation_seed_scheme_version'
        ],
        'epoch': int(payload['epoch']),
        'best_val_loss': checkpoint_best_val_loss,
        'seed': int(payload['seed']),
        'training_config': dict(training_config),
    }


def read_and_validate_evaluation_checkpoint(
    checkpoint_path,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> tuple:
    """Read a checkpoint on CPU and validate metadata before state loading."""
    _validate_protocol_pair(training_protocol, evaluation_protocol)
    checkpoint_path = Path(checkpoint_path)
    required_basename = evaluation_protocol['required_checkpoint_basename']
    if checkpoint_path.name != required_basename:
        raise M3EvaluationContractError(
            f'execute-test requires a checkpoint named exactly '
            f'{required_basename}; '
            f'got {checkpoint_path.name!r}. No fallback to last.pt is allowed.'
        )
    if not checkpoint_path.is_file():
        raise M3EvaluationContractError(
            f'evaluation checkpoint does not exist: {checkpoint_path}.'
        )
    payload, checkpoint_contract = _load_checkpoint_for_execution(checkpoint_path)
    if not checkpoint_contract['formal_protocol']:
        raise M3EvaluationContractError(
            'checkpoint formal_protocol must be true for execute-test.'
        )
    metadata = validate_evaluation_checkpoint_metadata(
        checkpoint_path,
        payload,
        training_protocol,
        evaluation_protocol,
        fold_id,
    )
    return payload, metadata


def _load_checkpoint_for_execution(checkpoint_path) -> tuple:
    """Load through the existing training checkpoint validator, lazily."""
    try:
        from training import _load_checkpoint_file, _validate_checkpoint

        payload = _load_checkpoint_file(checkpoint_path, map_location='cpu')
        checkpoint_contract = _validate_checkpoint(payload)
    except Exception as error:
        raise M3EvaluationContractError(
            f'checkpoint structural validation failed: {error}'
        ) from error
    return payload, checkpoint_contract


def _create_dataset_for_execution(data_root):
    """Create the real dataset only inside the explicit execution path."""
    from dataset import create_dataset

    return create_dataset(data_root)


def _build_models_from_checkpoint(payload, metadata, device):
    """Construct models and restore states only after metadata validation."""
    import torch

    from ct_encoder import CTEncoder
    from matching import PointCTMatcher
    from point_encoder import PointEncoder

    training_config = metadata['training_config']
    required_matcher_fields = ('temperature', 'sinkhorn_iterations', 'alpha_init')
    missing = [field for field in required_matcher_fields if field not in training_config]
    if missing:
        raise M3EvaluationContractError(
            f'checkpoint training_config is missing matcher fields: {missing}.'
        )
    cfg = make_cfg()
    point_encoder = PointEncoder(cfg)
    ct_encoder = CTEncoder(cfg)
    matcher = PointCTMatcher(
        projected_dim=cfg.point.projected_dim,
        temperature=training_config['temperature'],
        sinkhorn_iterations=training_config['sinkhorn_iterations'],
        alpha_init=training_config['alpha_init'],
    )
    try:
        point_encoder.load_state_dict(payload['point_encoder_state_dict'], strict=True)
        ct_encoder.load_state_dict(payload['ct_encoder_state_dict'], strict=True)
        matcher.load_state_dict(payload['matcher_state_dict'], strict=True)
    except Exception as error:
        raise M3EvaluationContractError(
            f'checkpoint model state restoration failed: {error}'
        ) from error
    modules = (point_encoder, ct_encoder, matcher)
    for module in modules:
        module.to(device=device, dtype=torch.float32)
        module.eval()
    return modules


def _synchronize(device) -> None:
    if device.type == 'cuda':
        import torch

        torch.cuda.synchronize(device)


def _run_inference_case(
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    evaluation_protocol,
    device,
) -> dict:
    """Run the frozen inference/matching/filtering/registration chain once."""
    import torch

    from dataset import m2_ct_collate_fn, m2_point_collate_fn
    from matching_filter import extract_dustbin_aware_mutual_correspondences
    from registration import estimate_weighted_point_to_ct_transform
    from training import _prepare_model_inputs, _require_encoder_outputs

    point_input, ct_input = _prepare_model_inputs(
        sample,
        m2_point_collate_fn,
        m2_ct_collate_fn,
        device,
    )

    _synchronize(device)
    start = time.perf_counter()
    point_output = point_encoder(point_input)
    ct_output = ct_encoder(ct_input)
    q, k, point_physical, ct_physical = _require_encoder_outputs(
        point_output,
        ct_output,
        device,
    )
    point_valid_mask = torch.ones((q.shape[0],), dtype=torch.bool, device=device)
    ct_valid_mask = torch.ones((k.shape[0],), dtype=torch.bool, device=device)
    matcher_output = matcher(
        q=q,
        k=k,
        point_valid_mask=point_valid_mask,
        ct_valid_mask=ct_valid_mask,
    )
    filter_output = extract_dustbin_aware_mutual_correspondences(
        matcher_output['log_assignment'],
        point_valid_mask=matcher_output['point_valid_mask'],
        ct_valid_mask=matcher_output['ct_valid_mask'],
        min_confidence=evaluation_protocol['matching_filter_min_confidence'],
    )
    registration_output = estimate_weighted_point_to_ct_transform(
        point_physical,
        ct_physical,
        filter_output['point_indices'],
        filter_output['ct_indices'],
        filter_output['confidence'],
    )
    if int(registration_output['num_correspondences']) != int(
        filter_output['num_correspondences']
    ):
        raise M3EvaluationContractError(
            'weighted registration correspondence count does not match mutual filtering.'
        )
    _synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if not math.isfinite(runtime_ms) or runtime_ms < 0.0:
        raise M3EvaluationContractError('inference runtime is invalid.')
    return {
        'point_physical': point_physical,
        'ct_physical': ct_physical,
        'filter_output': filter_output,
        'registration_output': registration_output,
        'inference_runtime_ms': runtime_ms,
    }


def _evaluate_test_case(
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
    subject_id = raw_sample.get('subject_id') if isinstance(raw_sample, Mapping) else None
    if subject_id != manifest_case['subject_id']:
        raise M3EvaluationContractError(
            'dataset subject does not match the formal test manifest: '
            f'expected={manifest_case["subject_id"]!r}, actual={subject_id!r}.'
        )
    augmented = augment_point_sample(
        raw_sample,
        seed=manifest_case['perturbation_seed'],
        max_rotation_deg=manifest_case['max_rotation_deg'],
        max_translation_mm=manifest_case['max_translation_mm'],
        scheme_version=training_protocol['seed_scheme_version'],
    )
    effective_gt = np.asarray(augmented['gt_transform'], dtype=np.float64)

    inference = _run_inference_case(
        augmented,
        point_encoder,
        ct_encoder,
        matcher,
        evaluation_protocol,
        device,
    )
    filter_output = inference['filter_output']
    inlier_metrics = correspondence_inlier_metrics(
        inference['point_physical'],
        inference['ct_physical'],
        filter_output['point_indices'],
        filter_output['ct_indices'],
        effective_gt,
        evaluation_protocol['correspondence_inlier_threshold_mm'],
    )
    registration_metrics = evaluate_registration_result(
        inference['registration_output'],
        effective_gt,
        evaluation_protocol['registration_rre_threshold_deg'],
        evaluation_protocol['registration_rte_threshold_mm'],
    )
    if (
        inlier_metrics['num_correspondences'] == 0
        and registration_metrics['solver_success']
    ):
        raise M3EvaluationContractError(
            'zero-correspondence registration must fail closed.'
        )

    return {
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol['evaluation_protocol_hash'],
        'fold_id': manifest_case['fold_id'],
        'subject_id': manifest_case['subject_id'],
        'severity': manifest_case['severity'],
        'variant_id': manifest_case['variant_id'],
        'perturbation_seed': manifest_case['perturbation_seed'],
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        **inlier_metrics,
        **registration_metrics,
        'inference_runtime_ms': inference['inference_runtime_ms'],
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            prefix=f'.{path.name}.',
            suffix='.tmp',
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(temporary_path, path)
    except Exception as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if isinstance(error, M3EvaluationContractError):
            raise
        raise M3EvaluationContractError(f'cannot write JSON {path}: {error}') from error


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            prefix=f'.{path.name}.',
            suffix='.tmp',
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        os.replace(temporary_path, path)
    except Exception as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if isinstance(error, M3EvaluationContractError):
            raise
        raise M3EvaluationContractError(f'cannot write JSONL {path}: {error}') from error


def run_manifest_only(
    training_protocol,
    evaluation_protocol,
    fold_id: str,
    output_dir,
) -> dict:
    """Generate one Fold manifest without dataset, checkpoint, model, or GPU access."""
    manifest = build_test_manifest(training_protocol, evaluation_protocol, fold_id)
    output_path = Path(output_dir) / 'test_manifest.json'
    _write_json(output_path, manifest)
    return {
        'mode': 'manifest-only',
        'fold_id': fold_id,
        'test_subject_ids': manifest['test_subject_ids'],
        'total_cases': manifest['total_cases'],
        'cases_per_subject': manifest['cases_per_subject'],
        'test_manifest_path': str(output_path.resolve()),
    }


def run_execute_test(
    *,
    data_root,
    checkpoint_path,
    output_dir,
    device_name: str,
    training_protocol,
    evaluation_protocol,
    fold_id: str,
) -> dict:
    """Execute formal test cases; callable only through explicit opt-in."""
    payload, checkpoint_metadata = read_and_validate_evaluation_checkpoint(
        checkpoint_path,
        training_protocol,
        evaluation_protocol,
        fold_id,
    )

    import torch

    from train_m3 import resolve_device
    from training import build_formal_subject_split

    device = resolve_device(device_name)
    point_encoder, ct_encoder, matcher = _build_models_from_checkpoint(
        payload,
        checkpoint_metadata,
        device,
    )
    dataset = _create_dataset_for_execution(data_root)
    split = build_formal_subject_split(dataset, training_protocol, fold_id)
    manifest = build_test_manifest(training_protocol, evaluation_protocol, fold_id)
    subject_to_index = dict(zip(split.test_subject_ids, split.test_indices))

    output_dir = Path(output_dir)
    _write_json(output_dir / 'test_manifest.json', manifest)
    cases = []
    with torch.no_grad():
        for manifest_case in manifest['cases']:
            subject_id = manifest_case['subject_id']
            raw_sample = dataset[subject_to_index[subject_id]]
            cases.append(
                _evaluate_test_case(
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

    summary = {
        'protocol_version': training_protocol['protocol_version'],
        'protocol_hash': training_protocol['protocol_hash'],
        'evaluation_protocol_version': evaluation_protocol[
            'evaluation_protocol_version'
        ],
        'evaluation_protocol_hash': evaluation_protocol['evaluation_protocol_hash'],
        'fold_id': fold_id,
        'test_subject_ids': list(split.test_subject_ids),
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        **aggregate_case_metrics(cases),
    }
    _write_jsonl(output_dir / 'cases.jsonl', cases)
    _write_json(output_dir / 'summary.json', summary)
    return summary


def run_evaluation(args) -> dict:
    """Resolve the explicitly selected safe mode and run it."""
    manifest_only = bool(getattr(args, 'manifest_only', False))
    execute_test = bool(getattr(args, 'execute_test', False))
    if manifest_only == execute_test:
        raise M3EvaluationContractError(
            'Explicitly choose exactly one mode: --manifest-only or --execute-test. '
            'Real test execution is disabled by default.'
        )

    training_protocol = load_training_protocol(args.protocol_manifest)
    evaluation_protocol = load_evaluation_protocol(args.evaluation_protocol)
    _validate_protocol_pair(training_protocol, evaluation_protocol)
    resolve_fold(training_protocol, args.fold_id)

    if manifest_only:
        return run_manifest_only(
            training_protocol,
            evaluation_protocol,
            args.fold_id,
            args.output_dir,
        )
    if args.checkpoint is None:
        raise M3EvaluationContractError('--execute-test requires --checkpoint.')
    return run_execute_test(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        training_protocol=training_protocol,
        evaluation_protocol=evaluation_protocol,
        fold_id=args.fold_id,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='M3-7 formal Point-CT evaluation infrastructure.'
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--protocol-manifest', type=Path, required=True)
    parser.add_argument('--evaluation-protocol', type=Path, required=True)
    parser.add_argument('--fold-id', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    modes = parser.add_mutually_exclusive_group()
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
    'TEST_MANIFEST_VERSION',
    'build_argument_parser',
    'build_test_manifest',
    'main',
    'read_and_validate_evaluation_checkpoint',
    'run_evaluation',
    'run_execute_test',
    'run_manifest_only',
    'validate_evaluation_checkpoint_metadata',
]
