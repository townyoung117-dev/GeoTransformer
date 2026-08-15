"""CLI orchestration for manual M3-6A and formal M3-6B training."""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Mapping

import torch

from config import DEFAULT_DATA_ROOT, make_cfg
from ct_encoder import CTEncoder
from dataset import create_dataset, m2_ct_collate_fn, m2_point_collate_fn
from matching import PointCTMatcher
from perturbation import augment_point_sample, derive_perturbation_seed
from point_encoder import PointEncoder
from training import (
    BATCH_SIZE,
    PRECISION,
    SMOKE_SPLIT_STATUS,
    TRAINING_DEFAULTS_STATUS,
    TRAINING_SMOKE_DEFAULT_LEARNING_RATE,
    TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY,
    M3TrainingContractError,
    TrainingConfig,
    aggregate_step_results,
    build_formal_subject_split,
    build_subject_split,
    create_optimizer,
    load_checkpoint,
    run_training_step,
    run_validation_step,
    save_epoch_checkpoints,
    set_random_seed,
    validate_training_config,
)
from training_protocol import (
    load_training_protocol,
    resolve_fold,
    train_perturbation_spec,
    validation_perturbation_specs,
)


MANUAL_PROTOCOL_STATUS = 'SMOKE/MANUAL SPLIT - NOT FORMAL EVALUATION PROTOCOL'
FORMAL_PROTOCOL_STATUS = 'FORMAL M3-6B 5-FOLD PROTOCOL'


def resolve_device(device_name):
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise M3TrainingContractError(f'invalid device: {device_name!r}.') from error
    if device.type not in {'cpu', 'cuda'}:
        raise M3TrainingContractError('M3-6A supports only CPU or CUDA FP32 execution.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise M3TrainingContractError('CUDA was requested but is unavailable.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        elif device.index < 0 or device.index >= torch.cuda.device_count():
            raise M3TrainingContractError(f'requested CUDA device does not exist: {device}.')
        torch.cuda.set_device(device)
    return device


def _require_positive_integer(value, name):
    if isinstance(value, bool):
        raise M3TrainingContractError(f'{name} must be a positive integer.')
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3TrainingContractError(f'{name} must be a positive integer.') from error
    if integer <= 0 or integer != value:
        raise M3TrainingContractError(f'{name} must be a positive integer.')
    return integer


def _training_config_record(
    training_config,
    *,
    seed,
    temperature,
    sinkhorn_iterations,
    alpha_init,
    primary_max_distance_mm,
    high_confidence_distance_mm,
    split_status,
):
    return {
        'learning_rate': training_config.learning_rate,
        'weight_decay': training_config.weight_decay,
        'batch_size': training_config.batch_size,
        'precision': training_config.precision,
        'seed': seed,
        'temperature': float(temperature),
        'sinkhorn_iterations': int(sinkhorn_iterations),
        'alpha_init': float(alpha_init),
        'primary_max_distance_mm': float(primary_max_distance_mm),
        'high_confidence_distance_mm': float(high_confidence_distance_mm),
        'hyperparameter_status': TRAINING_DEFAULTS_STATUS,
        'split_status': split_status,
    }


def resolve_training_mode(args):
    protocol_manifest = getattr(args, 'protocol_manifest', None)
    fold_id = getattr(args, 'fold_id', None)
    train_subjects = getattr(args, 'train_subjects', None)
    val_subjects = getattr(args, 'val_subjects', None)
    has_formal = protocol_manifest is not None or fold_id is not None
    has_manual = train_subjects is not None or val_subjects is not None
    if has_formal and has_manual:
        raise M3TrainingContractError(
            'formal --protocol-manifest/--fold-id mode is mutually exclusive with '
            '--train-subjects/--val-subjects.'
        )
    if has_formal:
        if protocol_manifest is None or fold_id is None:
            raise M3TrainingContractError(
                'formal mode requires both --protocol-manifest and --fold-id.'
            )
        return 'formal'
    if has_manual:
        if train_subjects is None or val_subjects is None:
            raise M3TrainingContractError(
                'manual mode requires both --train-subjects and --val-subjects.'
            )
        return 'manual'
    raise M3TrainingContractError(
        'choose formal --protocol-manifest/--fold-id or manual '
        '--train-subjects/--val-subjects mode.'
    )


def _resolve_training_contract(args, dataset):
    mode = resolve_training_mode(args)
    if mode == 'formal':
        protocol = load_training_protocol(args.protocol_manifest)
        split = build_formal_subject_split(dataset, protocol, args.fold_id)
        return mode, protocol, split
    split = build_subject_split(dataset, args.train_subjects, args.val_subjects)
    return mode, None, split


def _raw_subject_id(raw_sample, expected_subject_id=None):
    subject_id = raw_sample.get('subject_id') if isinstance(raw_sample, Mapping) else None
    if (
        not isinstance(subject_id, str)
        or not subject_id.strip()
        or subject_id != subject_id.strip()
    ):
        raise M3TrainingContractError('raw dataset sample requires a subject_id.')
    if expected_subject_id is not None and subject_id != expected_subject_id:
        raise M3TrainingContractError(
            f'dataset sample subject_id {subject_id!r} does not match ready record '
            f'{expected_subject_id!r}.'
        )
    return subject_id


def derive_train_sample_seed(protocol, fold_id, epoch, subject_id):
    fold = resolve_fold(protocol, fold_id)
    if subject_id not in fold['train_subject_ids']:
        raise M3TrainingContractError(
            f'subject {subject_id!r} is not a training subject in {fold_id}.'
        )
    spec = train_perturbation_spec(protocol)
    return derive_perturbation_seed(
        scheme_version=protocol['seed_scheme_version'],
        protocol_hash=protocol['protocol_hash'],
        fold_id=fold_id,
        root_seed=protocol['perturbation_root_seed'],
        purpose=spec['purpose'],
        epoch=epoch,
        subject_id=subject_id,
        severity=spec['severity'],
        variant_id=0,
    )


def augment_formal_training_sample(raw_sample, protocol, fold_id, epoch):
    subject_id = _raw_subject_id(raw_sample)
    spec = train_perturbation_spec(protocol)
    seed = derive_train_sample_seed(protocol, fold_id, epoch, subject_id)
    return augment_point_sample(
        raw_sample,
        seed=seed,
        max_rotation_deg=spec['max_rotation_deg'],
        max_translation_mm=spec['max_translation_mm'],
        scheme_version=protocol['seed_scheme_version'],
    )


def iter_formal_validation_samples(dataset, split, protocol):
    specs = validation_perturbation_specs(protocol)
    for subject_id, dataset_index in zip(split.val_subject_ids, split.val_indices):
        raw_sample = dataset[dataset_index]
        _raw_subject_id(raw_sample, expected_subject_id=subject_id)
        for spec in specs:
            seed = derive_perturbation_seed(
                scheme_version=protocol['seed_scheme_version'],
                protocol_hash=protocol['protocol_hash'],
                fold_id=split.fold_id,
                root_seed=protocol['perturbation_root_seed'],
                purpose=spec['purpose'],
                epoch=None,
                subject_id=subject_id,
                severity=spec['severity'],
                variant_id=0,
            )
            yield subject_id, spec['severity'], augment_point_sample(
                raw_sample,
                seed=seed,
                max_rotation_deg=spec['max_rotation_deg'],
                max_translation_mm=spec['max_translation_mm'],
                scheme_version=protocol['seed_scheme_version'],
            )


def _append_json_log(path, record):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')


def _checkpoint_fields(
    *,
    epoch,
    global_step,
    point_encoder,
    ct_encoder,
    matcher,
    optimizer,
    split,
    seed,
    training_config,
    protocol,
):
    fields = {
        'epoch': epoch,
        'global_step': global_step,
        'point_encoder': point_encoder,
        'ct_encoder': ct_encoder,
        'matcher': matcher,
        'optimizer': optimizer,
        'train_subject_ids': split.train_subject_ids,
        'val_subject_ids': split.val_subject_ids,
        'seed': seed,
        'training_config': training_config,
    }
    if protocol is not None:
        fields.update(
            {
                'formal_protocol': True,
                'protocol_version': protocol['protocol_version'],
                'protocol_hash': protocol['protocol_hash'],
                'fold_id': split.fold_id,
                'test_subject_ids': split.test_subject_ids,
                'perturbation_root_seed': protocol['perturbation_root_seed'],
                'perturbation_seed_scheme_version': protocol['seed_scheme_version'],
            }
        )
    return fields


def train(args):
    epochs = _require_positive_integer(args.epochs, 'epochs')
    mode = resolve_training_mode(args)
    seed = set_random_seed(args.seed)
    device = resolve_device(args.device)
    training_config = validate_training_config(
        TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            precision=args.precision,
        )
    )
    cfg = make_cfg()
    dataset = create_dataset(args.data_root)
    resolved_mode, protocol, split = _resolve_training_contract(args, dataset)
    if resolved_mode != mode:
        raise M3TrainingContractError('training mode resolution changed unexpectedly.')
    split_status = FORMAL_PROTOCOL_STATUS if mode == 'formal' else MANUAL_PROTOCOL_STATUS

    point_encoder = PointEncoder(cfg).to(device=device, dtype=torch.float32)
    ct_encoder = CTEncoder(cfg).to(device=device, dtype=torch.float32)
    matcher = PointCTMatcher(
        projected_dim=cfg.point.projected_dim,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        alpha_init=args.alpha_init,
    ).to(device=device, dtype=torch.float32)
    optimizer = create_optimizer(
        point_encoder,
        ct_encoder,
        matcher,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    config_record = _training_config_record(
        training_config,
        seed=seed,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        alpha_init=args.alpha_init,
        primary_max_distance_mm=cfg.gt_coarse.primary_max_distance_mm,
        high_confidence_distance_mm=cfg.gt_coarse.high_confidence_distance_mm,
        split_status=split_status,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    if args.resume is not None:
        formal_resume_fields = {}
        if protocol is not None:
            formal_resume_fields = {
                'formal_protocol': True,
                'protocol_version': protocol['protocol_version'],
                'protocol_hash': protocol['protocol_hash'],
                'fold_id': split.fold_id,
                'test_subject_ids': split.test_subject_ids,
                'perturbation_root_seed': protocol['perturbation_root_seed'],
                'perturbation_seed_scheme_version': protocol['seed_scheme_version'],
            }
        resume_state = load_checkpoint(
            args.resume,
            point_encoder,
            ct_encoder,
            matcher,
            optimizer,
            train_subject_ids=split.train_subject_ids,
            val_subject_ids=split.val_subject_ids,
            map_location=device,
            expected_training_config=config_record,
            **formal_resume_fields,
        )
        start_epoch = resume_state['epoch'] + 1
        global_step = resume_state['global_step']
        best_val_loss = resume_state['best_val_loss']
    if start_epoch >= epochs:
        raise M3TrainingContractError(
            f'checkpoint already reached epoch {start_epoch - 1}; target epochs={epochs}.'
        )

    print(TRAINING_DEFAULTS_STATUS)
    if mode == 'manual':
        print(SMOKE_SPLIT_STATUS)
    print(split_status)
    if protocol is not None:
        print(
            f'protocol_version={protocol["protocol_version"]} '
            f'protocol_hash={protocol["protocol_hash"]} fold_id={split.fold_id}'
        )
    print(
        f'device={device} batch_size={BATCH_SIZE} precision={PRECISION} seed={seed} '
        f'learning_rate={training_config.learning_rate:g} '
        f'weight_decay={training_config.weight_decay:g}'
    )
    print(
        f'train_subject_ids={list(split.train_subject_ids)} '
        f'val_subject_ids={list(split.val_subject_ids)}'
    )
    if protocol is not None:
        print(f'test_subject_ids={list(split.test_subject_ids)} (manifest only; not executed)')

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_indices = list(split.train_indices)
        random.Random(seed + epoch).shuffle(train_indices)

        point_encoder.train()
        ct_encoder.train()
        matcher.train()
        train_results = []
        for dataset_index in train_indices:
            raw_sample = dataset[dataset_index]
            if protocol is not None:
                expected_subject_id = dataset.records[dataset_index]['subject_id']
                _raw_subject_id(raw_sample, expected_subject_id=expected_subject_id)
                step_sample = augment_formal_training_sample(
                    raw_sample,
                    protocol,
                    split.fold_id,
                    epoch,
                )
            else:
                step_sample = raw_sample
            result = run_training_step(
                step_sample,
                point_encoder,
                ct_encoder,
                matcher,
                optimizer,
                point_collate_fn=m2_point_collate_fn,
                ct_collate_fn=m2_ct_collate_fn,
                primary_max_distance_mm=cfg.gt_coarse.primary_max_distance_mm,
                high_confidence_distance_mm=cfg.gt_coarse.high_confidence_distance_mm,
                device=device,
            )
            train_results.append({**result, 'loss': float(result['loss'].item())})
            global_step += 1

        point_encoder.eval()
        ct_encoder.eval()
        matcher.eval()
        val_results = []
        val_results_by_severity = {}
        if protocol is not None:
            validation_cases = iter_formal_validation_samples(dataset, split, protocol)
        else:
            validation_cases = (
                (
                    dataset.records[dataset_index]['subject_id'],
                    'manual',
                    dataset[dataset_index],
                )
                for dataset_index in split.val_indices
            )
        for _, severity, validation_sample in validation_cases:
            result = run_validation_step(
                validation_sample,
                point_encoder,
                ct_encoder,
                matcher,
                point_collate_fn=m2_point_collate_fn,
                ct_collate_fn=m2_ct_collate_fn,
                primary_max_distance_mm=cfg.gt_coarse.primary_max_distance_mm,
                high_confidence_distance_mm=cfg.gt_coarse.high_confidence_distance_mm,
                device=device,
            )
            recorded_result = {**result, 'loss': float(result['loss'].item())}
            val_results.append(recorded_result)
            val_results_by_severity.setdefault(severity, []).append(recorded_result)

        train_stats = aggregate_step_results(train_results)
        val_stats = aggregate_step_results(val_results)
        checkpoint_result = save_epoch_checkpoints(
            args.checkpoint_dir,
            val_loss=val_stats['mean_loss'],
            best_val_loss=best_val_loss,
            **_checkpoint_fields(
                epoch=epoch,
                global_step=global_step,
                point_encoder=point_encoder,
                ct_encoder=ct_encoder,
                matcher=matcher,
                optimizer=optimizer,
                split=split,
                seed=seed,
                training_config=config_record,
                protocol=protocol,
            ),
        )
        best_val_loss = checkpoint_result['best_val_loss']
        val_severity_means = {
            severity: aggregate_step_results(val_results_by_severity[severity])['mean_loss']
            if severity in val_results_by_severity
            else None
            for severity in ('mild', 'moderate', 'hard')
        }
        record = {
            'epoch': epoch,
            'global_step': global_step,
            'formal_protocol': protocol is not None,
            'split_status': split_status,
            'train_mean_loss': train_stats['mean_loss'],
            'val_mean_loss': val_stats['mean_loss'],
            'num_train_cases': train_stats['num_cases'],
            'num_val_cases': val_stats['num_cases'],
            'num_val_subjects': len(split.val_subject_ids),
            'num_val_perturbations': val_stats['num_cases'],
            'val_mild_mean_loss': val_severity_means['mild'],
            'val_moderate_mean_loss': val_severity_means['moderate'],
            'val_hard_mean_loss': val_severity_means['hard'],
            'train_supervised_points': train_stats['num_supervised_points'],
            'train_supervised_groups': train_stats['num_supervised_groups'],
            'train_collision_groups': train_stats['num_collision_groups'],
            'val_supervised_points': val_stats['num_supervised_points'],
            'val_supervised_groups': val_stats['num_supervised_groups'],
            'val_collision_groups': val_stats['num_collision_groups'],
            'runtime_seconds': time.perf_counter() - epoch_start,
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'best_val_loss': best_val_loss,
            'best_checkpoint_updated': checkpoint_result['improved'],
        }
        if protocol is not None:
            record.update(
                {
                    'protocol_version': protocol['protocol_version'],
                    'protocol_hash': protocol['protocol_hash'],
                    'fold_id': split.fold_id,
                    'test_subject_ids': list(split.test_subject_ids),
                }
            )
        print(json.dumps(record, sort_keys=True))
        _append_json_log(args.json_log, record)
    return {
        'epoch': epochs - 1,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'formal_protocol': protocol is not None,
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='M3 batch-size-one FP32 Point-CT training infrastructure.'
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        '--protocol-manifest',
        type=Path,
        help='Formal M3-6B protocol manifest; requires --fold-id.',
    )
    parser.add_argument('--fold-id', help='Formal fold key, for example Fold1.')
    parser.add_argument(
        '--train-subjects',
        nargs='+',
        help='Manual engineering-smoke subjects; mutually exclusive with formal mode.',
    )
    parser.add_argument(
        '--val-subjects',
        nargs='+',
        help='Manual engineering-smoke subjects; mutually exclusive with formal mode.',
    )
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--json-log', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=TRAINING_SMOKE_DEFAULT_LEARNING_RATE,
        help='Training smoke default only; not a frozen paper hyperparameter.',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY,
        help='Training smoke default only; not a frozen paper hyperparameter.',
    )
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--precision', default=PRECISION)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--sinkhorn-iterations', type=int, required=True)
    parser.add_argument('--alpha-init', type=float, required=True)
    return parser


def main():
    args = build_argument_parser().parse_args()
    train(args)


if __name__ == '__main__':
    main()
