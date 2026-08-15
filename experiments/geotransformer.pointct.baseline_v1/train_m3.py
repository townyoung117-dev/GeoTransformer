"""CLI orchestration for the M3-6A Point-CT training contract."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from config import DEFAULT_DATA_ROOT, make_cfg
from ct_encoder import CTEncoder
from dataset import create_dataset, m2_ct_collate_fn, m2_point_collate_fn
from matching import PointCTMatcher
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
    build_subject_split,
    create_optimizer,
    load_checkpoint,
    run_training_step,
    run_validation_step,
    save_epoch_checkpoints,
    set_random_seed,
    validate_training_config,
)


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
        'split_status': SMOKE_SPLIT_STATUS,
    }


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
):
    return {
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


def train(args):
    epochs = _require_positive_integer(args.epochs, 'epochs')
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
    split = build_subject_split(dataset, args.train_subjects, args.val_subjects)

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
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    if args.resume is not None:
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
        )
        start_epoch = resume_state['epoch'] + 1
        global_step = resume_state['global_step']
        best_val_loss = resume_state['best_val_loss']
    if start_epoch >= epochs:
        raise M3TrainingContractError(
            f'checkpoint already reached epoch {start_epoch - 1}; target epochs={epochs}.'
        )

    print(TRAINING_DEFAULTS_STATUS)
    print(SMOKE_SPLIT_STATUS)
    print(
        f'device={device} batch_size={BATCH_SIZE} precision={PRECISION} seed={seed} '
        f'learning_rate={training_config.learning_rate:g} '
        f'weight_decay={training_config.weight_decay:g}'
    )
    print(
        f'train_subject_ids={list(split.train_subject_ids)} '
        f'val_subject_ids={list(split.val_subject_ids)}'
    )

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_indices = list(split.train_indices)
        random.Random(seed + epoch).shuffle(train_indices)

        point_encoder.train()
        ct_encoder.train()
        matcher.train()
        train_results = []
        for dataset_index in train_indices:
            result = run_training_step(
                dataset[dataset_index],
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
        for dataset_index in split.val_indices:
            result = run_validation_step(
                dataset[dataset_index],
                point_encoder,
                ct_encoder,
                matcher,
                point_collate_fn=m2_point_collate_fn,
                ct_collate_fn=m2_ct_collate_fn,
                primary_max_distance_mm=cfg.gt_coarse.primary_max_distance_mm,
                high_confidence_distance_mm=cfg.gt_coarse.high_confidence_distance_mm,
                device=device,
            )
            val_results.append({**result, 'loss': float(result['loss'].item())})

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
            ),
        )
        best_val_loss = checkpoint_result['best_val_loss']
        record = {
            'epoch': epoch,
            'global_step': global_step,
            'train_mean_loss': train_stats['mean_loss'],
            'val_mean_loss': val_stats['mean_loss'],
            'num_train_cases': train_stats['num_cases'],
            'num_val_cases': val_stats['num_cases'],
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
        print(json.dumps(record, sort_keys=True))
        _append_json_log(args.json_log, record)
    return {
        'epoch': epochs - 1,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='M3-6A batch-size-one FP32 Point-CT training infrastructure.'
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--train-subjects', nargs='+', required=True)
    parser.add_argument('--val-subjects', nargs='+', required=True)
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
