"""Validate the frozen M2-3 coarse GT correspondence on all ready cases."""

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_DATA_ROOT, make_cfg
from dataset import create_dataset, m2_ct_collate_fn, m2_point_collate_fn
from gt_correspondence import build_coarse_gt_correspondence


EXPECTED_READY_SUBJECTS = 11
EXPECTED_SKIPPED_SUBJECT = 'Pat10'

# Broad audit guards catch coordinate-unit, transform-direction, or support
# regressions without turning the measured values into exact test constants.
EMPIRICAL_SANITY_RANGES = {
    'total_point_tokens': (1200, 1550),
    'high_confidence_coverage': (0.90, 0.99),
    'p50_mm': (8.5, 12.5),
    'p90_mm': (12.0, 16.0),
    'p95_mm': (13.0, 17.0),
    'p99_mm': (14.5, 17.5),
    'max_mm': (16.0, 17.5),
}


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _point_coarse_physical_mm(point_dict, scale_mm_to_m: float) -> np.ndarray:
    points = point_dict.get('points')
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise RuntimeError('Formal M2-1 preprocessing must provide four Point stages.')
    coarse_network = points[3]
    if hasattr(coarse_network, 'detach'):
        coarse_physical = coarse_network / scale_mm_to_m
        coarse_physical = _to_numpy(coarse_physical)
    else:
        coarse_physical = np.asarray(coarse_network) / np.float32(scale_mm_to_m)
    if coarse_physical.ndim != 2 or coarse_physical.shape[1] != 3:
        raise RuntimeError(
            f'Formal M2-1 coarse Point coordinates must have shape [Np,3]; got {coarse_physical.shape}.'
        )
    return coarse_physical


def _percentiles(distances: np.ndarray):
    return {
        'p50': float(np.percentile(distances, 50)),
        'p90': float(np.percentile(distances, 90)),
        'p95': float(np.percentile(distances, 95)),
        'p99': float(np.percentile(distances, 99)),
        'max': float(np.max(distances)),
    }


def _require_sanity(name: str, value: float):
    lower, upper = EMPIRICAL_SANITY_RANGES[name]
    if not lower <= value <= upper:
        raise RuntimeError(
            f'Global empirical sanity check failed for {name}: {value:.6f} is outside '
            f'[{lower}, {upper}]. Inspect the formal preprocessing and coordinate contracts.'
        )


def validate_gt_correspondence(data_root):
    cfg = make_cfg()
    dataset = create_dataset(data_root)
    if len(dataset) != EXPECTED_READY_SUBJECTS:
        raise RuntimeError(
            f'Expected {EXPECTED_READY_SUBJECTS} ready subjects; manifest selected {len(dataset)}.'
        )
    skipped_subject_ids = {record['subject_id'] for record in dataset.skipped_records}
    if EXPECTED_SKIPPED_SUBJECT not in skipped_subject_ids:
        raise RuntimeError(f'{EXPECTED_SKIPPED_SUBJECT} must be skipped by the manifest.')

    all_distances = []
    total_valid = 0
    total_high_confidence = 0
    total_point_tokens = 0
    passed = 0

    for case_index in range(len(dataset)):
        # Each formal collate receives exactly one raw case. Neither encoder
        # forward is needed: only the frozen preprocessed coarse coordinates
        # are consumed below.
        sample = dataset[case_index]
        point_batch = m2_point_collate_fn([sample])
        ct_batch = m2_ct_collate_fn([sample])
        subject_id = sample['subject_id']
        if point_batch['subject_id'] != subject_id or ct_batch['subject_id'] != subject_id:
            raise RuntimeError(f'{subject_id}: preprocessing changed the case identity.')
        if point_batch['physical_unit'] != 'mm' or ct_batch['physical_unit'] != 'mm':
            raise RuntimeError(f'{subject_id}: M2-3 requires both formal branches in physical mm.')

        point_phys = _point_coarse_physical_mm(
            point_batch['point'],
            cfg.point.point_network_scale_mm_to_m,
        )
        ct_phys = _to_numpy(ct_batch['ct']['ct_support_phys_20mm'])
        correspondence = build_coarse_gt_correspondence(
            Xp_phys_coarse=point_phys,
            Xv_phys_coarse=ct_phys,
            gt_transform=point_batch['gt_transform'],
            gt_transform_direction=point_batch['gt_transform_direction'],
            primary_max_distance_mm=cfg.gt_coarse.primary_max_distance_mm,
            high_confidence_distance_mm=cfg.gt_coarse.high_confidence_distance_mm,
        )

        point_count = int(correspondence['Xp_phys_coarse'].shape[0])
        ct_count = int(correspondence['Xv_phys_coarse'].shape[0])
        distances = correspondence['gt_primary_distance_mm']
        valid_count = int(np.count_nonzero(correspondence['gt_primary_valid']))
        high_confidence_count = int(np.count_nonzero(correspondence['gt_high_confidence']))
        if valid_count != point_count:
            invalid_count = point_count - valid_count
            raise RuntimeError(
                f'{subject_id}: {invalid_count}/{point_count} primary GT distances exceed '
                f'{cfg.gt_coarse.primary_max_distance_mm:.1f} mm.'
            )

        stats = _percentiles(distances)
        print(
            f'{subject_id}: Np={point_count} Nv={ct_count} '
            f'valid={valid_count}/{point_count} '
            f'high_confidence={high_confidence_count}/{point_count} '
            f'distance_mm[p50={stats["p50"]:.3f}, p90={stats["p90"]:.3f}, '
            f'p95={stats["p95"]:.3f}, max={stats["max"]:.3f}]'
        )

        all_distances.append(distances)
        total_point_tokens += point_count
        total_valid += valid_count
        total_high_confidence += high_confidence_count
        passed += 1

    if passed != EXPECTED_READY_SUBJECTS:
        raise RuntimeError(f'Only {passed}/{EXPECTED_READY_SUBJECTS} subjects completed validation.')

    global_distances = np.concatenate(all_distances)
    global_stats = _percentiles(global_distances)
    valid_coverage = total_valid / total_point_tokens
    high_confidence_coverage = total_high_confidence / total_point_tokens
    print(
        f'Global: total_point_tokens={total_point_tokens} '
        f'valid_coverage={valid_coverage:.4%} '
        f'high_confidence_coverage={high_confidence_coverage:.4%} '
        f'distance_mm[p50={global_stats["p50"]:.3f}, p90={global_stats["p90"]:.3f}, '
        f'p95={global_stats["p95"]:.3f}, p99={global_stats["p99"]:.3f}, '
        f'max={global_stats["max"]:.3f}]'
    )

    if total_valid != total_point_tokens:
        raise RuntimeError('Global valid coverage must be 100% for the current 11-case audit dataset.')
    _require_sanity('total_point_tokens', total_point_tokens)
    _require_sanity('high_confidence_coverage', high_confidence_coverage)
    for stat_name in ('p50', 'p90', 'p95', 'p99', 'max'):
        _require_sanity(f'{stat_name}_mm', global_stats[stat_name])

    print('11/11 M2 GT Coarse Correspondence validation: PASS')


def main():
    parser = argparse.ArgumentParser(
        description='Validate M2-3 physical nearest-one coarse GT labels on all ready subjects.'
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    validate_gt_correspondence(args.data_root)


if __name__ == '__main__':
    main()
