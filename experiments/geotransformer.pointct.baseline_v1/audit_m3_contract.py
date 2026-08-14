"""Audit M3-1 token, GT-primary, collision, and protocol contracts.

This module is deliberately statistics-only.  Real-case geometry is obtained
from the frozen M2 preprocessors and the frozen M2-3 nearest-one builder.
"""

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geotransformer.datasets.registration.pointct import find_subject_split_leakage

from config import DEFAULT_DATA_ROOT, make_cfg
from dataset import create_dataset, m2_ct_collate_fn, m2_point_collate_fn
from gt_correspondence import POINT_TO_CT_DIRECTION, build_coarse_gt_correspondence


EXPECTED_READY_SUBJECTS = 11
EXPECTED_SKIPPED_SUBJECT = 'Pat10'
FLOAT32_BYTES = 4
MEBIBYTE_BYTES = 1024 ** 2
DISTANCE_THRESHOLDS_MM = (5.0, 10.0, 15.0, 17.5, 20.0)
DISTANCE_PERCENTILES = (50, 75, 90, 95, 99)
IDENTITY_ATOL = 1e-6


class M3ContractAuditError(RuntimeError):
    pass


def _require_positive_count(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise M3ContractAuditError(f'{name} must be a positive integer.')
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3ContractAuditError(f'{name} must be a positive integer.') from error
    if count <= 0 or count != value:
        raise M3ContractAuditError(f'{name} must be a positive integer; got {value!r}.')
    return count


def calculate_matrix_memory(num_point_tokens, num_ct_tokens) -> Dict[str, float]:
    """Return single-matrix float32 sizes; these are not peak GPU memory."""
    num_point_tokens = _require_positive_count(num_point_tokens, 'num_point_tokens')
    num_ct_tokens = _require_positive_count(num_ct_tokens, 'num_ct_tokens')
    matrix_elements = num_point_tokens * num_ct_tokens
    dustbin_elements = (num_point_tokens + 1) * (num_ct_tokens + 1)
    return {
        'matrix_elements': matrix_elements,
        'raw_similarity_float32_MiB': matrix_elements * FLOAT32_BYTES / MEBIBYTE_BYTES,
        'dustbin_matrix_float32_MiB': dustbin_elements * FLOAT32_BYTES / MEBIBYTE_BYTES,
    }


def _require_index_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise M3ContractAuditError(f'{name} must be a one-dimensional integer array.')
    return array.astype(np.int64, copy=False)


def _require_bool_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.bool_:
        raise M3ContractAuditError(f'{name} must be a one-dimensional boolean array.')
    return array


def compute_primary_statistics(
    gt_primary_ct_index,
    gt_primary_valid,
    gt_high_confidence=None,
    *,
    num_ct_tokens: Optional[int] = None,
) -> Dict:
    """Compute sparse nearest-one multiplicity statistics over valid Points."""
    primary_indices = _require_index_vector(gt_primary_ct_index, 'gt_primary_ct_index')
    valid_mask = _require_bool_vector(gt_primary_valid, 'gt_primary_valid')
    if primary_indices.shape != valid_mask.shape:
        raise M3ContractAuditError('GT primary index/valid arrays must have identical shapes.')

    if gt_high_confidence is None:
        high_confidence = np.zeros_like(valid_mask)
    else:
        high_confidence = _require_bool_vector(gt_high_confidence, 'gt_high_confidence')
        if high_confidence.shape != valid_mask.shape:
            raise M3ContractAuditError('GT high-confidence/valid arrays must have identical shapes.')
        if np.any(high_confidence & ~valid_mask):
            raise M3ContractAuditError('High-confidence GT Points must also be primary-valid.')

    if num_ct_tokens is not None:
        num_ct_tokens = _require_positive_count(num_ct_tokens, 'num_ct_tokens')
        if np.any(primary_indices < 0) or np.any(primary_indices >= num_ct_tokens):
            raise M3ContractAuditError('GT primary CT index is outside the formal CT support range.')
    elif np.any(primary_indices < 0):
        raise M3ContractAuditError('GT primary CT indices must be non-negative.')

    valid_indices = primary_indices[valid_mask]
    valid_point_count = int(valid_indices.size)
    invalid_point_count = int(valid_mask.size - valid_point_count)
    high_confidence_count = int(np.count_nonzero(high_confidence & valid_mask))

    if valid_point_count == 0:
        return {
            'valid_point_count': 0,
            'invalid_point_count': invalid_point_count,
            'high_confidence_count': high_confidence_count,
            'unique_primary_ct_count': 0,
            'collision_group_count': 0,
            'collision_point_count': 0,
            'collision_point_ratio': 0.0,
            'excess_collision_count': 0,
            'excess_collision_ratio': 0.0,
            'multiplicity_max': 0,
            'multiplicity_mean': 0.0,
            'multiplicity_hist': {},
        }

    _, multiplicities = np.unique(valid_indices, return_counts=True)
    collision_multiplicities = multiplicities[multiplicities >= 2]
    unique_primary_ct_count = int(multiplicities.size)
    collision_group_count = int(collision_multiplicities.size)
    collision_point_count = int(collision_multiplicities.sum())
    excess_collision_count = valid_point_count - unique_primary_ct_count
    histogram_counts = Counter(int(value) for value in multiplicities.tolist())
    multiplicity_hist = {key: histogram_counts[key] for key in sorted(histogram_counts)}

    return {
        'valid_point_count': valid_point_count,
        'invalid_point_count': invalid_point_count,
        'high_confidence_count': high_confidence_count,
        'unique_primary_ct_count': unique_primary_ct_count,
        'collision_group_count': collision_group_count,
        'collision_point_count': collision_point_count,
        'collision_point_ratio': collision_point_count / valid_point_count,
        'excess_collision_count': excess_collision_count,
        'excess_collision_ratio': excess_collision_count / valid_point_count,
        'multiplicity_max': int(multiplicities.max()),
        'multiplicity_mean': float(multiplicities.mean()),
        'multiplicity_hist': multiplicity_hist,
    }


def compute_distance_statistics(gt_primary_distance_mm, gt_primary_valid) -> Dict:
    """Compute physical-mm distance statistics over primary-valid Points only."""
    distances = np.asarray(gt_primary_distance_mm)
    valid_mask = _require_bool_vector(gt_primary_valid, 'gt_primary_valid')
    if distances.ndim != 1 or distances.shape != valid_mask.shape:
        raise M3ContractAuditError('GT distance/valid arrays must be one-dimensional and shape-aligned.')
    if not np.issubdtype(distances.dtype, np.number):
        raise M3ContractAuditError('GT primary distances must be numeric physical millimetres.')
    distances = distances.astype(np.float64, copy=False)
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise M3ContractAuditError('GT primary distances must be finite and non-negative physical millimetres.')

    valid_distances = distances[valid_mask]
    output = {'distance_count': int(valid_distances.size)}
    if valid_distances.size == 0:
        for percentile in DISTANCE_PERCENTILES:
            output[f'p{percentile}_mm'] = None
        output['max_mm'] = None
        for threshold in DISTANCE_THRESHOLDS_MM:
            output[f'coverage_le_{threshold:g}_mm'] = 0.0
        return output

    percentile_values = np.percentile(valid_distances, DISTANCE_PERCENTILES)
    for percentile, value in zip(DISTANCE_PERCENTILES, percentile_values):
        output[f'p{percentile}_mm'] = float(value)
    output['max_mm'] = float(np.max(valid_distances))
    for threshold in DISTANCE_THRESHOLDS_MM:
        output[f'coverage_le_{threshold:g}_mm'] = float(np.mean(valid_distances <= threshold))
    return output


def is_identity_gt_transform(
    gt_transform,
    gt_transform_direction,
    *,
    atol: float = IDENTITY_ATOL,
) -> bool:
    """Validate a Point-to-CT rigid transform and classify identity numerically."""
    if gt_transform_direction != POINT_TO_CT_DIRECTION:
        raise M3ContractAuditError(
            f'gt_transform_direction must be exactly {POINT_TO_CT_DIRECTION!r}; '
            f'got {gt_transform_direction!r}. The transform is not inverted automatically.'
        )
    transform = np.asarray(gt_transform)
    if transform.shape != (4, 4) or not np.issubdtype(transform.dtype, np.number):
        raise M3ContractAuditError('gt_transform must be a numeric matrix with shape [4,4].')
    transform = transform.astype(np.float64, copy=False)
    if not np.all(np.isfinite(transform)):
        raise M3ContractAuditError('gt_transform contains NaN or Inf.')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=atol):
        raise M3ContractAuditError('gt_transform must have homogeneous last row [0,0,0,1].')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=atol):
        raise M3ContractAuditError('gt_transform rotation must be orthonormal.')
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=atol):
        raise M3ContractAuditError('gt_transform rotation must have determinant +1.')
    return bool(np.allclose(transform, np.eye(4), rtol=0.0, atol=atol))


def audit_subject_splits(records: Iterable[Mapping]) -> Dict:
    """Report a complete formal train/val/test subject split, without creating one."""
    records = list(records)
    leakage = find_subject_split_leakage(records)
    if leakage:
        raise M3ContractAuditError(f'Subject-level split leakage detected: {leakage}.')

    ready_records = [record for record in records if record.get('ready_for_baseline', True)]
    split_values = [record.get('split') for record in ready_records]
    assigned = [value for value in split_values if value is not None and str(value).strip()]
    if not assigned:
        return {'split_status': 'ABSENT', 'splits': []}
    if len(assigned) != len(split_values):
        raise M3ContractAuditError('Formal subject split is only partially assigned to ready subjects.')

    normalised = [str(value).strip().lower() for value in assigned]
    allowed = {'train', 'val', 'test'}
    unknown = sorted(set(normalised) - allowed)
    if unknown:
        raise M3ContractAuditError(f'Unsupported formal split labels: {unknown}.')
    missing = sorted(allowed - set(normalised))
    if missing:
        raise M3ContractAuditError(f'Formal subject split is missing required partitions: {missing}.')
    return {'split_status': 'PRESENT', 'splits': sorted(set(normalised))}


def find_evaluation_perturbation_manifests(
    project_root,
    data_root=None,
    explicit_manifest=None,
) -> List[Path]:
    """Find filename-matched, unverified perturbation-manifest candidates."""
    if explicit_manifest is not None:
        path = Path(explicit_manifest).expanduser().resolve()
        if not path.is_file():
            raise M3ContractAuditError(f'Explicit evaluation perturbation manifest is missing: {path}.')
        return [path]

    roots = [Path(project_root).expanduser().resolve()]
    if data_root is not None:
        resolved_data_root = Path(data_root).expanduser().resolve()
        if resolved_data_root not in roots:
            roots.append(resolved_data_root)

    candidates = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or '.git' in path.parts:
                continue
            name = path.name.lower()
            if path.suffix.lower() not in {'.json', '.yaml', '.yml'}:
                continue
            if 'perturb' in name and ('manifest' in name or 'eval' in name or 'evaluation' in name):
                candidates.add(path.resolve())
    return sorted(candidates, key=lambda path: str(path).lower())


def audit_evaluation_perturbation_protocol(
    project_root,
    data_root=None,
    explicit_manifest=None,
) -> Dict:
    """Fail closed while the formal evaluation perturbation schema is TBD."""
    candidates = find_evaluation_perturbation_manifests(
        project_root,
        data_root=data_root,
        explicit_manifest=explicit_manifest,
    )
    return {
        'perturbation_manifest_status': 'ABSENT',
        'unverified_candidates': candidates,
    }


def extract_subject_ids(dataset) -> tuple:
    """Extract ready/skipped subject ids from the frozen dataset record contract."""
    ready_ids = [record['subject_id'] for record in dataset.records]
    skipped_ids = [record['subject_id'] for record in dataset.skipped_records]
    return ready_ids, skipped_ids


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _point_coarse_physical_mm(point_dict, scale_mm_to_m: float) -> np.ndarray:
    """Read formal M2-1 coarse tokens and restore their frozen physical-mm view."""
    points = point_dict.get('points')
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise M3ContractAuditError('Formal M2-1 preprocessing must provide four Point stages.')
    coarse_network = points[3]
    if hasattr(coarse_network, 'detach'):
        coarse_physical = _to_numpy(coarse_network / scale_mm_to_m)
    else:
        coarse_physical = np.asarray(coarse_network) / np.float32(scale_mm_to_m)
    if coarse_physical.ndim != 2 or coarse_physical.shape[1] != 3 or coarse_physical.shape[0] == 0:
        raise M3ContractAuditError(
            f'Formal M2-1 coarse Point coordinates must have shape [Np,3], Np>0; '
            f'got {coarse_physical.shape}.'
        )
    if not np.all(np.isfinite(coarse_physical)):
        raise M3ContractAuditError('Formal M2-1 coarse physical Point coordinates contain NaN or Inf.')
    return coarse_physical.astype(np.float64, copy=True)


def audit_case(dataset, case_index: int, cfg) -> Dict:
    """Run one real case through the frozen M2/M2-3 contracts and retain only small statistics."""
    sample = None
    point_batch = None
    ct_batch = None
    correspondence = None
    try:
        sample = dataset[case_index]
        subject_id = sample['subject_id']
        identity_gt = is_identity_gt_transform(
            sample['gt_transform'],
            sample['gt_transform_direction'],
        )

        point_batch = m2_point_collate_fn([sample])
        if point_batch['subject_id'] != subject_id or point_batch['physical_unit'] != 'mm':
            raise M3ContractAuditError(f'{subject_id}: formal Point preprocessing changed identity or mm unit.')
        point_phys = _point_coarse_physical_mm(
            point_batch['point'],
            cfg.point.point_network_scale_mm_to_m,
        )
        gt_transform = np.asarray(point_batch['gt_transform'], dtype=np.float64).copy()
        gt_direction = point_batch['gt_transform_direction']
        point_batch = None
        gc.collect()

        ct_batch = m2_ct_collate_fn([sample])
        if ct_batch['subject_id'] != subject_id or ct_batch['physical_unit'] != 'mm':
            raise M3ContractAuditError(f'{subject_id}: formal CT preprocessing changed identity or mm unit.')
        ct_phys = _to_numpy(ct_batch['ct']['ct_support_phys_20mm']).astype(np.float64, copy=True)
        if ct_phys.ndim != 2 or ct_phys.shape[1] != 3 or ct_phys.shape[0] == 0:
            raise M3ContractAuditError(
                f'{subject_id}: formal M2-2 CT support must have shape [Nv,3], Nv>0; got {ct_phys.shape}.'
            )
        if not np.all(np.isfinite(ct_phys)):
            raise M3ContractAuditError(f'{subject_id}: formal M2-2 CT physical support contains NaN or Inf.')

        ct_batch = None
        sample = None
        gc.collect()

        correspondence = build_coarse_gt_correspondence(
            Xp_phys_coarse=point_phys,
            Xv_phys_coarse=ct_phys,
            gt_transform=gt_transform,
            gt_transform_direction=gt_direction,
            primary_max_distance_mm=cfg.gt_coarse.primary_max_distance_mm,
            high_confidence_distance_mm=cfg.gt_coarse.high_confidence_distance_mm,
        )

        num_point_tokens = int(point_phys.shape[0])
        num_ct_tokens = int(ct_phys.shape[0])
        matrix_stats = calculate_matrix_memory(num_point_tokens, num_ct_tokens)
        primary_stats = compute_primary_statistics(
            correspondence['gt_primary_ct_index'],
            correspondence['gt_primary_valid'],
            correspondence['gt_high_confidence'],
            num_ct_tokens=num_ct_tokens,
        )
        distance_stats = compute_distance_statistics(
            correspondence['gt_primary_distance_mm'],
            correspondence['gt_primary_valid'],
        )
        valid_distances = np.asarray(correspondence['gt_primary_distance_mm'])[
            np.asarray(correspondence['gt_primary_valid'], dtype=bool)
        ].astype(np.float64, copy=True)

        result = {
            'subject_id': subject_id,
            'Np': num_point_tokens,
            'Nv': num_ct_tokens,
            'identity_gt': identity_gt,
            'valid_distances_mm': valid_distances,
        }
        result.update(matrix_stats)
        result.update(primary_stats)
        result.update(distance_stats)
        return result
    finally:
        sample = None
        point_batch = None
        ct_batch = None
        correspondence = None
        gc.collect()


def aggregate_case_statistics(case_statistics: Sequence[Mapping]) -> Dict:
    if not case_statistics:
        raise M3ContractAuditError('M3-1 global audit requires at least one ready subject.')

    histogram = Counter()
    for statistics in case_statistics:
        histogram.update(statistics['multiplicity_hist'])
    multiplicity_hist = {key: histogram[key] for key in sorted(histogram)}

    total_valid = sum(int(item['valid_point_count']) for item in case_statistics)
    total_collision_points = sum(int(item['collision_point_count']) for item in case_statistics)
    total_excess = sum(int(item['excess_collision_count']) for item in case_statistics)
    distance_arrays = [item['valid_distances_mm'] for item in case_statistics if item['valid_distances_mm'].size]
    if distance_arrays:
        global_distances = np.concatenate(distance_arrays)
        global_distance_stats = compute_distance_statistics(
            global_distances,
            np.ones(global_distances.shape, dtype=bool),
        )
    else:
        global_distances = np.empty((0,), dtype=np.float64)
        global_distance_stats = compute_distance_statistics(
            global_distances,
            np.zeros((0,), dtype=bool),
        )

    output = {
        'ready_subjects': len(case_statistics),
        'total_Np': sum(int(item['Np']) for item in case_statistics),
        'total_Nv': sum(int(item['Nv']) for item in case_statistics),
        'total_matrix_elements': sum(int(item['matrix_elements']) for item in case_statistics),
        'max_case_matrix_elements': max(int(item['matrix_elements']) for item in case_statistics),
        'max_case_raw_similarity_float32_MiB': max(
            float(item['raw_similarity_float32_MiB']) for item in case_statistics
        ),
        'max_case_dustbin_matrix_float32_MiB': max(
            float(item['dustbin_matrix_float32_MiB']) for item in case_statistics
        ),
        'total_valid_points': total_valid,
        'total_invalid_points': sum(int(item['invalid_point_count']) for item in case_statistics),
        'total_high_confidence_points': sum(
            int(item['high_confidence_count']) for item in case_statistics
        ),
        'total_unique_primary_ct_count': sum(
            int(item['unique_primary_ct_count']) for item in case_statistics
        ),
        'total_collision_groups': sum(int(item['collision_group_count']) for item in case_statistics),
        'total_collision_points': total_collision_points,
        'global_collision_point_ratio': total_collision_points / total_valid if total_valid else 0.0,
        'total_excess_collision_count': total_excess,
        'global_excess_collision_ratio': total_excess / total_valid if total_valid else 0.0,
        'global_multiplicity_hist': multiplicity_hist,
        'global_multiplicity_max': max(int(item['multiplicity_max']) for item in case_statistics),
        'identity_gt_count': sum(bool(item['identity_gt']) for item in case_statistics),
        'non_identity_gt_count': sum(not bool(item['identity_gt']) for item in case_statistics),
        'direction_failures': 0,
        'global_valid_distances_mm': global_distances,
    }
    output['high_confidence_coverage'] = (
        output['total_high_confidence_points'] / total_valid if total_valid else 0.0
    )
    output.update(global_distance_stats)
    return output


def _format_histogram(histogram: Mapping[int, int]) -> str:
    return '{' + ', '.join(f'{key}:{histogram[key]}' for key in sorted(histogram)) + '}'


def _format_optional_mm(value) -> str:
    return 'NA' if value is None else f'{value:.3f}'


def print_case_statistics(statistics: Mapping):
    print(f'Subject {statistics["subject_id"]}:')
    print(
        f'  Np={statistics["Np"]} Nv={statistics["Nv"]} '
        f'matrix_elements={statistics["matrix_elements"]}'
    )
    print(
        f'  raw_similarity_float32_MiB={statistics["raw_similarity_float32_MiB"]:.6f} '
        f'dustbin_matrix_float32_MiB={statistics["dustbin_matrix_float32_MiB"]:.6f}'
    )
    print(
        f'  valid_point_count={statistics["valid_point_count"]} '
        f'invalid_point_count={statistics["invalid_point_count"]} '
        f'high_confidence_count={statistics["high_confidence_count"]}'
    )
    print(
        f'  unique_primary_ct_count={statistics["unique_primary_ct_count"]} '
        f'collision_group_count={statistics["collision_group_count"]} '
        f'collision_point_count={statistics["collision_point_count"]} '
        f'collision_point_ratio={statistics["collision_point_ratio"]:.6%}'
    )
    print(
        f'  excess_collision_count={statistics["excess_collision_count"]} '
        f'excess_collision_ratio={statistics["excess_collision_ratio"]:.6%}'
    )
    print(
        f'  multiplicity_max={statistics["multiplicity_max"]} '
        f'multiplicity_mean={statistics["multiplicity_mean"]:.6f} '
        f'multiplicity_hist={_format_histogram(statistics["multiplicity_hist"])}'
    )
    print(
        '  distance_mm['
        f'p50={_format_optional_mm(statistics["p50_mm"])}, '
        f'p75={_format_optional_mm(statistics["p75_mm"])}, '
        f'p90={_format_optional_mm(statistics["p90_mm"])}, '
        f'p95={_format_optional_mm(statistics["p95_mm"])}, '
        f'p99={_format_optional_mm(statistics["p99_mm"])}, '
        f'max={_format_optional_mm(statistics["max_mm"])}]'
    )
    print(
        '  coverage['
        f'<=5mm={statistics["coverage_le_5_mm"]:.6%}, '
        f'<=10mm={statistics["coverage_le_10_mm"]:.6%}, '
        f'<=15mm={statistics["coverage_le_15_mm"]:.6%}, '
        f'<=17.5mm={statistics["coverage_le_17.5_mm"]:.6%}, '
        f'<=20mm={statistics["coverage_le_20_mm"]:.6%}]'
    )


def print_global_statistics(statistics: Mapping, ready_ids: Sequence[str], skipped_ids: Sequence[str]):
    print('\nGlobal aggregate:')
    print(f'  ready_subjects={statistics["ready_subjects"]} ready_subject_ids={list(ready_ids)}')
    print(f'  skipped_subjects={len(skipped_ids)} skipped_subject_ids={list(skipped_ids)}')
    print(
        f'  total_Np={statistics["total_Np"]} total_Nv={statistics["total_Nv"]} '
        f'total_matrix_elements={statistics["total_matrix_elements"]}'
    )
    print(
        f'  max_case_matrix_elements={statistics["max_case_matrix_elements"]} '
        f'max_case_raw_similarity_float32_MiB='
        f'{statistics["max_case_raw_similarity_float32_MiB"]:.6f} '
        f'max_case_dustbin_matrix_float32_MiB='
        f'{statistics["max_case_dustbin_matrix_float32_MiB"]:.6f}'
    )
    print(
        f'  total_valid_points={statistics["total_valid_points"]} '
        f'total_invalid_points={statistics["total_invalid_points"]} '
        f'total_unique_primary_ct_count={statistics["total_unique_primary_ct_count"]}'
    )
    print(
        f'  total_collision_groups={statistics["total_collision_groups"]} '
        f'total_collision_points={statistics["total_collision_points"]} '
        f'global_collision_point_ratio={statistics["global_collision_point_ratio"]:.6%}'
    )
    print(
        f'  total_excess_collision_count={statistics["total_excess_collision_count"]} '
        f'global_excess_collision_ratio={statistics["global_excess_collision_ratio"]:.6%}'
    )
    print(
        f'  global_multiplicity_max={statistics["global_multiplicity_max"]} '
        f'global_multiplicity_hist={_format_histogram(statistics["global_multiplicity_hist"])}'
    )
    print(
        '  distance_mm['
        f'p50={_format_optional_mm(statistics["p50_mm"])}, '
        f'p75={_format_optional_mm(statistics["p75_mm"])}, '
        f'p90={_format_optional_mm(statistics["p90_mm"])}, '
        f'p95={_format_optional_mm(statistics["p95_mm"])}, '
        f'p99={_format_optional_mm(statistics["p99_mm"])}, '
        f'max={_format_optional_mm(statistics["max_mm"])}]'
    )
    print(
        '  coverage['
        f'<=5mm={statistics["coverage_le_5_mm"]:.6%}, '
        f'<=10mm={statistics["coverage_le_10_mm"]:.6%}, '
        f'<=15mm={statistics["coverage_le_15_mm"]:.6%}, '
        f'<=17.5mm={statistics["coverage_le_17.5_mm"]:.6%}, '
        f'<=20mm={statistics["coverage_le_20_mm"]:.6%}]'
    )
    print(f'  high_confidence_coverage={statistics["high_confidence_coverage"]:.6%} (diagnostic only)')
    print('\nGT audit:')
    print(
        f'  identity={statistics["identity_gt_count"]} '
        f'non_identity={statistics["non_identity_gt_count"]} '
        f'direction_failures={statistics["direction_failures"]}'
    )


def print_supervision_analysis(statistics: Mapping):
    print('\nM3-1 supervision strategy analysis:')
    print(
        '  A. Per-Point sparse NLL (-Z[i, gt_primary_ct_index[i]]): directly preserves every '
        'frozen nearest-one label, but '
        f'{statistics["total_excess_collision_count"]} Points '
        f'({statistics["global_excess_collision_ratio"]:.6%}) cannot simultaneously occupy '
        'their primary CT target under a strict one-to-one assignment.'
    )
    print(
        '  B. Collision-aware set/group supervision: represents '
        f'{statistics["total_collision_groups"]} observed collision groups involving '
        f'{statistics["total_collision_points"]} Points without inventing radius positives; '
        'exact objective semantics still require review.'
    )
    print(
        '  C. Deterministic one-to-one GT assignment: is assignment-compatible, but would discard '
        f'at least {statistics["total_excess_collision_count"]} frozen primary claims unless an additional '
        'candidate-assignment contract is approved.'
    )
    print(
        '  D. Pre-Sinkhorn row-wise sparse supervision or another partial-assignment strategy: avoids '
        'column-capacity conflict, but its consistency with later partial one-to-one inference remains TBD.'
    )
    print('Recommended L_match decision: TBD pending human review')


def audit_m3_contract(data_root, explicit_perturbation_manifest=None) -> Dict:
    cfg = make_cfg()
    dataset = create_dataset(data_root)
    if len(dataset) != EXPECTED_READY_SUBJECTS:
        raise M3ContractAuditError(
            f'Expected {EXPECTED_READY_SUBJECTS} ready subjects; manifest selected {len(dataset)}.'
        )
    ready_ids, skipped_ids = extract_subject_ids(dataset)
    if EXPECTED_SKIPPED_SUBJECT not in skipped_ids:
        raise M3ContractAuditError(f'{EXPECTED_SKIPPED_SUBJECT} must remain skipped/not-ready.')

    split_audit = audit_subject_splits(dataset.manifest.get('subjects', []))
    perturbation_audit = audit_evaluation_perturbation_protocol(
        PROJECT_ROOT,
        data_root=data_root,
        explicit_manifest=explicit_perturbation_manifest,
    )

    print(
        'Matrix memory note: reported float32 MiB values describe one theoretical matrix only; '
        'they are not training peak GPU memory.'
    )
    case_statistics = []
    for case_index in range(len(dataset)):
        statistics = audit_case(dataset, case_index, cfg)
        print_case_statistics(statistics)
        case_statistics.append(statistics)
        gc.collect()

    global_statistics = aggregate_case_statistics(case_statistics)
    print_global_statistics(global_statistics, ready_ids, skipped_ids)

    print('\nSubject split audit:')
    print(f'  split_status = {split_audit["split_status"]}')
    if split_audit['split_status'] == 'ABSENT':
        print('  Formal subject-level split is still TBD.')
    else:
        print(f'  splits={split_audit["splits"]}; subject-level leakage=none')

    print('\nEvaluation perturbation audit:')
    print(f'  perturbation_manifest_status = {perturbation_audit["perturbation_manifest_status"]}')
    if perturbation_audit['unverified_candidates']:
        print(
            '  unverified_candidates='
            f'{[str(path) for path in perturbation_audit["unverified_candidates"]]}'
        )
    print('  Formal non-identity evaluation perturbation protocol is TBD.')

    print_supervision_analysis(global_statistics)
    return {
        'cases': case_statistics,
        'global': global_statistics,
        'split_status': split_audit['split_status'],
        'perturbation_manifest_status': perturbation_audit['perturbation_manifest_status'],
        'perturbation_manifests': [],
        'unverified_perturbation_manifest_candidates': perturbation_audit['unverified_candidates'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Audit frozen Point-CT M3-1 token, collision, and protocol statistics.'
    )
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        '--perturbation-manifest',
        type=Path,
        default=None,
        help='Optional explicit existing evaluation perturbation manifest to audit; never created.',
    )
    args = parser.parse_args()
    audit_m3_contract(args.data_root, explicit_perturbation_manifest=args.perturbation_manifest)


if __name__ == '__main__':
    main()
