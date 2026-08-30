"""CPU validation gate for continuous CT-derived osseous support evidence.

The V2 score path receives only defective CT HU/header data, frozen CT support
locations, and the preregistered centre/tau/radius values. Defect annotations
are opened only after all V2 and historical V1 CT-only scores are complete, in
a separate diagnostic path for the frozen M4-2A independence comparison.
"""

import argparse
import ast
import gc
import inspect
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    CT_FOREGROUND_HU,
    aggregate_ct_defect_whole_cells,
    build_ct_support_20mm,
)
from defect_training import DEFECT_IDS  # noqa: E402
from geotransformer.datasets.registration.pointct.dataset import read_ct_nrrd  # noqa: E402
from m4_continuous_osseous_prior import (  # noqa: E402
    compute_continuous_osseous_support_score,
    compute_continuous_osseous_support_score_grid,
)


EXPECTED_BRANCH = 'm4_continuous_osseous_prior_validation_v2'
EXPECTED_BASE_HEAD = 'd3c294cf55bd4b6ad48a982f52b0ffba7293de6a'
AUTHORITATIVE_DATA_ROOT = Path(r'D:\Medical_AR_Facial_Data\outputs')
CLEAN10_SUBJECT_IDS = (
    'Pat1',
    'Pat2',
    'Pat3',
    'Pat4',
    'Pat5',
    'Pat7',
    'Pat8',
    'Pat9',
    'Pat11',
    'Pat12',
)

# Frozen before formal50 execution; see the repository-level V2 protocol.
CENTER_HU = 300.0
TAU_GRID_HU = (75.0, 100.0, 150.0)
RADIUS_GRID_MM = (15.0, 20.0, 25.0)
PRIMARY_TAU_HU = 100.0
PRIMARY_RADIUS_MM = 20.0
TOP_FRACTION = 0.20
M4_2A_DIAGNOSTIC_SIGMA_MM = 20.0

# Frozen V1 historical comparison family. It is not a V2 selection grid.
V1_HU_THRESHOLD_GRID = (200.0, 300.0, 400.0)
V1_PRIMARY_HU_THRESHOLD = 300.0
EXPECTED_V1_THRESHOLD_FAILURE_PATIENTS = ('Pat3', 'Pat7', 'Pat8', 'Pat12')

# Unchanged V1 gates, declared before formal50 V2 execution.
NONDEGENERATE_MIN_RANGE = 0.05
NONDEGENERATE_MIN_STD = 0.01
NONDEGENERATE_MAX_ENDPOINT_FRACTION = 0.98
CROSS_DEFECT_MIN_PATIENT_MEDIAN_RHO = 0.70
CROSS_DEFECT_MIN_PATIENT_MEDIAN_TOP_JACCARD = 0.50
TAU_MIN_PATIENT_MEDIAN_RHO = 0.85
RADIUS_MIN_PATIENT_MEDIAN_RHO = 0.70
INDEPENDENCE_MAX_PATIENT_MEDIAN_ABS_RHO = 0.95

ALLOWED_SCORE_PARAMETERS = (
    'ct_volume',
    'ct_spacing',
    'ct_origin',
    'ct_direction',
    'support_locations_mm',
    'center_hu',
    'tau_hu',
    'radius_mm',
)
FORBIDDEN_SCORE_PARAMETERS = {
    'mp',
    'mv',
    'mask',
    'point_mask',
    'ct_defect_mask',
    'gt_transform',
    'complete_ct',
    'complete_point',
    'complete_healthy_ct',
    'complete_healthy_point',
    'patient_id',
    'defect_id',
    'anchor',
    'anchor_xyz_mm',
    'mirror',
    'mirror_plane',
    'anatomical_axes',
    'template_id',
    'm4_2a_reliability',
    'reliability',
    'defect_distance',
    'm4_2a_defect_distance',
}


class ContinuousOsseousValidationError(RuntimeError):
    """Raised when the validation harness violates its fail-closed contract."""


def _finite_median(values: Iterable[float]):
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(median(finite)) if finite else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ContinuousOsseousValidationError(
            'Rank input must be a non-empty finite vector.'
        )
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rank_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size < 2:
        raise ContinuousOsseousValidationError(
            'Spearman inputs must share one-dimensional length >= 2.'
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ContinuousOsseousValidationError('Spearman inputs must be finite.')
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_centre = first_rank - first_rank.mean()
    second_centre = second_rank - second_rank.mean()
    denominator = math.sqrt(
        float(np.dot(first_centre, first_centre))
        * float(np.dot(second_centre, second_centre))
    )
    if denominator == 0.0:
        return None
    result = float(np.dot(first_centre, second_centre) / denominator)
    return float(np.clip(result, -1.0, 1.0))


def _top_token_ids(linear_ids: np.ndarray, scores: np.ndarray) -> set:
    linear_ids = np.asarray(linear_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if linear_ids.ndim != 1 or scores.shape != linear_ids.shape or linear_ids.size == 0:
        raise ContinuousOsseousValidationError('Top-token inputs are not aligned.')
    count = max(1, int(math.ceil(linear_ids.size * TOP_FRACTION)))
    order = np.lexsort((linear_ids, -scores))
    return set(int(value) for value in linear_ids[order[:count]])


def _jaccard(first: set, second: set):
    union = first | second
    return float(len(first & second) / len(union)) if union else None


def _dilate_6(binary: np.ndarray) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    dilated = binary.copy()
    dilated[1:, :, :] |= binary[:-1, :, :]
    dilated[:-1, :, :] |= binary[1:, :, :]
    dilated[:, 1:, :] |= binary[:, :-1, :]
    dilated[:, :-1, :] |= binary[:, 1:, :]
    dilated[:, :, 1:] |= binary[:, :, :-1]
    dilated[:, :, :-1] |= binary[:, :, 1:]
    return dilated


def extract_external_surface_sitk(
    ct_volume, foreground_hu=CT_FOREGROUND_HU
) -> np.ndarray:
    """Extract the frozen 6-connected boundary-air external CT surface."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise ContinuousOsseousValidationError(
            'SimpleITK is required for real NRRD validation.'
        ) from error
    volume = np.asarray(ct_volume)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise ContinuousOsseousValidationError('ct_volume must be non-empty 3D.')
    foreground = volume > float(foreground_hu)
    labels_image = sitk.ConnectedComponent(
        sitk.GetImageFromArray((~foreground).astype(np.uint8)), False
    )
    labels = sitk.GetArrayFromImage(labels_image)
    boundary_labels = set()
    for face in (
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ):
        boundary_labels.update(
            int(value) for value in np.unique(face) if int(value) != 0
        )
    outside_air = np.zeros(volume.shape, dtype=bool)
    for label in sorted(boundary_labels):
        outside_air |= labels == label
    external_surface = foreground & _dilate_6(outside_air)
    external_surface[0, :, :] |= foreground[0, :, :]
    external_surface[-1, :, :] |= foreground[-1, :, :]
    external_surface[:, 0, :] |= foreground[:, 0, :]
    external_surface[:, -1, :] |= foreground[:, -1, :]
    external_surface[:, :, 0] |= foreground[:, :, 0]
    external_surface[:, :, -1] |= foreground[:, :, -1]
    return external_surface


def compute_m4_2a_reliability_diagnostic(
    coordinates_mm,
    valid_flags,
    sigma_mm=M4_2A_DIAGNOSTIC_SIGMA_MM,
) -> np.ndarray:
    """CPU evaluation of the frozen M4-2A formula, diagnostic only."""
    coordinates = np.asarray(coordinates_mm, dtype=np.float64)
    valid = np.asarray(valid_flags)
    sigma = float(sigma_mm)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] == 0
        or coordinates.shape[1] != 3
        or not np.all(np.isfinite(coordinates))
    ):
        raise ContinuousOsseousValidationError(
            'M4-2A diagnostic coordinates are invalid.'
        )
    if valid.dtype != np.dtype(bool) or valid.shape != (coordinates.shape[0],):
        raise ContinuousOsseousValidationError(
            'M4-2A diagnostic flags are not token-aligned bool values.'
        )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ContinuousOsseousValidationError(
            'M4-2A diagnostic sigma must be finite and positive.'
        )
    reliability = np.zeros(coordinates.shape[0], dtype=np.float64)
    if not np.any(~valid):
        reliability[valid] = 1.0
        return reliability
    if not np.any(valid):
        return reliability
    excluded = coordinates[~valid]
    valid_indices = np.flatnonzero(valid)
    for start in range(0, valid_indices.size, 256):
        rows = valid_indices[start : start + 256]
        delta = coordinates[rows, np.newaxis, :] - excluded[np.newaxis, :, :]
        nearest = np.sqrt(
            np.min(np.einsum('...i,...i->...', delta, delta), axis=1)
        )
        reliability[rows] = -np.expm1(-0.5 * np.square(nearest / sigma))
    reliability = np.clip(reliability, 0.0, 1.0)
    if not np.all(np.isfinite(reliability)):
        raise ContinuousOsseousValidationError(
            'M4-2A diagnostic reliability is not finite.'
        )
    return reliability


def _case_paths(data_root: Path, subject_id: str, defect_id: str) -> Dict[str, Path]:
    defect_root = data_root / subject_id / 'defects' / defect_id
    return {
        'ct': defect_root / 'ct_defect.nrrd',
        'm4_annotation': defect_root / 'Mv_gt.nrrd',
    }


def audit_case_coverage(data_root: Path) -> dict:
    data_root = Path(data_root).resolve()
    records = []
    for subject_id in CLEAN10_SUBJECT_IDS:
        for defect_id in DEFECT_IDS:
            paths = _case_paths(data_root, subject_id, defect_id)
            ct_exists = paths['ct'].is_file()
            annotation_exists = paths['m4_annotation'].is_file()
            records.append(
                {
                    'subject_id': subject_id,
                    'defect_id': defect_id,
                    'ct_exists': ct_exists,
                    'm4_annotation_exists': annotation_exists,
                    'formal_case_ready': ct_exists and annotation_exists,
                    'ct_path': str(paths['ct']),
                    'm4_annotation_path': str(paths['m4_annotation']),
                }
            )
    return {
        'expected_subject_count': len(CLEAN10_SUBJECT_IDS),
        'expected_defects_per_subject': len(DEFECT_IDS),
        'expected_case_count': len(CLEAN10_SUBJECT_IDS) * len(DEFECT_IDS),
        'ready_case_count': sum(item['formal_case_ready'] for item in records),
        'ct_present_count': sum(item['ct_exists'] for item in records),
        'records': records,
        'pat6_excluded_from_formal_records': all(
            item['subject_id'] != 'Pat6' for item in records
        ),
        'pat6_preserved_on_disk': (data_root / 'Pat6').exists(),
    }


def _read_defective_ct(path: Path) -> dict:
    image = read_ct_nrrd(path)
    if tuple(image.get('array_axis_order', ())) != ('z', 'y', 'x'):
        raise ContinuousOsseousValidationError('CT array order must be [z,y,x].')
    if tuple(image.get('image_index_convention', ())) != ('x', 'y', 'z'):
        raise ContinuousOsseousValidationError(
            'CT image indices must use [x,y,z].'
        )
    return image


def _geometry_equal(first: Mapping, second: Mapping) -> bool:
    return (
        np.asarray(first['ct_volume']).shape == np.asarray(second['ct_volume']).shape
        and np.array_equal(first['ct_spacing'], second['ct_spacing'])
        and np.array_equal(first['ct_origin'], second['ct_origin'])
        and np.array_equal(first['ct_direction'], second['ct_direction'])
    )


def _score_statistics(score: np.ndarray) -> dict:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
        raise ContinuousOsseousValidationError(
            'Score statistics require a non-empty finite vector.'
        )
    endpoint_fraction = float(np.mean((score <= 0.01) | (score >= 0.99)))
    stats = {
        'token_count': int(score.size),
        'min': float(np.min(score)),
        'mean': float(np.mean(score)),
        'max': float(np.max(score)),
        'std': float(np.std(score)),
        'p10': float(np.percentile(score, 10)),
        'p50': float(np.percentile(score, 50)),
        'p90': float(np.percentile(score, 90)),
        'endpoint_fraction_le_0p01_or_ge_0p99': endpoint_fraction,
        'unique_value_count': int(np.unique(score).size),
    }
    stats['nondegenerate'] = bool(
        stats['max'] - stats['min'] >= NONDEGENERATE_MIN_RANGE
        and stats['std'] >= NONDEGENERATE_MIN_STD
        and endpoint_fraction <= NONDEGENERATE_MAX_ENDPOINT_FRACTION
        and stats['unique_value_count'] >= 3
    )
    return stats


def _sensitivity_pairs(score_grid, values, fixed_value, *, kind: str) -> list:
    pairs = []
    for first, second in itertools.combinations(values, 2):
        if kind == 'tau':
            first_score = score_grid[(first, fixed_value)]
            second_score = score_grid[(second, fixed_value)]
            first_key, second_key = 'first_tau_hu', 'second_tau_hu'
        elif kind == 'radius':
            first_score = score_grid[(fixed_value, first)]
            second_score = score_grid[(fixed_value, second)]
            first_key, second_key = 'first_radius_mm', 'second_radius_mm'
        elif kind == 'v1_threshold':
            first_score = score_grid[(first, fixed_value)]
            second_score = score_grid[(second, fixed_value)]
            first_key, second_key = 'first_hu', 'second_hu'
        else:
            raise ContinuousOsseousValidationError(f'Unknown sensitivity kind {kind}.')
        pairs.append(
            {
                first_key: first,
                second_key: second,
                'spearman_rho': spearman_rank_correlation(first_score, second_score),
            }
        )
    return pairs


def _compute_frozen_v1_hard_score_grid(
    ct_volume,
    ct_spacing,
    ct_origin,
    ct_direction,
    support_locations_mm,
) -> Dict[Tuple[float, float], np.ndarray]:
    """Recompute the frozen V1 formula for historical comparison only."""
    volume = np.asarray(ct_volume)
    spacing = np.asarray(ct_spacing, dtype=np.float64)
    origin = np.asarray(ct_origin, dtype=np.float64)
    direction = np.asarray(ct_direction, dtype=np.float64).reshape(3, 3)
    support = np.asarray(support_locations_mm, dtype=np.float64)
    image_affine = direction @ np.diag(spacing)
    support_index_xyz = np.linalg.solve(
        image_affine, (support - origin[np.newaxis, :]).T
    ).T
    scores = {
        (threshold, radius): np.empty(support.shape[0], dtype=np.float64)
        for radius in RADIUS_GRID_MM
        for threshold in V1_HU_THRESHOLD_GRID
    }
    max_radius = max(RADIUS_GRID_MM)
    max_delta = max_radius / spacing
    size_xyz = np.asarray(volume.shape[::-1], dtype=np.int64)
    tolerance = max(1.0, max_radius * max_radius) * 1e-12
    for row, centre_xyz in enumerate(support_index_xyz):
        lower = np.maximum(
            np.floor(centre_xyz - max_delta).astype(np.int64), 0
        )
        upper = np.minimum(
            np.ceil(centre_xyz + max_delta).astype(np.int64), size_xyz - 1
        )
        if np.any(lower > upper):
            raise ContinuousOsseousValidationError(
                'Historical V1 comparison found an empty neighborhood.'
            )
        x = np.arange(lower[0], upper[0] + 1, dtype=np.float64)
        y = np.arange(lower[1], upper[1] + 1, dtype=np.float64)
        z = np.arange(lower[2], upper[2] + 1, dtype=np.float64)
        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')
        delta = np.stack(
            (
                grid_x - centre_xyz[0],
                grid_y - centre_xyz[1],
                grid_z - centre_xyz[2],
            ),
            axis=-1,
        )
        physical_delta = delta @ image_affine.T
        distance_squared = np.einsum(
            '...i,...i->...', physical_delta, physical_delta
        )
        local_hu = volume[
            lower[2] : upper[2] + 1,
            lower[1] : upper[1] + 1,
            lower[0] : upper[0] + 1,
        ]
        for radius in RADIUS_GRID_MM:
            sphere_hu = local_hu[
                distance_squared <= radius * radius + tolerance
            ]
            if sphere_hu.size == 0:
                raise ContinuousOsseousValidationError(
                    'Historical V1 comparison found an empty physical sphere.'
                )
            for threshold in V1_HU_THRESHOLD_GRID:
                scores[(threshold, radius)][row] = float(
                    np.mean(sphere_hu >= threshold)
                )
    return scores


def _validate_one_case(record: Mapping) -> Tuple[dict, dict]:
    ct = _read_defective_ct(Path(record['ct_path']))
    volume = np.asarray(ct['ct_volume'])
    external_surface = extract_external_surface_sitk(volume)
    support = build_ct_support_20mm(
        external_surface,
        ct['ct_spacing'],
        ct['ct_origin'],
        ct['ct_direction'],
        volume_shape_zyx=volume.shape,
        physical_unit='mm',
        coordinate_system='LPS',
    )
    support_phys = support['ct_support_phys_20mm']

    # V2 score generation is completed using only defective CT data/header and
    # support locations. No M4-2A annotation has been opened at this point.
    score_grid = compute_continuous_osseous_support_score_grid(
        volume,
        ct['ct_spacing'],
        ct['ct_origin'],
        ct['ct_direction'],
        support_phys,
        CENTER_HU,
        TAU_GRID_HU,
        RADIUS_GRID_MM,
    )
    primary_score = score_grid[(PRIMARY_TAU_HU, PRIMARY_RADIUS_MM)]
    primary_stats = _score_statistics(primary_score)
    all_grid_numeric = all(
        score.shape == primary_score.shape
        and np.all(np.isfinite(score))
        and np.all((score > 0.0) & (score < 1.0))
        for score in score_grid.values()
    )
    tau_pairs = _sensitivity_pairs(
        score_grid, TAU_GRID_HU, PRIMARY_RADIUS_MM, kind='tau'
    )
    radius_pairs = _sensitivity_pairs(
        score_grid, RADIUS_GRID_MM, PRIMARY_TAU_HU, kind='radius'
    )

    # The historical V1 formula is recomputed only for direct comparison. It
    # cannot affect the V2 score or parameter selection.
    v1_grid = _compute_frozen_v1_hard_score_grid(
        volume,
        ct['ct_spacing'],
        ct['ct_origin'],
        ct['ct_direction'],
        support_phys,
    )
    v1_threshold_pairs = _sensitivity_pairs(
        v1_grid,
        V1_HU_THRESHOLD_GRID,
        PRIMARY_RADIUS_MM,
        kind='v1_threshold',
    )

    # Only after every CT-only score is complete is the M4-2A annotation read.
    annotation = _read_defective_ct(Path(record['m4_annotation_path']))
    if not _geometry_equal(ct, annotation):
        raise ContinuousOsseousValidationError(
            'M4 annotation geometry does not match defective CT geometry.'
        )
    mapping = aggregate_ct_defect_whole_cells(
        annotation['ct_volume'],
        volume.shape,
        ct['ct_spacing'],
        support['ct_support_linear_20mm'],
    )
    reliability = compute_m4_2a_reliability_diagnostic(
        support_phys, mapping['ct_intact_coarse']
    )
    independence_rho = spearman_rank_correlation(primary_score, reliability)

    case_report = {
        'subject_id': record['subject_id'],
        'defect_id': record['defect_id'],
        'status': 'PASS' if all_grid_numeric else 'FAIL',
        'ct_shape_zyx': list(volume.shape),
        'ct_dtype': str(volume.dtype),
        'ct_hu_min': float(np.min(volume)),
        'ct_hu_max': float(np.max(volume)),
        'ct_spacing_xyz_mm': [float(value) for value in ct['ct_spacing']],
        'ct_origin_xyz_mm': [float(value) for value in ct['ct_origin']],
        'ct_direction': np.asarray(ct['ct_direction']).tolist(),
        'support_token_count': int(support_phys.shape[0]),
        'score_token_count_exact': bool(
            primary_score.shape == (support_phys.shape[0],)
        ),
        'all_grid_numeric_contract': bool(all_grid_numeric),
        'primary_score_statistics': primary_stats,
        'tau_sensitivity': tau_pairs,
        'radius_sensitivity': radius_pairs,
        'historical_v1_threshold_sensitivity': v1_threshold_pairs,
        'm4_2a_independence': {
            'diagnostic_only': True,
            'annotation_opened_after_all_score_generation': True,
            'sigma_mm': M4_2A_DIAGNOSTIC_SIGMA_MM,
            'spearman_rho': independence_rho,
            'absolute_spearman_rho': (
                abs(independence_rho) if independence_rho is not None else None
            ),
        },
    }
    internal = {
        'subject_id': record['subject_id'],
        'defect_id': record['defect_id'],
        'support_linear': np.asarray(
            support['ct_support_linear_20mm'], dtype=np.int64
        ),
        'primary_score': primary_score.copy(),
    }
    del annotation, mapping, reliability, v1_grid, score_grid, external_surface
    del volume, ct
    gc.collect()
    return case_report, internal


def _cross_defect_patient_report(patient_cases: Sequence[dict]) -> dict:
    if len(patient_cases) != len(DEFECT_IDS):
        return {
            'status': 'INSUFFICIENT_CASES',
            'case_count': len(patient_cases),
            'expected_case_count': len(DEFECT_IDS),
            'pair_count': 0,
            'pair_metrics': [],
            'median_spearman_rho': None,
            'median_top20_jaccard': None,
        }
    pairs = []
    for first, second in itertools.combinations(patient_cases, 2):
        first_ids = first['support_linear']
        second_ids = second['support_linear']
        common, first_index, second_index = np.intersect1d(
            first_ids,
            second_ids,
            assume_unique=True,
            return_indices=True,
        )
        rho = None
        if common.size >= 2:
            rho = spearman_rank_correlation(
                first['primary_score'][first_index],
                second['primary_score'][second_index],
            )
        first_top = _top_token_ids(first_ids, first['primary_score'])
        second_top = _top_token_ids(second_ids, second['primary_score'])
        pairs.append(
            {
                'first_defect_id': first['defect_id'],
                'second_defect_id': second['defect_id'],
                'common_token_count': int(common.size),
                'support_jaccard': _jaccard(
                    set(int(value) for value in first_ids),
                    set(int(value) for value in second_ids),
                ),
                'spearman_rho_on_common_tokens': rho,
                'top20_jaccard': _jaccard(first_top, second_top),
            }
        )
    rho_median = _finite_median(item['spearman_rho_on_common_tokens'] for item in pairs)
    top_median = _finite_median(item['top20_jaccard'] for item in pairs)
    passed = bool(
        rho_median is not None
        and top_median is not None
        and rho_median >= CROSS_DEFECT_MIN_PATIENT_MEDIAN_RHO
        and top_median >= CROSS_DEFECT_MIN_PATIENT_MEDIAN_TOP_JACCARD
    )
    return {
        'status': 'PASS' if passed else 'FAIL',
        'case_count': len(patient_cases),
        'expected_case_count': len(DEFECT_IDS),
        'pair_count': len(pairs),
        'pair_metrics': pairs,
        'median_spearman_rho': rho_median,
        'median_top20_jaccard': top_median,
    }


def _pair_medians(cases: Sequence[dict], field: str, first_key: str, second_key: str) -> list:
    pairs = []
    for first, second in itertools.combinations(
        TAU_GRID_HU if field == 'tau_sensitivity' else (
            RADIUS_GRID_MM if field == 'radius_sensitivity' else V1_HU_THRESHOLD_GRID
        ),
        2,
    ):
        values = [
            pair['spearman_rho']
            for case in cases
            for pair in case[field]
            if pair[first_key] == first and pair[second_key] == second
        ]
        pairs.append(
            {
                first_key: first,
                second_key: second,
                'median_spearman_rho': _finite_median(values),
                'case_count': len(values),
            }
        )
    return pairs


def _patient_reports(case_reports: Sequence[dict], internal_cases: Sequence[dict]) -> list:
    output = []
    for subject_id in CLEAN10_SUBJECT_IDS:
        public = [
            case
            for case in case_reports
            if case.get('subject_id') == subject_id and case['status'] == 'PASS'
        ]
        internal = [
            case for case in internal_cases if case['subject_id'] == subject_id
        ]
        tau_values = [
            pair['spearman_rho'] for case in public for pair in case['tau_sensitivity']
        ]
        radius_values = [
            pair['spearman_rho']
            for case in public
            for pair in case['radius_sensitivity']
        ]
        v1_values = [
            pair['spearman_rho']
            for case in public
            for pair in case['historical_v1_threshold_sensitivity']
        ]
        independence_values = [
            case['m4_2a_independence']['absolute_spearman_rho'] for case in public
        ]
        tau_median = _finite_median(tau_values)
        radius_median = _finite_median(radius_values)
        v1_median = _finite_median(v1_values)
        independence_median = _finite_median(independence_values)
        output.append(
            {
                'subject_id': subject_id,
                'validated_case_count': len(public),
                'cross_defect_stability': _cross_defect_patient_report(internal),
                'tau_pair_patient_medians': _pair_medians(
                    public,
                    'tau_sensitivity',
                    'first_tau_hu',
                    'second_tau_hu',
                ),
                'median_tau_spearman_rho': tau_median,
                'tau_sensitivity_status': (
                    'PASS'
                    if tau_median is not None and tau_median >= TAU_MIN_PATIENT_MEDIAN_RHO
                    else 'FAIL'
                ),
                'radius_pair_patient_medians': _pair_medians(
                    public,
                    'radius_sensitivity',
                    'first_radius_mm',
                    'second_radius_mm',
                ),
                'median_radius_spearman_rho': radius_median,
                'radius_sensitivity_status': (
                    'PASS'
                    if radius_median is not None
                    and radius_median >= RADIUS_MIN_PATIENT_MEDIAN_RHO
                    else 'FAIL'
                ),
                'historical_v1_threshold_pair_patient_medians': _pair_medians(
                    public,
                    'historical_v1_threshold_sensitivity',
                    'first_hu',
                    'second_hu',
                ),
                'historical_v1_median_threshold_spearman_rho': v1_median,
                'historical_v1_threshold_status': (
                    'PASS'
                    if v1_median is not None and v1_median >= TAU_MIN_PATIENT_MEDIAN_RHO
                    else 'FAIL'
                ),
                'continuous_evidence_removed_v1_threshold_failure': bool(
                    v1_median is not None
                    and v1_median < TAU_MIN_PATIENT_MEDIAN_RHO
                    and tau_median is not None
                    and tau_median >= TAU_MIN_PATIENT_MEDIAN_RHO
                ),
                'median_abs_osseous_vs_m4_2a_spearman_rho': independence_median,
                'm4_2a_independence_status': (
                    'PASS'
                    if independence_median is not None
                    and independence_median < INDEPENDENCE_MAX_PATIENT_MEDIAN_ABS_RHO
                    else 'FAIL'
                ),
            }
        )
    return output


def _score_leakage_audit() -> dict:
    functions = (
        compute_continuous_osseous_support_score,
        compute_continuous_osseous_support_score_grid,
    )
    signatures = {
        function.__name__: list(inspect.signature(function).parameters)
        for function in functions
    }
    exact_signatures = all(
        tuple(parameters) == ALLOWED_SCORE_PARAMETERS
        for parameters in signatures.values()
    )
    actual = {
        name.lower() for parameters in signatures.values() for name in parameters
    }
    helper_path = EXPERIMENT_DIR / 'm4_continuous_osseous_prior.py'
    source = helper_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    imports = []
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg.lower())
        elif isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
    forbidden_parameters = sorted(actual & FORBIDDEN_SCORE_PARAMETERS)
    forbidden_identifiers = sorted(identifiers & FORBIDDEN_SCORE_PARAMETERS)
    allowed_import_roots = {'math', 'typing', 'numpy'}
    unexpected_imports = sorted(
        name for name in imports if name.split('.')[0] not in allowed_import_roots
    )
    passed = bool(
        exact_signatures
        and not forbidden_parameters
        and not forbidden_identifiers
        and not unexpected_imports
    )
    return {
        'allowed_score_parameters_exact': list(ALLOWED_SCORE_PARAMETERS),
        'score_function_parameters': signatures,
        'score_signatures_exact_match': exact_signatures,
        'forbidden_parameter_intersection': forbidden_parameters,
        'forbidden_ast_identifier_intersection': forbidden_identifiers,
        'helper_imports': sorted(imports),
        'unexpected_imports': unexpected_imports,
        'score_generation_reads_m4_2a': False,
        'score_generation_reads_defect_mask': False,
        'annotation_opened_only_after_v2_and_v1_ct_score_grids_completed': True,
        'pass': passed,
    }


def _git_value(*args) -> str:
    completed = subprocess.run(
        ('git',) + args,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ''


def _validation_counts(case_reports: Sequence[dict], patient_reports: Sequence[dict]) -> dict:
    numeric_pass = sum(
        case.get('status') == 'PASS'
        and case.get('all_grid_numeric_contract')
        and case.get('score_token_count_exact')
        for case in case_reports
    )
    nondegenerate_pass = sum(
        case.get('status') == 'PASS'
        and case['primary_score_statistics']['nondegenerate']
        for case in case_reports
    )
    return {
        'numeric_contract_pass_count': int(numeric_pass),
        'nondegenerate_pass_count': int(nondegenerate_pass),
        'cross_defect_patient_pass_count': sum(
            item['cross_defect_stability']['status'] == 'PASS'
            for item in patient_reports
        ),
        'tau_sensitivity_patient_pass_count': sum(
            item['tau_sensitivity_status'] == 'PASS' for item in patient_reports
        ),
        'radius_sensitivity_patient_pass_count': sum(
            item['radius_sensitivity_status'] == 'PASS' for item in patient_reports
        ),
        'm4_2a_independence_patient_pass_count': sum(
            item['m4_2a_independence_status'] == 'PASS' for item in patient_reports
        ),
    }


def _case_statistic_summary(case_reports: Sequence[dict]) -> dict:
    passing = [case for case in case_reports if case.get('status') == 'PASS']
    if not passing:
        return {}
    fields = ('min', 'mean', 'max', 'std', 'p10', 'p50', 'p90')
    return {
        field: {
            'minimum_across_cases': float(
                min(case['primary_score_statistics'][field] for case in passing)
            ),
            'median_across_cases': float(
                median(case['primary_score_statistics'][field] for case in passing)
            ),
            'maximum_across_cases': float(
                max(case['primary_score_statistics'][field] for case in passing)
            ),
        }
        for field in fields
    }


def _decision(
    data_root: Path,
    branch: str,
    head: str,
    coverage: Mapping,
    case_reports: Sequence[dict],
    patient_reports: Sequence[dict],
    leakage: Mapping,
) -> Tuple[str, list]:
    reasons = []
    if data_root != AUTHORITATIVE_DATA_ROOT.resolve():
        reasons.append(f'data root is not authoritative: {data_root}')
    if branch != EXPECTED_BRANCH:
        reasons.append(f'branch is {branch}, expected {EXPECTED_BRANCH}')
    if head != EXPECTED_BASE_HEAD:
        reasons.append(f'HEAD is {head}, expected legal base {EXPECTED_BASE_HEAD}')
    expected = coverage['expected_case_count']
    if coverage['ready_case_count'] != expected:
        reasons.append(
            f'formal coverage is {coverage["ready_case_count"]}/{expected}'
        )
    counts = _validation_counts(case_reports, patient_reports)
    if counts['numeric_contract_pass_count'] != expected:
        reasons.append(
            f'numeric contracts passed {counts["numeric_contract_pass_count"]}/{expected}'
        )
    if counts['nondegenerate_pass_count'] != expected:
        reasons.append(
            f'nondegenerate cases are {counts["nondegenerate_pass_count"]}/{expected}'
        )
    for patient in patient_reports:
        subject_id = patient['subject_id']
        if patient['cross_defect_stability']['status'] != 'PASS':
            reasons.append(f'{subject_id} cross-defect stability failed')
        if patient['tau_sensitivity_status'] != 'PASS':
            reasons.append(f'{subject_id} tau sensitivity failed')
        if patient['radius_sensitivity_status'] != 'PASS':
            reasons.append(f'{subject_id} radius sensitivity failed')
        if patient['m4_2a_independence_status'] != 'PASS':
            reasons.append(f'{subject_id} M4-2A independence failed')
    if not leakage.get('pass'):
        reasons.append('score helper leakage audit failed')
    if not coverage['pat6_excluded_from_formal_records']:
        reasons.append('Pat6 entered formal statistics')
    return ('PASS' if not reasons else 'FAIL'), reasons


def validate_continuous_osseous_prior(
    data_root, *, run_incomplete_diagnostic=False
) -> dict:
    data_root = Path(data_root).resolve()
    coverage = audit_case_coverage(data_root)
    leakage = _score_leakage_audit()
    case_reports = []
    internal_cases = []
    if (
        coverage['ready_case_count'] == coverage['expected_case_count']
        or run_incomplete_diagnostic
    ):
        for record in coverage['records']:
            if not record['formal_case_ready']:
                continue
            try:
                case_report, internal = _validate_one_case(record)
            except Exception as error:
                case_reports.append(
                    {
                        'subject_id': record['subject_id'],
                        'defect_id': record['defect_id'],
                        'status': 'FAIL',
                        'error_type': type(error).__name__,
                        'error': str(error),
                    }
                )
            else:
                case_reports.append(case_report)
                internal_cases.append(internal)
    patient_reports = _patient_reports(case_reports, internal_cases)
    branch = _git_value('branch', '--show-current')
    head = _git_value('rev-parse', 'HEAD')
    decision, failure_reasons = _decision(
        data_root,
        branch,
        head,
        coverage,
        case_reports,
        patient_reports,
        leakage,
    )
    counts = _validation_counts(case_reports, patient_reports)
    v1_failed = [
        item['subject_id']
        for item in patient_reports
        if item['historical_v1_threshold_status'] == 'FAIL'
    ]
    targeted = []
    for subject_id in EXPECTED_V1_THRESHOLD_FAILURE_PATIENTS:
        patient = next(
            item for item in patient_reports if item['subject_id'] == subject_id
        )
        targeted.append(
            {
                'subject_id': subject_id,
                'historical_v1_median_threshold_spearman_rho': patient[
                    'historical_v1_median_threshold_spearman_rho'
                ],
                'v2_median_tau_spearman_rho': patient['median_tau_spearman_rho'],
                'v1_failure_repaired': patient[
                    'continuous_evidence_removed_v1_threshold_failure'
                ],
            }
        )

    return {
        'validation_name': (
            'M4-2B2 Continuous CT-derived Osseous Support CPU Validation V2'
        ),
        'branch': branch,
        'head': head,
        'expected_branch': EXPECTED_BRANCH,
        'expected_base_head': EXPECTED_BASE_HEAD,
        'base_head_exact_match': head == EXPECTED_BASE_HEAD,
        'data_root': str(data_root),
        'authoritative_data_root_exact_match': (
            data_root == AUTHORITATIVE_DATA_ROOT.resolve()
        ),
        'coverage': coverage,
        'exact_evidence_formula': 'e(HU;c,tau) = sigmoid((HU-c)/tau)',
        'exact_support_score_formula': (
            's_j(c,tau,r) = mean({e(HU(x);c,tau): in-volume voxel centre x, '
            '||x-X_j||_2 <= r in physical mm})'
        ),
        'frozen_parameter_grid': {
            'center_hu': CENTER_HU,
            'tau_hu': list(TAU_GRID_HU),
            'physical_radius_mm': list(RADIUS_GRID_MM),
        },
        'primary_configuration': {
            'center_hu': CENTER_HU,
            'tau_hu': PRIMARY_TAU_HU,
            'radius_mm': PRIMARY_RADIUS_MM,
        },
        'predeclared_gates': {
            'nondegenerate_min_range': NONDEGENERATE_MIN_RANGE,
            'nondegenerate_min_std': NONDEGENERATE_MIN_STD,
            'nondegenerate_max_endpoint_fraction': NONDEGENERATE_MAX_ENDPOINT_FRACTION,
            'nondegenerate_min_unique_values': 3,
            'cross_defect_min_patient_median_spearman_rho': CROSS_DEFECT_MIN_PATIENT_MEDIAN_RHO,
            'cross_defect_min_patient_median_top20_jaccard': CROSS_DEFECT_MIN_PATIENT_MEDIAN_TOP_JACCARD,
            'tau_min_patient_median_spearman_rho': TAU_MIN_PATIENT_MEDIAN_RHO,
            'radius_min_patient_median_spearman_rho': RADIUS_MIN_PATIENT_MEDIAN_RHO,
            'independence_max_patient_median_abs_spearman_rho_strict': INDEPENDENCE_MAX_PATIENT_MEDIAN_ABS_RHO,
        },
        'validation_counts': counts,
        'primary_case_statistic_summary': _case_statistic_summary(case_reports),
        'case_statistics': case_reports,
        'patient_level_statistics': patient_reports,
        'v1_vs_v2': {
            'historical_v1_formula_recomputed_for_comparison_only': True,
            'v1_threshold_grid_hu': list(V1_HU_THRESHOLD_GRID),
            'v1_threshold_gate': TAU_MIN_PATIENT_MEDIAN_RHO,
            'recomputed_v1_failed_patients': v1_failed,
            'expected_frozen_v1_failed_patients': list(
                EXPECTED_V1_THRESHOLD_FAILURE_PATIENTS
            ),
            'recomputed_outcome_matches_frozen_v1': (
                set(v1_failed) == set(EXPECTED_V1_THRESHOLD_FAILURE_PATIENTS)
            ),
            'targeted_previous_failures': targeted,
            'all_previous_failures_repaired': all(
                item['v1_failure_repaired'] for item in targeted
            ),
        },
        'leakage_audit': leakage,
        'human_or_independent_anatomical_verification_still_required': True,
        'permitted_interpretation': (
            'stable input-only CT-derived continuous osseous support evidence candidate'
        ),
        'prohibited_interpretation': (
            'nasal/orbital/frontal or other structure-level anatomical localization'
        ),
        'CONTINUOUS_OSSEOUS_PRIOR_VALIDATION': decision,
        'ANATOMY_PRIOR_IMPLEMENTATION_READY': (
            'CONDITIONAL' if decision == 'PASS' else 'NO'
        ),
        'failure_reasons': failure_reasons,
        'GPU_USED': 'NO',
        'TRAINING_PERFORMED': 'NO',
        'AUTODL_USED': 'NO',
        'CHECKPOINT_GENERATED': 'NO',
        'COMMIT': 'NO',
        'PUSH': 'NO',
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the preregistered M4-2B2 CPU validation gate.'
    )
    parser.add_argument(
        '--data-root', type=Path, default=AUTHORITATIVE_DATA_ROOT
    )
    parser.add_argument(
        '--run-incomplete-diagnostic',
        action='store_true',
        help='Compute present cases while retaining the strict 50/50 FAIL gate.',
    )
    parser.add_argument('--output-json', type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    resolved_root = args.data_root.resolve()
    coverage = audit_case_coverage(resolved_root)
    print(f'DATA_ROOT = {resolved_root}', flush=True)
    print(f'READY_CASES = {coverage["ready_case_count"]}', flush=True)
    print(f'EXPECTED_CASES = {coverage["expected_case_count"]}', flush=True)
    report = validate_continuous_osseous_prior(
        resolved_root,
        run_incomplete_diagnostic=args.run_incomplete_diagnostic,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    return 0 if report['CONTINUOUS_OSSEOUS_PRIOR_VALIDATION'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'ContinuousOsseousValidationError',
    'audit_case_coverage',
    'compute_m4_2a_reliability_diagnostic',
    'extract_external_surface_sitk',
    'spearman_rank_correlation',
    'validate_continuous_osseous_prior',
]
