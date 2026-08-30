"""CPU validation gate for a defective-CT-only osseous support score.

The score path receives only defective CT HU/header data and the frozen CT
support physical locations.  Defect annotations are loaded afterwards, in a
separate diagnostic path, solely to compare against the existing M4-2A
defect-proximity reliability.
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
from m4_osseous_prior import (  # noqa: E402
    OsseousPriorContractError,
    compute_osseous_support_score,
    compute_osseous_support_score_grid,
)


EXPECTED_BASE_HEAD = 'd3c294cf55bd4b6ad48a982f52b0ffba7293de6a'
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
HU_THRESHOLD_GRID = (200.0, 300.0, 400.0)
RADIUS_GRID_MM = (15.0, 20.0, 25.0)
PRIMARY_HU_THRESHOLD = 300.0
PRIMARY_RADIUS_MM = 20.0
TOP_FRACTION = 0.20
M4_2A_DIAGNOSTIC_SIGMA_MM = 20.0

# These thresholds are declared before real-data execution.  They are
# descriptive engineering gates, not fitted statistical claims.
NONDEGENERATE_MIN_RANGE = 0.05
NONDEGENERATE_MIN_STD = 0.01
NONDEGENERATE_MAX_ENDPOINT_FRACTION = 0.98
CROSS_DEFECT_MIN_PATIENT_MEDIAN_RHO = 0.70
CROSS_DEFECT_MIN_PATIENT_MEDIAN_TOP_JACCARD = 0.50
THRESHOLD_MIN_PATIENT_MEDIAN_RHO = 0.85
RADIUS_MIN_PATIENT_MEDIAN_RHO = 0.70
INDEPENDENCE_MAX_PATIENT_MEDIAN_ABS_RHO = 0.95

FORBIDDEN_SCORE_PARAMETERS = {
    'gt_transform',
    'complete_healthy_ct',
    'complete_healthy_point',
    'mp',
    'mv',
    'm4_hard_mask',
    'm4_2a_defect_distance',
    'm4_2a_reliability',
    'anchor_xyz_mm',
    'anatomical_axes',
    'mirror_plane',
    'template_id',
    'anatomical_region',
    'patient_id',
    'defect_id',
    'held_out_test_statistics',
}


class OsseousValidationError(RuntimeError):
    """Raised when the validation harness itself violates its contract."""


def _finite_median(values: Iterable[float]):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(median(finite)) if finite else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise OsseousValidationError('Rank input must be a non-empty finite vector.')
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
        raise OsseousValidationError('Spearman inputs must share one-dimensional length >= 2.')
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise OsseousValidationError('Spearman inputs must be finite.')
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


def _top_token_ids(linear_ids: np.ndarray, scores: np.ndarray, fraction: float) -> set:
    linear_ids = np.asarray(linear_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if linear_ids.ndim != 1 or scores.shape != linear_ids.shape or linear_ids.size == 0:
        raise OsseousValidationError('Top-token inputs are not aligned.')
    count = max(1, int(math.ceil(linear_ids.size * fraction)))
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


def extract_external_surface_sitk(ct_volume, foreground_hu=CT_FOREGROUND_HU) -> np.ndarray:
    """Equivalent 6-connected external-air surface extraction via SimpleITK."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise OsseousValidationError(
            'SimpleITK is required for real NRRD validation and CPU connectivity.'
        ) from error
    volume = np.asarray(ct_volume)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise OsseousValidationError('ct_volume must be a non-empty 3D array.')
    foreground = volume > float(foreground_hu)
    background_image = sitk.GetImageFromArray((~foreground).astype(np.uint8))
    labels_image = sitk.ConnectedComponent(background_image, False)
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
        boundary_labels.update(int(value) for value in np.unique(face) if int(value) != 0)
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
    """NumPy/CPU evaluation of the existing M4-2A reliability formula."""
    coordinates = np.asarray(coordinates_mm, dtype=np.float64)
    valid = np.asarray(valid_flags)
    sigma = float(sigma_mm)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] == 0
        or coordinates.shape[1] != 3
        or not np.all(np.isfinite(coordinates))
    ):
        raise OsseousValidationError('M4-2A diagnostic coordinates are invalid.')
    if valid.dtype != np.dtype(bool) or valid.shape != (coordinates.shape[0],):
        raise OsseousValidationError('M4-2A diagnostic flags are not token-aligned bool values.')
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise OsseousValidationError('M4-2A diagnostic sigma must be finite and positive.')
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
        nearest = np.sqrt(np.min(np.einsum('...i,...i->...', delta, delta), axis=1))
        reliability[rows] = -np.expm1(-0.5 * np.square(nearest / sigma))
    reliability = np.clip(reliability, 0.0, 1.0)
    if not np.all(np.isfinite(reliability)):
        raise OsseousValidationError('M4-2A reliability diagnostic is not finite.')
    return reliability


def _case_paths(data_root: Path, subject_id: str, defect_id: str) -> Dict[str, Path]:
    defect_root = data_root / subject_id / 'defects' / defect_id
    return {
        'defect_root': defect_root,
        'ct': defect_root / 'ct_defect.nrrd',
        'm4_annotation': defect_root / 'Mv_gt.nrrd',
    }


def audit_case_coverage(data_root: Path) -> dict:
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
        'ready_case_count': sum(record['formal_case_ready'] for record in records),
        'ct_present_count': sum(record['ct_exists'] for record in records),
        'records': records,
        'pat6_excluded_from_formal_records': all(
            record['subject_id'] != 'Pat6' for record in records
        ),
        'pat6_preserved_on_disk': (data_root / 'Pat6').exists(),
    }


def _read_defective_ct(path: Path) -> dict:
    image = read_ct_nrrd(path)
    if tuple(image.get('array_axis_order', ())) != ('z', 'y', 'x'):
        raise OsseousValidationError('Defective CT array order must be [z,y,x].')
    if tuple(image.get('image_index_convention', ())) != ('x', 'y', 'z'):
        raise OsseousValidationError('Defective CT image indices must use [x,y,z].')
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
        raise OsseousValidationError('Score statistics require a non-empty finite vector.')
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


def _within_case_sensitivity(score_grid, *, fixed_radius, fixed_threshold) -> dict:
    threshold_pairs = []
    for first, second in itertools.combinations(HU_THRESHOLD_GRID, 2):
        threshold_pairs.append(
            {
                'first_hu': first,
                'second_hu': second,
                'spearman_rho': spearman_rank_correlation(
                    score_grid[(first, fixed_radius)],
                    score_grid[(second, fixed_radius)],
                ),
            }
        )
    radius_pairs = []
    for first, second in itertools.combinations(RADIUS_GRID_MM, 2):
        radius_pairs.append(
            {
                'first_radius_mm': first,
                'second_radius_mm': second,
                'spearman_rho': spearman_rank_correlation(
                    score_grid[(fixed_threshold, first)],
                    score_grid[(fixed_threshold, second)],
                ),
            }
        )
    return {'threshold_pairs': threshold_pairs, 'radius_pairs': radius_pairs}


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

    # Critical separation: this call is complete before the M4 annotation is
    # opened.  Only defective HU/header and frozen support locations enter it.
    score_grid = compute_osseous_support_score_grid(
        volume,
        ct['ct_spacing'],
        ct['ct_origin'],
        ct['ct_direction'],
        support_phys,
        hu_thresholds=HU_THRESHOLD_GRID,
        radius_values_mm=RADIUS_GRID_MM,
    )
    primary_score = score_grid[(PRIMARY_HU_THRESHOLD, PRIMARY_RADIUS_MM)]
    primary_stats = _score_statistics(primary_score)
    sensitivity = _within_case_sensitivity(
        score_grid,
        fixed_radius=PRIMARY_RADIUS_MM,
        fixed_threshold=PRIMARY_HU_THRESHOLD,
    )

    annotation = _read_defective_ct(Path(record['m4_annotation_path']))
    if not _geometry_equal(ct, annotation):
        raise OsseousValidationError('M4 annotation geometry does not match defective CT geometry.')
    mapping = aggregate_ct_defect_whole_cells(
        annotation['ct_volume'],
        volume.shape,
        ct['ct_spacing'],
        support['ct_support_linear_20mm'],
    )
    reliability = compute_m4_2a_reliability_diagnostic(
        support_phys,
        mapping['ct_intact_coarse'],
    )
    independence_rho = spearman_rank_correlation(primary_score, reliability)

    case_report = {
        'subject_id': record['subject_id'],
        'defect_id': record['defect_id'],
        'status': 'PASS',
        'ct_shape_zyx': list(volume.shape),
        'ct_dtype': str(volume.dtype),
        'ct_spacing_xyz_mm': [float(value) for value in ct['ct_spacing']],
        'ct_origin_xyz_mm': [float(value) for value in ct['ct_origin']],
        'ct_direction': np.asarray(ct['ct_direction']).tolist(),
        'support_token_count': int(support_phys.shape[0]),
        'primary_score_statistics': primary_stats,
        'threshold_sensitivity': sensitivity['threshold_pairs'],
        'radius_sensitivity': sensitivity['radius_pairs'],
        'm4_2a_independence': {
            'diagnostic_only': True,
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
        'support_linear': np.asarray(support['ct_support_linear_20mm'], dtype=np.int64),
        'primary_score': primary_score.copy(),
    }
    del annotation, mapping, reliability, score_grid, external_surface, volume, ct
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
        first_top = _top_token_ids(first_ids, first['primary_score'], TOP_FRACTION)
        second_top = _top_token_ids(second_ids, second['primary_score'], TOP_FRACTION)
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


def _patient_reports(case_reports: Sequence[dict], internal_cases: Sequence[dict]) -> list:
    output = []
    for subject_id in CLEAN10_SUBJECT_IDS:
        public = [case for case in case_reports if case.get('subject_id') == subject_id and case['status'] == 'PASS']
        internal = [case for case in internal_cases if case['subject_id'] == subject_id]
        threshold_values = [
            pair['spearman_rho']
            for case in public
            for pair in case['threshold_sensitivity']
        ]
        radius_values = [
            pair['spearman_rho']
            for case in public
            for pair in case['radius_sensitivity']
        ]
        independence_values = [
            case['m4_2a_independence']['absolute_spearman_rho'] for case in public
        ]
        output.append(
            {
                'subject_id': subject_id,
                'validated_case_count': len(public),
                'cross_defect_stability': _cross_defect_patient_report(internal),
                'median_threshold_spearman_rho': _finite_median(threshold_values),
                'median_radius_spearman_rho': _finite_median(radius_values),
                'median_abs_osseous_vs_m4_2a_spearman_rho': _finite_median(
                    independence_values
                ),
            }
        )
    return output


def _score_leakage_audit() -> dict:
    functions = (compute_osseous_support_score, compute_osseous_support_score_grid)
    signatures = {
        function.__name__: list(inspect.signature(function).parameters) for function in functions
    }
    actual = {name.lower() for names in signatures.values() for name in names}
    helper_path = EXPERIMENT_DIR / 'm4_osseous_prior.py'
    tree = ast.parse(helper_path.read_text(encoding='utf-8'))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    forbidden = sorted(actual & FORBIDDEN_SCORE_PARAMETERS)
    allowed_import_roots = {'math', 'typing', 'numpy'}
    unexpected_imports = sorted(
        name for name in imports if name.split('.')[0] not in allowed_import_roots
    )
    return {
        'score_function_parameters': signatures,
        'forbidden_parameter_intersection': forbidden,
        'helper_imports': sorted(imports),
        'unexpected_imports': unexpected_imports,
        'score_generation_reads_m4_annotation': False,
        'm4_annotation_opened_only_after_score_grid_completed': True,
        'pass': not forbidden and not unexpected_imports,
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


def _decision(
    coverage: Mapping,
    case_reports: Sequence[dict],
    patient_reports: Sequence[dict],
    leakage: Mapping,
) -> Tuple[str, list]:
    reasons = []
    if coverage['ready_case_count'] != coverage['expected_case_count']:
        reasons.append(
            f'formal coverage is {coverage["ready_case_count"]}/{coverage["expected_case_count"]}'
        )
    failures = [case for case in case_reports if case['status'] != 'PASS']
    if failures:
        reasons.append(f'{len(failures)} attempted cases failed computation')
    nondegenerate = [
        case for case in case_reports
        if case['status'] == 'PASS' and case['primary_score_statistics']['nondegenerate']
    ]
    if len(nondegenerate) != coverage['expected_case_count']:
        reasons.append(
            f'nondegenerate formal cases are {len(nondegenerate)}/{coverage["expected_case_count"]}'
        )
    for patient in patient_reports:
        subject_id = patient['subject_id']
        cross = patient['cross_defect_stability']
        if cross['status'] != 'PASS':
            reasons.append(f'{subject_id} cross-defect stability is {cross["status"]}')
        threshold_rho = patient['median_threshold_spearman_rho']
        if threshold_rho is None or threshold_rho < THRESHOLD_MIN_PATIENT_MEDIAN_RHO:
            reasons.append(f'{subject_id} threshold sensitivity gate failed')
        radius_rho = patient['median_radius_spearman_rho']
        if radius_rho is None or radius_rho < RADIUS_MIN_PATIENT_MEDIAN_RHO:
            reasons.append(f'{subject_id} radius sensitivity gate failed')
        independence = patient['median_abs_osseous_vs_m4_2a_spearman_rho']
        if independence is None or independence >= INDEPENDENCE_MAX_PATIENT_MEDIAN_ABS_RHO:
            reasons.append(f'{subject_id} M4-2A independence gate failed')
    if not leakage.get('pass'):
        reasons.append('score helper leakage audit failed')
    if not coverage['pat6_excluded_from_formal_records']:
        reasons.append('Pat6 entered formal statistics')
    return ('PASS' if not reasons else 'FAIL'), reasons


def validate_osseous_prior(data_root, *, run_incomplete_diagnostic=False) -> dict:
    data_root = Path(data_root).resolve()
    coverage = audit_case_coverage(data_root)
    leakage = _score_leakage_audit()
    case_reports = []
    internal_cases = []
    if coverage['ready_case_count'] == coverage['expected_case_count'] or run_incomplete_diagnostic:
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
    decision, failure_reasons = _decision(
        coverage,
        case_reports,
        patient_reports,
        leakage,
    )

    branch = _git_value('branch', '--show-current')
    head = _git_value('rev-parse', 'HEAD')
    return {
        'validation_name': 'M4-2B1 Defective-CT-only Osseous Support Prior CPU Validation Gate',
        'data_root': str(data_root),
        'branch': branch,
        'head': head,
        'expected_base_head': EXPECTED_BASE_HEAD,
        'base_head_exact_match': head == EXPECTED_BASE_HEAD,
        'exact_score_formula': (
            'score_j(h,r) = count({in-volume voxel centres x: '
            '||x-Xv_phys_coarse[j]||_2 <= r and HU(x) >= h}) / '
            'count({in-volume voxel centres x: ||x-Xv_phys_coarse[j]||_2 <= r})'
        ),
        'hu_threshold_grid': list(HU_THRESHOLD_GRID),
        'physical_radius_grid_mm': list(RADIUS_GRID_MM),
        'primary_configuration': {
            'hu_threshold': PRIMARY_HU_THRESHOLD,
            'radius_mm': PRIMARY_RADIUS_MM,
        },
        'predeclared_gates': {
            'nondegenerate_min_range': NONDEGENERATE_MIN_RANGE,
            'nondegenerate_min_std': NONDEGENERATE_MIN_STD,
            'nondegenerate_max_endpoint_fraction': NONDEGENERATE_MAX_ENDPOINT_FRACTION,
            'cross_defect_min_patient_median_spearman_rho': CROSS_DEFECT_MIN_PATIENT_MEDIAN_RHO,
            'cross_defect_min_patient_median_top20_jaccard': CROSS_DEFECT_MIN_PATIENT_MEDIAN_TOP_JACCARD,
            'threshold_min_patient_median_spearman_rho': THRESHOLD_MIN_PATIENT_MEDIAN_RHO,
            'radius_min_patient_median_spearman_rho': RADIUS_MIN_PATIENT_MEDIAN_RHO,
            'independence_max_patient_median_abs_spearman_rho': INDEPENDENCE_MAX_PATIENT_MEDIAN_ABS_RHO,
        },
        'coverage': coverage,
        'case_statistics': case_reports,
        'patient_level_statistics': patient_reports,
        'leakage_audit': leakage,
        'human_or_independent_anatomical_verification_still_required': True,
        'permitted_interpretation': 'reproducible CT-derived bone-like osseous support evidence only',
        'prohibited_interpretation': 'nasal/orbital/frontal anatomical localization',
        'OSSEOUS_PRIOR_VALIDATION': decision,
        'ANATOMY_PRIOR_IMPLEMENTATION_READY': 'CONDITIONAL' if decision == 'PASS' else 'NO',
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
        description='Run the M4-2B1 defective-CT-only CPU validation gate.'
    )
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'local_data')
    parser.add_argument(
        '--run-incomplete-diagnostic',
        action='store_true',
        help='Compute present formal cases while retaining the strict 50/50 FAIL gate.',
    )
    parser.add_argument('--output-json', type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = validate_osseous_prior(
        args.data_root,
        run_incomplete_diagnostic=args.run_incomplete_diagnostic,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    return 0 if report['OSSEOUS_PRIOR_VALIDATION'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'OsseousValidationError',
    'audit_case_coverage',
    'compute_m4_2a_reliability_diagnostic',
    'extract_external_surface_sitk',
    'spearman_rank_correlation',
    'validate_osseous_prior',
]
