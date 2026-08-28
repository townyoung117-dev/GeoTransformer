"""CPU-only synthetic audit for the independent M3 metric diagnostic."""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from defect_evaluation import load_defect_evaluation_protocol, run_protocol_audit
from m3_metric_diagnostic import (
    DIAGNOSTIC_TRE_FIELDS,
    EXPECTED_DEFECT_CONDITION_COUNT,
    EXPECTED_DEFECT_INSTANCE_COUNT,
    EXPECTED_PATIENT_COUNT,
    EXPECTED_TEST_CASE_COUNT,
    M3MetricDiagnosticContractError,
    build_pat6_forensic,
    compute_tre_diagnostics,
    legacy_parameter_rte_mm,
    load_diagnostic_protocol,
    make_rigid_transform,
    shift_transform_origin,
    validate_source_protocols,
)
from training_protocol import load_training_protocol


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_TRAINING_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
)
DEFAULT_EVALUATION_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_v1.json'
)
DEFAULT_DIAGNOSTIC_PROTOCOL = (
    EXPERIMENT_DIR / 'protocols' / 'm3_metric_diagnostic_v1.json'
)
FROZEN_DIFF_BASE = 'defect_evaluation_adapter'
FROZEN_FILES = (
    'experiments/geotransformer.pointct.baseline_v1/evaluation.py',
    'experiments/geotransformer.pointct.baseline_v1/evaluate_m3.py',
    'experiments/geotransformer.pointct.baseline_v1/defect_evaluation.py',
    'experiments/geotransformer.pointct.baseline_v1/evaluate_m3_defect.py',
    'experiments/geotransformer.pointct.baseline_v1/perturbation.py',
)


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _assert_close_metrics(first: dict, second: dict, atol: float = 1e-9) -> None:
    for field in DIAGNOSTIC_TRE_FIELDS:
        if not np.isclose(first[field], second[field], rtol=0.0, atol=atol):
            raise M3MetricDiagnosticContractError(
                f'synthetic invariance audit failed for {field}: '
                f'{first[field]!r} != {second[field]!r}.'
            )


def _synthetic_pat6_case(index: int, rre_deg: float) -> dict:
    success = True
    return {
        'case_key': f'Fold1|Pat6|defect_{index}|hard|{index}',
        'fold_id': 'Fold1',
        'subject_id': 'Pat6',
        'defect_id': f'defect_{index}',
        'severity': 'hard' if index % 2 else 'mild',
        'variant_id': index,
        'perturbation_seed': index,
        'solver_success': success,
        'solver_status': 'success',
        'registration_recall_hit': False,
        'rre_deg': rre_deg,
        'legacy_parameter_rte_mm': rre_deg * 2.0,
        'centroid_tre_mm': rre_deg / 2.0,
        'point_tre_mean_mm': rre_deg / 2.0 + 1.0,
        'point_tre_median_mm': rre_deg / 2.0,
        'point_tre_rmse_mm': rre_deg / 2.0 + 2.0,
        'point_tre_p95_mm': rre_deg / 2.0 + 3.0,
        'point_tre_max_mm': rre_deg / 2.0 + 4.0,
        'inlier_ratio': 0.5 - index * 0.1,
        'num_correspondences': 10 + index,
        'num_inliers': 5,
        'confidence_mean': 0.8 - index * 0.1,
        'confidence_median': 0.8,
        'confidence_p05': 0.5,
        'confidence_p95': 0.9,
        'confidence_min': 0.4,
        'confidence_max': 0.95,
        'source_centroid_norm_mm': 800.0,
        'origin_rotation_lever_proxy_mm': 100.0,
    }


def _frozen_baseline_diff_passes() -> bool:
    command = [
        'git',
        'diff',
        '--exit-code',
        FROZEN_DIFF_BASE,
        '--',
        *FROZEN_FILES,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except OSError as error:
        raise M3MetricDiagnosticContractError(
            f'cannot run frozen baseline Git audit: {error}'
        ) from error
    if result.returncode not in (0, 1):
        raise M3MetricDiagnosticContractError(
            'frozen baseline Git audit failed to execute: '
            f'{result.stderr.strip()}'
        )
    return result.returncode == 0


def run_audit(
    training_protocol_path=DEFAULT_TRAINING_PROTOCOL,
    evaluation_protocol_path=DEFAULT_EVALUATION_PROTOCOL,
    diagnostic_protocol_path=DEFAULT_DIAGNOSTIC_PROTOCOL,
) -> dict:
    """Run deterministic protocol/math/aggregation checks without real data."""
    training = load_training_protocol(training_protocol_path)
    evaluation = load_defect_evaluation_protocol(evaluation_protocol_path)
    diagnostic = load_diagnostic_protocol(diagnostic_protocol_path)
    validate_source_protocols(diagnostic, training, evaluation)
    protocol_audit = run_protocol_audit(training, evaluation)
    if protocol_audit['expected_test_case_count'] != EXPECTED_TEST_CASE_COUNT:
        raise M3MetricDiagnosticContractError(
            'frozen source protocol no longer expands to 825 cases.'
        )

    points = np.asarray(
        [
            [900.0, -600.0, 400.0],
            [910.0, -590.0, 405.0],
            [880.0, -605.0, 420.0],
            [895.0, -580.0, 390.0],
        ],
        dtype=np.float64,
    )
    identity = np.eye(4, dtype=np.float64)
    identical = compute_tre_diagnostics(
        points,
        identity,
        identity,
        solver_success=True,
    )
    identical_pass = all(identical[field] == 0.0 for field in DIAGNOSTIC_TRE_FIELDS)
    if not identical_pass:
        raise M3MetricDiagnosticContractError(
            'identical-transform synthetic TRE is not zero.'
        )

    translation = make_rigid_transform(np.eye(3), [3.0, 4.0, 12.0])
    translated = compute_tre_diagnostics(
        points,
        translation,
        identity,
        solver_success=True,
    )
    translation_pass = all(
        np.isclose(translated[field], 13.0, rtol=0.0, atol=1e-12)
        for field in DIAGNOSTIC_TRE_FIELDS
    )
    if not translation_pass:
        raise M3MetricDiagnosticContractError(
            'pure-translation synthetic TRE contract failed.'
        )

    predicted = make_rigid_transform(_rotation_z(7.0), [20.0, -5.0, 3.0])
    ground_truth = make_rigid_transform(_rotation_z(-2.0), [-3.0, 1.0, 8.0])
    original = compute_tre_diagnostics(
        points,
        predicted,
        ground_truth,
        solver_success=True,
    )
    shift = np.asarray([1000.0, -800.0, 500.0], dtype=np.float64)
    shifted = compute_tre_diagnostics(
        points - shift,
        shift_transform_origin(predicted, shift),
        shift_transform_origin(ground_truth, shift),
        solver_success=True,
    )
    _assert_close_metrics(original, shifted)

    legacy_before = legacy_parameter_rte_mm(
        predicted[:3, 3],
        ground_truth[:3, 3],
    )
    predicted_shifted = shift_transform_origin(predicted, shift)
    gt_shifted = shift_transform_origin(ground_truth, shift)
    legacy_after = legacy_parameter_rte_mm(
        predicted_shifted[:3, 3],
        gt_shifted[:3, 3],
    )
    legacy_sensitivity_pass = abs(legacy_after - legacy_before) > 100.0
    if not legacy_sensitivity_pass:
        raise M3MetricDiagnosticContractError(
            'legacy parameter RTE origin-sensitivity synthetic audit failed.'
        )

    failure = compute_tre_diagnostics(
        points,
        None,
        ground_truth,
        solver_success=False,
    )
    solver_failure_pass = all(
        failure[field] is None for field in DIAGNOSTIC_TRE_FIELDS
    ) and failure['legacy_parameter_rte_mm'] is None
    if not solver_failure_pass:
        raise M3MetricDiagnosticContractError(
            'solver failure fabricated a diagnostic metric.'
        )

    invalid_transform_pass = False
    invalid = identity.copy()
    invalid[0, 0] = 2.0
    try:
        compute_tre_diagnostics(
            points,
            invalid,
            ground_truth,
            solver_success=True,
        )
    except M3MetricDiagnosticContractError:
        invalid_transform_pass = True
    if not invalid_transform_pass:
        raise M3MetricDiagnosticContractError(
            'invalid transform did not fail closed.'
        )

    forensic = build_pat6_forensic(
        [
            _synthetic_pat6_case(0, 10.0),
            _synthetic_pat6_case(1, 45.0),
            _synthetic_pat6_case(2, 90.0),
        ],
        expected_case_count=3,
    )
    forensic_pass = (
        forensic['catastrophic_rotation_case_count'] == 2
        and np.isclose(forensic['catastrophic_rotation_case_ratio'], 2.0 / 3.0)
        and forensic['worst_10_cases'][0]['rre_deg'] == 90.0
    )
    if not forensic_pass:
        raise M3MetricDiagnosticContractError(
            'Pat6 forensic synthetic aggregation failed.'
        )

    frozen_diff_pass = _frozen_baseline_diff_passes()
    if not frozen_diff_pass:
        raise M3MetricDiagnosticContractError(
            'frozen baseline differs from defect_evaluation_adapter.'
        )
    return {
        'DIAGNOSTIC_PROTOCOL_AUDIT': 'PASS',
        'TRE_IDENTICAL_TRANSFORM_TEST': 'PASS',
        'TRE_TRANSLATION_TEST': 'PASS',
        'TRE_ORIGIN_SHIFT_INVARIANCE': 'PASS',
        'LEGACY_RTE_ORIGIN_SENSITIVITY_TEST': 'PASS',
        'SOLVER_FAILURE_POLICY': 'PASS',
        'INVALID_TRANSFORM_FAIL_CLOSED': 'PASS',
        'PAT6_FORENSIC_AGGREGATION_TEST': 'PASS',
        'EXPECTED_PATIENT_COUNT': EXPECTED_PATIENT_COUNT,
        'EXPECTED_DEFECT_CONDITION_COUNT': EXPECTED_DEFECT_CONDITION_COUNT,
        'EXPECTED_DEFECT_INSTANCE_COUNT': EXPECTED_DEFECT_INSTANCE_COUNT,
        'EXPECTED_TEST_CASE_COUNT': EXPECTED_TEST_CASE_COUNT,
        'FROZEN_BASELINE_DIFF': 'PASS',
        'LOCAL_REAL_DATA_AUDIT_RUN': False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CPU-only synthetic audit for M3 metric diagnostics.'
    )
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
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    result = run_audit(
        args.protocol_manifest,
        args.evaluation_protocol,
        args.diagnostic_protocol,
    )
    for key, value in result.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = value
        print(f'{key}={rendered}')
    return result


if __name__ == '__main__':
    main()


__all__ = ['build_argument_parser', 'main', 'run_audit']
