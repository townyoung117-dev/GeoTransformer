"""Real-data Point/CT matching, filtering, and registration smoke validation."""

import argparse
import gc
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import validate_m3_matching as matching_validation


torch = matching_validation.torch
DEFAULT_DATA_ROOT = matching_validation.DEFAULT_DATA_ROOT
EXPECTED_READY_SUBJECTS = matching_validation.EXPECTED_READY_SUBJECTS
EXPECTED_SKIPPED_SUBJECT = matching_validation.EXPECTED_SKIPPED_SUBJECT
PROJECTED_DIM = matching_validation.PROJECTED_DIM

SMOKE_TEMPERATURE = 0.1
SMOKE_SINKHORN_ITERATIONS = 20
SMOKE_ALPHA_INIT = 1.0
SMOKE_MIN_CONFIDENCE = None
SO3_NUMERICAL_TOLERANCE = 1e-4
SMOKE_PARAMETER_NOTICE = (
    'SMOKE VALIDATION ONLY: matcher parameters are NOT FROZEN baseline or paper values.'
)
IDENTITY_DIAGNOSTIC_NOTICE = 'SMOKE DIAGNOSTIC ONLY; NOT FROZEN EVALUATION.'


class M3RegistrationSmokeValidationError(RuntimeError):
    pass


extract_manifest_subject_ids = matching_validation.extract_manifest_subject_ids
measure_smoke_stage = matching_validation.measure_smoke_stage


def _require_fields(mapping, required, name: str):
    if not isinstance(mapping, dict):
        raise M3RegistrationSmokeValidationError(f'{name} must be a dictionary.')
    missing = [field for field in required if field not in mapping]
    if missing:
        raise M3RegistrationSmokeValidationError(
            f'{name} is missing required fields: {missing}.'
        )


def _load_runtime_components():
    components = matching_validation._load_runtime_components()
    from matching_filter import extract_dustbin_aware_mutual_correspondences
    from registration import estimate_weighted_point_to_ct_transform

    components.update(
        {
            'correspondence_filter': extract_dustbin_aware_mutual_correspondences,
            'registration_estimator': estimate_weighted_point_to_ct_transform,
        }
    )
    return components


def assemble_registration_inputs(point_encoder_output, ct_encoder_output, filter_output):
    _require_fields(point_encoder_output, ('Xp_phys_coarse',), 'Point encoder output')
    _require_fields(ct_encoder_output, ('Xv_phys_coarse',), 'CT encoder output')
    _require_fields(
        filter_output,
        ('point_indices', 'ct_indices', 'confidence'),
        'Correspondence filter output',
    )
    return {
        'point_physical': point_encoder_output['Xp_phys_coarse'],
        'ct_physical': ct_encoder_output['Xv_phys_coarse'],
        'point_indices': filter_output['point_indices'],
        'ct_indices': filter_output['ct_indices'],
        'weights': filter_output['confidence'],
    }


def _validate_physical_outputs(point_encoder_output, ct_encoder_output, num_point, num_ct, device):
    _require_fields(point_encoder_output, ('Xp_phys_coarse',), 'Point encoder output')
    _require_fields(ct_encoder_output, ('Xv_phys_coarse',), 'CT encoder output')
    point_physical = point_encoder_output['Xp_phys_coarse']
    ct_physical = ct_encoder_output['Xv_phys_coarse']
    matching_validation._finite_tensor('Xp_phys_coarse', point_physical)
    matching_validation._finite_tensor('Xv_phys_coarse', ct_physical)
    if tuple(point_physical.shape) != (num_point, 3):
        raise M3RegistrationSmokeValidationError(
            f'Xp_phys_coarse must have shape [{num_point},3]; got {tuple(point_physical.shape)}.'
        )
    if tuple(ct_physical.shape) != (num_ct, 3):
        raise M3RegistrationSmokeValidationError(
            f'Xv_phys_coarse must have shape [{num_ct},3]; got {tuple(ct_physical.shape)}.'
        )
    if point_physical.dtype != torch.float32 or ct_physical.dtype != torch.float32:
        raise M3RegistrationSmokeValidationError('Physical coarse coordinates must be float32.')
    if point_physical.device != device or ct_physical.device != device:
        raise M3RegistrationSmokeValidationError(
            'Physical coarse coordinates must remain on the requested device.'
        )


def _numpy_registration_arrays(rotation, translation):
    rotation_array = np.asarray(rotation)
    translation_array = np.asarray(translation)
    return rotation_array, translation_array


def validate_registration_output(registration_output, expected_device=None):
    required = (
        'success',
        'rotation',
        'translation',
        'failure_reason',
        'num_correspondences',
        'num_positive_weights',
        'weight_sum',
        'source_rank',
        'target_rank',
        'covariance_rank',
        'det_rotation',
    )
    _require_fields(registration_output, required, 'Registration output')
    success = registration_output['success']
    if not isinstance(success, bool):
        raise M3RegistrationSmokeValidationError('Registration success must be bool.')

    rotation = registration_output['rotation']
    translation = registration_output['translation']
    failure_reason = registration_output['failure_reason']
    if not success:
        if rotation is not None or translation is not None:
            raise M3RegistrationSmokeValidationError(
                'Failed registration must not provide a fallback transform.'
            )
        if not isinstance(failure_reason, str) or not failure_reason:
            raise M3RegistrationSmokeValidationError(
                'Failed registration must provide a non-empty failure_reason.'
            )
        return

    if failure_reason is not None:
        raise M3RegistrationSmokeValidationError(
            'Successful registration must have failure_reason=None.'
        )
    if int(registration_output['num_correspondences']) < 3:
        raise M3RegistrationSmokeValidationError(
            'Successful registration requires at least three correspondences.'
        )
    if int(registration_output['num_positive_weights']) < 3:
        raise M3RegistrationSmokeValidationError(
            'Successful registration requires at least three positive weights.'
        )
    successful_ranks = (
        registration_output['source_rank'],
        registration_output['target_rank'],
        registration_output['covariance_rank'],
    )
    if any(rank is None or int(rank) < 2 for rank in successful_ranks):
        raise M3RegistrationSmokeValidationError(
            'Successful registration requires source, target, and covariance rank >= 2.'
        )
    weight_sum = float(registration_output['weight_sum'])
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise M3RegistrationSmokeValidationError(
            'Successful registration requires a finite positive weight_sum.'
        )
    if expected_device is not None:
        if torch is None or not torch.is_tensor(rotation) or not torch.is_tensor(translation):
            raise M3RegistrationSmokeValidationError('Successful R/t must be torch.Tensor outputs.')
        if tuple(rotation.shape) != (3, 3) or tuple(translation.shape) != (3,):
            raise M3RegistrationSmokeValidationError('Successful R/t shapes must be [3,3] and [3].')
        if rotation.dtype != torch.float32 or translation.dtype != torch.float32:
            raise M3RegistrationSmokeValidationError('Successful R/t must use float32 dtype.')
        if rotation.device != expected_device or translation.device != expected_device:
            raise M3RegistrationSmokeValidationError(
                'Successful R/t must remain on the requested device.'
            )
        if not bool(torch.isfinite(rotation).all()) or not bool(torch.isfinite(translation).all()):
            raise M3RegistrationSmokeValidationError('Successful R/t must be finite.')
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        orthogonal = torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            identity,
            rtol=SO3_NUMERICAL_TOLERANCE,
            atol=SO3_NUMERICAL_TOLERANCE,
        )
        determinant = float(torch.linalg.det(rotation).item())
    else:
        rotation_array, translation_array = _numpy_registration_arrays(rotation, translation)
        if rotation_array.shape != (3, 3) or translation_array.shape != (3,):
            raise M3RegistrationSmokeValidationError('Successful R/t shapes must be [3,3] and [3].')
        if rotation_array.dtype != np.float32 or translation_array.dtype != np.float32:
            raise M3RegistrationSmokeValidationError('Successful R/t must use float32 dtype.')
        if not np.all(np.isfinite(rotation_array)) or not np.all(np.isfinite(translation_array)):
            raise M3RegistrationSmokeValidationError('Successful R/t must be finite.')
        orthogonal = np.allclose(
            rotation_array.T @ rotation_array,
            np.eye(3, dtype=np.float32),
            rtol=SO3_NUMERICAL_TOLERANCE,
            atol=SO3_NUMERICAL_TOLERANCE,
        )
        determinant = float(np.linalg.det(rotation_array))

    if not orthogonal or not math.isfinite(determinant):
        raise M3RegistrationSmokeValidationError('Successful rotation must be finite and orthogonal.')
    if abs(determinant - 1.0) > SO3_NUMERICAL_TOLERANCE:
        raise M3RegistrationSmokeValidationError('Successful rotation must have determinant +1.')
    reported_determinant = registration_output['det_rotation']
    if reported_determinant is None or not math.isfinite(float(reported_determinant)):
        raise M3RegistrationSmokeValidationError('Successful det_rotation must be finite.')
    if abs(float(reported_determinant) - determinant) > SO3_NUMERICAL_TOLERANCE:
        raise M3RegistrationSmokeValidationError('Reported det_rotation is inconsistent with R.')


def execute_registration_tail(
    log_assignment,
    point_valid_mask,
    ct_valid_mask,
    point_encoder_output,
    ct_encoder_output,
    correspondence_filter,
    registration_estimator,
    synchronize=None,
    expected_device=None,
):
    filter_output, filtering_time = measure_smoke_stage(
        lambda: correspondence_filter(
            log_assignment,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            min_confidence=SMOKE_MIN_CONFIDENCE,
        ),
        synchronize,
    )
    _require_fields(
        filter_output,
        (
            'point_indices',
            'ct_indices',
            'confidence',
            'num_mutual_before_confidence',
            'num_correspondences',
        ),
        'Correspondence filter output',
    )
    registration_inputs = assemble_registration_inputs(
        point_encoder_output,
        ct_encoder_output,
        filter_output,
    )
    registration_output, registration_time = measure_smoke_stage(
        lambda: registration_estimator(**registration_inputs),
        synchronize,
    )
    validate_registration_output(registration_output, expected_device=expected_device)

    num_correspondences = int(filter_output['num_correspondences'])
    if int(registration_output['num_correspondences']) != num_correspondences:
        raise M3RegistrationSmokeValidationError(
            'Filter and registration correspondence counts are inconsistent.'
        )
    if int(filter_output['num_mutual_before_confidence']) < num_correspondences:
        raise M3RegistrationSmokeValidationError(
            'num_mutual_before_confidence cannot be smaller than num_correspondences.'
        )
    return {
        'filter_output': filter_output,
        'registration_output': registration_output,
        'filtering_time': filtering_time,
        'registration_time': registration_time,
    }


def _tensor_to_nested_floats(value):
    if torch is not None and torch.is_tensor(value):
        if value.ndim == 1:
            return [float(value[index].item()) for index in range(value.shape[0])]
        return [
            [float(value[row, column].item()) for column in range(value.shape[1])]
            for row in range(value.shape[0])
        ]
    return np.asarray(value).astype(float).tolist()


def _identity_diagnostics(rotation, translation):
    if torch is not None and torch.is_tensor(rotation):
        trace = float(torch.trace(rotation).item())
        translation_norm = float(torch.linalg.vector_norm(translation).item())
    else:
        rotation_array, translation_array = _numpy_registration_arrays(rotation, translation)
        trace = float(np.trace(rotation_array))
        translation_norm = float(np.linalg.norm(translation_array))
    cosine = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    return {
        'identity_rotation_deviation_deg': math.degrees(math.acos(cosine)),
        'identity_translation_norm_mm': translation_norm,
    }


def build_registration_case_fields(tail_output):
    filter_output = tail_output['filter_output']
    registration_output = tail_output['registration_output']
    success = registration_output['success']
    fields = {
        'num_mutual_before_confidence': int(filter_output['num_mutual_before_confidence']),
        'num_correspondences': int(filter_output['num_correspondences']),
        'num_positive_weights': int(registration_output['num_positive_weights']),
        'registration_success': success,
        'failure_reason': registration_output['failure_reason'],
        'source_rank': registration_output['source_rank'],
        'target_rank': registration_output['target_rank'],
        'covariance_rank': registration_output['covariance_rank'],
        'det_rotation': registration_output['det_rotation'],
        'filtering_time': tail_output['filtering_time'],
        'registration_time': tail_output['registration_time'],
        'rotation': None,
        'translation': None,
        'identity_rotation_deviation_deg': None,
        'identity_translation_norm_mm': None,
    }
    if success:
        rotation = registration_output['rotation']
        translation = registration_output['translation']
        fields['rotation'] = _tensor_to_nested_floats(rotation)
        fields['translation'] = _tensor_to_nested_floats(translation)
        fields.update(_identity_diagnostics(rotation, translation))
    return fields


def capture_pipeline_case(subject_id, case_function):
    try:
        return case_function(), None
    except Exception as error:
        return None, {
            'subject_id': subject_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
        }


def validate_real_registration_case(
    sample,
    expected_subject_id,
    point_encoder,
    ct_encoder,
    matcher,
    point_collate,
    ct_collate,
    correspondence_filter,
    registration_estimator,
    device,
):
    point_batch = None
    ct_batch = None
    point_input = None
    ct_input = None
    point_encoder_output = None
    ct_encoder_output = None
    matcher_output = None
    tail_output = None
    synchronize = matching_validation._synchronize_callback(device)
    matching_validation._reset_peak_memory(device)
    if synchronize is not None:
        synchronize()
    case_start = time.perf_counter()

    try:
        subject_id = sample.get('subject_id')
        if subject_id != expected_subject_id:
            raise M3RegistrationSmokeValidationError(
                f'Manifest order mismatch: expected {expected_subject_id!r}, got {subject_id!r}.'
            )

        point_batch, point_preprocess_time = measure_smoke_stage(lambda: point_collate([sample]))
        ct_batch, ct_preprocess_time = measure_smoke_stage(lambda: ct_collate([sample]))
        tensor_adapter = lambda value, name: matching_validation._to_device_tensor(
            value,
            name,
            device,
        )
        point_input = matching_validation.assemble_point_encoder_input(
            point_batch['point'],
            tensor_adapter,
        )
        ct_input = matching_validation.assemble_ct_encoder_input(
            ct_batch['ct'],
            tensor_adapter,
        )

        point_encoder_output, point_encoder_time = measure_smoke_stage(
            lambda: point_encoder(point_input),
            synchronize,
        )
        ct_encoder_output, ct_encoder_time = measure_smoke_stage(
            lambda: ct_encoder(ct_input),
            synchronize,
        )
        q = point_encoder_output['Q']
        k = ct_encoder_output['K']
        num_point, num_ct = matching_validation.validate_descriptor_shapes(q, k)
        matching_validation._finite_tensor('Q', q)
        matching_validation._finite_tensor('K', k)
        if q.dtype != torch.float32 or k.dtype != torch.float32:
            raise M3RegistrationSmokeValidationError('Real encoder descriptors must be float32.')
        if q.device != device or k.device != device:
            raise M3RegistrationSmokeValidationError(
                'Real encoder descriptors must remain on the requested device.'
            )
        _validate_physical_outputs(
            point_encoder_output,
            ct_encoder_output,
            num_point,
            num_ct,
            device,
        )

        point_valid_mask = torch.ones((num_point,), dtype=torch.bool, device=device)
        ct_valid_mask = torch.ones((num_ct,), dtype=torch.bool, device=device)
        matcher_inputs = matching_validation.assemble_matcher_inputs(
            point_encoder_output,
            ct_encoder_output,
            point_valid_mask,
            ct_valid_mask,
        )
        matcher_output, matching_sinkhorn_time = measure_smoke_stage(
            lambda: matcher(**matcher_inputs),
            synchronize,
        )
        similarity = matcher_output['similarity']
        log_assignment = matcher_output['log_assignment']
        point_valid_mask = matcher_output['point_valid_mask']
        ct_valid_mask = matcher_output['ct_valid_mask']
        matching_validation.validate_matching_shapes(
            similarity,
            log_assignment,
            num_point,
            num_ct,
        )
        matching_validation._finite_tensor('S', similarity)
        matching_validation._finite_tensor('Z', log_assignment)
        if similarity.dtype != torch.float32 or log_assignment.dtype != torch.float32:
            raise M3RegistrationSmokeValidationError('Matcher S/Z outputs must be float32.')
        if any(
            value.device != device
            for value in (similarity, log_assignment, point_valid_mask, ct_valid_mask)
        ):
            raise M3RegistrationSmokeValidationError(
                'Matcher outputs must remain on the requested device.'
            )

        tail_output = execute_registration_tail(
            log_assignment,
            point_valid_mask,
            ct_valid_mask,
            point_encoder_output,
            ct_encoder_output,
            correspondence_filter,
            registration_estimator,
            synchronize=synchronize,
            expected_device=device,
        )
        if synchronize is not None:
            synchronize()
        total_case_time = time.perf_counter() - case_start
        return {
            'subject_id': subject_id,
            'Np': num_point,
            'Nv': num_ct,
            'point_preprocess_time': point_preprocess_time,
            'ct_preprocess_time': ct_preprocess_time,
            'point_encoder_time': point_encoder_time,
            'ct_encoder_time': ct_encoder_time,
            'matching_sinkhorn_time': matching_sinkhorn_time,
            'total_case_time': total_case_time,
            **build_registration_case_fields(tail_output),
        }
    finally:
        del point_batch
        del ct_batch
        del point_input
        del ct_input
        del point_encoder_output
        del ct_encoder_output
        del matcher_output
        del tail_output
        del sample
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _numeric_range(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return min(values), max(values)


def _runtime_statistics(case_statistics, field):
    values = [float(item[field]) for item in case_statistics]
    if not values:
        return {'total': 0.0, 'min': None, 'max': None, 'median': None}
    return {
        'total': sum(values),
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values),
    }


def aggregate_registration_statistics(
    ready_ids,
    case_statistics,
    pipeline_failures,
    total_runtime,
):
    correspondence_counts = [item['num_correspondences'] for item in case_statistics]
    registration_failures = [
        item for item in case_statistics if not item['registration_success']
    ]
    successful_cases = [item for item in case_statistics if item['registration_success']]
    source_rank_range = _numeric_range([item['source_rank'] for item in case_statistics])
    target_rank_range = _numeric_range([item['target_rank'] for item in case_statistics])
    covariance_rank_range = _numeric_range(
        [item['covariance_rank'] for item in case_statistics]
    )
    determinant_range = _numeric_range([item['det_rotation'] for item in successful_cases])
    identity_rotation_range = _numeric_range(
        [item['identity_rotation_deviation_deg'] for item in successful_cases]
    )
    identity_translation_range = _numeric_range(
        [item['identity_translation_norm_mm'] for item in successful_cases]
    )
    failure_histogram = Counter(item['failure_reason'] for item in registration_failures)
    return {
        'ready_subjects': len(ready_ids),
        'ready_subject_ids': list(ready_ids),
        'completed_subjects': len(case_statistics),
        'pipeline_failed_subjects': len(pipeline_failures),
        'pipeline_failed_subject_ids': [item['subject_id'] for item in pipeline_failures],
        'registration_success_count': len(successful_cases),
        'registration_failure_count': len(registration_failures),
        'failure_reason_histogram': dict(sorted(failure_histogram.items())),
        'total_correspondences': sum(correspondence_counts),
        'min_correspondences': min(correspondence_counts, default=None),
        'max_correspondences': max(correspondence_counts, default=None),
        'median_correspondences': (
            statistics.median(correspondence_counts) if correspondence_counts else None
        ),
        'min_source_rank': source_rank_range[0],
        'max_source_rank': source_rank_range[1],
        'min_target_rank': target_rank_range[0],
        'max_target_rank': target_rank_range[1],
        'min_covariance_rank': covariance_rank_range[0],
        'max_covariance_rank': covariance_rank_range[1],
        'min_successful_det_rotation': determinant_range[0],
        'max_successful_det_rotation': determinant_range[1],
        'total_runtime': float(total_runtime),
        'runtime_summary': {
            field: _runtime_statistics(case_statistics, field)
            for field in (
                'matching_sinkhorn_time',
                'filtering_time',
                'registration_time',
                'total_case_time',
            )
        },
        'identity_diagnostic_summary': {
            'min_rotation_deviation_deg': identity_rotation_range[0],
            'max_rotation_deviation_deg': identity_rotation_range[1],
            'min_translation_norm_mm': identity_translation_range[0],
            'max_translation_norm_mm': identity_translation_range[1],
        },
    }


def print_case_statistics(case_statistics):
    print(f'\nsubject_id = {case_statistics["subject_id"]}')
    print(f'  pipeline_status=COMPLETED Np={case_statistics["Np"]} Nv={case_statistics["Nv"]}')
    print(
        f'  mutual_before_confidence={case_statistics["num_mutual_before_confidence"]} '
        f'correspondences={case_statistics["num_correspondences"]} '
        f'positive_weights={case_statistics["num_positive_weights"]}'
    )
    print(
        f'  registration_success={case_statistics["registration_success"]} '
        f'failure_reason={case_statistics["failure_reason"]}'
    )
    print(
        f'  ranks[source={case_statistics["source_rank"]}, '
        f'target={case_statistics["target_rank"]}, '
        f'covariance={case_statistics["covariance_rank"]}] '
        f'det_rotation={case_statistics["det_rotation"]}'
    )
    print(
        '  runtime_seconds['
        f'matching_sinkhorn={case_statistics["matching_sinkhorn_time"]:.6f}, '
        f'filtering={case_statistics["filtering_time"]:.6f}, '
        f'registration={case_statistics["registration_time"]:.6f}, '
        f'total_case={case_statistics["total_case_time"]:.6f}]'
    )
    if case_statistics['registration_success']:
        print(f'  rotation={case_statistics["rotation"]}')
        print(f'  translation_mm={case_statistics["translation"]}')
        print(
            f'  identity_rotation_deviation_deg='
            f'{case_statistics["identity_rotation_deviation_deg"]:.8g} '
            f'identity_translation_norm_mm='
            f'{case_statistics["identity_translation_norm_mm"]:.8g}'
        )
        print(f'  {IDENTITY_DIAGNOSTIC_NOTICE}')


def print_global_statistics(global_statistics):
    print('\nGlobal M3-5 real-data registration smoke validation:')
    for field in (
        'ready_subjects',
        'ready_subject_ids',
        'completed_subjects',
        'pipeline_failed_subjects',
        'pipeline_failed_subject_ids',
        'registration_success_count',
        'registration_failure_count',
        'failure_reason_histogram',
        'total_correspondences',
        'min_correspondences',
        'max_correspondences',
        'median_correspondences',
        'min_source_rank',
        'max_source_rank',
        'min_target_rank',
        'max_target_rank',
        'min_covariance_rank',
        'max_covariance_rank',
        'min_successful_det_rotation',
        'max_successful_det_rotation',
        'total_runtime',
    ):
        print(f'  {field} = {global_statistics[field]}')
    print(f'  runtime_summary = {global_statistics["runtime_summary"]}')
    print(
        f'  identity_diagnostic_summary = '
        f'{global_statistics["identity_diagnostic_summary"]}'
    )
    print(f'  {IDENTITY_DIAGNOSTIC_NOTICE}')
    print('  Runtime measurements are smoke observations, not a formal benchmark.')


def validate_real_dataset(
    data_root=DEFAULT_DATA_ROOT,
    device_name='cuda',
    temperature=SMOKE_TEMPERATURE,
    sinkhorn_iterations=SMOKE_SINKHORN_ITERATIONS,
    alpha_init=SMOKE_ALPHA_INIT,
):
    components = _load_runtime_components()
    device = matching_validation.resolve_device(device_name)
    cfg = components['make_cfg']()
    dataset = components['create_dataset'](data_root)
    ready_ids, skipped_ids = extract_manifest_subject_ids(dataset)
    if len(dataset) != EXPECTED_READY_SUBJECTS or len(ready_ids) != EXPECTED_READY_SUBJECTS:
        raise M3RegistrationSmokeValidationError(
            f'Expected exactly {EXPECTED_READY_SUBJECTS} ready subjects; '
            f'dataset={len(dataset)} records={len(ready_ids)}.'
        )
    if EXPECTED_SKIPPED_SUBJECT not in skipped_ids:
        raise M3RegistrationSmokeValidationError(
            f'{EXPECTED_SKIPPED_SUBJECT} must remain present in skipped_records.'
        )

    point_encoder = components['PointEncoder'](cfg).to(device).eval()
    ct_encoder = components['CTEncoder'](cfg).to(device).eval()
    matcher = components['PointCTMatcher'](
        projected_dim=PROJECTED_DIM,
        temperature=temperature,
        sinkhorn_iterations=sinkhorn_iterations,
        alpha_init=alpha_init,
    ).to(device).eval()

    print(SMOKE_PARAMETER_NOTICE)
    print(
        f'  temperature={temperature} sinkhorn_iterations={sinkhorn_iterations} '
        f'alpha_init={alpha_init} min_confidence={SMOKE_MIN_CONFIDENCE}'
    )
    print('  Runtime measurements are smoke observations, not a formal benchmark.')

    case_statistics = []
    pipeline_failures = []
    validation_start = time.perf_counter()
    with torch.no_grad():
        for case_index, expected_subject_id in enumerate(ready_ids):
            def run_case():
                sample = dataset[case_index]
                return validate_real_registration_case(
                    sample,
                    expected_subject_id,
                    point_encoder,
                    ct_encoder,
                    matcher,
                    components['point_collate'],
                    components['ct_collate'],
                    components['correspondence_filter'],
                    components['registration_estimator'],
                    device,
                )

            statistics_dict, pipeline_failure = capture_pipeline_case(
                expected_subject_id,
                run_case,
            )
            if pipeline_failure is None:
                case_statistics.append(statistics_dict)
                print_case_statistics(statistics_dict)
            else:
                pipeline_failures.append(pipeline_failure)
                print(
                    f'\nsubject_id = {expected_subject_id}\n'
                    f'  pipeline_status=FAIL error={pipeline_failure["error_type"]}: '
                    f'{pipeline_failure["error_message"]}'
                )
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    total_runtime = time.perf_counter() - validation_start
    global_statistics = aggregate_registration_statistics(
        ready_ids,
        case_statistics,
        pipeline_failures,
        total_runtime,
    )
    print_global_statistics(global_statistics)
    if pipeline_failures or len(case_statistics) != EXPECTED_READY_SUBJECTS:
        raise M3RegistrationSmokeValidationError(
            'Real-data registration smoke validation has pipeline failures: '
            f'{global_statistics["pipeline_failed_subject_ids"]}.'
        )
    print(
        f'\n{EXPECTED_READY_SUBJECTS}/{EXPECTED_READY_SUBJECTS} M3-5 pipeline executions: PASS; '
        f'registration success={global_statistics["registration_success_count"]}, '
        f'mathematical failure={global_statistics["registration_failure_count"]}.'
    )
    return global_statistics


def main():
    parser = argparse.ArgumentParser(description='Run real-data M3-5 registration smoke validation.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--temperature', type=float, default=SMOKE_TEMPERATURE)
    parser.add_argument(
        '--sinkhorn-iterations',
        type=int,
        default=SMOKE_SINKHORN_ITERATIONS,
    )
    parser.add_argument('--alpha-init', type=float, default=SMOKE_ALPHA_INIT)
    args = parser.parse_args()
    validate_real_dataset(
        data_root=args.data_root,
        device_name=args.device,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        alpha_init=args.alpha_init,
    )


if __name__ == '__main__':
    main()
