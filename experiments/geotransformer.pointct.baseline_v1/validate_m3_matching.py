"""Real-data M3-2 Point/CT encoder and matching forward smoke validation."""

import argparse
import gc
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'local_data'
EXPECTED_READY_SUBJECTS = 11
EXPECTED_SKIPPED_SUBJECT = 'Pat10'
PROJECTED_DIM = 256
MEBIBYTE_BYTES = 1024 ** 2
DEFAULT_MARGINAL_TOLERANCE = 1e-3


class M3MatchingValidationError(RuntimeError):
    pass


def _require_positive_finite(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3MatchingValidationError(f'{name} must be finite and greater than zero.') from error
    if not math.isfinite(result) or result <= 0.0:
        raise M3MatchingValidationError(f'{name} must be finite and greater than zero.')
    return result


def extract_manifest_subject_ids(dataset):
    ready_ids = [record['subject_id'] for record in dataset.records]
    skipped_ids = [record['subject_id'] for record in dataset.skipped_records]
    return ready_ids, skipped_ids


def measure_smoke_stage(function, synchronize=None):
    if synchronize is not None:
        synchronize()
    start_time = time.perf_counter()
    result = function()
    if synchronize is not None:
        synchronize()
    return result, time.perf_counter() - start_time


def validate_descriptor_shapes(q, k):
    q_shape = tuple(q.shape)
    k_shape = tuple(k.shape)
    if len(q_shape) != 2 or q_shape[0] <= 0 or q_shape[1] != PROJECTED_DIM:
        raise M3MatchingValidationError(
            f'Q must have shape [Np,{PROJECTED_DIM}], Np>0; got {q_shape}.'
        )
    if len(k_shape) != 2 or k_shape[0] <= 0 or k_shape[1] != PROJECTED_DIM:
        raise M3MatchingValidationError(
            f'K must have shape [Nv,{PROJECTED_DIM}], Nv>0; got {k_shape}.'
        )
    return q_shape[0], k_shape[0]


def validate_matching_shapes(similarity, log_assignment, num_point: int, num_ct: int):
    expected_similarity = (num_point, num_ct)
    expected_assignment = (num_point + 1, num_ct + 1)
    if tuple(similarity.shape) != expected_similarity:
        raise M3MatchingValidationError(
            f'S must have shape {expected_similarity}; got {tuple(similarity.shape)}.'
        )
    if tuple(log_assignment.shape) != expected_assignment:
        raise M3MatchingValidationError(
            f'Z must have shape {expected_assignment}; got {tuple(log_assignment.shape)}.'
        )


def assemble_matcher_inputs(
    point_encoder_output,
    ct_encoder_output,
    point_valid_mask=None,
    ct_valid_mask=None,
):
    inputs = {
        'q': point_encoder_output['Q'],
        'k': ct_encoder_output['K'],
    }
    if point_valid_mask is not None:
        inputs['point_valid_mask'] = point_valid_mask
    if ct_valid_mask is not None:
        inputs['ct_valid_mask'] = ct_valid_mask
    return inputs


def _require_mapping_fields(mapping, required, name: str):
    missing = [field for field in required if field not in mapping]
    if missing:
        raise M3MatchingValidationError(f'{name} is missing required fields: {missing}.')


def assemble_point_encoder_input(point_branch, tensor_adapter):
    sequence_fields = ('points', 'neighbors', 'subsampling', 'upsampling')
    required = ('features', 'point_network_scale_mm_to_m') + sequence_fields
    _require_mapping_fields(point_branch, required, 'Point preprocessing output')
    output = {
        'features': tensor_adapter(point_branch['features'], 'features'),
        'point_network_scale_mm_to_m': point_branch['point_network_scale_mm_to_m'],
    }
    for field in sequence_fields:
        values = point_branch[field]
        if not isinstance(values, (list, tuple)):
            raise M3MatchingValidationError(f'Point field {field} must be a list or tuple.')
        output[field] = [
            tensor_adapter(value, f'{field}[{index}]')
            for index, value in enumerate(values)
        ]
    return output


def assemble_ct_encoder_input(ct_branch, tensor_adapter):
    tensor_fields = (
        'ct_context_features',
        'ct_context_indices',
        'ct_support_indices_20mm',
        'ct_support_linear_20mm',
        'ct_support_phys_20mm',
    )
    metadata_fields = (
        'ct_context_spatial_shape',
        'ct_support_spatial_shape_20mm',
        'physical_unit',
        'coordinate_system',
    )
    _require_mapping_fields(ct_branch, tensor_fields + metadata_fields, 'CT preprocessing output')
    output = {
        field: tensor_adapter(ct_branch[field], field)
        for field in tensor_fields
    }
    output.update({field: ct_branch[field] for field in metadata_fields})
    return output


def _max_or_zero(value):
    if value.size == 0:
        return 0.0
    return float(np.max(value))


def calculate_transport_qa(probability, point_valid_mask, ct_valid_mask):
    is_torch_tensor = torch is not None and torch.is_tensor(probability)
    if is_torch_tensor:
        if not torch.is_tensor(point_valid_mask) or not torch.is_tensor(ct_valid_mask):
            raise M3MatchingValidationError('Transport masks must be tensors when probability is a tensor.')
        if point_valid_mask.dtype != torch.bool or ct_valid_mask.dtype != torch.bool:
            raise M3MatchingValidationError('Transport masks must use bool dtype.')
        num_point = int(point_valid_mask.shape[0])
        num_ct = int(ct_valid_mask.shape[0])
        if tuple(probability.shape) != (num_point + 1, num_ct + 1):
            raise M3MatchingValidationError('Transport probability and structural masks are shape-inconsistent.')
        if not bool(torch.isfinite(probability).all()):
            raise M3MatchingValidationError('Transport probability contains NaN or Inf.')
        valid_point_count = int(point_valid_mask.sum().item())
        valid_ct_count = int(ct_valid_mask.sum().item())
        if valid_point_count == 0 or valid_ct_count == 0:
            raise M3MatchingValidationError('Transport QA requires valid tokens on both sides.')

        point_row_sums = probability[:num_point, :].sum(dim=1)
        ct_column_sums = probability[:, :num_ct].sum(dim=0)
        point_row_residual = torch.abs(point_row_sums[point_valid_mask] - 1.0).max().item()
        ct_column_residual = torch.abs(ct_column_sums[ct_valid_mask] - 1.0).max().item()
        invalid_point_mass = (
            torch.abs(point_row_sums[~point_valid_mask]).max().item()
            if bool((~point_valid_mask).any())
            else 0.0
        )
        invalid_ct_mass = (
            torch.abs(ct_column_sums[~ct_valid_mask]).max().item()
            if bool((~ct_valid_mask).any())
            else 0.0
        )
        point_dustbin_residual = abs(
            probability[num_point, :].sum().item() - valid_ct_count
        )
        ct_dustbin_residual = abs(
            probability[:, num_ct].sum().item() - valid_point_count
        )
        total_mass_residual = abs(
            probability.sum().item() - (valid_point_count + valid_ct_count)
        )
    else:
        probability = np.asarray(probability, dtype=np.float64)
        point_valid_mask = np.asarray(point_valid_mask)
        ct_valid_mask = np.asarray(ct_valid_mask)
        if point_valid_mask.ndim != 1 or point_valid_mask.dtype != np.bool_:
            raise M3MatchingValidationError('point_valid_mask must be a one-dimensional bool array.')
        if ct_valid_mask.ndim != 1 or ct_valid_mask.dtype != np.bool_:
            raise M3MatchingValidationError('ct_valid_mask must be a one-dimensional bool array.')
        num_point = int(point_valid_mask.shape[0])
        num_ct = int(ct_valid_mask.shape[0])
        if probability.shape != (num_point + 1, num_ct + 1):
            raise M3MatchingValidationError('Transport probability and structural masks are shape-inconsistent.')
        if not np.all(np.isfinite(probability)):
            raise M3MatchingValidationError('Transport probability contains NaN or Inf.')
        valid_point_count = int(np.count_nonzero(point_valid_mask))
        valid_ct_count = int(np.count_nonzero(ct_valid_mask))
        if valid_point_count == 0 or valid_ct_count == 0:
            raise M3MatchingValidationError('Transport QA requires valid tokens on both sides.')

        point_row_sums = probability[:num_point, :].sum(axis=1)
        ct_column_sums = probability[:, :num_ct].sum(axis=0)
        point_row_residual = _max_or_zero(np.abs(point_row_sums[point_valid_mask] - 1.0))
        ct_column_residual = _max_or_zero(np.abs(ct_column_sums[ct_valid_mask] - 1.0))
        invalid_point_mass = _max_or_zero(np.abs(point_row_sums[~point_valid_mask]))
        invalid_ct_mass = _max_or_zero(np.abs(ct_column_sums[~ct_valid_mask]))
        point_dustbin_residual = abs(float(probability[num_point, :].sum()) - valid_ct_count)
        ct_dustbin_residual = abs(float(probability[:, num_ct].sum()) - valid_point_count)
        total_mass_residual = abs(
            float(probability.sum()) - (valid_point_count + valid_ct_count)
        )

    residuals = {
        'max_abs_point_row_residual': float(point_row_residual),
        'max_abs_ct_column_residual': float(ct_column_residual),
        'point_dustbin_row_residual': float(point_dustbin_residual),
        'ct_dustbin_column_residual': float(ct_dustbin_residual),
        'total_mass_residual': float(total_mass_residual),
        'invalid_point_row_mass': float(invalid_point_mass),
        'invalid_ct_column_mass': float(invalid_ct_mass),
    }
    residuals['max_marginal_residual'] = max(residuals.values())
    return residuals


def record_helper_marginal_residuals(
    statistics_dict,
    helper_row_residual,
    helper_col_residual,
):
    helper_row_residual = float(helper_row_residual)
    helper_col_residual = float(helper_col_residual)
    if (
        not math.isfinite(helper_row_residual)
        or helper_row_residual < 0.0
        or not math.isfinite(helper_col_residual)
        or helper_col_residual < 0.0
    ):
        raise M3MatchingValidationError('Matcher helper marginal residuals must be finite and non-negative.')
    statistics_dict['helper_max_abs_row_residual'] = helper_row_residual
    statistics_dict['helper_max_abs_col_residual'] = helper_col_residual
    statistics_dict['max_marginal_residual'] = max(
        statistics_dict['max_marginal_residual'],
        helper_row_residual,
        helper_col_residual,
    )
    return statistics_dict


def validate_marginal_qa(statistics_dict, tolerance):
    tolerance = _require_positive_finite(tolerance, 'marginal_tolerance')
    residual_names = (
        'max_abs_point_row_residual',
        'max_abs_ct_column_residual',
        'point_dustbin_row_residual',
        'ct_dustbin_column_residual',
        'total_mass_residual',
        'invalid_point_row_mass',
        'invalid_ct_column_mass',
    )
    for name in residual_names:
        value = float(statistics_dict[name])
        if not math.isfinite(value) or value > tolerance:
            raise M3MatchingValidationError(
                f'{name}={value!r} exceeds marginal_tolerance={tolerance:g}.'
            )
    for name in ('helper_max_abs_row_residual', 'helper_max_abs_col_residual'):
        if name not in statistics_dict:
            continue
        value = float(statistics_dict[name])
        if not math.isfinite(value) or value > tolerance:
            raise M3MatchingValidationError(
                f'{name}={value!r} exceeds marginal_tolerance={tolerance:g}.'
            )


def _load_runtime_components():
    if torch is None:
        raise M3MatchingValidationError('Real M3-2 validation requires PyTorch.')
    from config import make_cfg
    from ct_encoder import CTEncoder
    from dataset import create_dataset, m2_ct_collate_fn, m2_point_collate_fn
    from matching import PointCTMatcher, compute_marginal_residual
    from point_encoder import PointEncoder

    return {
        'make_cfg': make_cfg,
        'create_dataset': create_dataset,
        'point_collate': m2_point_collate_fn,
        'ct_collate': m2_ct_collate_fn,
        'PointEncoder': PointEncoder,
        'CTEncoder': CTEncoder,
        'PointCTMatcher': PointCTMatcher,
        'compute_marginal_residual': compute_marginal_residual,
    }


def resolve_device(device_name: str):
    if torch is None:
        raise M3MatchingValidationError('Device resolution requires PyTorch.')
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise M3MatchingValidationError(f'Invalid device: {device_name!r}.') from error
    if device.type not in {'cpu', 'cuda'}:
        raise M3MatchingValidationError('Smoke validation supports only cpu or cuda devices.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise M3MatchingValidationError('CUDA was explicitly requested but is unavailable.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        elif device.index < 0 or device.index >= torch.cuda.device_count():
            raise M3MatchingValidationError(f'Requested CUDA device does not exist: {device}.')
        torch.cuda.set_device(device)
    return device


def _to_device_tensor(value, name: str, device):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        raise M3MatchingValidationError(f'{name} must be a tensor or NumPy array.')
    return tensor.to(device=device)


def _synchronize_callback(device):
    if device.type != 'cuda':
        return None
    return lambda: torch.cuda.synchronize(device)


def _reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mib(device):
    if device.type != 'cuda':
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / MEBIBYTE_BYTES,
        torch.cuda.max_memory_reserved(device) / MEBIBYTE_BYTES,
    )


def _finite_tensor(name: str, tensor):
    if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        raise M3MatchingValidationError(f'{name} must be a floating-point tensor.')
    if not bool(torch.isfinite(tensor).all()):
        raise M3MatchingValidationError(f'{name} contains NaN or Inf.')


def _tensor_statistics(tensor):
    return {
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()),
    }


def validate_real_case(
    sample,
    expected_subject_id,
    point_encoder,
    ct_encoder,
    matcher,
    point_collate,
    ct_collate,
    marginal_helper,
    device,
    marginal_tolerance,
):
    point_batch = None
    ct_batch = None
    point_input = None
    ct_input = None
    point_encoder_output = None
    ct_encoder_output = None
    matcher_inputs = None
    matcher_output = None
    q = None
    k = None
    similarity = None
    log_assignment = None
    probability = None
    point_valid_mask = None
    ct_valid_mask = None
    padded_point_mask = None
    padded_ct_mask = None
    valid_transport_mask = None
    ordinary_valid_mask = None
    ordinary_log_values = None
    helper_residual = None
    synchronize = _synchronize_callback(device)
    _reset_peak_memory(device)
    if synchronize is not None:
        synchronize()
    case_start = time.perf_counter()

    try:
        subject_id = sample.get('subject_id')
        if subject_id != expected_subject_id:
            raise M3MatchingValidationError(
                f'Manifest order mismatch: expected {expected_subject_id!r}, got {subject_id!r}.'
            )

        point_batch, point_preprocess_time = measure_smoke_stage(
            lambda: point_collate([sample])
        )
        ct_batch, ct_preprocess_time = measure_smoke_stage(
            lambda: ct_collate([sample])
        )

        tensor_adapter = lambda value, name: _to_device_tensor(value, name, device)
        point_input = assemble_point_encoder_input(point_batch['point'], tensor_adapter)
        ct_input = assemble_ct_encoder_input(ct_batch['ct'], tensor_adapter)

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
        num_point, num_ct = validate_descriptor_shapes(q, k)
        _finite_tensor('Q', q)
        _finite_tensor('K', k)
        if q.dtype != torch.float32 or k.dtype != torch.float32:
            raise M3MatchingValidationError('Real encoder Q and K outputs must be float32.')
        if q.device != device or k.device != device:
            raise M3MatchingValidationError('Q and K must remain on the requested device.')

        point_valid_mask = torch.ones((num_point,), dtype=torch.bool, device=device)
        ct_valid_mask = torch.ones((num_ct,), dtype=torch.bool, device=device)
        matcher_inputs = assemble_matcher_inputs(
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
        validate_matching_shapes(similarity, log_assignment, num_point, num_ct)
        if similarity.dtype != torch.float32 or log_assignment.dtype != torch.float32:
            raise M3MatchingValidationError('S and Z must be float32.')
        _finite_tensor('S', similarity)
        if any(
            tensor.device != device
            for tensor in (similarity, log_assignment, point_valid_mask, ct_valid_mask)
        ):
            raise M3MatchingValidationError('Matcher outputs must remain on the requested device.')
        if matcher.transport.alpha.device != device:
            raise M3MatchingValidationError('Matcher alpha must remain on the requested device.')

        padded_point_mask = torch.cat(
            [point_valid_mask, torch.ones((1,), dtype=torch.bool, device=device)]
        )
        padded_ct_mask = torch.cat(
            [ct_valid_mask, torch.ones((1,), dtype=torch.bool, device=device)]
        )
        valid_transport_mask = padded_point_mask[:, None] & padded_ct_mask[None, :]
        if not bool(torch.isfinite(log_assignment[valid_transport_mask]).all()):
            raise M3MatchingValidationError('Valid Z entries contain NaN or Inf.')

        ordinary_valid_mask = point_valid_mask[:, None] & ct_valid_mask[None, :]
        ordinary_log_values = log_assignment[:num_point, :num_ct][ordinary_valid_mask]
        if ordinary_log_values.numel() == 0 or not bool(torch.isfinite(ordinary_log_values).all()):
            raise M3MatchingValidationError('Ordinary valid Z entries must be finite and non-empty.')

        probability = torch.exp(log_assignment)
        transport_qa = calculate_transport_qa(
            probability,
            point_valid_mask,
            ct_valid_mask,
        )
        helper_residual = marginal_helper(
            log_assignment.unsqueeze(0),
            point_valid_mask.unsqueeze(0),
            ct_valid_mask.unsqueeze(0),
        )
        helper_point_residual = float(helper_residual['max_abs_row_residual'].item())
        helper_ct_residual = float(helper_residual['max_abs_col_residual'].item())
        record_helper_marginal_residuals(
            transport_qa,
            helper_point_residual,
            helper_ct_residual,
        )
        validate_marginal_qa(transport_qa, marginal_tolerance)

        if synchronize is not None:
            synchronize()
        total_case_time = time.perf_counter() - case_start
        peak_allocated_mib, peak_reserved_mib = _peak_memory_mib(device)
        similarity_statistics = _tensor_statistics(similarity)

        case_statistics = {
            'subject_id': subject_id,
            'Np': num_point,
            'Nv': num_ct,
            'Q_shape': tuple(q.shape),
            'K_shape': tuple(k.shape),
            'S_shape': tuple(similarity.shape),
            'Z_shape': tuple(log_assignment.shape),
            'Q_dtype': str(q.dtype),
            'K_dtype': str(k.dtype),
            'S_dtype': str(similarity.dtype),
            'Z_dtype': str(log_assignment.dtype),
            'device': str(device),
            'alpha_value': float(matcher.transport.alpha.detach().item()),
            'S_min': similarity_statistics['min'],
            'S_max': similarity_statistics['max'],
            'S_mean': similarity_statistics['mean'],
            'S_std': similarity_statistics['std'],
            'Z_ordinary_finite_min': float(ordinary_log_values.min().item()),
            'Z_ordinary_finite_max': float(ordinary_log_values.max().item()),
            'point_preprocess_time': point_preprocess_time,
            'ct_preprocess_time': ct_preprocess_time,
            'point_encoder_time': point_encoder_time,
            'ct_encoder_time': ct_encoder_time,
            'matching_sinkhorn_time': matching_sinkhorn_time,
            'total_case_time': total_case_time,
            'peak_allocated_MiB': peak_allocated_mib,
            'peak_reserved_MiB': peak_reserved_mib,
            **transport_qa,
        }
        return case_statistics
    finally:
        del point_batch
        del ct_batch
        del point_input
        del ct_input
        del point_encoder_output
        del ct_encoder_output
        del matcher_inputs
        del matcher_output
        del q
        del k
        del similarity
        del log_assignment
        del probability
        del point_valid_mask
        del ct_valid_mask
        del padded_point_mask
        del padded_ct_mask
        del valid_transport_mask
        del ordinary_valid_mask
        del ordinary_log_values
        del helper_residual
        del sample
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def print_case_statistics(case_statistics):
    print(f'\nsubject_id = {case_statistics["subject_id"]}')
    print(f'  Np={case_statistics["Np"]} Nv={case_statistics["Nv"]}')
    print(
        f'  Q_shape={case_statistics["Q_shape"]} K_shape={case_statistics["K_shape"]} '
        f'S_shape={case_statistics["S_shape"]} Z_shape={case_statistics["Z_shape"]}'
    )
    print(
        f'  Q_dtype={case_statistics["Q_dtype"]} K_dtype={case_statistics["K_dtype"]} '
        f'S_dtype={case_statistics["S_dtype"]} Z_dtype={case_statistics["Z_dtype"]}'
    )
    print(
        f'  device={case_statistics["device"]} '
        f'alpha_value={case_statistics["alpha_value"]:.8g} (smoke only; not frozen)'
    )
    print(
        '  similarity['
        f'min={case_statistics["S_min"]:.8g}, max={case_statistics["S_max"]:.8g}, '
        f'mean={case_statistics["S_mean"]:.8g}, std={case_statistics["S_std"]:.8g}]'
    )
    print(
        '  log_assignment_ordinary_finite['
        f'min={case_statistics["Z_ordinary_finite_min"]:.8g}, '
        f'max={case_statistics["Z_ordinary_finite_max"]:.8g}]'
    )
    print(
        '  marginal_residual['
        f'point_rows={case_statistics["max_abs_point_row_residual"]:.8g}, '
        f'ct_columns={case_statistics["max_abs_ct_column_residual"]:.8g}, '
        f'point_dustbin_row={case_statistics["point_dustbin_row_residual"]:.8g}, '
        f'ct_dustbin_column={case_statistics["ct_dustbin_column_residual"]:.8g}, '
        f'total_mass={case_statistics["total_mass_residual"]:.8g}, '
        f'helper_combined_rows={case_statistics["helper_max_abs_row_residual"]:.8g}, '
        f'helper_combined_columns={case_statistics["helper_max_abs_col_residual"]:.8g}]'
    )
    print(
        '  runtime_seconds['
        f'point_preprocess={case_statistics["point_preprocess_time"]:.6f}, '
        f'ct_preprocess={case_statistics["ct_preprocess_time"]:.6f}, '
        f'point_encoder={case_statistics["point_encoder_time"]:.6f}, '
        f'ct_encoder={case_statistics["ct_encoder_time"]:.6f}, '
        f'matching_sinkhorn={case_statistics["matching_sinkhorn_time"]:.6f}, '
        f'total_case={case_statistics["total_case_time"]:.6f}]'
    )
    print(
        f'  peak_allocated_MiB={case_statistics["peak_allocated_MiB"]:.3f} '
        f'peak_reserved_MiB={case_statistics["peak_reserved_MiB"]:.3f}'
    )


def _aggregate_statistics(ready_ids, case_statistics, failures, total_runtime):
    point_counts = [item['Np'] for item in case_statistics]
    ct_counts = [item['Nv'] for item in case_statistics]
    matching_times = [item['matching_sinkhorn_time'] for item in case_statistics]
    return {
        'ready_subjects': len(ready_ids),
        'completed_subjects': len(case_statistics),
        'failed_subjects': len(failures),
        'failed_subject_ids': list(failures),
        'total_point_tokens': sum(point_counts),
        'total_ct_tokens': sum(ct_counts),
        'min_Np': min(point_counts) if point_counts else None,
        'max_Np': max(point_counts) if point_counts else None,
        'min_Nv': min(ct_counts) if ct_counts else None,
        'max_Nv': max(ct_counts) if ct_counts else None,
        'rectangular_case_count': sum(
            item['Np'] != item['Nv'] for item in case_statistics
        ),
        'max_similarity_elements': max(
            (item['Np'] * item['Nv'] for item in case_statistics),
            default=0,
        ),
        'max_log_assignment_elements': max(
            ((item['Np'] + 1) * (item['Nv'] + 1) for item in case_statistics),
            default=0,
        ),
        'max_marginal_residual': max(
            (item['max_marginal_residual'] for item in case_statistics),
            default=0.0,
        ),
        'total_runtime': total_runtime,
        'median_matching_sinkhorn_time': (
            statistics.median(matching_times) if matching_times else None
        ),
        'max_matching_sinkhorn_time': max(matching_times, default=None),
        'max_peak_allocated_MiB': max(
            (item['peak_allocated_MiB'] for item in case_statistics),
            default=0.0,
        ),
        'max_peak_reserved_MiB': max(
            (item['peak_reserved_MiB'] for item in case_statistics),
            default=0.0,
        ),
    }


def print_global_statistics(global_statistics):
    print('\nGlobal M3-2 real-data smoke validation:')
    for name in (
        'ready_subjects',
        'completed_subjects',
        'failed_subjects',
        'failed_subject_ids',
        'total_point_tokens',
        'total_ct_tokens',
        'min_Np',
        'max_Np',
        'min_Nv',
        'max_Nv',
        'rectangular_case_count',
        'max_similarity_elements',
        'max_log_assignment_elements',
        'max_marginal_residual',
        'total_runtime',
        'median_matching_sinkhorn_time',
        'max_matching_sinkhorn_time',
        'max_peak_allocated_MiB',
        'max_peak_reserved_MiB',
    ):
        print(f'  {name} = {global_statistics[name]}')
    print('  Smoke runtime only; not formal benchmark.')


def validate_real_dataset(
    data_root,
    device_name,
    temperature,
    sinkhorn_iterations,
    alpha_init,
    marginal_tolerance=DEFAULT_MARGINAL_TOLERANCE,
):
    components = _load_runtime_components()
    marginal_tolerance = _require_positive_finite(
        marginal_tolerance,
        'marginal_tolerance',
    )
    device = resolve_device(device_name)
    cfg = components['make_cfg']()
    dataset = components['create_dataset'](data_root)
    ready_ids, skipped_ids = extract_manifest_subject_ids(dataset)
    if len(dataset) != EXPECTED_READY_SUBJECTS or len(ready_ids) != EXPECTED_READY_SUBJECTS:
        raise M3MatchingValidationError(
            f'Expected exactly {EXPECTED_READY_SUBJECTS} ready subjects; '
            f'dataset={len(dataset)} records={len(ready_ids)}.'
        )
    if EXPECTED_SKIPPED_SUBJECT not in skipped_ids:
        raise M3MatchingValidationError(
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
    if matcher.transport.alpha.device != device:
        raise M3MatchingValidationError('Matcher alpha did not move to the requested device.')

    print(
        'These M3-2 matcher hyperparameters are smoke-test values only and are NOT frozen '
        'production values.'
    )
    print(
        f'  temperature={temperature} sinkhorn_iterations={sinkhorn_iterations} '
        f'alpha_init={alpha_init} marginal_tolerance={marginal_tolerance}'
    )
    print('Smoke runtime only; not formal benchmark.')

    case_statistics = []
    failures = []
    validation_start = time.perf_counter()
    with torch.no_grad():
        for case_index, expected_subject_id in enumerate(ready_ids):
            sample = None
            try:
                sample = dataset[case_index]
                statistics_dict = validate_real_case(
                    sample,
                    expected_subject_id,
                    point_encoder,
                    ct_encoder,
                    matcher,
                    components['point_collate'],
                    components['ct_collate'],
                    components['compute_marginal_residual'],
                    device,
                    marginal_tolerance,
                )
                print_case_statistics(statistics_dict)
                case_statistics.append(statistics_dict)
            except Exception as error:
                failures.append(expected_subject_id)
                print(
                    f'\nsubject_id = {expected_subject_id}\n'
                    f'  status=FAIL error={type(error).__name__}: {error}'
                )
            finally:
                del sample
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    total_runtime = time.perf_counter() - validation_start
    global_statistics = _aggregate_statistics(
        ready_ids,
        case_statistics,
        failures,
        total_runtime,
    )
    print_global_statistics(global_statistics)
    if failures or len(case_statistics) != EXPECTED_READY_SUBJECTS:
        raise M3MatchingValidationError(
            f'M3-2 real-data validation failed for subjects: {failures}.'
        )
    print(f'\n{EXPECTED_READY_SUBJECTS}/{EXPECTED_READY_SUBJECTS} M3-2 Real Matching+Sinkhorn validation: PASS')
    return global_statistics


def main():
    parser = argparse.ArgumentParser(description='Run real-data M3-2 forward smoke validation.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--sinkhorn-iterations', type=int, required=True)
    parser.add_argument('--alpha-init', type=float, required=True)
    parser.add_argument(
        '--marginal-tolerance',
        type=float,
        default=DEFAULT_MARGINAL_TOLERANCE,
        help='Float32 smoke QA tolerance; not a model hyperparameter.',
    )
    args = parser.parse_args()
    validate_real_dataset(
        data_root=args.data_root,
        device_name=args.device,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        alpha_init=args.alpha_init,
        marginal_tolerance=args.marginal_tolerance,
    )


if __name__ == '__main__':
    main()
