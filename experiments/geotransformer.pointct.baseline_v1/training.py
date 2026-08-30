"""M3-6A/M3-6B Point-CT end-to-end training infrastructure.

The formal objective is the frozen M3-3 collision-aware matching loss.  Ground
truth is constructed from encoder-produced physical coarse coordinates only
after the matcher forward pass.
"""

import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from gt_correspondence import build_coarse_gt_correspondence
from m4_hard_constraint import (
    build_m4_mask_aware_gt,
    resolve_m4_matching_masks,
)
from m4_soft_modulation import (
    M4SoftModulationError,
    run_m4_soft_modulated_matching,
)
from m4_osseous_integration import (
    M4OsseousIntegrationError,
    run_m4_osseous_integrated_matching,
)
from matching_loss import compute_collision_aware_match_loss
from training_protocol import (
    M3TrainingProtocolError,
    resolve_fold,
    validate_dataset_ready_subjects,
)


TRAINING_SMOKE_DEFAULT_LEARNING_RATE = 1e-4
TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY = 1e-4
TRAINING_DEFAULTS_STATUS = 'TRAINING SMOKE DEFAULTS - NOT FROZEN PAPER HYPERPARAMETERS'
SMOKE_SPLIT_STATUS = 'SMOKE SPLIT ONLY - NOT FROZEN EVALUATION SPLIT'
SMOKE_CHECKPOINT_VERSION = 1
CHECKPOINT_VERSION = 2
BATCH_SIZE = 1
PRECISION = 'fp32'


class M3TrainingContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    """Configurable smoke settings; these are not frozen paper settings."""

    learning_rate: float = TRAINING_SMOKE_DEFAULT_LEARNING_RATE
    weight_decay: float = TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    precision: str = PRECISION


@dataclass(frozen=True)
class SubjectSplit:
    train_subject_ids: tuple
    val_subject_ids: tuple
    train_indices: tuple
    val_indices: tuple


@dataclass(frozen=True)
class FormalSubjectSplit:
    fold_id: str
    train_subject_ids: tuple
    val_subject_ids: tuple
    test_subject_ids: tuple
    train_indices: tuple
    val_indices: tuple
    test_indices: tuple


def _require_nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise M3TrainingContractError(f'{name} must be a non-negative integer.')
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3TrainingContractError(f'{name} must be a non-negative integer.') from error
    if integer < 0 or integer != value:
        raise M3TrainingContractError(f'{name} must be a non-negative integer.')
    return integer


def _require_finite_float(value, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise M3TrainingContractError(f'{name} must be a finite scalar.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3TrainingContractError(f'{name} must be a finite scalar.') from error
    if not math.isfinite(number) or number < 0.0 or (not allow_zero and number == 0.0):
        qualifier = 'non-negative' if allow_zero else 'greater than zero'
        raise M3TrainingContractError(f'{name} must be finite and {qualifier}.')
    return number


def _require_best_val_loss(value, name: str = 'best_val_loss') -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise M3TrainingContractError(f'{name} must be finite or positive infinity.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3TrainingContractError(
            f'{name} must be finite or positive infinity.'
        ) from error
    if math.isnan(number) or number == float('-inf'):
        raise M3TrainingContractError(f'{name} must be finite or positive infinity.')
    return number


def _require_any_finite_float(value, name: str) -> float:
    if isinstance(value, bool) or torch.is_tensor(value):
        raise M3TrainingContractError(f'{name} must be a finite scalar.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3TrainingContractError(f'{name} must be a finite scalar.') from error
    if not math.isfinite(number):
        raise M3TrainingContractError(f'{name} must be a finite scalar.')
    return number


def validate_training_config(config: TrainingConfig) -> TrainingConfig:
    if not isinstance(config, TrainingConfig):
        raise M3TrainingContractError('training config must be a TrainingConfig instance.')
    learning_rate = _require_finite_float(
        config.learning_rate,
        'learning_rate',
        allow_zero=False,
    )
    weight_decay = _require_finite_float(
        config.weight_decay,
        'weight_decay',
        allow_zero=True,
    )
    if config.batch_size != BATCH_SIZE:
        raise M3TrainingContractError('M3-6A requires batch_size=1.')
    if config.precision != PRECISION:
        raise M3TrainingContractError('M3-6A requires precision="fp32".')
    return TrainingConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=BATCH_SIZE,
        precision=PRECISION,
    )


def set_random_seed(seed: int) -> int:
    seed = _require_nonnegative_integer(seed, 'seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _validate_subject_id_list(subject_ids, name: str) -> tuple:
    if isinstance(subject_ids, (str, bytes)) or not isinstance(subject_ids, Sequence):
        raise M3TrainingContractError(f'{name} must be an explicit sequence of subject IDs.')
    normalized = []
    for value in subject_ids:
        if not isinstance(value, str) or not value.strip():
            raise M3TrainingContractError(f'{name} contains an invalid subject ID.')
        normalized.append(value.strip())
    if not normalized:
        raise M3TrainingContractError(f'{name} must not be empty.')
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise M3TrainingContractError(f'{name} contains duplicate subjects: {duplicates}.')
    return tuple(normalized)


def build_subject_split(dataset, train_subject_ids, val_subject_ids) -> SubjectSplit:
    """Resolve explicit subject-level splits against ready manifest records."""
    train_ids = _validate_subject_id_list(train_subject_ids, 'train_subject_ids')
    val_ids = _validate_subject_id_list(val_subject_ids, 'val_subject_ids')
    overlap = sorted(set(train_ids).intersection(val_ids))
    if overlap:
        raise M3TrainingContractError(f'train/val subject leakage detected: {overlap}.')

    records = getattr(dataset, 'records', None)
    if not isinstance(records, list):
        raise M3TrainingContractError('dataset must expose ready manifest records as a list.')
    subject_to_indices = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise M3TrainingContractError('dataset records must be mappings.')
        subject_id = record.get('subject_id')
        if (
            not isinstance(subject_id, str)
            or not subject_id.strip()
            or subject_id != subject_id.strip()
        ):
            raise M3TrainingContractError('every ready dataset record requires a subject_id.')
        subject_to_indices.setdefault(subject_id, []).append(index)

    requested = set(train_ids).union(val_ids)
    unknown = sorted(requested.difference(subject_to_indices))
    if unknown:
        raise M3TrainingContractError(
            f'split contains unknown or non-ready subjects: {unknown}.'
        )

    train_indices = tuple(
        index for subject_id in train_ids for index in subject_to_indices[subject_id]
    )
    val_indices = tuple(
        index for subject_id in val_ids for index in subject_to_indices[subject_id]
    )
    if not train_indices:
        raise M3TrainingContractError('resolved train split is empty.')
    if not val_indices:
        raise M3TrainingContractError('resolved val split is empty.')
    return SubjectSplit(train_ids, val_ids, train_indices, val_indices)


def build_formal_subject_split(dataset, protocol, fold_id) -> FormalSubjectSplit:
    """Resolve a formal fold only after exact ready-dataset validation."""
    records = getattr(dataset, 'records', None)
    if not isinstance(records, list):
        raise M3TrainingContractError('dataset must expose ready manifest records as a list.')
    subject_to_index = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise M3TrainingContractError('dataset records must be mappings.')
        subject_id = record.get('subject_id')
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise M3TrainingContractError('every ready dataset record requires a subject_id.')
        if subject_id in subject_to_index:
            raise M3TrainingContractError(
                f'formal dataset contains duplicate subject record: {subject_id!r}.'
            )
        subject_to_index[subject_id] = index
    try:
        validate_dataset_ready_subjects(protocol, tuple(subject_to_index))
        fold = resolve_fold(protocol, fold_id)
    except M3TrainingProtocolError as error:
        raise M3TrainingContractError(f'formal protocol validation failed: {error}') from error

    def indices(split_name):
        return tuple(subject_to_index[subject_id] for subject_id in fold[split_name])

    return FormalSubjectSplit(
        fold_id=fold['fold_id'],
        train_subject_ids=fold['train_subject_ids'],
        val_subject_ids=fold['val_subject_ids'],
        test_subject_ids=fold['test_subject_ids'],
        train_indices=indices('train_subject_ids'),
        val_indices=indices('val_subject_ids'),
        test_indices=indices('test_subject_ids'),
    )


def _trainable_parameters(module, module_name: str):
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise M3TrainingContractError(f'{module_name} has no trainable parameters.')
    return parameters


def collect_trainable_parameters(point_encoder, ct_encoder, matcher):
    groups = (
        ('PointEncoder', point_encoder),
        ('CTEncoder', ct_encoder),
        ('PointCTMatcher', matcher),
    )
    parameters = []
    seen = set()
    for module_name, module in groups:
        for parameter in _trainable_parameters(module, module_name):
            identity = id(parameter)
            if identity in seen:
                raise M3TrainingContractError(
                    f'a trainable parameter is shared across formal modules: {module_name}.'
                )
            seen.add(identity)
            parameters.append(parameter)
    return parameters


def create_optimizer(
    point_encoder,
    ct_encoder,
    matcher,
    *,
    learning_rate=TRAINING_SMOKE_DEFAULT_LEARNING_RATE,
    weight_decay=TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY,
):
    learning_rate = _require_finite_float(
        learning_rate,
        'learning_rate',
        allow_zero=False,
    )
    weight_decay = _require_finite_float(
        weight_decay,
        'weight_decay',
        allow_zero=True,
    )
    parameters = collect_trainable_parameters(point_encoder, ct_encoder, matcher)
    if not parameters:
        raise M3TrainingContractError('optimizer received no parameters.')
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _optimizer_parameter_ids(optimizer):
    parameter_ids = []
    for group in optimizer.param_groups:
        parameter_ids.extend(id(parameter) for parameter in group.get('params', ()))
    return parameter_ids


def _validate_optimizer_coverage(optimizer, formal_parameters):
    optimizer_ids = _optimizer_parameter_ids(optimizer)
    if not optimizer_ids:
        raise M3TrainingContractError('optimizer contains no parameters.')
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise M3TrainingContractError('optimizer contains duplicate parameters.')
    if set(optimizer_ids) != {id(parameter) for parameter in formal_parameters}:
        raise M3TrainingContractError(
            'optimizer parameters must exactly cover PointEncoder, CTEncoder, and PointCTMatcher.'
        )


def _to_device_tensor(value, name: str, device):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        raise M3TrainingContractError(f'{name} must be a tensor or NumPy array.')
    return tensor.to(device=device)


def _require_fields(mapping, fields, name: str):
    if not isinstance(mapping, Mapping):
        raise M3TrainingContractError(f'{name} must be a mapping.')
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise M3TrainingContractError(f'{name} is missing fields: {missing}.')


def _mapping_enabled(branch, artifact_fields, branch_name):
    enabled = branch.get('m4_defect_mapping_enabled', False)
    if not isinstance(enabled, bool):
        raise M3TrainingContractError(f'{branch_name} m4_defect_mapping_enabled must be bool.')
    present = [field for field in artifact_fields if field in branch]
    if not enabled and present:
        raise M3TrainingContractError(
            f'{branch_name} coarse defect mapping artifacts require explicit M4 mapping enablement.'
        )
    if enabled:
        _require_fields(branch, artifact_fields, f'{branch_name} M4 mapping output')
    return enabled


def assemble_point_encoder_input(point_branch, device):
    sequence_fields = ('points', 'neighbors', 'subsampling', 'upsampling')
    _require_fields(
        point_branch,
        ('features', 'point_network_scale_mm_to_m') + sequence_fields,
        'Point preprocessing output',
    )
    inputs = {
        'features': _to_device_tensor(point_branch['features'], 'features', device),
        'point_network_scale_mm_to_m': point_branch['point_network_scale_mm_to_m'],
    }
    for field in sequence_fields:
        values = point_branch[field]
        if not isinstance(values, (list, tuple)):
            raise M3TrainingContractError(f'Point field {field} must be a list or tuple.')
        inputs[field] = [
            _to_device_tensor(value, f'{field}[{index}]', device)
            for index, value in enumerate(values)
        ]
    mapping_fields = (
        'point_intact_coarse',
        'point_raw_total_count_coarse',
        'point_raw_defect_count_coarse',
    )
    if _mapping_enabled(point_branch, mapping_fields, 'Point'):
        inputs['m4_defect_mapping_enabled'] = True
        inputs.update(
            {
                field: _to_device_tensor(point_branch[field], field, device)
                for field in mapping_fields
            }
        )
    return inputs


def assemble_ct_encoder_input(ct_branch, device):
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
    _require_fields(
        ct_branch,
        tensor_fields + metadata_fields,
        'CT preprocessing output',
    )
    inputs = {
        field: _to_device_tensor(ct_branch[field], field, device)
        for field in tensor_fields
    }
    inputs.update({field: ct_branch[field] for field in metadata_fields})
    mapping_fields = (
        'ct_intact_coarse',
        'ct_raw_total_count_coarse',
        'ct_raw_defect_count_coarse',
    )
    if _mapping_enabled(ct_branch, mapping_fields, 'CT'):
        inputs['m4_defect_mapping_enabled'] = True
        inputs.update(
            {
                field: _to_device_tensor(ct_branch[field], field, device)
                for field in mapping_fields
            }
        )
    return inputs


def _model_device(module, module_name: str):
    parameters = _trainable_parameters(module, module_name)
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise M3TrainingContractError(f'{module_name} trainable parameters span multiple devices.')
    return next(iter(devices))


def _validate_fp32_modules(point_encoder, ct_encoder, matcher, device):
    for module_name, module in (
        ('PointEncoder', point_encoder),
        ('CTEncoder', ct_encoder),
        ('PointCTMatcher', matcher),
    ):
        if _model_device(module, module_name) != device:
            raise M3TrainingContractError(f'{module_name} is not on the requested device.')
        for parameter in _trainable_parameters(module, module_name):
            if parameter.dtype != torch.float32:
                raise M3TrainingContractError(
                    f'{module_name} trainable parameters must use float32.'
                )


def _sample_subject_id(sample):
    if not isinstance(sample, Mapping):
        raise M3TrainingContractError('sample must be a mapping.')
    subject_id = sample.get('subject_id')
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise M3TrainingContractError('sample requires a non-empty subject_id.')
    return subject_id


def _prepare_model_inputs(
    sample,
    point_collate_fn,
    ct_collate_fn,
    device,
    *,
    include_osseous_ct_metadata=False,
):
    if not isinstance(include_osseous_ct_metadata, bool):
        raise M3TrainingContractError('include_osseous_ct_metadata must be bool.')
    subject_id = _sample_subject_id(sample)
    point_batch = point_collate_fn([sample])
    ct_batch = ct_collate_fn([sample])
    _require_fields(point_batch, ('subject_id', 'point'), 'Point batch')
    _require_fields(ct_batch, ('subject_id', 'ct'), 'CT batch')
    if point_batch['subject_id'] != subject_id or ct_batch['subject_id'] != subject_id:
        raise M3TrainingContractError('preprocessing changed the sample subject identity.')
    point_input = assemble_point_encoder_input(point_batch['point'], device)
    ct_input = assemble_ct_encoder_input(ct_batch['ct'], device)
    if not include_osseous_ct_metadata:
        return point_input, ct_input

    raw_fields = ('ct_volume', 'ct_spacing', 'ct_origin', 'ct_direction')
    _require_fields(ct_batch['ct'], raw_fields, 'M4 osseous defective CT input')
    osseous_ct_metadata = {field: ct_batch['ct'][field] for field in raw_fields}
    return point_input, ct_input, osseous_ct_metadata


def _require_encoder_outputs(point_output, ct_output, device):
    _require_fields(point_output, ('Q', 'Xp_phys_coarse'), 'PointEncoder output')
    _require_fields(ct_output, ('K', 'Xv_phys_coarse'), 'CTEncoder output')
    q = point_output['Q']
    k = ct_output['K']
    point_phys = point_output['Xp_phys_coarse']
    ct_phys = ct_output['Xv_phys_coarse']
    for name, tensor in (
        ('Q', q),
        ('K', k),
        ('Xp_phys_coarse', point_phys),
        ('Xv_phys_coarse', ct_phys),
    ):
        if not torch.is_tensor(tensor):
            raise M3TrainingContractError(f'{name} must be a torch.Tensor.')
        if tensor.device != device:
            raise M3TrainingContractError(f'{name} is on the wrong device.')
        if tensor.dtype != torch.float32:
            raise M3TrainingContractError(f'{name} must use float32.')
        if not bool(torch.isfinite(tensor).all()):
            raise M3TrainingContractError(f'{name} contains NaN or Inf.')
    if q.ndim != 2 or k.ndim != 2 or q.shape[0] == 0 or k.shape[0] == 0:
        raise M3TrainingContractError('Q and K must be non-empty descriptor matrices.')
    if point_phys.shape != (q.shape[0], 3):
        raise M3TrainingContractError('Point descriptors and physical coordinates are inconsistent.')
    if ct_phys.shape != (k.shape[0], 3):
        raise M3TrainingContractError('CT descriptors and physical coordinates are inconsistent.')
    return q, k, point_phys, ct_phys


def _build_gt_labels(
    sample,
    point_phys,
    ct_phys,
    device,
    primary_max_distance_mm,
    high_confidence_distance_mm,
    *,
    m4_hard_constraint_enabled,
    point_valid_mask,
    ct_valid_mask,
):
    _require_fields(
        sample,
        ('gt_transform', 'gt_transform_direction'),
        'sample GT metadata',
    )
    correspondence_kwargs = {
        'Xp_phys_coarse': point_phys.detach().cpu().numpy(),
        'Xv_phys_coarse': ct_phys.detach().cpu().numpy(),
        'gt_transform': sample['gt_transform'],
        'gt_transform_direction': sample['gt_transform_direction'],
        'primary_max_distance_mm': primary_max_distance_mm,
        'high_confidence_distance_mm': high_confidence_distance_mm,
    }
    if m4_hard_constraint_enabled:
        correspondence = build_m4_mask_aware_gt(
            **correspondence_kwargs,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
        )
    else:
        # Preserve the frozen M3 builder call exactly when no M4 artifacts exist.
        correspondence = build_coarse_gt_correspondence(**correspondence_kwargs)
    _require_fields(
        correspondence,
        ('gt_primary_ct_index', 'gt_primary_valid', 'gt_high_confidence'),
        'GT correspondence output',
    )
    primary_index = torch.as_tensor(
        correspondence['gt_primary_ct_index'],
        dtype=torch.int64,
        device=device,
    )
    primary_valid = torch.as_tensor(
        correspondence['gt_primary_valid'],
        dtype=torch.bool,
        device=device,
    )
    high_confidence = np.asarray(correspondence['gt_high_confidence'])
    if high_confidence.shape != (point_phys.shape[0],) or high_confidence.dtype != np.bool_:
        raise M3TrainingContractError('gt_high_confidence must be a bool array with shape [Np].')
    if primary_index.shape != (point_phys.shape[0],):
        raise M3TrainingContractError('gt_primary_ct_index must have shape [Np].')
    if primary_valid.shape != (point_phys.shape[0],):
        raise M3TrainingContractError('gt_primary_valid must have shape [Np].')
    return primary_index, primary_valid, int(np.count_nonzero(high_confidence))


def _forward_and_loss(
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    *,
    point_collate_fn,
    ct_collate_fn,
    primary_max_distance_mm,
    high_confidence_distance_mm,
    device,
    m4_soft_modulation_enabled=False,
    m4_soft_sigma_mm=None,
    m4_soft_strength=None,
    m4_osseous_prior_enabled=False,
    m4_osseous_strength=None,
):
    if not isinstance(m4_soft_modulation_enabled, bool):
        raise M3TrainingContractError('m4_soft_modulation_enabled must be bool.')
    if not isinstance(m4_osseous_prior_enabled, bool):
        raise M3TrainingContractError('m4_osseous_prior_enabled must be bool.')
    if m4_osseous_prior_enabled and not m4_soft_modulation_enabled:
        raise M3TrainingContractError(
            'M4 osseous prior requires active M4 soft modulation.'
        )
    if m4_osseous_prior_enabled:
        m4_osseous_strength = _require_finite_float(
            m4_osseous_strength,
            'm4_osseous_strength',
            allow_zero=True,
        )
        point_input, ct_input, osseous_ct_metadata = _prepare_model_inputs(
            sample,
            point_collate_fn,
            ct_collate_fn,
            device,
            include_osseous_ct_metadata=True,
        )
    else:
        if m4_osseous_strength is not None:
            raise M3TrainingContractError(
                'm4_osseous_strength requires m4_osseous_prior_enabled=true.'
            )
        point_input, ct_input = _prepare_model_inputs(
            sample,
            point_collate_fn,
            ct_collate_fn,
            device,
        )
        osseous_ct_metadata = None
    point_output = point_encoder(point_input)
    ct_output = ct_encoder(ct_input)
    q, k, point_phys, ct_phys = _require_encoder_outputs(point_output, ct_output, device)
    hard_constraint = resolve_m4_matching_masks(point_output, ct_output, device)
    point_valid_mask = hard_constraint['point_valid_mask']
    ct_valid_mask = hard_constraint['ct_valid_mask']

    soft_diagnostics = {
        'm4_soft_modulation_enabled': False,
        'm4_soft_modulation_active': False,
        'm4_soft_sigma_mm': None,
        'm4_soft_strength': None,
        'point_reliability_min': None,
        'point_reliability_mean': None,
        'point_reliability_max': None,
        'ct_reliability_min': None,
        'ct_reliability_mean': None,
        'ct_reliability_max': None,
        'pair_reliability_min': None,
        'pair_reliability_mean': None,
        'pair_reliability_max': None,
        'base_similarity_mean': None,
        'modulated_similarity_mean': None,
        'soft_similarity_changed': False,
    }
    osseous_diagnostics = {
        'm4_osseous_prior_enabled': False,
        'm4_osseous_prior_active': False,
        'm4_osseous_strength': None,
        'osseous_strength': None,
        'osseous_center_hu': None,
        'osseous_tau_hu': None,
        'osseous_radius_mm': None,
        'osseous_score_min': None,
        'osseous_score_mean': None,
        'osseous_score_max': None,
        'osseous_pair_reliability_min': None,
        'osseous_pair_reliability_mean': None,
        'osseous_pair_reliability_max': None,
        'osseous_penalty_min': None,
        'osseous_penalty_mean': None,
        'osseous_penalty_max': None,
        'final_modulation_min': None,
        'final_modulation_mean': None,
        'final_modulation_max': None,
        'final_similarity_mean': None,
        'osseous_similarity_changed': False,
    }
    if m4_osseous_prior_enabled:
        if hard_constraint['enabled'] is not True:
            raise M3TrainingContractError(
                'M4 osseous prior requires active M4 defect mapping and hard masks.'
            )
        try:
            matcher_output = run_m4_osseous_integrated_matching(
                q=q,
                k=k,
                point_coordinates_mm=point_phys,
                ct_token_locations_mm=ct_phys,
                point_valid_mask=point_valid_mask,
                ct_valid_mask=ct_valid_mask,
                sigma_mm=m4_soft_sigma_mm,
                soft_strength=m4_soft_strength,
                osseous_strength=m4_osseous_strength,
                matcher=matcher,
                defective_ct_volume=osseous_ct_metadata['ct_volume'],
                ct_spacing=osseous_ct_metadata['ct_spacing'],
                ct_origin=osseous_ct_metadata['ct_origin'],
                ct_direction=osseous_ct_metadata['ct_direction'],
                support_locations_mm=ct_input['ct_support_phys_20mm'],
            )
        except M4OsseousIntegrationError as error:
            raise M3TrainingContractError(
                f'M4 osseous-integrated matching failed: {error}'
            ) from error
        for field in tuple(soft_diagnostics):
            if field == 'modulated_similarity_mean':
                soft_diagnostics[field] = matcher_output['soft_similarity_mean']
            else:
                soft_diagnostics[field] = matcher_output[field]
        for field in tuple(osseous_diagnostics):
            osseous_diagnostics[field] = matcher_output[field]
    elif m4_soft_modulation_enabled:
        if hard_constraint['enabled'] is not True:
            raise M3TrainingContractError(
                'M4 soft modulation requires active M4 defect mapping and hard masks.'
            )
        try:
            matcher_output = run_m4_soft_modulated_matching(
                q=q,
                k=k,
                point_coordinates_mm=point_phys,
                ct_coordinates_mm=ct_phys,
                point_valid_mask=point_valid_mask,
                ct_valid_mask=ct_valid_mask,
                sigma_mm=m4_soft_sigma_mm,
                strength=m4_soft_strength,
                matcher=matcher,
            )
        except M4SoftModulationError as error:
            raise M3TrainingContractError(
                f'M4 soft-modulated matching failed: {error}'
            ) from error
        for field in tuple(soft_diagnostics):
            soft_diagnostics[field] = matcher_output[field]
    else:
        # Preserve the frozen M3/M4-1 matcher route when the explicitly
        # optional soft path is disabled.
        matcher_output = matcher(
            q=q,
            k=k,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
        )
    _require_fields(
        matcher_output,
        ('log_assignment', 'point_valid_mask', 'ct_valid_mask'),
        'PointCTMatcher output',
    )
    log_assignment = matcher_output['log_assignment']
    matcher_point_valid_mask = matcher_output['point_valid_mask']
    matcher_ct_valid_mask = matcher_output['ct_valid_mask']
    if (
        not torch.is_tensor(matcher_point_valid_mask)
        or not torch.equal(matcher_point_valid_mask, point_valid_mask)
    ):
        raise M3TrainingContractError(
            'PointCTMatcher changed the resolved Point hard-constraint mask.'
        )
    if (
        not torch.is_tensor(matcher_ct_valid_mask)
        or not torch.equal(matcher_ct_valid_mask, ct_valid_mask)
    ):
        raise M3TrainingContractError(
            'PointCTMatcher changed the resolved CT hard-constraint mask.'
        )
    if not torch.is_tensor(log_assignment) or log_assignment.device != device:
        raise M3TrainingContractError('log_assignment must be a tensor on the requested device.')
    if log_assignment.dtype != torch.float32:
        raise M3TrainingContractError('log_assignment must use float32.')
    if log_assignment.shape != (q.shape[0] + 1, k.shape[0] + 1):
        raise M3TrainingContractError('log_assignment shape is inconsistent with Q and K.')

    primary_index, primary_valid, high_confidence_count = _build_gt_labels(
        sample,
        point_phys,
        ct_phys,
        device,
        primary_max_distance_mm,
        high_confidence_distance_mm,
        m4_hard_constraint_enabled=hard_constraint['enabled'],
        point_valid_mask=point_valid_mask,
        ct_valid_mask=ct_valid_mask,
    )
    loss_output = compute_collision_aware_match_loss(
        log_assignment,
        primary_index,
        primary_valid,
        point_valid_mask=point_valid_mask,
        ct_valid_mask=ct_valid_mask,
    )
    _require_fields(loss_output, ('loss',), 'M3-3 loss output')
    loss = loss_output['loss']
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise M3TrainingContractError('matching loss must be a scalar tensor.')
    if loss.dtype != torch.float32:
        raise M3TrainingContractError('matching loss must use float32.')
    if loss.device != device:
        raise M3TrainingContractError('matching loss must remain on the requested device.')
    if not bool(torch.isfinite(loss)):
        raise M3TrainingContractError('matching loss is NaN or Inf.')

    result = dict(loss_output)
    result.update(
        {
            'loss': loss,
            'subject_id': _sample_subject_id(sample),
            'Np': int(q.shape[0]),
            'Nv': int(k.shape[0]),
            'num_high_confidence_points': high_confidence_count,
            'm4_hard_constraint_enabled': hard_constraint['enabled'],
            'point_total_tokens': hard_constraint['point_total_count'],
            'point_intact_tokens': hard_constraint['point_intact_count'],
            'point_excluded_tokens': hard_constraint['point_excluded_count'],
            'ct_total_tokens': hard_constraint['ct_total_count'],
            'ct_intact_tokens': hard_constraint['ct_intact_count'],
            'ct_excluded_tokens': hard_constraint['ct_excluded_count'],
            'num_supervised_points_after_mask': int(
                loss_output['num_supervised_points']
            ),
            **soft_diagnostics,
            **osseous_diagnostics,
        }
    )
    return result


def _finite_gradient(gradient):
    if gradient.is_sparse:
        gradient = gradient.coalesce().values()
    return bool(torch.isfinite(gradient).all())


def _validate_gradients(module, module_name: str):
    parameters = _trainable_parameters(module, module_name)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise M3TrainingContractError(f'{module_name} received no gradient.')
    if not all(_finite_gradient(gradient) for gradient in gradients):
        raise M3TrainingContractError(f'{module_name} produced a NaN or Inf gradient.')


def run_training_step(
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    optimizer,
    *,
    point_collate_fn,
    ct_collate_fn,
    primary_max_distance_mm,
    high_confidence_distance_mm,
    device,
    m4_soft_modulation_enabled=False,
    m4_soft_sigma_mm=None,
    m4_soft_strength=None,
    m4_osseous_prior_enabled=False,
    m4_osseous_strength=None,
):
    """Run one batch-size-one FP32 optimizer step with the formal M3-3 loss."""
    device = torch.device(device)
    formal_parameters = collect_trainable_parameters(point_encoder, ct_encoder, matcher)
    _validate_optimizer_coverage(optimizer, formal_parameters)
    _validate_fp32_modules(point_encoder, ct_encoder, matcher, device)
    point_encoder.train()
    ct_encoder.train()
    matcher.train()
    optimizer.zero_grad(set_to_none=True)
    result = _forward_and_loss(
        sample,
        point_encoder,
        ct_encoder,
        matcher,
        point_collate_fn=point_collate_fn,
        ct_collate_fn=ct_collate_fn,
        primary_max_distance_mm=primary_max_distance_mm,
        high_confidence_distance_mm=high_confidence_distance_mm,
        device=device,
        m4_soft_modulation_enabled=m4_soft_modulation_enabled,
        m4_soft_sigma_mm=m4_soft_sigma_mm,
        m4_soft_strength=m4_soft_strength,
        m4_osseous_prior_enabled=m4_osseous_prior_enabled,
        m4_osseous_strength=m4_osseous_strength,
    )
    loss = result['loss']
    if not loss.requires_grad:
        raise M3TrainingContractError('training loss must require gradients.')
    loss.backward()
    _validate_gradients(point_encoder, 'PointEncoder')
    _validate_gradients(ct_encoder, 'CTEncoder')
    _validate_gradients(matcher, 'PointCTMatcher')
    optimizer.step()
    for parameter in formal_parameters:
        if not bool(torch.isfinite(parameter).all()):
            raise M3TrainingContractError('optimizer step produced a NaN or Inf parameter.')
    return result


def run_validation_step(
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    *,
    point_collate_fn,
    ct_collate_fn,
    primary_max_distance_mm,
    high_confidence_distance_mm,
    device,
    m4_soft_modulation_enabled=False,
    m4_soft_sigma_mm=None,
    m4_soft_strength=None,
    m4_osseous_prior_enabled=False,
    m4_osseous_strength=None,
):
    """Run one batch-size-one FP32 validation forward and restore prior modes."""
    device = torch.device(device)
    _validate_fp32_modules(point_encoder, ct_encoder, matcher, device)
    modules = (point_encoder, ct_encoder, matcher)
    previous_modes = tuple(module.training for module in modules)
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            result = _forward_and_loss(
                sample,
                point_encoder,
                ct_encoder,
                matcher,
                point_collate_fn=point_collate_fn,
                ct_collate_fn=ct_collate_fn,
                primary_max_distance_mm=primary_max_distance_mm,
                high_confidence_distance_mm=high_confidence_distance_mm,
                device=device,
                m4_soft_modulation_enabled=m4_soft_modulation_enabled,
                m4_soft_sigma_mm=m4_soft_sigma_mm,
                m4_soft_strength=m4_soft_strength,
                m4_osseous_prior_enabled=m4_osseous_prior_enabled,
                m4_osseous_strength=m4_osseous_strength,
            )
        if result['loss'].requires_grad:
            raise M3TrainingContractError('validation loss must not require gradients.')
        return result
    finally:
        for module, was_training in zip(modules, previous_modes):
            module.train(was_training)


def aggregate_step_results(results):
    results = list(results)
    if not results:
        raise M3TrainingContractError('cannot aggregate an empty case list.')
    losses = []
    for result in results:
        loss = result.get('loss')
        if torch.is_tensor(loss):
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise M3TrainingContractError('case result contains an invalid loss.')
            losses.append(float(loss.item()))
        else:
            losses.append(_require_any_finite_float(loss, 'case loss'))
    return {
        'mean_loss': float(sum(losses) / len(losses)),
        'num_cases': len(results),
        'num_supervised_points': sum(int(item['num_supervised_points']) for item in results),
        'num_supervised_groups': sum(int(item['num_supervised_groups']) for item in results),
        'num_collision_groups': sum(int(item['num_collision_groups']) for item in results),
    }


def _normalize_training_config(training_config):
    if is_dataclass(training_config):
        training_config = asdict(training_config)
    if not isinstance(training_config, Mapping):
        raise M3TrainingContractError('training_config must be a mapping or dataclass.')
    try:
        return json.loads(json.dumps(dict(training_config), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise M3TrainingContractError('training_config must contain JSON-serializable values.') from error


def _checkpoint_payload(
    *,
    epoch,
    global_step,
    point_encoder,
    ct_encoder,
    matcher,
    optimizer,
    train_subject_ids,
    val_subject_ids,
    seed,
    training_config,
    best_val_loss,
    formal_protocol=False,
    protocol_version=None,
    protocol_hash=None,
    fold_id=None,
    test_subject_ids=None,
    perturbation_root_seed=None,
    perturbation_seed_scheme_version=None,
):
    epoch = _require_nonnegative_integer(epoch, 'epoch')
    global_step = _require_nonnegative_integer(global_step, 'global_step')
    seed = _require_nonnegative_integer(seed, 'seed')
    train_ids = _validate_subject_id_list(train_subject_ids, 'train_subject_ids')
    val_ids = _validate_subject_id_list(val_subject_ids, 'val_subject_ids')
    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise M3TrainingContractError('checkpoint split contains train/val leakage.')
    best_val_loss = _require_best_val_loss(best_val_loss)
    if not isinstance(formal_protocol, bool):
        raise M3TrainingContractError('formal_protocol must be a boolean.')
    payload = {
        'checkpoint_version': CHECKPOINT_VERSION if formal_protocol else SMOKE_CHECKPOINT_VERSION,
        'epoch': epoch,
        'global_step': global_step,
        'point_encoder_state_dict': point_encoder.state_dict(),
        'ct_encoder_state_dict': ct_encoder.state_dict(),
        'matcher_state_dict': matcher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_subject_ids': list(train_ids),
        'val_subject_ids': list(val_ids),
        'seed': seed,
        'training_config': _normalize_training_config(training_config),
        'best_val_loss': best_val_loss,
    }
    formal_values = (
        protocol_version,
        protocol_hash,
        fold_id,
        test_subject_ids,
        perturbation_root_seed,
        perturbation_seed_scheme_version,
    )
    if not formal_protocol:
        if any(value is not None for value in formal_values):
            raise M3TrainingContractError(
                'manual/smoke checkpoints must not contain formal protocol metadata.'
            )
        return payload

    for value, name in (
        (protocol_version, 'protocol_version'),
        (protocol_hash, 'protocol_hash'),
        (fold_id, 'fold_id'),
        (perturbation_seed_scheme_version, 'perturbation_seed_scheme_version'),
    ):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise M3TrainingContractError(f'{name} must be a non-empty formal checkpoint string.')
    if len(protocol_hash) != 64 or any(
        character not in '0123456789abcdef' for character in protocol_hash
    ):
        raise M3TrainingContractError(
            'protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    test_ids = _validate_subject_id_list(test_subject_ids, 'test_subject_ids')
    if set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise M3TrainingContractError('formal checkpoint split contains test leakage.')
    root_seed = _require_nonnegative_integer(
        perturbation_root_seed,
        'perturbation_root_seed',
    )
    payload.update(
        {
            'formal_protocol': True,
            'protocol_version': protocol_version,
            'protocol_hash': protocol_hash,
            'fold_id': fold_id,
            'test_subject_ids': list(test_ids),
            'perturbation_root_seed': root_seed,
            'perturbation_seed_scheme_version': perturbation_seed_scheme_version,
        }
    )
    return payload


def save_checkpoint(path, **checkpoint_fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(**checkpoint_fields)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f'.{path.name}.',
            suffix='.tmp',
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except Exception as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if isinstance(error, M3TrainingContractError):
            raise
        raise M3TrainingContractError(f'cannot save checkpoint {path}: {error}') from error
    return path


def _load_checkpoint_file(path, map_location):
    try:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)
    except Exception as error:
        raise M3TrainingContractError(f'cannot load checkpoint {path}: {error}') from error


def _validate_checkpoint(payload):
    required = {
        'checkpoint_version',
        'epoch',
        'global_step',
        'point_encoder_state_dict',
        'ct_encoder_state_dict',
        'matcher_state_dict',
        'optimizer_state_dict',
        'train_subject_ids',
        'val_subject_ids',
        'seed',
        'training_config',
        'best_val_loss',
    }
    if not isinstance(payload, Mapping):
        raise M3TrainingContractError('checkpoint payload must be a mapping.')
    missing = sorted(required.difference(payload))
    if missing:
        raise M3TrainingContractError(f'checkpoint is malformed; missing fields: {missing}.')
    checkpoint_version = payload['checkpoint_version']
    if isinstance(checkpoint_version, bool) or checkpoint_version not in {
        SMOKE_CHECKPOINT_VERSION,
        CHECKPOINT_VERSION,
    }:
        raise M3TrainingContractError('checkpoint version is unsupported.')
    _require_nonnegative_integer(payload['epoch'], 'checkpoint epoch')
    _require_nonnegative_integer(payload['global_step'], 'checkpoint global_step')
    _require_nonnegative_integer(payload['seed'], 'checkpoint seed')
    train_ids = _validate_subject_id_list(payload['train_subject_ids'], 'checkpoint train_subject_ids')
    val_ids = _validate_subject_id_list(payload['val_subject_ids'], 'checkpoint val_subject_ids')
    if set(train_ids).intersection(val_ids):
        raise M3TrainingContractError('checkpoint split contains train/val leakage.')
    for field in (
        'point_encoder_state_dict',
        'ct_encoder_state_dict',
        'matcher_state_dict',
        'optimizer_state_dict',
        'training_config',
    ):
        if not isinstance(payload[field], Mapping):
            raise M3TrainingContractError(f'checkpoint field {field} must be a mapping.')
    _require_best_val_loss(payload['best_val_loss'], 'checkpoint best_val_loss')
    if checkpoint_version == SMOKE_CHECKPOINT_VERSION:
        if payload.get('formal_protocol') is True:
            raise M3TrainingContractError(
                'checkpoint v1 cannot claim to be a formal protocol checkpoint.'
            )
        return {
            'formal_protocol': False,
            'train_subject_ids': train_ids,
            'val_subject_ids': val_ids,
            'test_subject_ids': (),
        }

    formal_fields = {
        'formal_protocol',
        'protocol_version',
        'protocol_hash',
        'fold_id',
        'test_subject_ids',
        'perturbation_root_seed',
        'perturbation_seed_scheme_version',
    }
    formal_missing = sorted(formal_fields.difference(payload))
    if formal_missing:
        raise M3TrainingContractError(
            f'formal checkpoint is malformed; missing fields: {formal_missing}.'
        )
    if payload['formal_protocol'] is not True:
        raise M3TrainingContractError('checkpoint v2 requires formal_protocol=true.')
    for field in (
        'protocol_version',
        'protocol_hash',
        'fold_id',
        'perturbation_seed_scheme_version',
    ):
        value = payload[field]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise M3TrainingContractError(f'checkpoint field {field} must be a non-empty string.')
    if len(payload['protocol_hash']) != 64 or any(
        character not in '0123456789abcdef' for character in payload['protocol_hash']
    ):
        raise M3TrainingContractError(
            'checkpoint protocol_hash must be a lowercase SHA-256 hex digest.'
        )
    test_ids = _validate_subject_id_list(payload['test_subject_ids'], 'checkpoint test_subject_ids')
    if set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise M3TrainingContractError('formal checkpoint split contains test leakage.')
    _require_nonnegative_integer(
        payload['perturbation_root_seed'],
        'checkpoint perturbation_root_seed',
    )
    return {
        'formal_protocol': True,
        'train_subject_ids': train_ids,
        'val_subject_ids': val_ids,
        'test_subject_ids': test_ids,
    }


def load_checkpoint(
    path,
    point_encoder,
    ct_encoder,
    matcher,
    optimizer,
    *,
    train_subject_ids,
    val_subject_ids,
    map_location,
    expected_training_config=None,
    formal_protocol=False,
    protocol_version=None,
    protocol_hash=None,
    fold_id=None,
    test_subject_ids=None,
    perturbation_root_seed=None,
    perturbation_seed_scheme_version=None,
):
    path = Path(path)
    if not path.is_file():
        raise M3TrainingContractError(f'checkpoint does not exist: {path}.')
    payload = _load_checkpoint_file(path, map_location)
    checkpoint_contract = _validate_checkpoint(payload)
    current_train_ids = _validate_subject_id_list(train_subject_ids, 'train_subject_ids')
    current_val_ids = _validate_subject_id_list(val_subject_ids, 'val_subject_ids')
    if (
        checkpoint_contract['train_subject_ids'] != current_train_ids
        or checkpoint_contract['val_subject_ids'] != current_val_ids
    ):
        raise M3TrainingContractError('checkpoint split does not match the current split contract.')
    if not isinstance(formal_protocol, bool):
        raise M3TrainingContractError('formal_protocol must be a boolean.')
    if formal_protocol:
        if not checkpoint_contract['formal_protocol']:
            raise M3TrainingContractError(
                'old manual/smoke checkpoint cannot resume into formal protocol mode.'
            )
        current_test_ids = _validate_subject_id_list(test_subject_ids, 'test_subject_ids')
        comparisons = (
            ('protocol_version', protocol_version),
            ('protocol_hash', protocol_hash),
            ('fold_id', fold_id),
            ('perturbation_root_seed', perturbation_root_seed),
            ('perturbation_seed_scheme_version', perturbation_seed_scheme_version),
        )
        for field, expected in comparisons:
            if payload[field] != expected:
                raise M3TrainingContractError(
                    f'checkpoint {field} does not match the current formal protocol.'
                )
        if checkpoint_contract['test_subject_ids'] != current_test_ids:
            raise M3TrainingContractError(
                'checkpoint test_subject_ids do not match the current formal protocol.'
            )
    elif checkpoint_contract['formal_protocol']:
        raise M3TrainingContractError(
            'formal protocol checkpoint cannot resume into manual/smoke mode.'
        )
    if expected_training_config is not None:
        expected = _normalize_training_config(expected_training_config)
        if payload['training_config'] != expected:
            raise M3TrainingContractError('checkpoint training_config does not match the current config.')
    try:
        point_encoder.load_state_dict(payload['point_encoder_state_dict'], strict=True)
        ct_encoder.load_state_dict(payload['ct_encoder_state_dict'], strict=True)
        matcher.load_state_dict(payload['matcher_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    except Exception as error:
        raise M3TrainingContractError(f'checkpoint state restoration failed: {error}') from error
    return {
        'epoch': int(payload['epoch']),
        'global_step': int(payload['global_step']),
        'best_val_loss': float(payload['best_val_loss']),
        'seed': int(payload['seed']),
        'training_config': dict(payload['training_config']),
        'formal_protocol': bool(checkpoint_contract['formal_protocol']),
    }


def save_epoch_checkpoints(
    checkpoint_dir,
    *,
    val_loss,
    best_val_loss,
    **checkpoint_fields,
):
    val_loss = _require_any_finite_float(val_loss, 'val_loss')
    best_val_loss = _require_best_val_loss(best_val_loss)
    improved = val_loss < best_val_loss
    new_best_val_loss = val_loss if improved else best_val_loss
    checkpoint_dir = Path(checkpoint_dir)
    fields = dict(checkpoint_fields)
    fields['best_val_loss'] = new_best_val_loss
    last_path = save_checkpoint(checkpoint_dir / 'last.pt', **fields)
    best_path = None
    if improved:
        best_path = save_checkpoint(checkpoint_dir / 'best_val_loss.pt', **fields)
    return {
        'improved': improved,
        'best_val_loss': new_best_val_loss,
        'last_path': last_path,
        'best_path': best_path,
    }


__all__ = [
    'BATCH_SIZE',
    'CHECKPOINT_VERSION',
    'FormalSubjectSplit',
    'M3TrainingContractError',
    'PRECISION',
    'SMOKE_CHECKPOINT_VERSION',
    'SMOKE_SPLIT_STATUS',
    'SubjectSplit',
    'TRAINING_DEFAULTS_STATUS',
    'TRAINING_SMOKE_DEFAULT_LEARNING_RATE',
    'TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY',
    'TrainingConfig',
    'aggregate_step_results',
    'assemble_ct_encoder_input',
    'assemble_point_encoder_input',
    'build_formal_subject_split',
    'build_subject_split',
    'collect_trainable_parameters',
    'create_optimizer',
    'load_checkpoint',
    'run_training_step',
    'run_validation_step',
    'save_checkpoint',
    'save_epoch_checkpoints',
    'set_random_seed',
    'validate_training_config',
]
