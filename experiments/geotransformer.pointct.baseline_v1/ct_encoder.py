from typing import Sequence

import torch
import torch.nn as nn


class CTEncoderContractError(RuntimeError):
    pass


class CTEncoderDependencyError(RuntimeError):
    pass


def _load_spconv():
    try:
        import spconv.pytorch as spconv
    except ModuleNotFoundError as error:
        raise CTEncoderDependencyError(
            'M2-2 CTEncoder requires spconv.pytorch. Provision it on the GPU server; '
            'this project does not install dependencies automatically.'
        ) from error
    return spconv


def _shape_tuple(value, name: str):
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise CTEncoderContractError(f'{name} must contain three positive integers.') from error
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise CTEncoderContractError(f'{name} must contain three positive integers; got {shape}.')
    return shape


def recover_x20_support_rows(x20_indices_zyx, support_shape, support_linear):
    """Recover x20 feature rows in authoritative support order with fail-closed validation."""
    support_shape = _shape_tuple(support_shape, 'ct_support_spatial_shape_20mm')
    if not torch.is_tensor(x20_indices_zyx) or x20_indices_zyx.ndim != 2 or x20_indices_zyx.shape[1] != 3:
        raise CTEncoderContractError('x20 active indices must be a torch tensor with shape [N,3].')
    if x20_indices_zyx.dtype not in (torch.int32, torch.int64) or x20_indices_zyx.shape[0] == 0:
        raise CTEncoderContractError('x20 active indices must be a non-empty integer tensor.')
    limits = torch.as_tensor(support_shape, dtype=x20_indices_zyx.dtype, device=x20_indices_zyx.device)
    if bool(torch.any(x20_indices_zyx < 0)) or bool(torch.any(x20_indices_zyx >= limits)):
        raise CTEncoderContractError('x20 active indices contain an out-of-bounds grid index.')
    if not torch.is_tensor(support_linear) or support_linear.ndim != 1 or support_linear.shape[0] == 0:
        raise CTEncoderContractError('ct_support_linear_20mm must be a non-empty one-dimensional tensor.')
    if support_linear.dtype not in (torch.int32, torch.int64):
        raise CTEncoderContractError('ct_support_linear_20mm must use an integer dtype.')
    if support_linear.device != x20_indices_zyx.device:
        raise CTEncoderContractError('x20 active indices and support keys must be on the same device.')

    _, size_y, size_x = support_shape
    indices_i64 = x20_indices_zyx.to(dtype=torch.int64)
    x20_linear = indices_i64[:, 0] * size_y * size_x + indices_i64[:, 1] * size_x + indices_i64[:, 2]
    sorted_linear, sorted_order = torch.sort(x20_linear)
    if sorted_linear.shape[0] > 1 and bool(torch.any(sorted_linear[1:] == sorted_linear[:-1])):
        raise CTEncoderContractError('x20 active linear keys must be unique.')

    support_linear_i64 = support_linear.to(dtype=torch.int64)
    positions = torch.searchsorted(sorted_linear, support_linear_i64)
    in_range = positions < sorted_linear.shape[0]
    found = torch.zeros_like(in_range, dtype=torch.bool)
    found[in_range] = sorted_linear[positions[in_range]] == support_linear_i64[in_range]
    if not bool(torch.all(found)):
        missing_linear = support_linear_i64[~found].detach().cpu().tolist()
        raise CTEncoderContractError(
            f'x20 is missing {len(missing_linear)} external-surface support tokens; '
            f'first missing linear indices={missing_linear[:8]}.'
        )

    feature_rows = sorted_order[positions]
    if not torch.equal(x20_linear[feature_rows], support_linear_i64):
        raise CTEncoderContractError('Recovered x20 feature rows do not match authoritative support order.')
    return feature_rows


def validate_ct_defect_mapping(ct_dict, support_count, device):
    fields = (
        'ct_intact_coarse',
        'ct_raw_total_count_coarse',
        'ct_raw_defect_count_coarse',
    )
    enabled = ct_dict.get('m4_defect_mapping_enabled', False)
    if not isinstance(enabled, bool):
        raise CTEncoderContractError('m4_defect_mapping_enabled must be bool.')
    present = [name for name in fields if name in ct_dict]
    if not enabled:
        if present:
            raise CTEncoderContractError('CT coarse defect mapping artifacts require explicit M4 mapping enablement.')
        return {}
    missing = [name for name in fields if name not in ct_dict]
    if missing:
        raise CTEncoderContractError(f'CT M4 mapping input is missing fields: {missing}.')

    intact = ct_dict['ct_intact_coarse']
    total = ct_dict['ct_raw_total_count_coarse']
    defect = ct_dict['ct_raw_defect_count_coarse']
    for name, tensor, dtype in (
        ('ct_intact_coarse', intact, torch.bool),
        ('ct_raw_total_count_coarse', total, torch.int64),
        ('ct_raw_defect_count_coarse', defect, torch.int64),
    ):
        if not torch.is_tensor(tensor) or tensor.shape != (support_count,) or tensor.dtype != dtype:
            raise CTEncoderContractError(
                f'{name} must have shape ({support_count},) and dtype {dtype}.'
            )
        if tensor.device != device:
            raise CTEncoderContractError(f'{name} must share the authoritative CT support device.')
    if bool(torch.any(total <= 0)) or bool(torch.any(defect < 0)) or bool(torch.any(defect > total)):
        raise CTEncoderContractError('CT coarse defect contributor counts are invalid.')
    if not torch.equal(intact, defect == 0):
        raise CTEncoderContractError('ct_intact_coarse is inconsistent with raw defect counts.')
    return {name: ct_dict[name] for name in fields}


class CTEncoder(nn.Module):
    """Frozen Baseline V1 sparse CT encoder and external-support extractor."""

    def __init__(self, cfg):
        super().__init__()
        ct_cfg = cfg.ct
        frozen = {
            'physical_unit': 'mm',
            'foreground_hu': -500.0,
            'context_grid_mm': 5.0,
            'support_grid_mm': 20.0,
            'input_dim': 1,
            'stem_dim': 32,
            'mid_dim': 64,
            'coarse_raw_dim': 128,
            'projected_dim': 256,
            'hu_clip_min': -500.0,
            'hu_clip_max': 2000.0,
        }
        for name, expected in frozen.items():
            actual = getattr(ct_cfg, name)
            if actual != expected:
                raise CTEncoderContractError(
                    f'M2-2 cfg.ct.{name} must remain frozen at {expected!r}; got {actual!r}.'
                )

        self.input_dim = int(ct_cfg.input_dim)
        self.stem_dim = int(ct_cfg.stem_dim)
        self.mid_dim = int(ct_cfg.mid_dim)
        self.coarse_raw_dim = int(ct_cfg.coarse_raw_dim)
        self.projected_dim = int(ct_cfg.projected_dim)
        spconv = _load_spconv()

        self.encoder_5mm = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.input_dim,
                self.stem_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='ct_subm_5mm',
            ),
            nn.BatchNorm1d(self.stem_dim),
            nn.ReLU(),
        )
        self.encoder_10mm = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.stem_dim,
                self.mid_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                indice_key='ct_down_10mm',
            ),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
            spconv.SubMConv3d(
                self.mid_dim,
                self.mid_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='ct_subm_10mm',
            ),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
        )
        self.encoder_20mm = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.mid_dim,
                self.coarse_raw_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                indice_key='ct_down_20mm',
            ),
            nn.BatchNorm1d(self.coarse_raw_dim),
            nn.ReLU(),
            spconv.SubMConv3d(
                self.coarse_raw_dim,
                self.coarse_raw_dim,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='ct_subm_20mm',
            ),
            nn.BatchNorm1d(self.coarse_raw_dim),
            nn.ReLU(),
        )
        self.ct_proj = nn.Linear(self.coarse_raw_dim, self.projected_dim)
        self._sparse_tensor_cls = spconv.SparseConvTensor

    @staticmethod
    def _require_tensor(name, tensor, ndim=None):
        if not torch.is_tensor(tensor):
            raise CTEncoderContractError(f'{name} must be a torch.Tensor.')
        if ndim is not None and tensor.ndim != ndim:
            raise CTEncoderContractError(f'{name} must have {ndim} dimensions; got {tensor.ndim}.')
        if not torch.isfinite(tensor).all():
            raise CTEncoderContractError(f'{name} contains NaN or Inf.')

    @staticmethod
    def _linear_indices(indices_zyx, spatial_shape_zyx: Sequence[int]):
        _, size_y, size_x = spatial_shape_zyx
        indices = indices_zyx.to(dtype=torch.int64)
        return indices[:, 0] * size_y * size_x + indices[:, 1] * size_x + indices[:, 2]

    @staticmethod
    def _validate_indices(name, indices, spatial_shape):
        if not torch.is_tensor(indices) or indices.ndim != 2 or indices.shape[1] != 3:
            raise CTEncoderContractError(f'{name} must be a torch tensor with shape [N,3].')
        if indices.dtype not in (torch.int32, torch.int64):
            raise CTEncoderContractError(f'{name} must use an integer dtype.')
        if indices.shape[0] == 0:
            raise CTEncoderContractError(f'{name} must contain at least one active cell.')
        limits = torch.as_tensor(spatial_shape, dtype=indices.dtype, device=indices.device)
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= limits)):
            raise CTEncoderContractError(f'{name} contains an out-of-bounds grid index.')

    def forward(self, ct_dict):
        """Encode only ``data_dict['ct']``; batch size is frozen at one."""
        if ct_dict.get('physical_unit') != 'mm':
            raise CTEncoderContractError('CTEncoder requires an explicit physical_unit="mm".')
        coordinate_system = ct_dict.get('coordinate_system')
        if not isinstance(coordinate_system, str) or (
            'lps' not in coordinate_system.lower()
            and 'left-posterior-superior' not in coordinate_system.lower()
        ):
            raise CTEncoderContractError('CTEncoder requires explicit LPS physical coordinates.')

        required = (
            'ct_context_features',
            'ct_context_indices',
            'ct_context_spatial_shape',
            'ct_support_indices_20mm',
            'ct_support_linear_20mm',
            'ct_support_spatial_shape_20mm',
            'ct_support_phys_20mm',
        )
        missing = [name for name in required if name not in ct_dict]
        if missing:
            raise CTEncoderContractError(f'CTEncoder input is missing fields: {missing}.')

        features = ct_dict['ct_context_features']
        context_indices = ct_dict['ct_context_indices']
        support_indices = ct_dict['ct_support_indices_20mm']
        support_linear = ct_dict['ct_support_linear_20mm']
        support_phys = ct_dict['ct_support_phys_20mm']
        context_shape = _shape_tuple(ct_dict['ct_context_spatial_shape'], 'ct_context_spatial_shape')
        support_shape = _shape_tuple(
            ct_dict['ct_support_spatial_shape_20mm'],
            'ct_support_spatial_shape_20mm',
        )
        expected_support_shape = tuple((size + 3) // 4 for size in context_shape)
        if support_shape != expected_support_shape:
            raise CTEncoderContractError(
                f'20 mm support shape {support_shape} does not match two stride-2 levels '
                f'from 5 mm context shape {context_shape}: {expected_support_shape}.'
            )

        self._require_tensor('ct_context_features', features, ndim=2)
        if features.dtype != torch.float32 or features.shape[1] != self.input_dim:
            raise CTEncoderContractError(
                f'ct_context_features must be float32 [N,{self.input_dim}]; got '
                f'{features.dtype} {tuple(features.shape)}.'
            )
        if features.shape[0] == 0:
            raise CTEncoderContractError('ct_context_features must contain at least one active cell.')
        if bool(torch.any(features < 0.0)) or bool(torch.any(features > 1.0)):
            raise CTEncoderContractError('ct_context_features must contain normalised HU values in [0,1].')
        self._validate_indices('ct_context_indices', context_indices, context_shape)
        self._validate_indices('ct_support_indices_20mm', support_indices, support_shape)
        if features.shape[0] != context_indices.shape[0]:
            raise CTEncoderContractError('CT context feature/index counts are inconsistent.')
        if context_indices.device != features.device:
            raise CTEncoderContractError('CT context features and indices must be on the same device.')
        context_linear = self._linear_indices(context_indices, context_shape)
        if context_linear.shape[0] > 1 and not bool(torch.all(context_linear[1:] > context_linear[:-1])):
            raise CTEncoderContractError('CT context indices must be unique in deterministic ascending order.')

        if not torch.is_tensor(support_linear) or support_linear.ndim != 1:
            raise CTEncoderContractError('ct_support_linear_20mm must be a torch tensor with shape [Nv].')
        if support_linear.dtype not in (torch.int32, torch.int64):
            raise CTEncoderContractError('ct_support_linear_20mm must use an integer dtype.')
        self._require_tensor('ct_support_phys_20mm', support_phys, ndim=2)
        if not torch.is_floating_point(support_phys) or support_phys.shape[1] != 3:
            raise CTEncoderContractError('ct_support_phys_20mm must be floating point with shape [Nv,3].')
        support_count = int(support_linear.shape[0])
        if support_count == 0:
            raise CTEncoderContractError('CT external-surface support must contain at least one token.')
        if support_indices.shape[0] != support_count or support_phys.shape[0] != support_count:
            raise CTEncoderContractError('CT support index, linear-index, and physical-coordinate counts differ.')
        if support_indices.device != features.device or support_linear.device != features.device:
            raise CTEncoderContractError('All CT sparse/support indices must be on the feature device.')
        if support_phys.device != features.device:
            raise CTEncoderContractError('CT support physical coordinates must be on the feature device.')

        computed_support_linear = self._linear_indices(support_indices, support_shape)
        support_linear_i64 = support_linear.to(dtype=torch.int64)
        if not torch.equal(computed_support_linear, support_linear_i64):
            raise CTEncoderContractError('CT support grid indices and linear indices are inconsistent.')
        if support_count > 1 and not bool(torch.all(support_linear_i64[1:] > support_linear_i64[:-1])):
            raise CTEncoderContractError('CT support linear indices must be unique and ascending.')
        mapping_output = validate_ct_defect_mapping(ct_dict, support_count, features.device)

        batch_column = torch.zeros(
            (context_indices.shape[0], 1),
            dtype=torch.int32,
            device=context_indices.device,
        )
        sparse_indices = torch.cat(
            (batch_column, context_indices.to(dtype=torch.int32)),
            dim=1,
        ).contiguous()
        sparse_input = self._sparse_tensor_cls(
            features=features,
            indices=sparse_indices,
            spatial_shape=list(context_shape),
            batch_size=1,
        )
        x5 = self.encoder_5mm(sparse_input)
        x10 = self.encoder_10mm(x5)
        x20 = self.encoder_20mm(x10)
        for name, sparse_tensor, expected_dim in (
            ('x5.features', x5, self.stem_dim),
            ('x10.features', x10, self.mid_dim),
            ('x20.features', x20, self.coarse_raw_dim),
        ):
            self._require_tensor(name, sparse_tensor.features, ndim=2)
            if sparse_tensor.features.shape[0] == 0 or sparse_tensor.features.shape[1] != expected_dim:
                raise CTEncoderContractError(
                    f'{name} must have shape [N,{expected_dim}], N>0; '
                    f'got {tuple(sparse_tensor.features.shape)}.'
                )

        x20_shape = _shape_tuple(x20.spatial_shape, 'x20.spatial_shape')
        if x20_shape != support_shape:
            raise CTEncoderContractError(
                f'x20 spatial shape {x20_shape} does not match support shape {support_shape}.'
            )
        if x20.indices.ndim != 2 or x20.indices.shape[1] != 4:
            raise CTEncoderContractError('x20 sparse indices must have [batch,z,y,x] columns.')
        if not bool(torch.all(x20.indices[:, 0] == 0)):
            raise CTEncoderContractError('M2-2 CTEncoder supports batch_size=1 only.')

        feature_rows = recover_x20_support_rows(
            x20.indices[:, 1:],
            support_shape,
            support_linear_i64,
        )
        V_raw = x20.features[feature_rows]
        Xv_phys_coarse = support_phys.to(dtype=features.dtype)
        K = self.ct_proj(V_raw)

        for name, tensor in (
            ('V_raw', V_raw),
            ('K', K),
            ('Xv_phys_coarse', Xv_phys_coarse),
        ):
            self._require_tensor(name, tensor, ndim=2)
        if V_raw.shape != (support_count, self.coarse_raw_dim):
            raise CTEncoderContractError(
                f'V_raw must have shape [{support_count},{self.coarse_raw_dim}]; got {tuple(V_raw.shape)}.'
            )
        if K.shape != (support_count, self.projected_dim):
            raise CTEncoderContractError(
                f'K must have shape [{support_count},{self.projected_dim}]; got {tuple(K.shape)}.'
            )
        if Xv_phys_coarse.shape != (support_count, 3):
            raise CTEncoderContractError(
                f'Xv_phys_coarse must have shape [{support_count},3]; got {tuple(Xv_phys_coarse.shape)}.'
            )

        output = {
            'V_raw': V_raw,
            'K': K,
            'Xv_phys_coarse': Xv_phys_coarse,
            'x5_active_count': int(x5.indices.shape[0]),
            'x10_active_count': int(x10.indices.shape[0]),
            'x20_active_count': int(x20.indices.shape[0]),
            'support_count': support_count,
        }
        output.update(mapping_output)
        return output


def create_ct_encoder(cfg):
    return CTEncoder(cfg)


__all__ = [
    'CTEncoder',
    'CTEncoderContractError',
    'CTEncoderDependencyError',
    'create_ct_encoder',
    'recover_x20_support_rows',
    'validate_ct_defect_mapping',
]
