import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geotransformer.datasets.registration.pointct import PointCTDataset, pointct_collate_fn

from config import (
    CT_CONTEXT_GRID_MM,
    CT_FOREGROUND_HU,
    CT_HU_CLIP_MAX,
    CT_HU_CLIP_MIN,
    CT_PHYSICAL_UNIT,
    CT_SUPPORT_GRID_MM,
    POINT_INIT_RADIUS_M,
    POINT_INIT_VOXEL_SIZE_M,
    POINT_NEIGHBOR_LIMITS,
    POINT_NETWORK_SCALE_MM_TO_M,
    POINT_NUM_STAGES,
    POINT_PHYSICAL_UNIT,
)


class PointPreprocessingContractError(RuntimeError):
    pass


class CTPreprocessingContractError(RuntimeError):
    pass


class CTDefectAggregationContractError(RuntimeError):
    pass


def _require_explicit_mapping_flag(enable_m4_defect_mapping) -> bool:
    if not isinstance(enable_m4_defect_mapping, bool):
        raise TypeError('enable_m4_defect_mapping must be an explicit bool.')
    return enable_m4_defect_mapping


def create_dataset(data_root=None, **kwargs):
    """Create the manifest-driven M1 dataset without defining train/val/test splits."""
    if data_root is None:
        data_root = PROJECT_ROOT / 'local_data'
    return PointCTDataset(data_root, **kwargs)


def _require_vector(value, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise CTPreprocessingContractError(f'{name} must be a finite vector with shape (3,).')
    if positive and np.any(array <= 0):
        raise CTPreprocessingContractError(f'{name} must contain positive values.')
    return array


def _require_direction(value) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape == (9,):
        direction = direction.reshape(3, 3)
    if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
        raise CTPreprocessingContractError('ct_direction must be finite with shape (3,3).')
    if abs(float(np.linalg.det(direction))) < 1e-12:
        raise CTPreprocessingContractError('ct_direction must be invertible.')
    return direction


def _validate_physical_contract(physical_unit, coordinate_system=None):
    if physical_unit is None:
        raise CTPreprocessingContractError('M2-2 CT preprocessing requires an explicit physical_unit.')
    if physical_unit != CT_PHYSICAL_UNIT:
        raise CTPreprocessingContractError(
            f'M2-2 CT preprocessing requires physical_unit={CT_PHYSICAL_UNIT!r}; got {physical_unit!r}.'
        )
    if coordinate_system is not None:
        if not isinstance(coordinate_system, str) or not coordinate_system.strip():
            raise CTPreprocessingContractError('CT coordinate_system must be an explicit non-empty string.')
        normalised = coordinate_system.lower()
        if 'lps' not in normalised and 'left-posterior-superior' not in normalised:
            raise CTPreprocessingContractError(
                f'M2-2 CT physical coordinates must use LPS; got {coordinate_system!r}.'
            )


def _validate_volume_and_spacing(ct_volume, ct_spacing) -> Tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(ct_volume)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise CTPreprocessingContractError('ct_volume must be a non-empty 3D array in [z,y,x] order.')
    if not np.issubdtype(volume.dtype, np.number):
        raise CTPreprocessingContractError('ct_volume must contain numeric HU values.')
    if np.issubdtype(volume.dtype, np.floating) and not np.all(np.isfinite(volume)):
        raise CTPreprocessingContractError('ct_volume contains NaN or Inf.')
    spacing = _require_vector(ct_spacing, 'ct_spacing', positive=True)
    return volume, spacing


def _grid_spatial_shape_zyx(volume_shape_zyx, spacing_xyz, grid_mm: float) -> np.ndarray:
    if not np.isfinite(grid_mm) or grid_mm <= 0:
        raise CTPreprocessingContractError('CT grid size must be finite and positive.')
    # NumPy storage is [z,y,x], but spacing is an image-axis [x,y,z] vector.
    max_image_displacement_xyz = (np.asarray(volume_shape_zyx[::-1]) - 1) * spacing_xyz
    spatial_shape_xyz = np.floor(max_image_displacement_xyz / grid_mm).astype(np.int64) + 1
    return spatial_shape_xyz[::-1].copy()


def _linear_to_indices_zyx(linear: np.ndarray, spatial_shape_zyx: np.ndarray) -> np.ndarray:
    _, size_y, size_x = (int(value) for value in spatial_shape_zyx)
    qx = linear % size_x
    quotient = linear // size_x
    qy = quotient % size_y
    qz = quotient // size_y
    return np.stack((qz, qy, qx), axis=1)


def _whole_cell_axis_ranges(axis_size: int, spacing_mm: float) -> Dict[int, Tuple[int, int]]:
    """Map an image-axis cell index to its half-open raw-index range."""
    raw_indices = np.arange(axis_size, dtype=np.float64)
    cell_indices = np.floor(raw_indices * spacing_mm / CT_SUPPORT_GRID_MM).astype(np.int64)
    unique_cells, starts, counts = np.unique(cell_indices, return_index=True, return_counts=True)
    return {
        int(cell): (int(start), int(start + count))
        for cell, start, count in zip(unique_cells, starts, counts)
    }


def aggregate_ct_defect_whole_cells(
    ct_defect_mask,
    volume_shape_zyx,
    ct_spacing,
    ct_support_linear_20mm,
) -> Dict[str, np.ndarray]:
    """Aggregate intact/defect values over entire existing 20 mm image-axis cells.

    The authoritative support keys determine output row order. Cell membership is
    defined only by raw integer image indices and spacing; CT intensity, origin,
    direction, surface support, and encoder receptive fields are not inputs.
    """
    shape_array = np.asarray(volume_shape_zyx)
    if (
        shape_array.shape != (3,)
        or not np.issubdtype(shape_array.dtype, np.integer)
        or np.any(shape_array <= 0)
    ):
        raise CTDefectAggregationContractError(
            'volume_shape_zyx must contain three positive integer sizes in [z,y,x] order.'
        )
    volume_shape = tuple(int(value) for value in shape_array)

    mask = np.asarray(ct_defect_mask)
    if mask.ndim != 3 or mask.shape != volume_shape:
        raise CTDefectAggregationContractError(
            f'ct_defect_mask must have shape {volume_shape}; got {mask.shape}.'
        )
    if mask.dtype == np.bool_:
        intact_mask = mask
    elif mask.dtype == np.uint8:
        if not np.all((mask == 0) | (mask == 1)):
            raise CTDefectAggregationContractError('uint8 ct_defect_mask values must be binary {0,1}.')
        intact_mask = mask.astype(np.bool_, copy=False)
    else:
        raise CTDefectAggregationContractError('ct_defect_mask dtype must be bool or uint8.')

    try:
        spacing = _require_vector(ct_spacing, 'ct_spacing', positive=True)
        spatial_shape = _grid_spatial_shape_zyx(volume_shape, spacing, CT_SUPPORT_GRID_MM)
    except CTPreprocessingContractError as error:
        raise CTDefectAggregationContractError(str(error)) from error

    support = np.asarray(ct_support_linear_20mm)
    if support.ndim != 1 or support.size == 0 or not np.issubdtype(support.dtype, np.integer):
        raise CTDefectAggregationContractError(
            'ct_support_linear_20mm must be a non-empty one-dimensional integer array.'
        )
    support_linear = support.astype(np.int64, copy=False)
    if not np.array_equal(support, support_linear):
        raise CTDefectAggregationContractError('ct_support_linear_20mm contains values outside int64 range.')
    # Use np.unique only as validation; its sorted output never becomes row identity.
    if np.unique(support_linear).size != support_linear.size:
        raise CTDefectAggregationContractError('ct_support_linear_20mm must contain unique support keys.')
    if support_linear.size > 1 and np.any(support_linear[1:] <= support_linear[:-1]):
        raise CTDefectAggregationContractError(
            'ct_support_linear_20mm authoritative keys must be strictly increasing.'
        )

    total_cells = int(np.prod(spatial_shape, dtype=np.int64))
    if np.any(support_linear < 0) or np.any(support_linear >= total_cells):
        raise CTDefectAggregationContractError(
            f'ct_support_linear_20mm keys must be in [0, {total_cells}).'
        )
    support_indices = _linear_to_indices_zyx(support_linear, spatial_shape)

    z_ranges = _whole_cell_axis_ranges(volume_shape[0], spacing[2])
    y_ranges = _whole_cell_axis_ranges(volume_shape[1], spacing[1])
    x_ranges = _whole_cell_axis_ranges(volume_shape[2], spacing[0])

    raw_total_count = np.empty(support_linear.size, dtype=np.int64)
    raw_defect_count = np.empty(support_linear.size, dtype=np.int64)
    for row, (qz, qy, qx) in enumerate(support_indices):
        z_range = z_ranges.get(int(qz))
        y_range = y_ranges.get(int(qy))
        x_range = x_ranges.get(int(qx))
        if z_range is None or y_range is None or x_range is None:
            raise CTDefectAggregationContractError(
                f'ct_support_linear_20mm[{row}]={support_linear[row]} has no existing raw voxel.'
            )

        z_start, z_stop = z_range
        y_start, y_stop = y_range
        x_start, x_stop = x_range
        cell = intact_mask[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        total_count = int(cell.size)
        defect_count = total_count - int(np.count_nonzero(cell))
        if total_count <= 0 or defect_count < 0 or defect_count > total_count:
            raise CTDefectAggregationContractError('Invalid whole-cell aggregation count invariant.')
        raw_total_count[row] = total_count
        raw_defect_count[row] = defect_count

    ct_intact_coarse = raw_defect_count == 0
    return {
        'ct_intact_coarse': ct_intact_coarse,
        'ct_raw_total_count_coarse': raw_total_count,
        'ct_raw_defect_count_coarse': raw_defect_count,
    }


def build_ct_context_5mm(
    ct_volume,
    ct_spacing,
    *,
    physical_unit,
    foreground_hu: float = CT_FOREGROUND_HU,
    grid_mm: float = CT_CONTEXT_GRID_MM,
    hu_clip_min: float = CT_HU_CLIP_MIN,
    hu_clip_max: float = CT_HU_CLIP_MAX,
) -> Dict[str, np.ndarray]:
    """Aggregate foreground HU into deterministic 5 mm image-axis cells."""
    _validate_physical_contract(physical_unit)
    volume, spacing = _validate_volume_and_spacing(ct_volume, ct_spacing)
    if not np.isfinite(foreground_hu):
        raise CTPreprocessingContractError('foreground_hu must be finite.')
    if not np.isfinite(hu_clip_min) or not np.isfinite(hu_clip_max) or hu_clip_min >= hu_clip_max:
        raise CTPreprocessingContractError('HU clipping bounds must be finite and increasing.')

    spatial_shape = _grid_spatial_shape_zyx(volume.shape, spacing, grid_mm)
    size_z, size_y, size_x = (int(value) for value in spatial_shape)
    total_cells = size_z * size_y * size_x
    hu_sums = np.zeros(total_cells, dtype=np.float64)
    counts = np.zeros(total_cells, dtype=np.int64)

    # Work in small z slabs so real 500x500 CT volumes do not require a large
    # global foreground-coordinate array. Axis conversion remains explicit:
    # array [z,y,x] -> image-axis displacement [x*sx,y*sy,z*sz].
    slab_depth = 16
    for z_start in range(0, volume.shape[0], slab_depth):
        z_stop = min(z_start + slab_depth, volume.shape[0])
        slab = volume[z_start:z_stop]
        local_z, voxel_y, voxel_x = np.nonzero(slab > foreground_hu)
        if local_z.size == 0:
            continue
        voxel_z = local_z + z_start
        qx = np.floor(voxel_x * spacing[0] / grid_mm).astype(np.int64)
        qy = np.floor(voxel_y * spacing[1] / grid_mm).astype(np.int64)
        qz = np.floor(voxel_z * spacing[2] / grid_mm).astype(np.int64)
        linear = qz * size_y * size_x + qy * size_x + qx

        occupied, inverse = np.unique(linear, return_inverse=True)
        slab_counts = np.bincount(inverse)
        slab_sums = np.bincount(
            inverse,
            weights=slab[local_z, voxel_y, voxel_x].astype(np.float64, copy=False),
        )
        counts[occupied] += slab_counts
        hu_sums[occupied] += slab_sums

    occupied_linear = np.flatnonzero(counts)
    if occupied_linear.size == 0:
        raise CTPreprocessingContractError(
            f'CT has no foreground voxels satisfying HU > {foreground_hu}.'
        )
    mean_hu = hu_sums[occupied_linear] / counts[occupied_linear]
    normalised_hu = (np.clip(mean_hu, hu_clip_min, hu_clip_max) - hu_clip_min) / (
        hu_clip_max - hu_clip_min
    )
    return {
        'ct_context_features': normalised_hu.astype(np.float32).reshape(-1, 1),
        'ct_context_indices': _linear_to_indices_zyx(occupied_linear, spatial_shape).astype(np.int32),
        'ct_context_linear': occupied_linear.astype(np.int64, copy=False),
        'ct_context_spatial_shape': spatial_shape.astype(np.int64, copy=False),
    }


def _dilate_6(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :, :] |= mask[:-1, :, :]
    dilated[:-1, :, :] |= mask[1:, :, :]
    dilated[:, 1:, :] |= mask[:, :-1, :]
    dilated[:, :-1, :] |= mask[:, 1:, :]
    dilated[:, :, 1:] |= mask[:, :, :-1]
    dilated[:, :, :-1] |= mask[:, :, 1:]
    return dilated


def find_boundary_connected_outside_air(foreground: np.ndarray) -> np.ndarray:
    """Return background connected to a volume face under 6-neighbour moves."""
    foreground = np.asarray(foreground, dtype=bool)
    if foreground.ndim != 3 or any(size <= 0 for size in foreground.shape):
        raise CTPreprocessingContractError('foreground must be a non-empty 3D mask.')
    background = ~foreground
    seeds = np.zeros_like(foreground, dtype=bool)
    seeds[0, :, :] |= background[0, :, :]
    seeds[-1, :, :] |= background[-1, :, :]
    seeds[:, 0, :] |= background[:, 0, :]
    seeds[:, -1, :] |= background[:, -1, :]
    seeds[:, :, 0] |= background[:, :, 0]
    seeds[:, :, -1] |= background[:, :, -1]

    try:
        from scipy.ndimage import binary_propagation, generate_binary_structure
    except ModuleNotFoundError:
        # NumPy fallback keeps lightweight contract tests runnable. Real-volume
        # validation should use SciPy's compiled propagation implementation.
        outside_air = seeds
        while True:
            propagated = _dilate_6(outside_air) & background
            if np.array_equal(propagated, outside_air):
                return outside_air
            outside_air = propagated
    structure = generate_binary_structure(3, 1)
    return binary_propagation(seeds, structure=structure, mask=background)


def extract_external_surface(ct_volume, foreground_hu: float = CT_FOREGROUND_HU) -> np.ndarray:
    """Select foreground adjacent to boundary-connected air via 6-neighbours."""
    volume = np.asarray(ct_volume)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise CTPreprocessingContractError('ct_volume must be a non-empty 3D array.')
    foreground = volume > foreground_hu
    outside_air = find_boundary_connected_outside_air(foreground)
    external_surface = foreground & _dilate_6(outside_air)

    # A foreground voxel directly on a volume face has no in-volume air voxel
    # on its outward side, so the frozen support definition includes it here.
    external_surface[0, :, :] |= foreground[0, :, :]
    external_surface[-1, :, :] |= foreground[-1, :, :]
    external_surface[:, 0, :] |= foreground[:, 0, :]
    external_surface[:, -1, :] |= foreground[:, -1, :]
    external_surface[:, :, 0] |= foreground[:, :, 0]
    external_surface[:, :, -1] |= foreground[:, :, -1]
    return external_surface


def build_ct_support_20mm(
    external_surface,
    ct_spacing,
    ct_origin,
    ct_direction,
    *,
    volume_shape_zyx,
    physical_unit,
    coordinate_system,
    grid_mm: float = CT_SUPPORT_GRID_MM,
) -> Dict[str, np.ndarray]:
    """Quantise external surface and attach deterministic LPS/mm coordinates."""
    _validate_physical_contract(physical_unit, coordinate_system)
    spacing = _require_vector(ct_spacing, 'ct_spacing', positive=True)
    origin = _require_vector(ct_origin, 'ct_origin')
    direction = _require_direction(ct_direction)
    surface = np.asarray(external_surface, dtype=bool)
    expected_shape = tuple(int(value) for value in volume_shape_zyx)
    if surface.ndim != 3 or surface.shape != expected_shape:
        raise CTPreprocessingContractError(
            f'external_surface shape {surface.shape} does not match CT shape {expected_shape}.'
        )
    spatial_shape = _grid_spatial_shape_zyx(surface.shape, spacing, grid_mm)
    _, size_y, size_x = (int(value) for value in spatial_shape)

    voxel_z, voxel_y, voxel_x = np.nonzero(surface)
    if voxel_z.size == 0:
        raise CTPreprocessingContractError('CT external surface produced no support voxels.')
    qx = np.floor(voxel_x * spacing[0] / grid_mm).astype(np.int64)
    qy = np.floor(voxel_y * spacing[1] / grid_mm).astype(np.int64)
    qz = np.floor(voxel_z * spacing[2] / grid_mm).astype(np.int64)
    linear = qz * size_y * size_x + qy * size_x + qx
    support_linear = np.unique(linear)
    support_indices = _linear_to_indices_zyx(support_linear, spatial_shape)

    # The representative is the complete 20 mm image-axis cell centre even
    # when the last cell extends beyond the original array. This is a coarse
    # token coordinate, not an original voxel centre; no voxel offset is added.
    displacement_xyz_mm = (support_indices[:, ::-1].astype(np.float64) + 0.5) * grid_mm
    support_phys = origin[np.newaxis, :] + displacement_xyz_mm @ direction.T
    if not np.all(np.isfinite(support_phys)):
        raise CTPreprocessingContractError('CT support physical coordinates contain NaN or Inf.')
    return {
        'ct_support_indices_20mm': support_indices.astype(np.int32),
        'ct_support_linear_20mm': support_linear.astype(np.int64, copy=False),
        'ct_support_spatial_shape_20mm': spatial_shape.astype(np.int64, copy=False),
        'ct_support_phys_20mm': support_phys,
    }


def m2_ct_collate_fn(samples: List[Mapping], enable_m4_defect_mapping: bool = False):
    """Build M2-2 CT inputs while preserving the frozen M1 dual branches."""
    enable_m4_defect_mapping = _require_explicit_mapping_flag(enable_m4_defect_mapping)
    if len(samples) != 1:
        raise CTPreprocessingContractError(
            f'M2-2 CT preprocessing requires batch_size=1; got {len(samples)} samples.'
        )
    sample = samples[0]
    if enable_m4_defect_mapping:
        if not isinstance(sample.get('defect_id'), str) or not sample['defect_id'].strip():
            raise CTPreprocessingContractError('M4 CT mapping requires an explicit defect_id.')
        if 'ct_defect_mask' not in sample:
            raise CTPreprocessingContractError('M4 CT mapping requires ct_defect_mask.')
    if 'physical_unit' not in sample:
        raise CTPreprocessingContractError('M2-2 CT preprocessing requires an explicit physical_unit.')
    if 'coordinate_system' not in sample:
        raise CTPreprocessingContractError('M2-2 CT preprocessing requires an explicit coordinate_system.')

    collated = pointct_collate_fn(samples)
    ct_branch = collated['ct']
    if tuple(ct_branch.get('array_axis_order', ())) != ('z', 'y', 'x'):
        raise CTPreprocessingContractError('CT NumPy array_axis_order must be [z,y,x].')
    if tuple(ct_branch.get('image_index_convention', ())) != ('x', 'y', 'z'):
        raise CTPreprocessingContractError('CT image_index_convention must be [x,y,z].')
    volume, spacing = _validate_volume_and_spacing(ct_branch['ct_volume'], ct_branch['ct_spacing'])
    if tuple(np.asarray(ct_branch['ct_shape']).tolist()) != volume.shape:
        raise CTPreprocessingContractError('ct_shape does not match ct_volume.')
    _validate_physical_contract(sample['physical_unit'], sample['coordinate_system'])

    context = build_ct_context_5mm(
        volume,
        spacing,
        physical_unit=sample['physical_unit'],
    )
    external_surface = extract_external_surface(volume)
    support = build_ct_support_20mm(
        external_surface,
        spacing,
        ct_branch['ct_origin'],
        ct_branch['ct_direction'],
        volume_shape_zyx=volume.shape,
        physical_unit=sample['physical_unit'],
        coordinate_system=sample['coordinate_system'],
    )
    ct_branch.update(context)
    ct_branch.update(support)
    ct_branch['physical_unit'] = sample['physical_unit']
    ct_branch['coordinate_system'] = sample['coordinate_system']

    if enable_m4_defect_mapping:
        ct_mapping = aggregate_ct_defect_whole_cells(
            ct_branch['ct_defect_mask'],
            volume.shape,
            spacing,
            support['ct_support_linear_20mm'],
        )
        support_count = int(support['ct_support_linear_20mm'].shape[0])
        expected_shape = (support_count,)
        expected_dtypes = {
            'ct_intact_coarse': np.dtype(bool),
            'ct_raw_total_count_coarse': np.dtype(np.int64),
            'ct_raw_defect_count_coarse': np.dtype(np.int64),
        }
        for name, expected_dtype in expected_dtypes.items():
            value = ct_mapping.get(name)
            if not isinstance(value, np.ndarray) or value.shape != expected_shape or value.dtype != expected_dtype:
                raise CTPreprocessingContractError(
                    f'{name} must have shape {expected_shape} and dtype {expected_dtype}.'
                )
        ct_branch.update(ct_mapping)
        ct_branch['m4_defect_mapping_enabled'] = True

    if 'ct_metadata' in sample:
        ct_branch['ct_metadata'] = sample['ct_metadata']
    if 'pair_metadata' in sample:
        collated['pair_metadata'] = sample['pair_metadata']
    return collated


def _validate_point_sample(sample: Mapping) -> np.ndarray:
    if 'point_xyz_phys' not in sample:
        raise PointPreprocessingContractError('M2 Point preprocessing requires point_xyz_phys.')
    point_xyz_phys = np.asarray(sample['point_xyz_phys'])
    if point_xyz_phys.ndim != 2 or point_xyz_phys.shape[1] != 3 or point_xyz_phys.shape[0] == 0:
        raise PointPreprocessingContractError(
            f'point_xyz_phys must have shape [N,3], N>0; got {point_xyz_phys.shape}.'
        )
    if not np.issubdtype(point_xyz_phys.dtype, np.number) or not np.all(np.isfinite(point_xyz_phys)):
        raise PointPreprocessingContractError('point_xyz_phys must contain finite numeric values.')

    if 'physical_unit' not in sample:
        raise PointPreprocessingContractError(
            'M2-1 Point preprocessing requires an explicit physical_unit before mm-to-m scaling.'
        )
    physical_unit = sample['physical_unit']
    if physical_unit != POINT_PHYSICAL_UNIT:
        raise PointPreprocessingContractError(
            f'M2-1 Point preprocessing requires physical_unit={POINT_PHYSICAL_UNIT!r}; got {physical_unit!r}.'
        )

    point_normal = sample.get('point_normal')
    if point_normal is not None and np.asarray(point_normal).shape != point_xyz_phys.shape:
        raise PointPreprocessingContractError(
            f'point_normal shape {np.asarray(point_normal).shape} does not match point_xyz_phys.'
        )
    return point_xyz_phys


def _load_single_stack_collate_fn():
    # The compiled GeoTransformer extension is only needed when real KPConv
    # preprocessing runs. Keeping this import local lets CPU-only synthetic
    # contract tests run on Windows without pretending the extension exists.
    from geotransformer.utils.data import single_collate_fn_stack_mode

    return single_collate_fn_stack_mode


def _load_single_stack_collate_with_parent_fn():
    from geotransformer.utils.data import single_collate_fn_stack_mode_with_parent

    return single_collate_fn_stack_mode_with_parent


def m2_point_collate_fn(
    samples: List[Mapping],
    precompute_data: bool = True,
    stack_collate_fn: Optional[Callable] = None,
    enable_m4_defect_mapping: bool = False,
):
    """Build the frozen M2-1 single-cloud KPConv input without touching CT data.

    This is deliberately separate from the frozen M1 ``pointct_collate_fn``.
    Only ``points`` and all-ones ``feats`` are sent to GeoTransformer's
    single-cloud stack-mode collate. Every original Point/CT/GT/metadata field
    remains in the returned dictionary under its original physical semantics.
    """
    enable_m4_defect_mapping = _require_explicit_mapping_flag(enable_m4_defect_mapping)
    if len(samples) != 1:
        raise PointPreprocessingContractError(
            f'M2-1 Point preprocessing requires batch_size=1; got {len(samples)} samples.'
        )

    sample = samples[0]
    if enable_m4_defect_mapping:
        if not precompute_data:
            raise PointPreprocessingContractError('M4 Point mapping requires precompute_data=True.')
        if not isinstance(sample.get('defect_id'), str) or not sample['defect_id'].strip():
            raise PointPreprocessingContractError('M4 Point mapping requires an explicit defect_id.')
        if 'point_defect_mask' not in sample:
            raise PointPreprocessingContractError('M4 Point mapping requires point_defect_mask.')
    point_xyz_phys = _validate_point_sample(sample)
    point_xyz_net = np.multiply(
        point_xyz_phys,
        np.float32(POINT_NETWORK_SCALE_MM_TO_M),
        dtype=np.float32,
    )
    point_features = np.ones((point_xyz_phys.shape[0], 1), dtype=np.float32)

    if stack_collate_fn is None:
        if enable_m4_defect_mapping:
            stack_collate_fn = _load_single_stack_collate_with_parent_fn()
        else:
            stack_collate_fn = _load_single_stack_collate_fn()
    point_stack = stack_collate_fn(
        [{'points': point_xyz_net, 'feats': point_features}],
        num_stages=POINT_NUM_STAGES,
        voxel_size=POINT_INIT_VOXEL_SIZE_M,
        search_radius=POINT_INIT_RADIUS_M,
        neighbor_limits=list(POINT_NEIGHBOR_LIMITS),
        precompute_data=precompute_data,
    )

    if enable_m4_defect_mapping:
        from geotransformer.modules.ops import aggregate_point_defect_hierarchy

        try:
            import torch
        except ModuleNotFoundError as error:
            raise PointPreprocessingContractError('M4 Point mapping requires PyTorch.') from error
        required_provenance = ('points', 'lengths', 'parent_indices')
        missing = [name for name in required_provenance if name not in point_stack]
        if missing:
            raise PointPreprocessingContractError(
                f'M4 Point hierarchy is missing same-pass provenance fields: {missing}.'
            )
        points = point_stack['points']
        lengths = point_stack['lengths']
        parent_indices = point_stack['parent_indices']
        if not isinstance(points, (list, tuple)) or len(points) != POINT_NUM_STAGES:
            raise PointPreprocessingContractError('M4 Point hierarchy must contain four actual point stages.')
        raw_mask = torch.as_tensor(sample['point_defect_mask'], device=points[0].device)
        point_mapping = aggregate_point_defect_hierarchy(
            raw_mask,
            points,
            lengths,
            parent_indices,
        )
        coarse_count = int(points[3].shape[0])
        expected_shape = (coarse_count,)
        expected_dtypes = {
            'point_intact_coarse': torch.bool,
            'point_raw_total_count_coarse': torch.int64,
            'point_raw_defect_count_coarse': torch.int64,
        }
        for name, expected_dtype in expected_dtypes.items():
            value = point_mapping.get(name)
            if not torch.is_tensor(value) or value.shape != expected_shape or value.dtype != expected_dtype:
                raise PointPreprocessingContractError(
                    f'{name} must have shape {expected_shape} and dtype {expected_dtype}.'
                )
        point_stack.update(point_mapping)
        point_stack['m4_defect_mapping_enabled'] = True

    collated = pointct_collate_fn(samples)
    point_branch = collated['point']
    point_branch.update(point_stack)
    point_branch['point_xyz_phys'] = sample['point_xyz_phys']
    point_branch['point_normal'] = sample.get('point_normal')
    point_branch['point_xyz_net'] = point_stack['points'][0] if precompute_data else point_stack['points']
    point_branch['point_network_scale_mm_to_m'] = POINT_NETWORK_SCALE_MM_TO_M

    # M1 intentionally keeps a minimal contract. Preserve available source
    # metadata without flattening Point/CT tensors into a shared namespace.
    if 'ct_metadata' in sample:
        collated['ct']['ct_metadata'] = sample['ct_metadata']
    if 'pair_metadata' in sample:
        collated['pair_metadata'] = sample['pair_metadata']
    return collated


__all__ = [
    'CTDefectAggregationContractError',
    'CTPreprocessingContractError',
    'PointPreprocessingContractError',
    'aggregate_ct_defect_whole_cells',
    'build_ct_context_5mm',
    'build_ct_support_20mm',
    'create_dataset',
    'extract_external_surface',
    'find_boundary_connected_outside_air',
    'm2_ct_collate_fn',
    'm2_point_collate_fn',
    'pointct_collate_fn',
]
