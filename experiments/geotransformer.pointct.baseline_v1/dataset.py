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


def m2_ct_collate_fn(samples: List[Mapping]):
    """Build M2-2 CT inputs while preserving the frozen M1 dual branches."""
    if len(samples) != 1:
        raise CTPreprocessingContractError(
            f'M2-2 CT preprocessing requires batch_size=1; got {len(samples)} samples.'
        )
    sample = samples[0]
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


def m2_point_collate_fn(
    samples: List[Mapping],
    precompute_data: bool = True,
    stack_collate_fn: Optional[Callable] = None,
):
    """Build the frozen M2-1 single-cloud KPConv input without touching CT data.

    This is deliberately separate from the frozen M1 ``pointct_collate_fn``.
    Only ``points`` and all-ones ``feats`` are sent to GeoTransformer's
    single-cloud stack-mode collate. Every original Point/CT/GT/metadata field
    remains in the returned dictionary under its original physical semantics.
    """
    if len(samples) != 1:
        raise PointPreprocessingContractError(
            f'M2-1 Point preprocessing requires batch_size=1; got {len(samples)} samples.'
        )

    sample = samples[0]
    point_xyz_phys = _validate_point_sample(sample)
    point_xyz_net = np.multiply(
        point_xyz_phys,
        np.float32(POINT_NETWORK_SCALE_MM_TO_M),
        dtype=np.float32,
    )
    point_features = np.ones((point_xyz_phys.shape[0], 1), dtype=np.float32)

    if stack_collate_fn is None:
        stack_collate_fn = _load_single_stack_collate_fn()
    point_stack = stack_collate_fn(
        [{'points': point_xyz_net, 'feats': point_features}],
        num_stages=POINT_NUM_STAGES,
        voxel_size=POINT_INIT_VOXEL_SIZE_M,
        search_radius=POINT_INIT_RADIUS_M,
        neighbor_limits=list(POINT_NEIGHBOR_LIMITS),
        precompute_data=precompute_data,
    )

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
    'CTPreprocessingContractError',
    'PointPreprocessingContractError',
    'build_ct_context_5mm',
    'build_ct_support_20mm',
    'create_dataset',
    'extract_external_surface',
    'find_boundary_connected_outside_air',
    'm2_ct_collate_fn',
    'm2_point_collate_fn',
    'pointct_collate_fn',
]
