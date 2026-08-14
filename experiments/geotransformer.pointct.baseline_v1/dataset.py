import sys
from pathlib import Path
from typing import Callable, List, Mapping, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geotransformer.datasets.registration.pointct import PointCTDataset, pointct_collate_fn

from config import (
    POINT_INIT_RADIUS_M,
    POINT_INIT_VOXEL_SIZE_M,
    POINT_NEIGHBOR_LIMITS,
    POINT_NETWORK_SCALE_MM_TO_M,
    POINT_NUM_STAGES,
    POINT_PHYSICAL_UNIT,
)


class PointPreprocessingContractError(RuntimeError):
    pass


def create_dataset(data_root=None, **kwargs):
    """Create the manifest-driven M1 dataset without defining train/val/test splits."""
    if data_root is None:
        data_root = PROJECT_ROOT / 'local_data'
    return PointCTDataset(data_root, **kwargs)


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
    'PointPreprocessingContractError',
    'create_dataset',
    'm2_point_collate_fn',
    'pointct_collate_fn',
]
