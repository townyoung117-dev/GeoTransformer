import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geotransformer.datasets.registration.pointct import PointCTDataset, pointct_collate_fn


def create_dataset(data_root=None, **kwargs):
    """Create the manifest-driven M1 dataset without defining train/val/test splits."""
    if data_root is None:
        data_root = PROJECT_ROOT / 'local_data'
    return PointCTDataset(data_root, **kwargs)


__all__ = ['create_dataset', 'pointct_collate_fn']

