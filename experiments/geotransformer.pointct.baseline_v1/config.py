from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'local_data'
M1_BATCH_SIZE = 1

# M2-1 frozen Point/KPConv contract. Physical coordinates remain in millimetres;
# only the explicitly named network coordinates are scaled to metres.
POINT_PHYSICAL_UNIT = 'mm'
POINT_NETWORK_SCALE_MM_TO_M = 0.001
POINT_NUM_STAGES = 4
POINT_INIT_VOXEL_SIZE_M = 0.0025
POINT_KERNEL_SIZE = 15
POINT_BASE_RADIUS = 2.5
POINT_INIT_RADIUS_M = 0.00625
POINT_BASE_SIGMA = 2.0
POINT_INIT_SIGMA_M = 0.005

# Calibrated on all 11 ready real subjects with GeoTransformer neighbor
# calibration keep_ratio=0.8. These limits are frozen for Baseline V1.
POINT_NEIGHBOR_LIMITS = [177, 32, 33, 34]

POINT_INPUT_DIM = 1
POINT_INIT_DIM = 64
POINT_GROUP_NORM = 32
POINT_FINE_FEATURE_DIM = 256
POINT_COARSE_RAW_DIM = 1024
POINT_PROJECTED_DIM = 256


_C = SimpleNamespace()

_C.data = SimpleNamespace()
_C.data.dataset_root = str(DEFAULT_DATA_ROOT)
_C.data.batch_size = M1_BATCH_SIZE
_C.data.physical_unit = POINT_PHYSICAL_UNIT

_C.point = SimpleNamespace()
_C.point.physical_unit = POINT_PHYSICAL_UNIT
_C.point.point_network_scale_mm_to_m = POINT_NETWORK_SCALE_MM_TO_M
_C.point.num_stages = POINT_NUM_STAGES
_C.point.init_voxel_size_m = POINT_INIT_VOXEL_SIZE_M
_C.point.kernel_size = POINT_KERNEL_SIZE
_C.point.base_radius = POINT_BASE_RADIUS
_C.point.init_radius_m = POINT_INIT_RADIUS_M
_C.point.base_sigma = POINT_BASE_SIGMA
_C.point.init_sigma_m = POINT_INIT_SIGMA_M
_C.point.neighbor_limits = list(POINT_NEIGHBOR_LIMITS)
_C.point.input_dim = POINT_INPUT_DIM
_C.point.init_dim = POINT_INIT_DIM
_C.point.group_norm = POINT_GROUP_NORM
_C.point.fine_feature_dim = POINT_FINE_FEATURE_DIM
_C.point.coarse_raw_dim = POINT_COARSE_RAW_DIM
_C.point.projected_dim = POINT_PROJECTED_DIM


def make_cfg():
    return _C
