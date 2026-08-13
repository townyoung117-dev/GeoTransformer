from itertools import product
from typing import Sequence, Tuple, Union

import numpy as np


AxisOrder = Tuple[str, str, str]
AxisOrderLike = Union[str, Sequence[str]]


def normalize_axis_order(axis_order: AxisOrderLike) -> AxisOrder:
    """Return an explicit three-axis permutation such as ("z", "y", "x")."""
    if isinstance(axis_order, str):
        text = axis_order.strip().lower()
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
        axes = tuple(part.strip().strip("'\"") for part in text.split(','))
    else:
        axes = tuple(str(axis).strip().lower() for axis in axis_order)
    if len(axes) != 3 or set(axes) != {'x', 'y', 'z'}:
        raise ValueError(f'axis_order must be a permutation of x, y, z; got {axis_order!r}.')
    return axes


def _as_coordinate_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f'{name} must have shape (..., 3); got {array.shape}.')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains NaN or Inf.')
    return array


def _validate_geometry(spacing, origin, direction):
    spacing_array = _as_coordinate_array(spacing, 'spacing')
    origin_array = _as_coordinate_array(origin, 'origin')
    if spacing_array.shape != (3,) or origin_array.shape != (3,):
        raise ValueError('spacing and origin must each have shape (3,).')
    if np.any(spacing_array <= 0):
        raise ValueError(f'spacing must be strictly positive; got {spacing_array}.')

    direction_array = np.asarray(direction, dtype=np.float64)
    if direction_array.shape != (3, 3):
        raise ValueError(f'direction must have shape (3, 3); got {direction_array.shape}.')
    if not np.all(np.isfinite(direction_array)):
        raise ValueError('direction contains NaN or Inf.')
    if abs(np.linalg.det(direction_array)) < 1e-12:
        raise ValueError('direction must be invertible.')
    return spacing_array, origin_array, direction_array


def array_index_to_image_index(array_index, array_axis_order: AxisOrderLike) -> np.ndarray:
    """Convert array subscripts in the declared storage order to image [x, y, z] indices."""
    array_index = _as_coordinate_array(array_index, 'array_index')
    axis_order = normalize_axis_order(array_axis_order)
    positions = {axis: index for index, axis in enumerate(axis_order)}
    return np.stack(
        [array_index[..., positions['x']], array_index[..., positions['y']], array_index[..., positions['z']]],
        axis=-1,
    )


def image_index_to_array_index(image_index, array_axis_order: AxisOrderLike) -> np.ndarray:
    """Convert image [x, y, z] indices to array subscripts in the declared storage order."""
    image_index = _as_coordinate_array(image_index, 'image_index')
    image_positions = {'x': 0, 'y': 1, 'z': 2}
    axis_order = normalize_axis_order(array_axis_order)
    return np.stack([image_index[..., image_positions[axis]] for axis in axis_order], axis=-1)


def image_index_to_physical(image_index, spacing, origin, direction) -> np.ndarray:
    """Apply x_phys = origin + direction @ (image_index * spacing), without a half-voxel offset."""
    image_index = _as_coordinate_array(image_index, 'image_index')
    spacing, origin, direction = _validate_geometry(spacing, origin, direction)
    return origin + (image_index * spacing) @ direction.T


def physical_to_image_index(physical_coordinate, spacing, origin, direction) -> np.ndarray:
    """Map physical coordinates to continuous image [x, y, z] indices."""
    physical_coordinate = _as_coordinate_array(physical_coordinate, 'physical_coordinate')
    spacing, origin, direction = _validate_geometry(spacing, origin, direction)
    shifted = physical_coordinate - origin
    scaled_index = np.linalg.solve(direction, shifted.reshape(-1, 3).T).T.reshape(shifted.shape)
    return scaled_index / spacing


def validate_rigid_transform(transform, atol: float = 1e-6) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f'transform must have shape (4, 4); got {transform.shape}.')
    if not np.all(np.isfinite(transform)):
        raise ValueError('transform contains NaN or Inf.')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError(f'transform bottom row must be [0, 0, 0, 1]; got {transform[3]}.')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError('transform rotation is not orthonormal.')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f'transform rotation must have determinant +1; got {np.linalg.det(rotation)}.')
    return transform


def apply_transform(points, transform) -> np.ndarray:
    """Apply a Point-to-target rigid transform to physical points with shape (..., 3)."""
    points = _as_coordinate_array(points, 'points')
    transform = validate_rigid_transform(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(transform) -> np.ndarray:
    """Invert a validated 4x4 rigid transform."""
    transform = validate_rigid_transform(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_normals(normals, transform_or_rotation) -> np.ndarray:
    """Rotate normals; translation is deliberately ignored."""
    normals = _as_coordinate_array(normals, 'normals')
    value = np.asarray(transform_or_rotation, dtype=np.float64)
    if value.shape == (4, 4):
        rotation = validate_rigid_transform(value)[:3, :3]
    elif value.shape == (3, 3):
        rotation = value
        if not np.all(np.isfinite(rotation)):
            raise ValueError('rotation contains NaN or Inf.')
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
            raise ValueError('rotation is not orthonormal.')
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError('rotation must have determinant +1.')
    else:
        raise ValueError(f'transform_or_rotation must have shape (4, 4) or (3, 3); got {value.shape}.')
    return normals @ rotation.T


def ct_physical_bounding_box(array_shape, array_axis_order, spacing, origin, direction):
    """Return min/max physical coordinates of voxel centers at the eight array corners."""
    shape = np.asarray(array_shape)
    if shape.shape != (3,) or not np.issubdtype(shape.dtype, np.number):
        raise ValueError(f'array_shape must contain three numeric dimensions; got {array_shape!r}.')
    if not np.all(np.isfinite(shape)) or np.any(shape <= 0) or not np.all(shape == np.floor(shape)):
        raise ValueError(f'array_shape must contain positive integer dimensions; got {array_shape!r}.')
    shape = shape.astype(np.int64)
    axis_order = normalize_axis_order(array_axis_order)
    array_corners = np.asarray(list(product(*[(0, int(size - 1)) for size in shape])), dtype=np.float64)
    image_corners = array_index_to_image_index(array_corners, axis_order)
    physical_corners = image_index_to_physical(image_corners, spacing, origin, direction)
    return physical_corners.min(axis=0), physical_corners.max(axis=0)
