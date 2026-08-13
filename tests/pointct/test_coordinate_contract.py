import unittest

import numpy as np

from geotransformer.datasets.registration.pointct.coordinates import (
    apply_transform,
    array_index_to_image_index,
    ct_physical_bounding_box,
    image_index_to_array_index,
    image_index_to_physical,
    invert_transform,
    physical_to_image_index,
    transform_normals,
)


class CoordinateContractTest(unittest.TestCase):
    def setUp(self):
        self.spacing = np.asarray([2.0, 3.0, 4.0])
        self.origin = np.asarray([10.0, -5.0, 100.0])
        self.direction = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_array_image_axis_mapping(self):
        array_index = np.asarray([[30.0, 20.0, 10.0]])
        image_index = array_index_to_image_index(array_index, '[z,y,x]')
        np.testing.assert_array_equal(image_index, [[10.0, 20.0, 30.0]])
        np.testing.assert_array_equal(
            image_index_to_array_index(image_index, '[z,y,x]'),
            array_index,
        )

    def test_image_physical_basis_and_round_trip_with_non_identity_direction(self):
        image_indices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected = np.asarray(
            [
                [10.0, -5.0, 100.0],
                [10.0, -3.0, 100.0],
                [7.0, -5.0, 100.0],
                [10.0, -5.0, 104.0],
            ]
        )
        physical = image_index_to_physical(
            image_indices,
            self.spacing,
            self.origin,
            self.direction,
        )
        np.testing.assert_allclose(physical, expected)
        np.testing.assert_allclose(
            physical_to_image_index(physical, self.spacing, self.origin, self.direction),
            image_indices,
        )

    def test_ct_bounding_box_uses_voxel_centers_without_half_voxel_offset(self):
        bbox_min, bbox_max = ct_physical_bounding_box(
            [2, 3, 4],
            '[z,y,x]',
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            np.eye(3),
        )
        np.testing.assert_array_equal(bbox_min, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(bbox_max, [3.0, 2.0, 1.0])

    def test_point_to_ct_transform_and_inverse(self):
        rotation = self.direction
        translation = np.asarray([5.0, -2.0, 7.0])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        point = np.asarray([[1.0, 0.0, 2.0]])
        expected_ct = np.asarray([[5.0, -1.0, 9.0]])
        np.testing.assert_allclose(apply_transform(point, transform), expected_ct)
        np.testing.assert_allclose(
            apply_transform(expected_ct, invert_transform(transform)),
            point,
        )
        np.testing.assert_array_equal(transform[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_normal_transform_ignores_translation(self):
        transform = np.eye(4)
        transform[:3, :3] = self.direction
        transform[:3, 3] = [1000.0, -2000.0, 3000.0]
        normal = np.asarray([[1.0, 0.0, 0.0]])
        rotated = transform_normals(normal, transform)
        np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]])
        self.assertAlmostEqual(float(np.linalg.norm(rotated)), 1.0)
        np.testing.assert_allclose(transform_normals(rotated, invert_transform(transform)), normal)


if __name__ == '__main__':
    unittest.main()

