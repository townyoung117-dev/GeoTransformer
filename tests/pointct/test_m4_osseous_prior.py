import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import extract_external_surface
from m4_osseous_prior import (
    OsseousPriorContractError,
    compute_osseous_support_score,
    compute_osseous_support_score_grid,
)
from validate_m4_osseous_prior import (
    CLEAN10_SUBJECT_IDS,
    audit_case_coverage,
    compute_m4_2a_reliability_diagnostic,
    extract_external_surface_sitk,
    spearman_rank_correlation,
)


class OsseousSupportScoreTest(unittest.TestCase):
    @staticmethod
    def score(
        volume,
        support,
        *,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        threshold=300.0,
        radius=1.0,
    ):
        return compute_osseous_support_score(
            volume,
            np.asarray(spacing, dtype=np.float64),
            np.asarray(origin, dtype=np.float64),
            np.asarray(direction, dtype=np.float64),
            np.asarray(support, dtype=np.float64),
            hu_threshold=threshold,
            radius_mm=radius,
        )

    def test_int16_and_int32_hu_and_threshold_boundary(self):
        for dtype in (np.int16, np.int32):
            with self.subTest(dtype=dtype):
                volume = np.zeros((3, 3, 3), dtype=dtype)
                volume[1, 1, 1] = 300
                volume[1, 1, 2] = 400
                score = self.score(volume, [[1.0, 1.0, 1.0]])
                self.assertEqual(score.dtype, np.float64)
                self.assertAlmostEqual(float(score[0]), 2.0 / 7.0)

                above_boundary = self.score(
                    volume,
                    [[1.0, 1.0, 1.0]],
                    threshold=300.0001,
                )
                self.assertAlmostEqual(float(above_boundary[0]), 1.0 / 7.0)

    def test_anisotropic_spacing_uses_physical_radius(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, 1] = 500
        volume[1, 1, 0] = 500
        volume[1, 1, 2] = 500
        score = self.score(
            volume,
            [[2.0, 1.0, 1.0]],
            spacing=(2.0, 1.0, 1.0),
            radius=1.1,
        )
        # The +/-x voxels are 2 mm away and must not enter the 1.1 mm sphere.
        self.assertAlmostEqual(float(score[0]), 1.0 / 5.0)

    def test_origin_translation_preserves_score_when_support_moves_with_header(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, 1] = 500
        first = self.score(volume, [[1.0, 1.0, 1.0]], radius=0.1)
        translated_origin = np.asarray([10.0, -20.0, 30.0])
        second = self.score(
            volume,
            [translated_origin + np.asarray([1.0, 1.0, 1.0])],
            origin=translated_origin,
            radius=0.1,
        )
        np.testing.assert_array_equal(first, second)

    def test_legal_rotated_direction_maps_physical_support_to_correct_voxel(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, 2] = 500
        direction = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        image_index_xyz = np.asarray([2.0, 1.0, 1.0])
        support = direction @ image_index_xyz
        score = self.score(
            volume,
            [support],
            direction=direction,
            radius=0.1,
        )
        np.testing.assert_array_equal(score, np.asarray([1.0]))

    def test_physical_radius_includes_axis_voxels_and_excludes_diagonals(self):
        volume = np.full((3, 3, 3), 500, dtype=np.int16)
        score = self.score(volume, [[1.0, 1.0, 1.0]], radius=1.0)
        self.assertEqual(float(score[0]), 1.0)

        volume[1, 1, 1] = 0
        score = self.score(volume, [[1.0, 1.0, 1.0]], radius=1.0)
        self.assertAlmostEqual(float(score[0]), 6.0 / 7.0)

    def test_score_range_finite_count_and_grid_consistency(self):
        volume = np.arange(64, dtype=np.int16).reshape(4, 4, 4) * 20
        support = np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        grid = compute_osseous_support_score_grid(
            volume,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            np.eye(3),
            support,
            hu_thresholds=(200, 300, 400),
            radius_values_mm=(1.0, 1.5),
        )
        self.assertEqual(set(grid), {(200.0, 1.0), (300.0, 1.0), (400.0, 1.0),
                                     (200.0, 1.5), (300.0, 1.5), (400.0, 1.5)})
        for value in grid.values():
            self.assertEqual(value.shape, (2,))
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertTrue(np.all((value >= 0.0) & (value <= 1.0)))
        single = self.score(volume, support, threshold=300, radius=1.5)
        np.testing.assert_array_equal(single, grid[(300.0, 1.5)])

    def test_empty_neighborhood_fails_closed(self):
        with self.assertRaisesRegex(OsseousPriorContractError, 'empty physical neighborhood'):
            self.score(
                np.zeros((2, 2, 2), dtype=np.int16),
                [[100.0, 100.0, 100.0]],
                radius=1.0,
            )

    def test_invalid_spacing_fails_closed(self):
        volume = np.zeros((2, 2, 2), dtype=np.int16)
        for spacing in ((0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (np.nan, 1.0, 1.0)):
            with self.subTest(spacing=spacing):
                with self.assertRaises(OsseousPriorContractError):
                    self.score(volume, [[0.0, 0.0, 0.0]], spacing=spacing)

    def test_invalid_direction_fails_closed(self):
        volume = np.zeros((2, 2, 2), dtype=np.int16)
        directions = (
            np.zeros((3, 3)),
            np.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.full((3, 3), np.nan),
            np.eye(2),
        )
        for direction in directions:
            with self.subTest(direction=direction):
                with self.assertRaises(OsseousPriorContractError):
                    self.score(volume, [[0.0, 0.0, 0.0]], direction=direction)

    def test_invalid_hu_fails_closed(self):
        invalid = (
            np.zeros((2, 2, 2), dtype=bool),
            np.zeros((2, 2, 2), dtype=np.complex64),
            np.asarray([[['bad']]], dtype=object),
            np.full((2, 2, 2), np.nan, dtype=np.float32),
        )
        for volume in invalid:
            with self.subTest(dtype=volume.dtype):
                with self.assertRaises(OsseousPriorContractError):
                    self.score(volume, [[0.0, 0.0, 0.0]])

    def test_shape_mismatch_and_invalid_scalar_fail_closed(self):
        volume = np.zeros((2, 2, 2), dtype=np.int16)
        for support in ([], [0.0, 0.0, 0.0], [[0.0, 0.0]], [[np.nan, 0.0, 0.0]]):
            with self.subTest(support=support):
                with self.assertRaises(OsseousPriorContractError):
                    self.score(volume, support)
        for threshold, radius in ((np.nan, 1.0), (300.0, 0.0), (300.0, -1.0)):
            with self.subTest(threshold=threshold, radius=radius):
                with self.assertRaises(OsseousPriorContractError):
                    self.score(
                        volume,
                        [[0.0, 0.0, 0.0]],
                        threshold=threshold,
                        radius=radius,
                    )

    def test_score_api_rejects_forbidden_annotation_and_identity_arguments(self):
        parameters = set(inspect.signature(compute_osseous_support_score).parameters)
        forbidden = {
            'mask',
            'gt_transform',
            'complete_ct',
            'complete_point',
            'Mp',
            'Mv',
            'reliability',
            'anchor_xyz_mm',
            'anatomical_axes',
            'mirror_plane',
            'template_id',
            'anatomical_region',
            'patient_id',
            'defect_id',
        }
        self.assertFalse(parameters & forbidden)
        with self.assertRaises(TypeError):
            compute_osseous_support_score(
                np.zeros((2, 2, 2), dtype=np.int16),
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                np.eye(3),
                [[0.0, 0.0, 0.0]],
                hu_threshold=300,
                radius_mm=1.0,
                mask=np.ones((2, 2, 2), dtype=bool),
            )


class OsseousValidationUtilityTest(unittest.TestCase):
    def test_sitk_external_surface_matches_existing_reference_on_small_volume(self):
        volume = np.full((7, 7, 7), -1000, dtype=np.int16)
        volume[1:6, 1:6, 1:6] = 500
        volume[3, 3, 3] = -1000  # sealed internal air must not become external
        expected = extract_external_surface(volume)
        actual = extract_external_surface_sitk(volume)
        np.testing.assert_array_equal(actual, expected)

    def test_spearman_handles_ties_and_constant_input(self):
        self.assertAlmostEqual(
            spearman_rank_correlation([1, 2, 2, 4], [10, 20, 20, 40]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_rank_correlation([1, 2, 3], [3, 2, 1]),
            -1.0,
        )
        self.assertIsNone(spearman_rank_correlation([1, 1, 1], [1, 2, 3]))

    def test_numpy_m4_2a_reliability_matches_frozen_formula(self):
        coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        valid = np.asarray([True, False, True])
        reliability = compute_m4_2a_reliability_diagnostic(
            coordinates,
            valid,
            sigma_mm=4.0,
        )
        expected = np.asarray(
            [1.0 - np.exp(-0.5 * (1.0 / 4.0) ** 2), 0.0,
             1.0 - np.exp(-0.5 * (3.0 / 4.0) ** 2)]
        )
        np.testing.assert_allclose(reliability, expected, rtol=1e-14, atol=1e-14)

    def test_coverage_is_exact_clean10_by_five_and_pat6_never_enters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Pat6').mkdir()
            first_defect = 'defect_001_left_maxilla_cheek_small'
            ready = root / CLEAN10_SUBJECT_IDS[0] / 'defects' / first_defect
            ready.mkdir(parents=True)
            (ready / 'ct_defect.nrrd').touch()
            (ready / 'Mv_gt.nrrd').touch()
            coverage = audit_case_coverage(root)
        self.assertEqual(coverage['expected_subject_count'], 10)
        self.assertEqual(coverage['expected_defects_per_subject'], 5)
        self.assertEqual(coverage['expected_case_count'], 50)
        self.assertEqual(coverage['ready_case_count'], 1)
        self.assertTrue(coverage['pat6_excluded_from_formal_records'])
        self.assertTrue(coverage['pat6_preserved_on_disk'])


if __name__ == '__main__':
    unittest.main()
