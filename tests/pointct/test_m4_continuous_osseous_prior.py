import ast
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
from m4_continuous_osseous_prior import (
    ContinuousOsseousPriorContractError,
    compute_continuous_osseous_support_score,
    compute_continuous_osseous_support_score_grid,
    continuous_hu_evidence,
)
from validate_m4_continuous_osseous_prior import (
    ALLOWED_SCORE_PARAMETERS,
    CLEAN10_SUBJECT_IDS,
    audit_case_coverage,
    compute_m4_2a_reliability_diagnostic,
    extract_external_surface_sitk,
    spearman_rank_correlation,
)


class ContinuousHUEvidenceTest(unittest.TestCase):
    def test_sigmoid_is_monotonic_finite_and_strictly_open_unit(self):
        hu = np.asarray([-3000.0, -1000.0, 0.0, 299.0, 300.0, 301.0, 1000.0, 3000.0])
        evidence = continuous_hu_evidence(hu, 300.0, 100.0)
        self.assertTrue(np.all(np.diff(evidence) > 0.0))
        self.assertTrue(np.all(np.isfinite(evidence)))
        self.assertTrue(np.all((evidence > 0.0) & (evidence < 1.0)))
        self.assertEqual(float(evidence[4]), 0.5)

    def test_extreme_hu_is_numerically_stable(self):
        limit = np.finfo(np.float64).max
        evidence = continuous_hu_evidence([-limit, -1e12, 1e12, limit], 300.0, 75.0)
        self.assertTrue(np.all(np.isfinite(evidence)))
        self.assertTrue(np.all((evidence > 0.0) & (evidence < 1.0)))
        self.assertTrue(np.all(np.diff(evidence) >= 0.0))

    def test_nonpositive_or_nonfinite_tau_fails_closed(self):
        for tau in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(tau=tau):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    continuous_hu_evidence([300.0], 300.0, tau)

    def test_nonfinite_hu_or_center_fails_closed(self):
        for hu in ([np.nan], [np.inf], [-np.inf]):
            with self.subTest(hu=hu):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    continuous_hu_evidence(hu, 300.0, 100.0)
        for center in (np.nan, np.inf, -np.inf):
            with self.subTest(center=center):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    continuous_hu_evidence([300.0], center, 100.0)


class ContinuousOsseousScoreTest(unittest.TestCase):
    @staticmethod
    def score(
        volume,
        support,
        *,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        center=300.0,
        tau=100.0,
        radius=1.0,
    ):
        return compute_continuous_osseous_support_score(
            volume,
            spacing,
            origin,
            direction,
            support,
            center,
            tau,
            radius,
        )

    def test_int16_and_int32_ct_match_manual_mean(self):
        for dtype in (np.int16, np.int32):
            with self.subTest(dtype=dtype):
                volume = np.zeros((3, 3, 3), dtype=dtype)
                volume[1, 1, 1] = 300
                volume[1, 1, 2] = 400
                score = self.score(volume, [[1.0, 1.0, 1.0]])
                sphere = [300.0, 400.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                expected = np.mean(continuous_hu_evidence(sphere, 300.0, 100.0))
                self.assertEqual(score.dtype, np.float64)
                self.assertAlmostEqual(float(score[0]), float(expected), places=15)

    def test_anisotropic_spacing_uses_physical_radius(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, :] = 1000
        score = self.score(
            volume,
            [[2.0, 1.0, 1.0]],
            spacing=(2.0, 1.0, 1.0),
            radius=1.1,
        )
        expected_hu = [1000.0, 0.0, 0.0, 0.0, 0.0]
        expected = np.mean(continuous_hu_evidence(expected_hu, 300.0, 100.0))
        self.assertAlmostEqual(float(score[0]), float(expected), places=15)

    def test_nonzero_origin_translation_preserves_score(self):
        volume = np.arange(27, dtype=np.int16).reshape(3, 3, 3) * 50
        first = self.score(volume, [[1.0, 1.0, 1.0]])
        origin = np.asarray([10.0, -20.0, 30.0])
        second = self.score(volume, [origin + [1.0, 1.0, 1.0]], origin=origin)
        np.testing.assert_array_equal(first, second)

    def test_orthonormal_direction_maps_support_to_correct_voxel(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, 2] = 800
        direction = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        support = direction @ np.asarray([2.0, 1.0, 1.0])
        score = self.score(volume, [support], direction=direction, radius=0.1)
        np.testing.assert_array_equal(
            score, continuous_hu_evidence([800.0], 300.0, 100.0)
        )

    def test_physical_sphere_includes_axes_and_excludes_diagonals(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, 1] = 900
        score = self.score(volume, [[1.0, 1.0, 1.0]], radius=1.0)
        expected_hu = [900.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected = np.mean(continuous_hu_evidence(expected_hu, 300.0, 100.0))
        self.assertAlmostEqual(float(score[0]), float(expected), places=15)

    def test_score_shape_range_count_and_grid_consistency(self):
        volume = np.arange(64, dtype=np.int16).reshape(4, 4, 4) * 20
        support = np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        grid = compute_continuous_osseous_support_score_grid(
            volume,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            np.eye(3),
            support,
            300.0,
            (75.0, 100.0, 150.0),
            (1.0, 1.5),
        )
        self.assertEqual(len(grid), 6)
        for value in grid.values():
            self.assertEqual(value.shape, (2,))
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertTrue(np.all((value > 0.0) & (value < 1.0)))
        single = self.score(volume, support, tau=100.0, radius=1.5)
        np.testing.assert_array_equal(single, grid[(100.0, 1.5)])

    def test_empty_neighborhood_fails_closed(self):
        with self.assertRaisesRegex(
            ContinuousOsseousPriorContractError, 'empty physical neighborhood'
        ):
            self.score(
                np.zeros((2, 2, 2), dtype=np.int16),
                [[100.0, 100.0, 100.0]],
            )

    def test_nan_inf_and_nonreal_ct_fail_closed(self):
        invalid = (
            np.zeros((2, 2, 2), dtype=bool),
            np.zeros((2, 2, 2), dtype=np.complex64),
            np.asarray([[['bad']]], dtype=object),
            np.full((2, 2, 2), np.nan, dtype=np.float32),
            np.full((2, 2, 2), np.inf, dtype=np.float64),
        )
        for volume in invalid:
            with self.subTest(dtype=volume.dtype):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    self.score(volume, [[0.0, 0.0, 0.0]])

    def test_invalid_geometry_fails_closed(self):
        volume = np.zeros((2, 2, 2), dtype=np.int16)
        invalid_kwargs = (
            {'spacing': (0.0, 1.0, 1.0)},
            {'spacing': (np.nan, 1.0, 1.0)},
            {'origin': (np.inf, 0.0, 0.0)},
            {'direction': np.zeros((3, 3))},
            {'direction': np.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    self.score(volume, [[0.0, 0.0, 0.0]], **kwargs)

    def test_invalid_support_and_scalar_parameters_fail_closed(self):
        volume = np.zeros((2, 2, 2), dtype=np.int16)
        for support in ([], [0.0, 0.0, 0.0], [[np.nan, 0.0, 0.0]]):
            with self.subTest(support=support):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    self.score(volume, support)
        for center, tau, radius in (
            (np.nan, 100.0, 1.0),
            (300.0, 0.0, 1.0),
            (300.0, -1.0, 1.0),
            (300.0, 100.0, 0.0),
            (300.0, 100.0, np.inf),
        ):
            with self.subTest(center=center, tau=tau, radius=radius):
                with self.assertRaises(ContinuousOsseousPriorContractError):
                    self.score(
                        volume,
                        [[0.0, 0.0, 0.0]],
                        center=center,
                        tau=tau,
                        radius=radius,
                    )

    def test_same_hu_different_valid_geometry_is_correct(self):
        volume = np.full((3, 3, 3), 450, dtype=np.int16)
        expected = float(continuous_hu_evidence([450.0], 300.0, 100.0)[0])
        first = self.score(volume, [[1.0, 1.0, 1.0]], radius=1.2)
        direction = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        origin = np.asarray([8.0, -4.0, 12.0])
        spacing = np.asarray([2.0, 1.0, 0.5])
        support = origin + direction @ (np.asarray([1.0, 1.0, 1.0]) * spacing)
        second = self.score(
            volume,
            [support],
            spacing=spacing,
            origin=origin,
            direction=direction,
            radius=1.2,
        )
        self.assertAlmostEqual(float(first[0]), expected, places=15)
        self.assertAlmostEqual(float(second[0]), expected, places=15)

    def test_tau_change_actually_changes_score(self):
        volume = np.full((2, 2, 2), 600, dtype=np.int16)
        first = self.score(volume, [[0.0, 0.0, 0.0]], tau=75.0, radius=0.1)
        second = self.score(volume, [[0.0, 0.0, 0.0]], tau=150.0, radius=0.1)
        self.assertNotEqual(float(first[0]), float(second[0]))

    def test_radius_change_actually_changes_score(self):
        volume = np.zeros((3, 3, 3), dtype=np.int16)
        volume[1, 1, 1] = 1000
        small = self.score(volume, [[1.0, 1.0, 1.0]], radius=0.1)
        large = self.score(volume, [[1.0, 1.0, 1.0]], radius=1.0)
        self.assertGreater(float(small[0]), float(large[0]))

    def test_score_api_only_accepts_frozen_allowed_parameters(self):
        functions = (
            compute_continuous_osseous_support_score,
            compute_continuous_osseous_support_score_grid,
        )
        for function in functions:
            self.assertEqual(
                tuple(inspect.signature(function).parameters),
                tuple(ALLOWED_SCORE_PARAMETERS),
            )
        with self.assertRaises(TypeError):
            compute_continuous_osseous_support_score(
                np.zeros((2, 2, 2), dtype=np.int16),
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                np.eye(3),
                [[0.0, 0.0, 0.0]],
                300.0,
                100.0,
                1.0,
                mask=np.ones((2, 2, 2), dtype=bool),
            )

    def test_helper_ast_has_no_training_matching_mask_import_or_argument(self):
        helper = EXPERIMENT_DIR / 'm4_continuous_osseous_prior.py'
        tree = ast.parse(helper.read_text(encoding='utf-8'))
        imports = []
        arguments = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
            elif isinstance(node, ast.arg):
                arguments.add(node.arg.lower())
        self.assertTrue(
            all(name.split('.')[0] in {'math', 'typing', 'numpy'} for name in imports)
        )
        self.assertFalse(
            arguments
            & {
                'mask',
                'mp',
                'mv',
                'gt_transform',
                'patient_id',
                'defect_id',
                'reliability',
                'defect_distance',
            }
        )


class ContinuousOsseousValidationUtilityTest(unittest.TestCase):
    def test_sitk_external_surface_matches_frozen_reference(self):
        volume = np.full((7, 7, 7), -1000, dtype=np.int16)
        volume[1:6, 1:6, 1:6] = 500
        volume[3, 3, 3] = -1000
        np.testing.assert_array_equal(
            extract_external_surface_sitk(volume), extract_external_surface(volume)
        )

    def test_spearman_handles_ties_reversal_and_constant_input(self):
        self.assertAlmostEqual(
            spearman_rank_correlation([1, 2, 2, 4], [10, 20, 20, 40]), 1.0
        )
        self.assertAlmostEqual(
            spearman_rank_correlation([1, 2, 3], [3, 2, 1]), -1.0
        )
        self.assertIsNone(spearman_rank_correlation([1, 1, 1], [1, 2, 3]))

    def test_m4_2a_diagnostic_matches_frozen_formula(self):
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        )
        valid = np.asarray([True, False, True])
        actual = compute_m4_2a_reliability_diagnostic(coordinates, valid, sigma_mm=4.0)
        expected = np.asarray(
            [
                1.0 - np.exp(-0.5 * (1.0 / 4.0) ** 2),
                0.0,
                1.0 - np.exp(-0.5 * (3.0 / 4.0) ** 2),
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_coverage_is_clean10_by_five_and_pat6_never_enters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Pat6').mkdir()
            defect = 'defect_001_left_maxilla_cheek_small'
            ready = root / CLEAN10_SUBJECT_IDS[0] / 'defects' / defect
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
