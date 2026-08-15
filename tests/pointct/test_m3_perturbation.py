import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from perturbation import (
    AXIS_RETRY_LIMIT,
    M3PerturbationContractError,
    POINT_TO_CT_DIRECTION,
    PROVENANCE_FIELD,
    SEED_SCHEME_VERSION,
    apply_point_rigid_perturbation,
    augment_point_sample,
    compose_effective_point_to_ct_gt,
    derive_perturbation_seed,
    sample_rigid_perturbation,
)


class M3PerturbationTest(unittest.TestCase):
    @staticmethod
    def _seed_payload(**overrides):
        payload = {
            'scheme_version': SEED_SCHEME_VERSION,
            'protocol_hash': 'protocol-sha256-value',
            'fold_id': 'Fold1',
            'root_seed': 2026,
            'purpose': 'train',
            'epoch': 4,
            'subject_id': 'Pat6',
            'severity': None,
            'variant_id': None,
        }
        payload.update(overrides)
        return payload

    @staticmethod
    def _rotation_z(angle_deg):
        angle = np.deg2rad(angle_deg)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_y(angle_deg):
        angle = np.deg2rad(angle_deg)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.asarray(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _sample(cls, xyz_dtype=np.float32, normal_dtype=np.float32, gt_transform=None):
        xyz = np.asarray(
            [
                [10.0, 20.0, 30.0],
                [14.0, 20.0, 30.0],
                [10.0, 26.0, 30.0],
                [10.0, 20.0, 38.0],
            ],
            dtype=xyz_dtype,
        )
        normals = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=normal_dtype,
        )
        normals /= np.linalg.norm(normals.astype(np.float64), axis=1, keepdims=True).astype(
            normal_dtype
        )
        if gt_transform is None:
            gt_transform = np.eye(4, dtype=np.float64)
        return {
            'subject_id': 'Pat6',
            'point_xyz_phys': xyz,
            'point_normal': normals,
            'gt_transform': np.asarray(gt_transform).copy(),
            'gt_transform_direction': POINT_TO_CT_DIRECTION,
            'physical_unit': 'mm',
            'ct_marker': np.arange(4, dtype=np.int16),
        }

    @staticmethod
    def _apply_transform(points, transform):
        points = np.asarray(points, dtype=np.float64)
        transform = np.asarray(transform, dtype=np.float64)
        return points @ transform[:3, :3].T + transform[:3, 3]

    def test_seed_matches_required_canonical_json_sha256_uint64(self):
        payload = self._seed_payload(epoch=None, severity='hard', variant_id=2)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        expected = int.from_bytes(
            hashlib.sha256(canonical.encode('utf-8')).digest()[:8],
            byteorder='big',
            signed=False,
        )
        self.assertEqual(derive_perturbation_seed(**payload), expected)
        self.assertGreaterEqual(expected, 0)
        self.assertLessEqual(expected, (1 << 64) - 1)

    def test_same_payload_produces_identical_seed_and_perturbation(self):
        seed_a = derive_perturbation_seed(**self._seed_payload())
        seed_b = derive_perturbation_seed(**self._seed_payload())
        self.assertEqual(seed_a, seed_b)
        first = sample_rigid_perturbation(seed_a, 20.0, 20.0)
        second = sample_rigid_perturbation(seed_b, 20.0, 20.0)
        self.assertEqual(first['angle_deg'], second['angle_deg'])
        for key in ('axis', 'translation_mm', 'R_aug'):
            np.testing.assert_array_equal(first[key], second[key])

    def test_changed_epoch_changes_seed(self):
        first = derive_perturbation_seed(**self._seed_payload(epoch=1))
        second = derive_perturbation_seed(**self._seed_payload(epoch=2))
        self.assertNotEqual(first, second)

    def test_changed_subject_changes_seed(self):
        first = derive_perturbation_seed(**self._seed_payload(subject_id='Pat1'))
        second = derive_perturbation_seed(**self._seed_payload(subject_id='Pat2'))
        self.assertNotEqual(first, second)

    def test_changed_fold_changes_seed(self):
        first = derive_perturbation_seed(**self._seed_payload(fold_id='Fold1'))
        second = derive_perturbation_seed(**self._seed_payload(fold_id='Fold2'))
        self.assertNotEqual(first, second)

    def test_changed_purpose_changes_seed(self):
        first = derive_perturbation_seed(**self._seed_payload(purpose='train'))
        second = derive_perturbation_seed(**self._seed_payload(purpose='validation'))
        self.assertNotEqual(first, second)

    def test_optional_severity_and_variant_are_not_dropped(self):
        base = derive_perturbation_seed(
            **self._seed_payload(epoch=None, severity=None, variant_id=None)
        )
        severity = derive_perturbation_seed(
            **self._seed_payload(epoch=None, severity='mild', variant_id=None)
        )
        variant = derive_perturbation_seed(
            **self._seed_payload(epoch=None, severity=None, variant_id=0)
        )
        self.assertEqual(len({base, severity, variant}), 3)

    def test_epoch_none_allows_fixed_validation_or_test_seed(self):
        payload = self._seed_payload(purpose='validation', epoch=None, severity='moderate', variant_id=1)
        self.assertEqual(
            derive_perturbation_seed(**payload),
            derive_perturbation_seed(**payload),
        )

    def test_sampling_is_independent_of_global_numpy_state(self):
        seed = derive_perturbation_seed(**self._seed_payload())
        np.random.seed(1)
        first = sample_rigid_perturbation(seed, 20.0, 20.0)
        np.random.seed(987654)
        second = sample_rigid_perturbation(seed, 20.0, 20.0)
        self.assertEqual(first['angle_deg'], second['angle_deg'])
        np.testing.assert_array_equal(first['axis'], second['axis'])
        np.testing.assert_array_equal(first['translation_mm'], second['translation_mm'])

    def test_sampled_rotation_is_so3_and_axis_is_unit(self):
        for seed in range(32):
            perturbation = sample_rigid_perturbation(seed, 20.0, 20.0)
            rotation = perturbation['R_aug']
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(perturbation['axis'])), 1.0, places=12)
        self.assertEqual(AXIS_RETRY_LIMIT, 16)

    def test_sampling_respects_angle_and_per_axis_translation_bounds(self):
        max_rotation = 17.25
        max_translation = 12.5
        for seed in range(256):
            perturbation = sample_rigid_perturbation(seed, max_rotation, max_translation)
            self.assertLessEqual(abs(perturbation['angle_deg']), max_rotation)
            self.assertTrue(np.all(np.abs(perturbation['translation_mm']) <= max_translation))

    def test_rotation_is_centered_on_float64_full_resolution_centroid(self):
        sample = self._sample()
        rotation = self._rotation_z(73.0)
        output = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            rotation,
            np.zeros(3),
        )
        expected_centroid = np.mean(sample['point_xyz_phys'], axis=0, dtype=np.float64)
        self.assertEqual(output['centroid_phys_mm'].dtype, np.float64)
        np.testing.assert_allclose(output['centroid_phys_mm'], expected_centroid, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            np.mean(output['point_xyz_phys'], axis=0, dtype=np.float64),
            expected_centroid,
            rtol=0.0,
            atol=1e-6,
        )

    def test_identity_rotation_moves_centroid_by_physical_mm_translation(self):
        sample = self._sample(xyz_dtype=np.float64)
        translation = np.asarray([20.0, -7.5, 3.25], dtype=np.float64)
        output = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            np.eye(3),
            translation,
        )
        before = np.mean(sample['point_xyz_phys'], axis=0, dtype=np.float64)
        after = np.mean(output['point_xyz_phys'], axis=0, dtype=np.float64)
        np.testing.assert_allclose(after, before + translation, rtol=0.0, atol=1e-12)

    def test_row_implementation_matches_column_vector_contract(self):
        sample = self._sample(xyz_dtype=np.float64)
        rotation = self._rotation_y(-31.0)
        translation = np.asarray([4.0, -2.0, 9.0])
        output = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            rotation,
            translation,
        )
        centroid = np.mean(sample['point_xyz_phys'], axis=0, dtype=np.float64)
        expected = np.stack(
            [rotation @ (point - centroid) + centroid + translation for point in sample['point_xyz_phys']]
        )
        np.testing.assert_allclose(output['point_xyz_phys'], expected, rtol=0.0, atol=1e-12)

    def test_normals_use_the_same_rotation_as_points(self):
        sample = self._sample(normal_dtype=np.float64)
        rotation = self._rotation_z(90.0)
        output = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            rotation,
            np.asarray([3.0, 4.0, 5.0]),
        )
        expected = sample['point_normal'] @ rotation.T
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(output['point_normal'], expected, rtol=0.0, atol=1e-12)

    def test_normal_output_is_translation_invariant(self):
        sample = self._sample()
        rotation = self._rotation_z(37.0)
        first = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            rotation,
            np.zeros(3),
        )
        second = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            rotation,
            np.asarray([100.0, -200.0, 300.0]),
        )
        np.testing.assert_array_equal(first['point_normal'], second['point_normal'])

    def test_rotated_normals_are_unit_normalized(self):
        sample = self._sample(normal_dtype=np.float64)
        sample['point_normal'] *= np.asarray([[1.01], [0.99], [1.005], [0.995]])
        output = apply_point_rigid_perturbation(
            sample['point_xyz_phys'],
            sample['point_normal'],
            self._rotation_y(13.0),
            np.zeros(3),
        )
        np.testing.assert_allclose(
            np.linalg.norm(output['point_normal'], axis=1),
            np.ones(4),
            rtol=0.0,
            atol=1e-12,
        )

    def test_none_normal_remains_none(self):
        sample = self._sample()
        sample['point_normal'] = None
        output = augment_point_sample(
            sample,
            seed=17,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        self.assertIsNone(output['point_normal'])

    def test_zero_perturbation_has_identity_rotation_and_zero_translation(self):
        sample = self._sample()
        output = augment_point_sample(
            sample,
            seed=99,
            max_rotation_deg=0.0,
            max_translation_mm=0.0,
        )
        provenance = output[PROVENANCE_FIELD]
        np.testing.assert_array_equal(np.asarray(provenance['R_aug']), np.eye(3))
        np.testing.assert_array_equal(np.asarray(provenance['translation_mm']), np.zeros(3))
        np.testing.assert_array_equal(output['point_xyz_phys'], sample['point_xyz_phys'])
        np.testing.assert_allclose(output['gt_transform'], sample['gt_transform'], atol=0.0, rtol=0.0)
        original_direction = sample['point_normal'] / np.linalg.norm(
            sample['point_normal'].astype(np.float64), axis=1, keepdims=True
        )
        np.testing.assert_allclose(output['point_normal'], original_direction, atol=1e-7, rtol=0.0)

    def test_identity_original_gt_maps_augmented_points_back_to_original_ct_frame(self):
        sample = self._sample(xyz_dtype=np.float64, normal_dtype=np.float64)
        augmented = augment_point_sample(
            sample,
            seed=12345,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        actual = self._apply_transform(augmented['point_xyz_phys'], augmented['gt_transform'])
        expected = self._apply_transform(sample['point_xyz_phys'], sample['gt_transform'])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)

    def test_nonidentity_original_gt_composition_is_exact(self):
        original_gt = np.eye(4, dtype=np.float64)
        original_gt[:3, :3] = self._rotation_z(28.0)
        original_gt[:3, 3] = [42.0, -17.0, 8.5]
        sample = self._sample(
            xyz_dtype=np.float64,
            normal_dtype=np.float64,
            gt_transform=original_gt,
        )
        augmented = augment_point_sample(
            sample,
            seed=98765,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        actual = self._apply_transform(augmented['point_xyz_phys'], augmented['gt_transform'])
        expected = self._apply_transform(sample['point_xyz_phys'], original_gt)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)

    def test_effective_gt_matches_explicit_transform_composition(self):
        original_gt = np.eye(4, dtype=np.float64)
        original_gt[:3, :3] = self._rotation_y(-19.0)
        original_gt[:3, 3] = [1.0, 2.0, 3.0]
        rotation = self._rotation_z(11.0)
        b_aug = np.asarray([7.0, -4.0, 2.5])
        effective = compose_effective_point_to_ct_gt(
            original_gt,
            rotation,
            b_aug,
            gt_transform_direction=POINT_TO_CT_DIRECTION,
        )
        augmentation = np.eye(4, dtype=np.float64)
        augmentation[:3, :3] = rotation
        augmentation[:3, 3] = b_aug
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ b_aug
        np.testing.assert_allclose(effective, original_gt @ inverse, rtol=0.0, atol=1e-12)

    def test_wrong_point_to_ct_direction_fails_closed(self):
        with self.assertRaisesRegex(M3PerturbationContractError, 'Point Cloud -> CT'):
            compose_effective_point_to_ct_gt(
                np.eye(4),
                np.eye(3),
                np.zeros(3),
                gt_transform_direction='CT -> Point Cloud',
            )

    def test_augmentation_does_not_mutate_raw_sample_or_source_arrays(self):
        sample = self._sample()
        xyz_before = sample['point_xyz_phys'].copy()
        normal_before = sample['point_normal'].copy()
        gt_before = sample['gt_transform'].copy()
        ct_before = sample['ct_marker'].copy()
        augmented = augment_point_sample(
            sample,
            seed=314159,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        np.testing.assert_array_equal(sample['point_xyz_phys'], xyz_before)
        np.testing.assert_array_equal(sample['point_normal'], normal_before)
        np.testing.assert_array_equal(sample['gt_transform'], gt_before)
        np.testing.assert_array_equal(sample['ct_marker'], ct_before)
        self.assertIsNot(augmented, sample)
        self.assertIsNot(augmented['point_xyz_phys'], sample['point_xyz_phys'])
        self.assertIsNot(augmented['point_normal'], sample['point_normal'])
        self.assertIsNot(augmented['gt_transform'], sample['gt_transform'])
        self.assertIs(augmented['ct_marker'], sample['ct_marker'])

    def test_xyz_floating_dtype_is_preserved(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                sample = self._sample(xyz_dtype=dtype)
                augmented = augment_point_sample(
                    sample,
                    seed=100,
                    max_rotation_deg=20.0,
                    max_translation_mm=20.0,
                )
                self.assertEqual(augmented['point_xyz_phys'].dtype, np.dtype(dtype))

    def test_normal_floating_dtype_is_preserved(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                sample = self._sample(normal_dtype=dtype)
                augmented = augment_point_sample(
                    sample,
                    seed=101,
                    max_rotation_deg=20.0,
                    max_translation_mm=20.0,
                )
                self.assertEqual(augmented['point_normal'].dtype, np.dtype(dtype))

    def test_effective_gt_is_float64(self):
        sample = self._sample()
        sample['gt_transform'] = np.eye(4, dtype=np.float32)
        augmented = augment_point_sample(
            sample,
            seed=102,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        self.assertEqual(augmented['gt_transform'].dtype, np.float64)

    def test_integer_xyz_fails_closed(self):
        sample = self._sample()
        sample['point_xyz_phys'] = sample['point_xyz_phys'].astype(np.int32)
        with self.assertRaisesRegex(M3PerturbationContractError, 'floating dtype'):
            augment_point_sample(
                sample,
                seed=1,
                max_rotation_deg=20.0,
                max_translation_mm=20.0,
            )

    def test_integer_normals_fail_closed(self):
        sample = self._sample()
        sample['point_normal'] = sample['point_normal'].astype(np.int32)
        with self.assertRaisesRegex(M3PerturbationContractError, 'floating dtype'):
            augment_point_sample(
                sample,
                seed=1,
                max_rotation_deg=20.0,
                max_translation_mm=20.0,
            )

    def test_nonfinite_xyz_normal_and_gt_fail_closed(self):
        cases = []
        xyz_sample = self._sample()
        xyz_sample['point_xyz_phys'][0, 0] = np.nan
        cases.append(xyz_sample)
        normal_sample = self._sample()
        normal_sample['point_normal'][0, 0] = np.inf
        cases.append(normal_sample)
        gt_sample = self._sample()
        gt_sample['gt_transform'][0, 3] = np.nan
        cases.append(gt_sample)
        for sample in cases:
            with self.subTest(field=[key for key, value in sample.items() if isinstance(value, np.ndarray)]):
                with self.assertRaisesRegex(M3PerturbationContractError, 'NaN or Inf|finite'):
                    augment_point_sample(
                        sample,
                        seed=1,
                        max_rotation_deg=20.0,
                        max_translation_mm=20.0,
                    )

    def test_degenerate_normal_fails_closed(self):
        sample = self._sample()
        sample['point_normal'][2] = 0.0
        with self.assertRaisesRegex(M3PerturbationContractError, 'greater than 1e-12'):
            augment_point_sample(
                sample,
                seed=1,
                max_rotation_deg=20.0,
                max_translation_mm=20.0,
            )

    def test_invalid_sampling_bounds_and_seed_fail_closed(self):
        for name, values in (
            ('rotation', (float('nan'), float('inf'), -1.0)),
            ('translation', (float('nan'), float('inf'), -1.0)),
        ):
            for value in values:
                with self.subTest(name=name, value=value):
                    kwargs = {'max_rotation_deg': 20.0, 'max_translation_mm': 20.0}
                    kwargs[f'max_{name}_deg' if name == 'rotation' else 'max_translation_mm'] = value
                    with self.assertRaises(M3PerturbationContractError):
                        sample_rigid_perturbation(1, **kwargs)
        for seed in (-1, 1 << 64, True):
            with self.subTest(seed=seed):
                with self.assertRaises(M3PerturbationContractError):
                    sample_rigid_perturbation(seed, 20.0, 20.0)

    def test_provenance_is_complete_json_safe_and_deterministic(self):
        sample = self._sample()
        first = augment_point_sample(
            sample,
            seed=555,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        second = augment_point_sample(
            sample,
            seed=555,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        first_provenance = first[PROVENANCE_FIELD]
        second_provenance = second[PROVENANCE_FIELD]
        self.assertEqual(first_provenance, second_provenance)
        self.assertEqual(
            set(first_provenance),
            {
                'scheme_version',
                'seed',
                'axis',
                'angle_deg',
                'translation_mm',
                'centroid_phys_mm',
                'R_aug',
                'b_aug',
                'max_rotation_deg',
                'max_translation_mm',
            },
        )
        self.assertEqual(first_provenance['scheme_version'], SEED_SCHEME_VERSION)
        json.dumps(first_provenance, sort_keys=True)

    def test_already_augmented_sample_fails_closed(self):
        sample = self._sample()
        augmented = augment_point_sample(
            sample,
            seed=6,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        with self.assertRaisesRegex(M3PerturbationContractError, 'already marked'):
            augment_point_sample(
                augmented,
                seed=7,
                max_rotation_deg=20.0,
                max_translation_mm=20.0,
            )

    def test_missing_sample_fields_fail_closed(self):
        sample = self._sample()
        del sample['gt_transform']
        with self.assertRaisesRegex(M3PerturbationContractError, 'missing fields'):
            augment_point_sample(
                sample,
                seed=1,
                max_rotation_deg=20.0,
                max_translation_mm=20.0,
            )

    def test_source_is_numpy_only_and_stops_at_perturbation_primitive(self):
        source_path = EXPERIMENT_DIR / 'perturbation.py'
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        imported = set()
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lower())
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id.lower())
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr.lower())
        self.assertNotIn('torch', imported)
        self.assertNotIn('scipy', imported)
        self.assertNotIn('random', imported)
        self.assertNotIn('hash', called_names)
        self.assertNotIn('seed', called_names)
        self.assertNotIn('rand', called_names)
        prohibited_terms = {
            'kpconv',
            'ct preprocessing',
            'matcher',
            'sinkhorn',
            'matching_loss',
            'matching_filter',
            'registration',
            'ransac',
            'icp',
            'defect',
            'anatomical',
            'entropy',
            'fusion',
        }
        for term in prohibited_terms:
            self.assertNotIn(term, source.lower())
        self.assertIn('np.random.Generator(np.random.PCG64(seed))', source)


if __name__ == '__main__':
    unittest.main()
