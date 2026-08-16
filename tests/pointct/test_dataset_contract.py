import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geotransformer.datasets.registration.pointct.dataset import (
    DatasetContractError,
    PointCTDataset,
    assert_no_subject_split_leakage,
    find_subject_split_leakage,
    pointct_collate_fn,
)


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        ready_dir = self.root / 'ReadyCase'
        ready_dir.mkdir()
        (self.root / 'UnlistedCase').mkdir()

        self.volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        self.defect_volume = (self.volume + 100).astype(np.int16)
        self.ct_defect_mask_uint8 = np.asarray(
            [
                [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
            ],
            dtype=np.uint8,
        )
        self.spacing = np.asarray([1.0, 2.0, 3.0])
        self.mv_spacing = self.spacing.copy()
        self.origin = np.asarray([10.0, 20.0, 30.0])
        self.direction = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        xyz = np.asarray([[10.0, 20.0, 30.0], [8.0, 20.0, 30.0], [10.0, 23.0, 33.0]], dtype=np.float32)
        normal = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.complete_xyz = xyz.copy()
        self.complete_normal = normal.copy()
        self.defect_xyz = xyz + np.float32(0.25)
        self.defect_normal = normal.copy()
        self.point_defect_mask_uint8 = np.asarray([1, 0, 1], dtype=np.uint8)
        np.savez(
            ready_dir / 'pointcloud.npz',
            xyz=xyz,
            normal=normal,
            metadata_version=np.asarray('v2'),
            sampling_method=np.asarray('deterministic_test'),
            point_count=np.asarray(3),
        )
        self._write_nrrd_header(ready_dir / 'ct.nrrd')
        pair_metadata = {
            'subject_id': 'SubjectA',
            'case_id': 'CaseA',
            'derived_sample_id': 'complete',
            'canonical_ct_file': 'ct.nrrd',
            'canonical_pointcloud_file': 'pointcloud.npz',
            'coordinate_system': 'left-posterior-superior (LPS) physical coordinates',
            'physical_unit': 'mm',
            'point_to_ct_relationship': 'Points are already in the CT physical coordinate system.',
            'point_ct_alignment': 'already_in_same_physical_coordinate_system',
            'point_to_ct_transform': None,
            'transform_direction': None,
            'ct_spacing': self.spacing.tolist(),
            'ct_origin': self.origin.tolist(),
            'ct_direction': self.direction.reshape(-1).tolist(),
            'ct_array_shape': list(self.volume.shape),
            'ct_array_axis_order': '[z,y,x]',
            'point_count': 3,
            'point_xyz_key': 'xyz',
            'point_normal_key': 'normal',
            'point_xyz_coordinate_semantics': '[x,y,z] physical coordinates in mm',
            'pointcloud_npz_keys': ['xyz', 'normal', 'metadata_version', 'sampling_method', 'point_count'],
        }
        ct_metadata = {
            'shape': list(self.volume.shape),
            'dtype': 'int16',
            'spacing': self.spacing.tolist(),
            'origin': self.origin.tolist(),
            'direction': self.direction.reshape(-1).tolist(),
            'physical_unit': 'mm',
            'array_axis_order': '[z,y,x]',
            'image_index_convention': '[x,y,z]',
            'nrrd_space': 'left-posterior-superior',
        }
        self._write_json(ready_dir / 'pair_metadata.json', pair_metadata)
        self._write_json(ready_dir / 'ct_metadata.json', ct_metadata)
        self._write_json(
            ready_dir / 'qc.json',
            {
                'subject_id': 'SubjectA',
                'processing_success': True,
                'ct_readable': True,
                'pointcloud_readable': True,
                'coordinate_contract_resolved': True,
                'point_ct_alignment_resolved': True,
                'ready_for_baseline': True,
            },
        )
        manifest = {
            'num_subjects_total': 2,
            'num_subjects_ready': 1,
            'physical_unit': 'mm',
            'subjects': [
                {
                    'subject_id': 'SubjectA',
                    'case_id': 'CaseA',
                    'derived_sample_id': 'complete',
                    'ready_for_baseline': True,
                    'ct_path': 'ReadyCase/ct.nrrd',
                    'pointcloud_path': 'ReadyCase/pointcloud.npz',
                    'pair_metadata_path': 'ReadyCase/pair_metadata.json',
                    'ct_metadata_path': 'ReadyCase/ct_metadata.json',
                    'qc_path': 'ReadyCase/qc.json',
                },
                {
                    'subject_id': 'SubjectB',
                    'case_id': 'CaseB',
                    'derived_sample_id': 'complete',
                    'ready_for_baseline': False,
                    'ct_path': 'Missing/ct.nrrd',
                    'pointcloud_path': 'Missing/pointcloud.npz',
                    'pair_metadata_path': 'Missing/pair_metadata.json',
                    'ct_metadata_path': 'Missing/ct_metadata.json',
                    'qc_path': 'Missing/qc.json',
                },
            ],
        }
        self._write_json(self.root / 'dataset_manifest.json', manifest)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def _write_nrrd_header(
        self,
        path,
        *,
        scalar_type='short',
        spacing=None,
        origin=None,
        direction=None,
        space='left-posterior-superior',
    ):
        spacing = self.spacing if spacing is None else np.asarray(spacing, dtype=np.float64)
        origin = self.origin if origin is None else np.asarray(origin, dtype=np.float64)
        direction = self.direction if direction is None else np.asarray(direction, dtype=np.float64)
        scaled_directions = direction * spacing[np.newaxis, :]
        directions = ' '.join(
            f'({scaled_directions[0, axis]},{scaled_directions[1, axis]},{scaled_directions[2, axis]})'
            for axis in range(3)
        )
        path.write_text(
            '\n'.join(
                [
                    'NRRD0004',
                    f'type: {scalar_type}',
                    'dimension: 3',
                    f'space: {space}',
                    'sizes: 4 3 2',
                    f'space directions: {directions}',
                    f'space origin: ({origin[0]},{origin[1]},{origin[2]})',
                    '',
                    '',
                ]
            ),
            encoding='ascii',
        )

    def _fake_ct_reader(self, _path):
        name = Path(_path).name
        if name == 'ct_defect.nrrd':
            volume = self.defect_volume.copy()
        elif name == 'Mv_gt.nrrd':
            volume = self.ct_defect_mask_uint8.copy()
        else:
            volume = self.volume.copy()
        spacing = self.mv_spacing.copy() if name == 'Mv_gt.nrrd' else self.spacing.copy()
        return {
            'ct_volume': volume,
            'ct_spacing': spacing,
            'ct_origin': self.origin.copy(),
            'ct_direction': self.direction.copy(),
            'array_axis_order': ('z', 'y', 'x'),
            'image_index_convention': ('x', 'y', 'z'),
        }

    def _write_valid_defect(self, defect_id='DefectA'):
        defect_dir = self.root / 'ReadyCase' / 'defects' / defect_id
        defect_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            defect_dir / 'face_pointcloud_defect.npz',
            xyz=self.defect_xyz,
            normal=self.defect_normal,
            patient_id=np.asarray('SubjectA'),
            defect_id=np.asarray(defect_id),
            coordinate_system=np.asarray('CT physical coordinates'),
            unit=np.asarray('mm'),
            sampling_method=np.asarray('deterministic_defect_test'),
        )
        self._write_nrrd_header(defect_dir / 'ct_defect.nrrd')
        self._write_nrrd_header(defect_dir / 'Mv_gt.nrrd', scalar_type='unsigned char')
        np.save(defect_dir / 'Mp_gt.npy', self.point_defect_mask_uint8)
        metadata = {
            'patient_id': 'SubjectA',
            'defect_id': defect_id,
            'formal_transform': 'Identity',
            'ct_geometry': {
                'size_xyz': [4, 3, 2],
                'spacing_xyz_mm': self.spacing.tolist(),
                'origin_xyz_mm': self.origin.tolist(),
                'direction': self.direction.reshape(-1).tolist(),
            },
            'defect_pointcloud': {
                'point_count': int(self.defect_xyz.shape[0]),
                'sampling_method': 'deterministic_defect_test',
            },
            'sampling_method': 'deterministic_defect_test',
            'Mp': {
                'shape': [int(self.point_defect_mask_uint8.shape[0])],
                'zero_count': int(np.count_nonzero(self.point_defect_mask_uint8 == 0)),
                'one_count': int(np.count_nonzero(self.point_defect_mask_uint8 == 1)),
                'convention': '0 = defect / unreliable; 1 = intact / stable',
                'point_defect_surface_distance_mm': 2.0,
                'point_boundary_distance_mm': 5.0,
            },
            'Mv': {
                'shape_zyx': list(self.ct_defect_mask_uint8.shape),
                'zero_count': int(np.count_nonzero(self.ct_defect_mask_uint8 == 0)),
                'one_count': int(np.count_nonzero(self.ct_defect_mask_uint8 == 1)),
                'convention': '0 = defect / invalid region; 1 = intact / valid region',
            },
        }
        self._write_json(defect_dir / 'metadata_defect.json', metadata)
        return defect_dir

    def _read_defect_metadata(self, defect_dir):
        return json.loads((defect_dir / 'metadata_defect.json').read_text(encoding='utf-8'))

    def _write_defect_pointcloud(
        self,
        defect_dir,
        *,
        defect_id='DefectA',
        patient_id='SubjectA',
        coordinate_system='CT physical coordinates',
    ):
        np.savez(
            defect_dir / 'face_pointcloud_defect.npz',
            xyz=self.defect_xyz,
            normal=self.defect_normal,
            patient_id=np.asarray(patient_id),
            defect_id=np.asarray(defect_id),
            coordinate_system=np.asarray(coordinate_system),
            unit=np.asarray('mm'),
            sampling_method=np.asarray('deterministic_defect_test'),
        )

    def test_manifest_is_the_only_source_of_ready_cases(self):
        dataset = PointCTDataset(self.root, ct_reader=self._fake_ct_reader)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.subjects_total_in_manifest, 2)
        self.assertEqual(dataset.ready_subject_ids, ['SubjectA'])
        self.assertEqual(dataset.skipped_subject_ids, ['SubjectB'])

    def test_dataset_loads_physical_contract_and_identity_gt(self):
        dataset = PointCTDataset(self.root, ct_reader=self._fake_ct_reader)
        sample = dataset[0]
        self.assertEqual(sample['subject_id'], 'SubjectA')
        self.assertEqual(sample['case_id'], 'CaseA')
        self.assertEqual(sample['derived_sample_id'], 'complete')
        self.assertEqual(sample['point_xyz_phys'].shape, (3, 3))
        self.assertEqual(sample['point_normal'].shape, (3, 3))
        self.assertEqual(sample['ct_volume'].shape, (2, 3, 4))
        self.assertEqual(sample['physical_unit'], 'mm')
        self.assertEqual(sample['gt_transform_direction'], 'Point Cloud -> CT')
        np.testing.assert_array_equal(sample['gt_transform'], np.eye(4))
        self.assertNotIn('point_xyz_net', sample)

    def test_independent_batch_size_one_collate(self):
        dataset = PointCTDataset(self.root, ct_reader=self._fake_ct_reader)
        batch = pointct_collate_fn([dataset[0]])
        self.assertIn('point', batch)
        self.assertIn('ct', batch)
        self.assertIn('point_xyz_phys', batch['point'])
        self.assertIn('ct_volume', batch['ct'])
        self.assertNotIn('points', batch)
        with self.assertRaises(DatasetContractError):
            pointct_collate_fn([dataset[0], dataset[0]])

    def test_b0_does_not_require_or_scan_defects(self):
        invalid_defect_dir = self.root / 'ReadyCase' / 'defects' / 'IncompleteDefect'
        invalid_defect_dir.mkdir(parents=True)
        self.assertEqual(list(invalid_defect_dir.iterdir()), [])
        dataset = PointCTDataset(self.root, ct_reader=self._fake_ct_reader, defect_variants=None)
        sample = dataset[0]
        np.testing.assert_array_equal(sample['point_xyz_phys'], self.complete_xyz)
        np.testing.assert_array_equal(sample['point_normal'], self.complete_normal)
        np.testing.assert_array_equal(sample['ct_volume'], self.volume)
        self.assertNotIn('defect_id', sample)
        self.assertNotIn('point_defect_mask', sample)
        self.assertNotIn('ct_defect_mask', sample)

    def test_valid_explicit_defect_variant_loads_raw_contract(self):
        self._write_valid_defect()
        dataset = PointCTDataset(
            self.root,
            ct_reader=self._fake_ct_reader,
            defect_variants=[('SubjectA', 'DefectA')],
        )
        sample = dataset[0]

        self.assertEqual(sample['subject_id'], 'SubjectA')
        self.assertEqual(sample['defect_id'], 'DefectA')
        np.testing.assert_array_equal(sample['point_xyz_phys'], self.defect_xyz)
        np.testing.assert_array_equal(sample['point_normal'], self.defect_normal)
        np.testing.assert_array_equal(sample['ct_volume'], self.defect_volume)
        self.assertEqual(sample['point_defect_mask'].dtype, np.dtype(bool))
        self.assertEqual(sample['ct_defect_mask'].dtype, np.dtype(bool))
        np.testing.assert_array_equal(sample['point_defect_mask'], self.point_defect_mask_uint8.astype(bool))
        np.testing.assert_array_equal(sample['ct_defect_mask'], self.ct_defect_mask_uint8.astype(bool))
        np.testing.assert_array_equal(sample['gt_transform'], np.eye(4))
        self.assertEqual(sample['gt_transform_direction'], 'Point Cloud -> CT')

        batch = pointct_collate_fn([sample])
        self.assertEqual(batch['defect_id'], 'DefectA')
        self.assertIs(batch['point']['point_defect_mask'], sample['point_defect_mask'])
        self.assertIs(batch['ct']['ct_defect_mask'], sample['ct_defect_mask'])

    def test_defect_selection_is_never_implicit(self):
        self._write_valid_defect()
        b0_dataset = PointCTDataset(self.root, ct_reader=self._fake_ct_reader)
        self.assertNotIn('defect_id', b0_dataset[0])

        with self.assertRaisesRegex(DatasetContractError, 'at least one explicit'):
            PointCTDataset(self.root, ct_reader=self._fake_ct_reader, defect_variants=[])
        with self.assertRaisesRegex(DatasetContractError, 'non-empty string'):
            PointCTDataset(self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', None)])
        with self.assertRaisesRegex(DatasetContractError, 'Duplicate explicit defect selection'):
            PointCTDataset(
                self.root,
                ct_reader=self._fake_ct_reader,
                defect_variants=[('SubjectA', 'DefectA'), ('SubjectA', 'DefectA')],
            )

    def test_same_subject_can_select_multiple_explicit_variants(self):
        self._write_valid_defect('DefectA')
        self._write_valid_defect('DefectB')
        dataset = PointCTDataset(
            self.root,
            ct_reader=self._fake_ct_reader,
            defect_variants=[('SubjectA', 'DefectA'), ('SubjectA', 'DefectB')],
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual([dataset[index]['defect_id'] for index in range(2)], ['DefectA', 'DefectB'])

    def test_missing_variant_or_core_artifact_fails_closed(self):
        with self.assertRaisesRegex(DatasetContractError, 'does not exist'):
            PointCTDataset(
                self.root,
                ct_reader=self._fake_ct_reader,
                defect_variants=[('SubjectA', 'MissingDefect')],
            )

        defect_dir = self._write_valid_defect()
        (defect_dir / 'Mv_gt.nrrd').unlink()
        with self.assertRaisesRegex(DatasetContractError, 'missing core artifact'):
            PointCTDataset(
                self.root,
                ct_reader=self._fake_ct_reader,
                defect_variants=[('SubjectA', 'DefectA')],
            )

    def test_subject_and_defect_identity_mismatch_fail_closed(self):
        defect_dir = self._write_valid_defect()
        metadata = self._read_defect_metadata(defect_dir)
        metadata['patient_id'] = 'OtherSubject'
        self._write_json(defect_dir / 'metadata_defect.json', metadata)
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, 'patient_id'):
            dataset[0]

        metadata['patient_id'] = 'SubjectA'
        self._write_json(defect_dir / 'metadata_defect.json', metadata)
        self._write_defect_pointcloud(defect_dir, defect_id='OtherDefect')
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, 'NPZ defect_id'):
            dataset[0]

    def test_mp_wrong_dtype_fails_closed(self):
        defect_dir = self._write_valid_defect()
        np.save(defect_dir / 'Mp_gt.npy', self.point_defect_mask_uint8.astype(np.float32))
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, 'on-disk dtype must be uint8'):
            dataset[0]

    def test_mp_non_binary_fails_closed(self):
        defect_dir = self._write_valid_defect()
        np.save(defect_dir / 'Mp_gt.npy', np.asarray([1, 2, 0], dtype=np.uint8))
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, 'binary values'):
            dataset[0]

    def test_mp_wrong_length_fails_closed(self):
        defect_dir = self._write_valid_defect()
        np.save(defect_dir / 'Mp_gt.npy', np.asarray([1, 0], dtype=np.uint8))
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, 'does not match defective point count'):
            dataset[0]

    def test_defective_point_explicit_lps_coordinate_system_is_accepted(self):
        defect_dir = self._write_valid_defect()
        self._write_defect_pointcloud(defect_dir, coordinate_system='LPS physical coordinates')
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        self.assertEqual(dataset[0]['defect_id'], 'DefectA')

    def test_defective_point_non_lps_coordinate_system_fails_closed(self):
        defect_dir = self._write_valid_defect()
        invalid_coordinate_systems = (
            'CT RAS physical coordinates',
            'right-anterior-superior physical coordinates',
            'not LPS physical coordinates',
            'RAS-to-LPS physical coordinates',
        )
        for coordinate_system in invalid_coordinate_systems:
            with self.subTest(coordinate_system=coordinate_system):
                self._write_defect_pointcloud(defect_dir, coordinate_system=coordinate_system)
                dataset = PointCTDataset(
                    self.root,
                    ct_reader=self._fake_ct_reader,
                    defect_variants=[('SubjectA', 'DefectA')],
                )
                with self.assertRaisesRegex(DatasetContractError, 'not the accepted CT/LPS physical frame'):
                    dataset[0]

    def test_ct_mv_geometry_mismatch_fails_closed(self):
        defect_dir = self._write_valid_defect()
        self.mv_spacing = np.asarray([1.0, 2.0, 4.0])
        self._write_nrrd_header(
            defect_dir / 'Mv_gt.nrrd',
            scalar_type='unsigned char',
            spacing=self.mv_spacing,
        )
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, '^ct_defect/Mv_gt spacing mismatch\\.$'):
            dataset[0]

    def test_unsupported_formal_transform_fails_closed(self):
        defect_dir = self._write_valid_defect()
        metadata = self._read_defect_metadata(defect_dir)
        metadata['formal_transform'] = np.eye(4).tolist()
        self._write_json(defect_dir / 'metadata_defect.json', metadata)
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, 'only "Identity" is frozen'):
            dataset[0]

    def test_metadata_mask_convention_mismatch_fails_closed(self):
        defect_dir = self._write_valid_defect()
        metadata = self._read_defect_metadata(defect_dir)
        metadata['Mp']['convention'] = '0 = intact; 1 = defect'
        self._write_json(defect_dir / 'metadata_defect.json', metadata)
        dataset = PointCTDataset(
            self.root, ct_reader=self._fake_ct_reader, defect_variants=[('SubjectA', 'DefectA')]
        )
        with self.assertRaisesRegex(DatasetContractError, '0=defect and 1=intact'):
            dataset[0]

    def test_subject_level_leakage_check(self):
        records = [
            {'subject_id': 'SubjectA', 'derived_sample_id': 'complete', 'split': 'train'},
            {'subject_id': 'SubjectA', 'derived_sample_id': 'defect_1', 'split': 'test'},
        ]
        self.assertEqual(find_subject_split_leakage(records), {'SubjectA': ['test', 'train']})
        with self.assertRaises(DatasetContractError):
            assert_no_subject_split_leakage(records)
        assert_no_subject_split_leakage(
            [
                {'subject_id': 'SubjectA', 'derived_sample_id': 'complete', 'split': 'train'},
                {'subject_id': 'SubjectA', 'derived_sample_id': 'defect_1', 'split': 'train'},
            ]
        )


if __name__ == '__main__':
    unittest.main()
