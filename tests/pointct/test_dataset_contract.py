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
        self.spacing = np.asarray([1.0, 2.0, 3.0])
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

    def _write_nrrd_header(self, path):
        scaled_directions = self.direction * self.spacing[np.newaxis, :]
        directions = ' '.join(
            f'({scaled_directions[0, axis]},{scaled_directions[1, axis]},{scaled_directions[2, axis]})'
            for axis in range(3)
        )
        path.write_text(
            '\n'.join(
                [
                    'NRRD0004',
                    'type: short',
                    'dimension: 3',
                    'space: left-posterior-superior',
                    'sizes: 4 3 2',
                    f'space directions: {directions}',
                    f'space origin: ({self.origin[0]},{self.origin[1]},{self.origin[2]})',
                    '',
                    '',
                ]
            ),
            encoding='ascii',
        )

    def _fake_ct_reader(self, _path):
        return {
            'ct_volume': self.volume.copy(),
            'ct_spacing': self.spacing.copy(),
            'ct_origin': self.origin.copy(),
            'ct_direction': self.direction.copy(),
            'array_axis_order': ('z', 'y', 'x'),
            'image_index_convention': ('x', 'y', 'z'),
        }

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
