import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geotransformer.datasets.registration.pointct import (
    DatasetContractError,
    NRRDDependencyError,
    PointCTDataset,
    ct_physical_bounding_box,
)


def _as_list(array):
    return np.asarray(array).tolist()


def _normal_statistics(normal):
    if normal is None:
        return {'available': False}
    norms = np.linalg.norm(normal, axis=1)
    return {
        'available': True,
        'shape': list(normal.shape),
        'dtype': str(normal.dtype),
        'norm_min': float(norms.min()),
        'norm_mean': float(norms.mean()),
        'norm_max': float(norms.max()),
    }


def validate_dataset(data_root):
    dataset = PointCTDataset(data_root)
    report = {
        'status': 'PASS',
        'subjects_total_in_manifest': dataset.subjects_total_in_manifest,
        'subjects_ready': len(dataset),
        'subjects_skipped_not_ready': len(dataset.skipped_records),
        'subjects_loaded_successfully': 0,
        'subjects_failed_to_load': 0,
        'subjects_blocked_by_dependency': 0,
        'skipped_subject_ids': dataset.skipped_subject_ids,
        'subjects': [],
    }

    for index, record in enumerate(dataset.records):
        subject = {'subject_id': record['subject_id'], 'status': 'PASS'}
        try:
            metadata = dataset.load_case_metadata(index)
            point = dataset.load_pointcloud(index)
            point_min = point['point_xyz_phys'].min(axis=0)
            point_max = point['point_xyz_phys'].max(axis=0)
            point_mean = point['point_xyz_phys'].mean(axis=0)
            ct_bbox_min, ct_bbox_max = ct_physical_bounding_box(
                metadata['ct_shape'],
                metadata['array_axis_order'],
                metadata['ct_spacing'],
                metadata['ct_origin'],
                metadata['ct_direction'],
            )
            bbox_tolerance = 1e-3
            point_inside_ct_bbox = bool(
                np.all(point_min >= ct_bbox_min - bbox_tolerance)
                and np.all(point_max <= ct_bbox_max + bbox_tolerance)
            )
            if not point_inside_ct_bbox:
                raise DatasetContractError(
                    f'{record["subject_id"]}: Point physical bbox is outside the CT physical bbox.'
                )

            subject.update(
                {
                    'metadata_contract_status': 'PASS',
                    'coordinate_system': metadata['coordinate_system'],
                    'physical_unit': metadata['physical_unit'],
                    'gt_transform': _as_list(metadata['gt_transform']),
                    'gt_transform_direction': metadata['gt_transform_direction'],
                    'ct': {
                        'shape': _as_list(metadata['ct_shape']),
                        'spacing': _as_list(metadata['ct_spacing']),
                        'origin': _as_list(metadata['ct_origin']),
                        'direction': _as_list(metadata['ct_direction']),
                        'array_axis_order': list(metadata['array_axis_order']),
                        'image_index_convention': list(metadata['image_index_convention']),
                        'physical_bbox_min': _as_list(ct_bbox_min),
                        'physical_bbox_max': _as_list(ct_bbox_max),
                    },
                    'point': {
                        'point_count': point['point_count'],
                        'xyz_shape': list(point['point_xyz_phys'].shape),
                        'xyz_dtype': str(point['point_xyz_phys'].dtype),
                        'xyz_min': _as_list(point_min),
                        'xyz_max': _as_list(point_max),
                        'xyz_mean': _as_list(point_mean),
                        'physical_bbox_inside_ct': point_inside_ct_bbox,
                        'normal': _normal_statistics(point['point_normal']),
                    },
                }
            )

            try:
                sample = dataset[index]
            except NRRDDependencyError as error:
                subject['status'] = 'TEST_BLOCKED_BY_DEPENDENCY'
                subject['dependency_error'] = str(error)
                report['subjects_blocked_by_dependency'] += 1
            else:
                volume = sample['ct_volume']
                subject['ct'].update(
                    {
                        'dtype': str(volume.dtype),
                        'intensity_min': float(volume.min()),
                        'intensity_max': float(volume.max()),
                    }
                )
                if sample['subject_id'] != record['subject_id']:
                    raise DatasetContractError('Loaded sample identity does not match manifest.')
                report['subjects_loaded_successfully'] += 1
        except NRRDDependencyError as error:
            subject['status'] = 'TEST_BLOCKED_BY_DEPENDENCY'
            subject['dependency_error'] = str(error)
            report['subjects_blocked_by_dependency'] += 1
        except Exception as error:
            subject['status'] = 'FAIL'
            subject['error'] = f'{type(error).__name__}: {error}'
            report['subjects_failed_to_load'] += 1
        report['subjects'].append(subject)

    if report['subjects_failed_to_load']:
        report['status'] = 'FAIL'
    elif report['subjects_blocked_by_dependency']:
        report['status'] = 'TEST_BLOCKED_BY_DEPENDENCY'
    return report


def main():
    parser = argparse.ArgumentParser(description='Validate M1 Point-CT physical coordinate contracts.')
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'local_data')
    args = parser.parse_args()
    try:
        report = validate_dataset(args.data_root)
    except Exception as error:
        report = {'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
