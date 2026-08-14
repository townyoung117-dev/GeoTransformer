import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_DATA_ROOT, make_cfg
from dataset import create_dataset, m2_point_collate_fn
from point_encoder import create_point_encoder


EXPECTED_READY_SUBJECTS = 11
EXPECTED_SKIPPED_SUBJECT = 'Pat10'


def _resolve_device(requested_device):
    if requested_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return device


def _move_point_stack_to_device(point_dict, device):
    # CT arrays and medical metadata deliberately stay on CPU and unmodified.
    point_dict['features'] = point_dict['features'].to(device)
    for key in ('points', 'lengths', 'neighbors', 'subsampling', 'upsampling'):
        point_dict[key] = [tensor.to(device) for tensor in point_dict[key]]
    point_dict['point_xyz_net'] = point_dict['points'][0]
    return point_dict


def validate_point_encoder(data_root, requested_device='auto'):
    cfg = make_cfg()
    dataset = create_dataset(data_root)
    if len(dataset) != EXPECTED_READY_SUBJECTS:
        raise RuntimeError(
            f'Expected {EXPECTED_READY_SUBJECTS} ready subjects; manifest selected {len(dataset)}.'
        )
    skipped_subject_ids = {record['subject_id'] for record in dataset.skipped_records}
    if EXPECTED_SKIPPED_SUBJECT not in skipped_subject_ids:
        raise RuntimeError(f'{EXPECTED_SKIPPED_SUBJECT} must be skipped by the manifest.')

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=m2_point_collate_fn,
    )
    if data_loader.batch_size != 1:
        raise RuntimeError('M2-1 validation requires batch_size=1.')

    device = _resolve_device(requested_device)
    model = create_point_encoder(cfg).to(device)
    model.eval()

    passed = 0
    with torch.no_grad():
        for data_dict in data_loader:
            subject_id = data_dict['subject_id']
            point_dict = _move_point_stack_to_device(data_dict['point'], device)
            output = model(point_dict)

            stage_point_counts = [int(points.shape[0]) for points in point_dict['points']]
            P_raw = output['P_raw']
            Q = output['Q']
            Xp_net_coarse = output['Xp_net_coarse']
            Xp_phys_coarse = output['Xp_phys_coarse']
            finite = all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in (P_raw, Q, Xp_net_coarse, Xp_phys_coarse)
            )

            expected_n = stage_point_counts[3]
            if tuple(P_raw.shape) != (expected_n, 1024):
                raise RuntimeError(f'{subject_id}: invalid P_raw shape {tuple(P_raw.shape)}.')
            if tuple(Q.shape) != (expected_n, 256):
                raise RuntimeError(f'{subject_id}: invalid Q shape {tuple(Q.shape)}.')
            if tuple(Xp_phys_coarse.shape) != (expected_n, 3):
                raise RuntimeError(
                    f'{subject_id}: invalid Xp_phys_coarse shape {tuple(Xp_phys_coarse.shape)}.'
                )
            if not finite:
                raise RuntimeError(f'{subject_id}: Point encoder output contains NaN or Inf.')
            if not torch.allclose(
                Xp_phys_coarse * cfg.point.point_network_scale_mm_to_m,
                Xp_net_coarse,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise RuntimeError(f'{subject_id}: coarse physical/network coordinates are inconsistent.')

            print(
                f'{subject_id}: stage_point_counts={stage_point_counts} '
                f'P_raw={list(P_raw.shape)} Q={list(Q.shape)} '
                f'Xp_phys_coarse={list(Xp_phys_coarse.shape)} finite={finite}'
            )
            passed += 1

    if passed != EXPECTED_READY_SUBJECTS:
        raise RuntimeError(f'Only {passed}/{EXPECTED_READY_SUBJECTS} subjects passed validation.')
    print('11/11 M2 Point Encoder validation: PASS')


def main():
    parser = argparse.ArgumentParser(description='Validate M2-1 Point/KPConv encoder on all ready subjects.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', default='auto', help='Torch device, for example cuda, cuda:0, cpu, or auto.')
    args = parser.parse_args()
    validate_point_encoder(args.data_root, args.device)


if __name__ == '__main__':
    main()
