import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_DATA_ROOT, make_cfg
from ct_encoder import create_ct_encoder
from dataset import create_dataset, m2_ct_collate_fn


EXPECTED_READY_SUBJECTS = 11
EXPECTED_SKIPPED_SUBJECT = 'Pat10'


def _resolve_device(requested_device):
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return device


def _move_ct_encoder_inputs_to_device(ct_dict, device):
    tensor_fields = {
        'ct_context_features': torch.float32,
        'ct_context_indices': torch.int32,
        'ct_support_indices_20mm': torch.int32,
        'ct_support_linear_20mm': torch.int64,
        'ct_support_phys_20mm': torch.float64,
    }
    for name, dtype in tensor_fields.items():
        ct_dict[name] = torch.as_tensor(ct_dict[name], dtype=dtype, device=device)
    return ct_dict


def validate_ct_encoder(data_root, requested_device='cuda'):
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
        collate_fn=m2_ct_collate_fn,
    )
    if data_loader.batch_size != 1:
        raise RuntimeError('M2-2 validation requires batch_size=1.')

    device = _resolve_device(requested_device)
    model = create_ct_encoder(cfg).to(device)
    model.eval()

    passed = 0
    with torch.no_grad():
        for data_dict in data_loader:
            subject_id = data_dict['subject_id']
            ct_dict = _move_ct_encoder_inputs_to_device(data_dict['ct'], device)
            context_count = int(ct_dict['ct_context_features'].shape[0])

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(ct_dict)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - started
            peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                if device.type == 'cuda'
                else 0.0
            )

            V_raw = output['V_raw']
            K = output['K']
            Xv_phys_coarse = output['Xv_phys_coarse']
            support_count = output['support_count']
            finite = all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in (V_raw, K, Xv_phys_coarse)
            )
            if support_count <= 0:
                raise RuntimeError(f'{subject_id}: support_count must be positive.')
            if output['x5_active_count'] != context_count:
                raise RuntimeError(
                    f'{subject_id}: x5 active count {output["x5_active_count"]} '
                    f'does not match context count {context_count}.'
                )
            if tuple(V_raw.shape) != (support_count, 128):
                raise RuntimeError(f'{subject_id}: invalid V_raw shape {tuple(V_raw.shape)}.')
            if tuple(K.shape) != (support_count, 256):
                raise RuntimeError(f'{subject_id}: invalid K shape {tuple(K.shape)}.')
            if tuple(Xv_phys_coarse.shape) != (support_count, 3):
                raise RuntimeError(
                    f'{subject_id}: invalid Xv_phys_coarse shape {tuple(Xv_phys_coarse.shape)}.'
                )
            if not finite:
                raise RuntimeError(f'{subject_id}: CT encoder output contains NaN or Inf.')

            print(
                f'{subject_id}: context_5mm={output["x5_active_count"]} '
                f'x10={output["x10_active_count"]} x20={output["x20_active_count"]} '
                f'support={support_count} V_raw={list(V_raw.shape)} K={list(K.shape)} '
                f'Xv_phys_coarse={list(Xv_phys_coarse.shape)} finite={finite} '
                f'forward_time_s={forward_seconds:.3f} peak_gpu_memory_mb={peak_memory_mb:.1f}'
            )
            passed += 1

    if passed != EXPECTED_READY_SUBJECTS:
        raise RuntimeError(f'Only {passed}/{EXPECTED_READY_SUBJECTS} subjects passed validation.')
    print('11/11 M2 CT Encoder validation: PASS')


def main():
    parser = argparse.ArgumentParser(description='Validate M2-2 sparse CT encoder on all ready subjects.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', default='cuda', help='Torch CUDA device, for example cuda or cuda:0.')
    args = parser.parse_args()
    validate_ct_encoder(args.data_root, args.device)


if __name__ == '__main__':
    main()
