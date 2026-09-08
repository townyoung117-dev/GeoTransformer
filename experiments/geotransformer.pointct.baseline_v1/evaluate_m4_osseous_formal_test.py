"""Explicit future execution only. Do not invoke --evaluate during protocol building."""

import argparse

import m4_osseous_formal_protocol as contract
import bind_m4_osseous_formal_checkpoints as checkpoints


def metrics_from_inference(inference, effective_gt, evaluation):
    """Use unchanged M3 metrics on one M4 output; accepts synthetic CPU fixtures."""
    from evaluation import correspondence_inlier_metrics, evaluate_registration_result
    from m3_metric_diagnostic import compute_tre_diagnostics, make_rigid_transform
    filtered, registered = inference['filter_output'], inference['registration_output']
    inliers = correspondence_inlier_metrics(
        inference['point_physical'], inference['ct_physical'], filtered['point_indices'],
        filtered['ct_indices'], effective_gt, evaluation['correspondence_inlier_threshold_mm'])
    registration = evaluate_registration_result(
        registered, effective_gt, evaluation['registration_rre_threshold_deg'],
        evaluation['registration_rte_threshold_mm'])
    # M3 formal evaluator converts malformed success transforms to explicit failure.
    # Preserve that result and keep all associated error diagnostics null.
    success = registration['solver_success']
    if success and inliers['num_correspondences'] < 3:
        raise contract.ContractError('solver success violates unchanged correspondence minimum')
    transform = make_rigid_transform(registered['rotation'], registered['translation']) if success else None
    tre = compute_tre_diagnostics(inference['point_physical'], transform, effective_gt, solver_success=success)
    return {**inliers, **registration,
            **{key: tre[key] for key in contract.ERROR_METRICS if key not in ('rre_deg', 'rte_mm')},
            'inference_runtime_ms': inference['inference_runtime_ms']}


def evaluate_case(raw_sample, case, sources, modules, device):
    from perturbation import augment_point_sample
    from evaluate_m4_osseous_prior import run_m4_osseous_prior_inference
    for key in ('subject_id', 'defect_id'):
        contract.equal(raw_sample.get(key), case[key], 'sample ' + key)
    augmented = augment_point_sample(
        raw_sample, seed=case['perturbation_seed'], max_rotation_deg=case['max_rotation_deg'],
        max_translation_mm=case['max_translation_mm'], scheme_version=sources['clean10']['seed_scheme_version'])
    model = contract.expected_model(sources)
    output = run_m4_osseous_prior_inference(
        sample=augmented, point_encoder=modules[0], ct_encoder=modules[1], matcher=modules[2],
        evaluation_protocol=sources['evaluation'], device=device,
        sigma_mm=model['soft_sigma_mm'], soft_strength=model['soft_strength'], osseous_strength=model['lambda_oss'])
    for key, value in {'m4_hard_constraint_active': True, 'm4_soft_modulation_active': True,
                       'm4_osseous_prior_active': True, 'm4_soft_sigma_mm': 60.0,
                       'm4_soft_strength': 2.0, 'm4_osseous_strength': 0.5,
                       'osseous_center_hu': 300.0, 'osseous_tau_hu': 100.0,
                       'osseous_radius_mm': 20.0}.items():
        contract.equal(output[key], value, 'M4 inference ' + key)
    return metrics_from_inference(output, augmented['gt_transform'], sources['evaluation'])


def execute_fold(sources, fold, payload, metadata, device_name, data_root):
    """All runtime imports and sample access are behind committed binding verification."""
    import torch
    from dataset import create_dataset
    from evaluate_m3 import _build_models_from_checkpoint
    device = torch.device(device_name)
    # Resolve a bare CUDA device exactly as the existing M4 validation producer.
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    modules = _build_models_from_checkpoint(payload, metadata, device)
    variants = [(patient, defect) for patient in sources['clean10']['folds'][fold]['test_subject_ids']
                for defect in sources['evaluation']['required_defect_ids']]
    dataset = create_dataset(data_root, defect_variants=variants)
    records = [dataset.get_record(index) for index in range(len(dataset))]
    index_map = {(row['subject_id'], row['defect_id']): index for index, row in enumerate(records)}
    if len(records) != len(variants) or len(index_map) != len(variants) or set(index_map) != set(variants):
        raise contract.ContractError('dataset must contain only current fold frozen test instances')
    result = []
    with torch.no_grad():
        for case in contract.manifest(sources, fold):
            raw = dataset[index_map[case['subject_id'], case['defect_id']]]
            result.append({**{key: case[key] for key in (*contract.IDENTITY, 'case_key')},
                           **evaluate_case(raw, case, sources, modules, device)})
    return result


def evaluate(fold, data_root, device_name):
    protocol, sources = contract.load_protocol()
    if fold not in sources['clean10']['folds']:
        raise contract.ContractError('unknown fold')
    binding = checkpoints.read_binding(sources)
    directory = contract.OUTPUT_ROOT + fold.lower() + '/'
    # Exclusive directory reservation prevents concurrent or partial-run overwrite.
    target = contract.safe_path(directory)
    if target.exists():
        raise contract.ContractError('formal fold output exists; refusing overwrite')
    _, payload = checkpoints.audit_all(sources, binding, wanted_fold=fold)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    rows = execute_fold(sources, fold, payload, binding['folds'][fold], device_name, data_root)
    for row in rows:
        row.update(artifact_scope='formal_test', protocol_version=contract.VERSION,
                   protocol_sha256=protocol['protocol_sha256'], checkpoint_binding_sha256=binding['binding_sha256'],
                   checkpoint_sha256=binding['folds'][fold]['checkpoint_sha256'], lambda_oss=0.5)
    contract.validate_rows(rows, sources, fold, binding)
    contract.write_once(directory + 'formal_test_cases.jsonl',
                        ''.join(contract.canonical(row) + '\n' for row in rows).encode())
    from aggregate_m4_osseous_formal_test import summarize
    contract.write_json_once(directory + 'formal_test_summary.json', summarize(rows))
    return {'fold_id': fold, 'case_count': len(rows), 'FORMAL_TEST_ACCESSED': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--evaluate', action='store_true', required=True)
    parser.add_argument('--fold', choices=['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'], required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'cuda:0'], default='cuda')
    args = parser.parse_args(argv)
    print(contract.canonical(evaluate(args.fold, args.data_root, args.device)))


if __name__ == '__main__':
    main()
