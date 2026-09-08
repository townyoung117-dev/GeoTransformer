"""CPU-only checkpoint binding/verification. No dataset or training entrypoint."""

import argparse
import io
import math
from collections.abc import Mapping

import m4_osseous_formal_protocol as contract


def expected_config(sources, fold):
    s = sources['selection_v1']['training_settings']
    clean = sources['clean10']
    provenance = {'adapter': 'pointct_defect_training_v1', 'fold_id': fold,
                  'defect_ids': sources['evaluation']['required_defect_ids']}
    for partition in ('train', 'val', 'test'):
        provenance[partition + '_instances'] = [
            {'subject_id': patient, 'defect_id': defect, 'fold_id': fold, 'partition': partition}
            for patient in clean['folds'][fold][partition + '_subject_ids']
            for defect in provenance['defect_ids']]
    return {
        'learning_rate': s['optimizer']['learning_rate'], 'weight_decay': s['optimizer']['weight_decay'],
        'batch_size': s['batch_size'], 'precision': s['precision'], 'seed': s['seed'],
        'temperature': s['temperature'], 'sinkhorn_iterations': s['sinkhorn_iterations'], 'alpha_init': s['alpha_init'],
        'primary_max_distance_mm': 17.5, 'high_confidence_distance_mm': 15.0,
        'hyperparameter_status': 'TRAINING SMOKE DEFAULTS - NOT FROZEN PAPER HYPERPARAMETERS',
        'split_status': 'FORMAL M3-6B 5-FOLD PROTOCOL',
        'enable_m4_defect_mapping': True, 'm4_hard_constraint_active': True,
        'm4_soft_modulation_enabled': True, 'm4_soft_modulation_active': True,
        'm4_soft_sigma_mm': s['m4_soft']['sigma_mm'], 'm4_soft_strength': s['m4_soft']['strength'],
        'm4_soft_hyperparameter_status': 'M4 DEVELOPMENT HYPERPARAMETERS - NOT FROZEN PAPER HYPERPARAMETERS',
        'm4_osseous_prior_enabled': True, 'm4_osseous_prior_active': True, 'm4_osseous_strength': 0.5,
        'osseous_center_hu': s['osseous_score']['center_hu'], 'osseous_tau_hu': s['osseous_score']['tau_hu'],
        'osseous_radius_mm': s['osseous_score']['radius_mm'],
        'm4_osseous_hyperparameter_status': 'DEVELOPMENT_ONLY - NOT FROZEN PAPER HYPERPARAMETER',
        'defect_training_provenance': provenance,
    }


def audit_metadata(payload, log_rows, sources, fold, checkpoint_path, checkpoint_sha, log_sha):
    """Pure CPU metadata contract; unit tests pass synthetic mappings here."""
    paths = contract.checkpoint_paths(sources)
    if fold not in paths or checkpoint_path != paths[fold]:
        raise contract.ContractError('wrong fold/checkpoint path or basename')
    for value in (checkpoint_sha, log_sha):
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise contract.ContractError('invalid checkpoint/log SHA')
    clean = sources['clean10']
    config = expected_config(sources, fold)
    expected = {'checkpoint_version': 2, 'formal_protocol': True,
                'protocol_version': clean['protocol_version'], 'protocol_hash': clean['protocol_hash'],
                'fold_id': fold, 'seed': config['seed'],
                'perturbation_root_seed': clean['perturbation_root_seed'],
                'perturbation_seed_scheme_version': clean['seed_scheme_version'],
                'train_subject_ids': clean['folds'][fold]['train_subject_ids'],
                'val_subject_ids': clean['folds'][fold]['val_subject_ids'], 'training_config': config}
    for key, value in expected.items():
        contract.equal(payload.get(key), value, 'checkpoint ' + key)
    for key in ('point_encoder_state_dict', 'ct_encoder_state_dict', 'matcher_state_dict', 'optimizer_state_dict'):
        if not isinstance(payload.get(key), Mapping) or not payload[key]:
            raise contract.ContractError('missing checkpoint state: ' + key)
    if type(payload.get('global_step')) is not int or payload['global_step'] < 0:
        raise contract.ContractError('invalid global step')
    groups = payload['optimizer_state_dict'].get('param_groups')
    if not isinstance(groups, list) or not groups:
        raise contract.ContractError('missing optimizer groups')
    for group in groups:
        contract.equal(group.get('lr'), config['learning_rate'], 'optimizer learning rate')
        contract.equal(group.get('weight_decay'), config['weight_decay'], 'optimizer decay')
    epochs = sources['selection_v1']['training_settings']['epochs']
    if len(log_rows) != epochs:
        raise contract.ContractError('training log must contain frozen full epoch budget')
    best, best_epoch = math.inf, None
    for index, row in enumerate(log_rows):
        contract.canonical(row)
        for key, value in {'epoch': index, 'formal_protocol': True, 'protocol_version': clean['protocol_version'],
                           'protocol_hash': clean['protocol_hash'], 'fold_id': fold,
                           'num_val_cases': 30, 'num_val_subjects': 2, 'num_val_perturbations': 30,
                           'learning_rate': config['learning_rate'],
                           'defect_training_provenance': config['defect_training_provenance']}.items():
            contract.equal(row.get(key), value, 'training log ' + key)
        for key in ('enable_m4_defect_mapping', 'm4_hard_constraint_active', 'm4_soft_modulation_enabled',
                    'm4_soft_modulation_active', 'm4_soft_sigma_mm', 'm4_soft_strength',
                    'm4_osseous_prior_enabled', 'm4_osseous_prior_active', 'm4_osseous_strength',
                    'osseous_center_hu', 'osseous_tau_hu', 'osseous_radius_mm'):
            contract.equal(row.get(key), config[key], 'training log ' + key)
        value = row.get('val_mean_loss')
        contract.finite(value, 'validation objective', nonnegative=False)
        improved = value < best
        if improved:
            best, best_epoch = value, index
        contract.equal(row.get('best_checkpoint_updated'), improved, 'strict-less best checkpoint')
        contract.equal(row.get('best_val_loss'), best, 'running validation minimum')
    contract.equal(payload.get('epoch'), best_epoch, 'best epoch')
    contract.equal(payload.get('best_val_loss'), best, 'best validation objective')
    return {'fold_id': fold, 'checkpoint_path': checkpoint_path, 'checkpoint_sha256': checkpoint_sha,
            'best_epoch': best_epoch, 'best_val_objective': best,
            'training_config': config, 'training_config_sha256': contract.digest(config),
            'train_subject_ids': expected['train_subject_ids'], 'val_subject_ids': expected['val_subject_ids'],
            'lambda_oss': 0.5, 'hard_enabled': True, 'soft_enabled': True, 'osseous_enabled': True,
            'training_log_sha256': log_sha}


def decode_checkpoint(raw):
    import torch
    # Deserialize precisely the buffer whose SHA was checked, on CPU only.
    # No arbitrary pickle fallback is allowed.
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def validate_binding(binding, sources):
    if set(binding) != {'binding_version', 'binding_status', 'formal_protocol_sha256', 'formal_test_accessed',
                        'gpu_used', 'folds', 'binding_sha256'}:
        raise contract.ContractError('unexpected checkpoint binding schema')
    contract.equal(binding['binding_sha256'], contract.digest(binding, 'binding_sha256'), 'binding SHA')
    contract.equal(binding['binding_version'], contract.VERSION + '_checkpoint_binding_v1', 'binding version')
    contract.equal(binding['binding_status'], 'FROZEN_BEFORE_FIRST_M4_FORMAL_TEST', 'binding status')
    contract.equal(binding['formal_protocol_sha256'], contract.PROTOCOL_SHA, 'binding formal SHA')
    contract.equal(binding['formal_test_accessed'], False, 'binding firewall')
    contract.equal(binding['gpu_used'], False, 'CPU binding')
    contract.equal(sorted(binding['folds']), sorted(sources['clean10']['folds']), 'binding folds')
    for fold, row in binding['folds'].items():
        contract.equal(row['checkpoint_path'], contract.checkpoint_paths(sources)[fold], 'binding checkpoint path')
        contract.equal(row['training_config'], expected_config(sources, fold), 'binding config')
        contract.equal(row['training_config_sha256'], contract.digest(row['training_config']), 'binding config SHA')
    return binding


def read_binding(sources, *, committed=True):
    if committed:
        contract.require_committed(contract.PROTOCOL_PATH)
        contract.require_committed(contract.BINDING_PATH)
    return validate_binding(contract.parse(contract.read_file(contract.BINDING_PATH)), sources)


def audit_all(sources, binding=None, wanted_fold=None):
    folds, selected_payload = {}, None
    for fold, relative in contract.checkpoint_paths(sources).items():
        raw = contract.read_file(relative)
        checkpoint_sha = contract.sha(raw)
        if binding:
            contract.equal(checkpoint_sha, binding['folds'][fold]['checkpoint_sha256'], 'checkpoint SHA')
        payload = decode_checkpoint(raw)
        log_path = relative.rsplit('/checkpoints/', 1)[0] + '/train.jsonl'
        raw_log = contract.read_file(log_path)
        rows = [contract.parse(line) for line in raw_log.splitlines()]
        folds[fold] = audit_metadata(payload, rows, sources, fold, relative, checkpoint_sha, contract.sha(raw_log))
        if binding:
            contract.equal(folds[fold], binding['folds'][fold], 'bound checkpoint metadata')
        if fold == wanted_fold:
            selected_payload = payload
        del payload, raw
    return folds, selected_payload


def bind():
    _, sources = contract.load_protocol()
    if contract.safe_path(contract.BINDING_PATH).exists():
        raise contract.ContractError('checkpoint binding exists; refusing overwrite')
    folds, _ = audit_all(sources)
    result = {'binding_version': contract.VERSION + '_checkpoint_binding_v1',
              'binding_status': 'FROZEN_BEFORE_FIRST_M4_FORMAL_TEST',
              'formal_protocol_sha256': contract.PROTOCOL_SHA, 'formal_test_accessed': False,
              'gpu_used': False, 'folds': folds}
    result['binding_sha256'] = contract.digest(result)
    validate_binding(result, sources)
    contract.write_json_once(contract.BINDING_PATH, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--checkpoint-bind', action='store_true')
    mode.add_argument('--verify-checkpoints', action='store_true')
    args = parser.parse_args(argv)
    if args.checkpoint_bind:
        result = bind()
    else:
        _, sources = contract.load_protocol()
        binding = read_binding(sources)
        audit_all(sources, binding)
        result = {'CHECKPOINT_BINDING': 'PASS', 'binding_sha256': binding['binding_sha256'],
                  'FORMAL_TEST_ACCESSED': False, 'GPU_USED': False}
    print(contract.canonical(result))


if __name__ == '__main__':
    main()
