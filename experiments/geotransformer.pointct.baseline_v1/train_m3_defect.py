"""Formal PointCT training entry for 11 patients x 5 defect conditions.

The existing ``train_m3`` loop is reused through a narrow in-process adapter so
the frozen complete-subject entry and its formal split function remain
unchanged.  This file is a training entry; contract audits should use
``audit_defect_training_adapter.py`` and do not call :func:`main`.
"""

import argparse
import json
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import train_m3 as complete_training
from dataset import create_dataset, m2_ct_collate_fn, m2_point_collate_fn
from defect_training import (
    DefectTrainingContractError,
    build_formal_defect_split,
    build_split_provenance,
    create_formal_defect_dataset,
)
from perturbation import augment_point_sample, derive_perturbation_seed
from training_protocol import load_training_protocol, validation_perturbation_specs


DEFECT_PROTOCOL_STATUS = 'FORMAL M3 DEFECT TRAINING - PATIENT-LEVEL 5-FOLD SPLIT'


def _validate_raw_defect_identity(raw_sample, record):
    complete_training._raw_subject_id(raw_sample, expected_subject_id=record['subject_id'])
    defect_id = raw_sample.get('defect_id') if isinstance(raw_sample, dict) else None
    if defect_id != record['defect_id']:
        raise DefectTrainingContractError(
            f'dataset sample defect_id {defect_id!r} does not match record {record["defect_id"]!r}.'
        )
    return record['subject_id'], record['defect_id']


def iter_formal_defect_validation_samples(dataset, split, protocol):
    """Yield every validation defect under every frozen validation severity."""
    specs = validation_perturbation_specs(protocol)
    for dataset_index in split.val_indices:
        record = dataset.records[dataset_index]
        raw_sample = dataset[dataset_index]
        subject_id, defect_id = _validate_raw_defect_identity(raw_sample, record)
        for spec in specs:
            seed = derive_perturbation_seed(
                scheme_version=protocol['seed_scheme_version'],
                protocol_hash=protocol['protocol_hash'],
                fold_id=split.fold_id,
                root_seed=protocol['perturbation_root_seed'],
                purpose=spec['purpose'],
                epoch=None,
                subject_id=subject_id,
                severity=spec['severity'],
                variant_id=0,
            )
            yield (subject_id, defect_id), spec['severity'], augment_point_sample(
                raw_sample,
                seed=seed,
                max_rotation_deg=spec['max_rotation_deg'],
                max_translation_mm=spec['max_translation_mm'],
                scheme_version=protocol['seed_scheme_version'],
            )


@contextmanager
def _install_defect_adapter(protocol, *, enable_m4_defect_mapping):
    """Temporarily inject defect-aware boundaries into the reused M3 loop."""
    originals = {
        'create_dataset': complete_training.create_dataset,
        '_resolve_training_contract': complete_training._resolve_training_contract,
        'iter_formal_validation_samples': complete_training.iter_formal_validation_samples,
        'm2_point_collate_fn': complete_training.m2_point_collate_fn,
        'm2_ct_collate_fn': complete_training.m2_ct_collate_fn,
        '_training_config_record': complete_training._training_config_record,
        '_append_json_log': complete_training._append_json_log,
    }
    state = {'split': None, 'provenance': None}

    def defect_dataset_factory(data_root):
        return create_formal_defect_dataset(
            data_root,
            protocol,
            dataset_factory=create_dataset,
        )

    def defect_contract_resolver(args, dataset):
        if complete_training.resolve_training_mode(args) != 'formal':
            raise DefectTrainingContractError('defect training supports formal protocol mode only.')
        requested_protocol = load_training_protocol(args.protocol_manifest)
        if requested_protocol['protocol_hash'] != protocol['protocol_hash']:
            raise DefectTrainingContractError('training protocol changed during adapter setup.')
        split = build_formal_defect_split(dataset, requested_protocol, args.fold_id)
        provenance = build_split_provenance(split)
        state['split'] = split
        state['provenance'] = provenance
        print(DEFECT_PROTOCOL_STATUS)
        print(
            json.dumps(
                {
                    'adapter': provenance['adapter'],
                    'fold_id': split.fold_id,
                    'defect_ids': provenance['defect_ids'],
                    'train_instance_count': len(split.train_indices),
                    'val_instance_count': len(split.val_indices),
                    'test_instance_count': len(split.test_indices),
                    'enable_m4_defect_mapping': enable_m4_defect_mapping,
                },
                sort_keys=True,
            )
        )
        return 'formal', requested_protocol, split

    def defect_training_config_record(*args, **kwargs):
        record = originals['_training_config_record'](*args, **kwargs)
        if state['provenance'] is None:
            raise DefectTrainingContractError('defect split provenance was not resolved.')
        record['defect_training_provenance'] = state['provenance']
        record['enable_m4_defect_mapping'] = enable_m4_defect_mapping
        return record

    def append_defect_json_log(path, record):
        enriched = dict(record)
        if state['provenance'] is not None:
            enriched['defect_training_provenance'] = state['provenance']
            enriched['enable_m4_defect_mapping'] = enable_m4_defect_mapping
        return originals['_append_json_log'](path, enriched)

    complete_training.create_dataset = defect_dataset_factory
    complete_training._resolve_training_contract = defect_contract_resolver
    complete_training.iter_formal_validation_samples = iter_formal_defect_validation_samples
    complete_training.m2_point_collate_fn = partial(
        m2_point_collate_fn,
        enable_m4_defect_mapping=enable_m4_defect_mapping,
    )
    complete_training.m2_ct_collate_fn = partial(
        m2_ct_collate_fn,
        enable_m4_defect_mapping=enable_m4_defect_mapping,
    )
    complete_training._training_config_record = defect_training_config_record
    complete_training._append_json_log = append_defect_json_log
    try:
        yield state
    finally:
        for name, value in originals.items():
            setattr(complete_training, name, value)


def run_defect_training(args):
    """Run the reused M3 loop over the formal 55-record defect dataset."""
    protocol = load_training_protocol(args.protocol_manifest)
    enabled = getattr(args, 'enable_m4_defect_mapping', False)
    if not isinstance(enabled, bool):
        raise DefectTrainingContractError('enable_m4_defect_mapping must be bool.')
    with _install_defect_adapter(protocol, enable_m4_defect_mapping=enabled) as state:
        result = complete_training.train(args)
    output = dict(result)
    output['defect_training_provenance'] = state['provenance']
    output['enable_m4_defect_mapping'] = enabled
    return output


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Formal patient-level PointCT training over 55 explicit defect instances.'
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol-manifest', type=Path, required=True)
    parser.add_argument('--fold-id', required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--json-log', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=complete_training.TRAINING_SMOKE_DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=complete_training.TRAINING_SMOKE_DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument('--batch-size', type=int, default=complete_training.BATCH_SIZE)
    parser.add_argument('--precision', default=complete_training.PRECISION)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--sinkhorn-iterations', type=int, required=True)
    parser.add_argument('--alpha-init', type=float, required=True)
    parser.add_argument(
        '--enable-m4-defect-mapping',
        action='store_true',
        help='Explicitly derive M4 coarse mapping fields; omitted keeps frozen M3 preprocessing.',
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    run_defect_training(args)


if __name__ == '__main__':
    main()
