import argparse
import copy
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
TRAINING_PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
EVALUATION_PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_defect_eval_v1.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import defect_evaluation
import evaluate_m3_defect
from audit_defect_evaluation_adapter import run_audit
from defect_evaluation import (
    CASES_PER_DEFECT_INSTANCE,
    EXPECTED_TEST_CASE_COUNT,
    DefectEvaluationContractError,
    build_defect_test_manifest,
    expected_defect_training_provenance,
    resolve_fold_artifacts,
    run_protocol_audit,
    validate_defect_checkpoint_metadata,
    validate_defect_evaluation_protocol,
    validate_expanded_partitions,
    validate_jsonl_best_checkpoint,
)
from defect_training import (
    DEFECT_CONDITION_COUNT,
    DEFECT_IDS,
    EXPECTED_DEFECT_INSTANCE_COUNT,
    READY_PATIENT_COUNT,
    DefectTrainingContractError,
    build_defect_variants,
    build_formal_defect_split,
)
from evaluation import aggregate_case_metrics
from training_protocol import FOLD_SUBJECTS, load_training_protocol, resolve_fold
from defect_evaluation import load_defect_evaluation_protocol


class FakeDataset:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)


class DefectEvaluationStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_protocol = load_training_protocol(TRAINING_PROTOCOL_PATH)
        cls.evaluation_protocol = load_defect_evaluation_protocol(
            EVALUATION_PROTOCOL_PATH
        )
        cls.variants = build_defect_variants(
            tuple(cls.training_protocol['ready_subject_ids'])
        )
        cls.records = [
            {'subject_id': subject_id, 'defect_id': defect_id}
            for subject_id, defect_id in cls.variants
        ]

    def test_frozen_counts_are_11_patients_5_defects_and_55_instances(self):
        self.assertEqual(READY_PATIENT_COUNT, 11)
        self.assertEqual(DEFECT_CONDITION_COUNT, 5)
        self.assertEqual(EXPECTED_DEFECT_INSTANCE_COUNT, 55)
        self.assertEqual(len(self.variants), 55)
        self.assertEqual(len(set(self.variants)), 55)

    def test_all_fold_test_instance_and_case_counts(self):
        expected = {
            'Fold1': (15, 225),
            'Fold2': (10, 150),
            'Fold3': (10, 150),
            'Fold4': (10, 150),
            'Fold5': (10, 150),
        }
        for fold_id, (instance_count, case_count) in expected.items():
            with self.subTest(fold_id=fold_id):
                manifest = build_defect_test_manifest(
                    self.training_protocol,
                    self.evaluation_protocol,
                    fold_id,
                )
                self.assertEqual(manifest['total_test_instances'], instance_count)
                self.assertEqual(manifest['total_cases'], case_count)
                self.assertEqual(
                    manifest['cases_per_defect_instance'],
                    CASES_PER_DEFECT_INSTANCE,
                )

    def test_global_protocol_audit_is_exactly_55_instances_and_825_cases(self):
        result = run_protocol_audit(
            self.training_protocol,
            self.evaluation_protocol,
        )
        self.assertEqual(result['expected_test_instance_count'], 55)
        self.assertEqual(result['expected_test_case_count'], 825)
        self.assertEqual(EXPECTED_TEST_CASE_COUNT, 825)
        self.assertTrue(result['patient_level_leakage_pass'])
        self.assertTrue(result['instance_level_leakage_pass'])
        self.assertTrue(result['pair_uniqueness_pass'])
        self.assertTrue(result['perturbation_count_pass'])

    def test_every_patient_has_all_five_defects_in_one_test_fold(self):
        subject_folds = defaultdict(set)
        subject_defects = defaultdict(set)
        for fold_id in FOLD_SUBJECTS:
            manifest = build_defect_test_manifest(
                self.training_protocol,
                self.evaluation_protocol,
                fold_id,
            )
            for case in manifest['cases']:
                subject_folds[case['subject_id']].add(fold_id)
                subject_defects[case['subject_id']].add(case['defect_id'])
        self.assertEqual(set(subject_folds), set(self.training_protocol['ready_subject_ids']))
        self.assertTrue(all(len(folds) == 1 for folds in subject_folds.values()))
        self.assertTrue(
            all(defects == set(DEFECT_IDS) for defects in subject_defects.values())
        )

    def test_patient_level_leakage_fails_closed(self):
        fold = resolve_fold(self.training_protocol, 'Fold1')

        def expand(subject_ids):
            return tuple(
                (subject_id, defect_id)
                for subject_id in subject_ids
                for defect_id in DEFECT_IDS
            )

        train = expand(fold['train_subject_ids']) + expand(('Pat12',))
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            'patient-level leakage',
        ):
            validate_expanded_partitions(
                'Fold1',
                tuple(self.training_protocol['ready_subject_ids']),
                train,
                expand(fold['val_subject_ids']),
                expand(fold['test_subject_ids']),
            )

    def test_subject_defect_duplicate_fails_closed(self):
        records = list(self.records) + [dict(self.records[0])]
        with self.assertRaisesRegex(DefectTrainingContractError, 'duplicate instance'):
            build_formal_defect_split(
                FakeDataset(records),
                self.training_protocol,
                'Fold1',
            )

    def test_missing_defect_condition_fails_closed(self):
        with self.assertRaisesRegex(
            DefectTrainingContractError,
            'exactly once|exactly 55',
        ):
            build_formal_defect_split(
                FakeDataset(self.records[:-1]),
                self.training_protocol,
                'Fold1',
            )

    def test_unknown_defect_id_fails_closed(self):
        records = list(self.records)
        records[0] = {
            'subject_id': records[0]['subject_id'],
            'defect_id': 'defect_001_right_maxilla_cheek_large',
        }
        with self.assertRaisesRegex(DefectTrainingContractError, 'unknown defect_id'):
            build_formal_defect_split(
                FakeDataset(records),
                self.training_protocol,
                'Fold1',
            )

    def test_all_825_case_keys_are_unique(self):
        keys = []
        for fold_id in FOLD_SUBJECTS:
            manifest = build_defect_test_manifest(
                self.training_protocol,
                self.evaluation_protocol,
                fold_id,
            )
            keys.extend(case['case_key'] for case in manifest['cases'])
        self.assertEqual(len(keys), 825)
        self.assertEqual(len(set(keys)), 825)

    def test_same_patient_severity_variant_reuses_one_perturbation_across_defects(self):
        manifest = build_defect_test_manifest(
            self.training_protocol,
            self.evaluation_protocol,
            'Fold1',
        )
        groups = defaultdict(list)
        for case in manifest['cases']:
            groups[
                (
                    case['subject_id'],
                    case['severity'],
                    case['variant_index'],
                )
            ].append(case)
        self.assertTrue(groups)
        for cases in groups.values():
            self.assertEqual(len(cases), 5)
            self.assertEqual(len({case['perturbation_seed'] for case in cases}), 1)
            self.assertEqual(
                len({case['perturbation_identifier'] for case in cases}),
                1,
            )
            self.assertEqual(
                len({case['perturbation_angle_deg'] for case in cases}),
                1,
            )
            self.assertEqual(
                len(
                    {
                        tuple(case['perturbation_translation_mm'])
                        for case in cases
                    }
                ),
                1,
            )
            self.assertEqual({case['defect_id'] for case in cases}, set(DEFECT_IDS))

    def test_wrong_defect_evaluation_protocol_hash_fails_closed(self):
        protocol = copy.deepcopy(self.evaluation_protocol)
        protocol['evaluation_protocol_hash'] = 'f' * 64
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            'evaluation_protocol_hash mismatch',
        ):
            validate_defect_evaluation_protocol(protocol)

    def test_audit_truthfully_reports_real_checks_not_run(self):
        result = run_audit(
            self.training_protocol,
            self.evaluation_protocol,
        )
        self.assertTrue(result['protocol_audit_run'])
        self.assertFalse(result['real_55_data_audit_run'])
        self.assertFalse(result['real_checkpoint_audit_run'])

    def test_original_metric_aggregator_is_reused(self):
        self.assertIs(defect_evaluation.aggregate_case_metrics, aggregate_case_metrics)


class DefectEvaluationCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_protocol = load_training_protocol(TRAINING_PROTOCOL_PATH)
        cls.evaluation_protocol = load_defect_evaluation_protocol(
            EVALUATION_PROTOCOL_PATH
        )

    def _payload(self, fold_id='Fold1', epoch=7, best_val_loss=1.0):
        fold = resolve_fold(self.training_protocol, fold_id)
        required = self.evaluation_protocol['required_formal_training']
        training_config = {
            field: required[field]
            for field in (
                'learning_rate',
                'weight_decay',
                'batch_size',
                'precision',
                'seed',
                'temperature',
                'sinkhorn_iterations',
                'alpha_init',
            )
        }
        training_config.update(
            {
                'defect_training_provenance': expected_defect_training_provenance(
                    self.training_protocol,
                    fold_id,
                ),
                'enable_m4_defect_mapping': False,
            }
        )
        return {
            'formal_protocol': True,
            'protocol_version': self.training_protocol['protocol_version'],
            'protocol_hash': self.training_protocol['protocol_hash'],
            'fold_id': fold_id,
            'train_subject_ids': list(fold['train_subject_ids']),
            'val_subject_ids': list(fold['val_subject_ids']),
            'test_subject_ids': list(fold['test_subject_ids']),
            'perturbation_root_seed': self.training_protocol[
                'perturbation_root_seed'
            ],
            'perturbation_seed_scheme_version': self.training_protocol[
                'seed_scheme_version'
            ],
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'seed': required['seed'],
            'training_config': training_config,
        }

    def _metadata(self, payload=None, fold_id='Fold1'):
        if payload is None:
            payload = self._payload(fold_id)
        return validate_defect_checkpoint_metadata(
            Path('best_val_loss.pt'),
            payload,
            self.training_protocol,
            self.evaluation_protocol,
            fold_id,
        )

    def _jsonl_records(self, fold_id='Fold1', best_epoch=7):
        fold = resolve_fold(self.training_protocol, fold_id)
        provenance = expected_defect_training_provenance(
            self.training_protocol,
            fold_id,
        )
        records = []
        running_best = float('inf')
        for epoch in range(20):
            val_loss = 1.0 if epoch == best_epoch else 100.0 + epoch
            updated = val_loss < running_best
            running_best = min(running_best, val_loss)
            records.append(
                {
                    'epoch': epoch,
                    'formal_protocol': True,
                    'protocol_version': self.training_protocol['protocol_version'],
                    'protocol_hash': self.training_protocol['protocol_hash'],
                    'fold_id': fold_id,
                    'test_subject_ids': list(fold['test_subject_ids']),
                    'enable_m4_defect_mapping': False,
                    'defect_training_provenance': provenance,
                    'val_mean_loss': val_loss,
                    'best_val_loss': running_best,
                    'best_checkpoint_updated': updated,
                }
            )
        return records

    def test_checkpoint_fold_mismatch_fails_closed(self):
        payload = self._payload('Fold2')
        with self.assertRaisesRegex(DefectEvaluationContractError, 'fold_id'):
            self._metadata(payload, fold_id='Fold1')

    def test_checkpoint_protocol_mismatch_fails_closed(self):
        payload = self._payload()
        payload['protocol_hash'] = 'a' * 64
        with self.assertRaisesRegex(DefectEvaluationContractError, 'protocol_hash'):
            self._metadata(payload)

    def test_checkpoint_seed_mismatch_fails_closed(self):
        payload = self._payload()
        payload['seed'] += 1
        payload['training_config']['seed'] = payload['seed']
        with self.assertRaisesRegex(DefectEvaluationContractError, 'checkpoint seed'):
            self._metadata(payload)

    def test_checkpoint_training_hyperparameter_mismatch_fails_closed(self):
        payload = self._payload()
        payload['training_config']['temperature'] = 0.2
        with self.assertRaisesRegex(DefectEvaluationContractError, 'temperature'):
            self._metadata(payload)

    def test_m4_mapping_true_checkpoint_fails_closed(self):
        payload = self._payload()
        payload['training_config']['enable_m4_defect_mapping'] = True
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            'enable_m4_defect_mapping=false',
        ):
            self._metadata(payload)

    def test_old_complete_subject_checkpoint_is_not_accepted(self):
        payload = self._payload()
        payload['training_config'].pop('defect_training_provenance')
        payload['training_config'].pop('enable_m4_defect_mapping')
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            'old complete-subject checkpoints',
        ):
            self._metadata(payload)

    def test_checkpoint_defect_provenance_mismatch_fails_closed(self):
        payload = self._payload()
        payload['training_config']['defect_training_provenance'][
            'adapter'
        ] = 'pointct_complete_subject'
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            'defect_training_provenance',
        ):
            self._metadata(payload)

    def test_jsonl_best_epoch_and_checkpoint_epoch_match_passes(self):
        metadata = self._metadata()
        result = validate_jsonl_best_checkpoint(
            self._jsonl_records(),
            metadata,
            self.training_protocol,
            self.evaluation_protocol,
            'Fold1',
        )
        self.assertEqual(result['jsonl_epoch_count'], 20)
        self.assertEqual(result['jsonl_best_epoch'], 7)
        self.assertEqual(result['jsonl_min_val_mean_loss'], 1.0)
        self.assertTrue(result['best_checkpoint_cross_check_pass'])

    def test_jsonl_best_epoch_checkpoint_epoch_mismatch_fails_closed(self):
        metadata = self._metadata(self._payload(epoch=6))
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            'best epoch does not match checkpoint epoch',
        ):
            validate_jsonl_best_checkpoint(
                self._jsonl_records(best_epoch=7),
                metadata,
                self.training_protocol,
                self.evaluation_protocol,
                'Fold1',
            )

    def test_jsonl_must_contain_all_20_epochs(self):
        metadata = self._metadata()
        with self.assertRaisesRegex(DefectEvaluationContractError, 'exactly 20'):
            validate_jsonl_best_checkpoint(
                self._jsonl_records()[:-1],
                metadata,
                self.training_protocol,
                self.evaluation_protocol,
                'Fold1',
            )

    def test_checkpoint_and_log_paths_resolve_from_root(self):
        checkpoint, log = resolve_fold_artifacts(
            Path('/formal-checkpoints'),
            'Fold3',
            self.evaluation_protocol,
        )
        self.assertEqual(checkpoint.name, 'best_val_loss.pt')
        self.assertEqual(checkpoint.parent.name, 'Fold3')
        self.assertEqual(log.name, 'Fold3.jsonl')


class DefectEvaluationCLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_protocol = load_training_protocol(TRAINING_PROTOCOL_PATH)
        cls.evaluation_protocol = load_defect_evaluation_protocol(
            EVALUATION_PROTOCOL_PATH
        )

    def test_no_explicit_mode_fails_before_execution(self):
        args = argparse.Namespace(manifest_only=False, execute_test=False)
        with mock.patch.object(evaluate_m3_defect, 'run_execute_test') as execute:
            with self.assertRaisesRegex(
                DefectEvaluationContractError,
                'explicitly choose',
            ):
                evaluate_m3_defect.run_evaluation(args)
        execute.assert_not_called()

    def test_manifest_only_fold1_does_not_read_data_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                manifest_only=True,
                execute_test=False,
                all_folds=False,
                fold_id='Fold1',
                protocol_manifest=TRAINING_PROTOCOL_PATH,
                evaluation_protocol=EVALUATION_PROTOCOL_PATH,
                output_dir=Path(directory),
                data_root=Path('must-not-be-read'),
                checkpoint_root=Path('must-not-be-read'),
                json_log_root=None,
                device='cuda',
            )
            with mock.patch.object(
                evaluate_m3_defect,
                '_create_defect_dataset_for_execution',
                side_effect=AssertionError('dataset read'),
            ), mock.patch.object(
                defect_evaluation,
                'read_and_validate_defect_checkpoint',
                side_effect=AssertionError('checkpoint read'),
            ):
                result = evaluate_m3_defect.run_evaluation(args)
            self.assertEqual(result['mode'], 'manifest-only')
            self.assertEqual(result['folds']['Fold1']['test_instances'], 15)
            self.assertEqual(result['folds']['Fold1']['test_cases'], 225)
            self.assertTrue(
                (Path(directory) / 'Fold1' / 'test_manifest.json').is_file()
            )

    def test_execute_test_requires_checkpoint_root(self):
        args = argparse.Namespace(
            manifest_only=False,
            execute_test=True,
            all_folds=False,
            fold_id='Fold1',
            protocol_manifest=TRAINING_PROTOCOL_PATH,
            evaluation_protocol=EVALUATION_PROTOCOL_PATH,
            output_dir=Path('unused'),
            data_root=Path('unused'),
            checkpoint_root=None,
            json_log_root=None,
            device='cuda',
        )
        with self.assertRaisesRegex(
            DefectEvaluationContractError,
            '--checkpoint-root',
        ):
            evaluate_m3_defect.run_evaluation(args)


if __name__ == '__main__':
    unittest.main()
