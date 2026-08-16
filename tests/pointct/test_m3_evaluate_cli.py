import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
TRAINING_PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
EVALUATION_PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_7_eval_v1.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import evaluate_m3
from evaluation import M3EvaluationContractError, load_evaluation_protocol
from training_protocol import load_training_protocol, resolve_fold


CHECKPOINT_VERSION = 2
SMOKE_CHECKPOINT_VERSION = 1
EXPECTED_FORMAL_BESTS = {
    'Fold1': {'epoch': 11, 'best_val_loss': 2.187397321065267},
    'Fold2': {'epoch': 17, 'best_val_loss': 2.3390124638875327},
    'Fold3': {'epoch': 19, 'best_val_loss': 2.2600975036621094},
    'Fold4': {'epoch': 14, 'best_val_loss': 2.5588058630625405},
    'Fold5': {'epoch': 18, 'best_val_loss': 3.0356096691555448},
}


class M3EvaluateManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_protocol = load_training_protocol(TRAINING_PROTOCOL_PATH)
        cls.evaluation_protocol = load_evaluation_protocol(EVALUATION_PROTOCOL_PATH)

    def test_manifest_has_fifteen_cases_per_subject(self):
        for fold_id in self.training_protocol['folds']:
            with self.subTest(fold_id=fold_id):
                manifest = evaluate_m3.build_test_manifest(
                    self.training_protocol,
                    self.evaluation_protocol,
                    fold_id,
                )
                expected_subjects = resolve_fold(
                    self.training_protocol,
                    fold_id,
                )['test_subject_ids']
                self.assertEqual(manifest['test_subject_ids'], list(expected_subjects))
                self.assertEqual(manifest['cases_per_subject'], 15)
                self.assertEqual(
                    manifest['total_cases'],
                    15 * len(expected_subjects),
                )

    def test_manifest_is_deterministic(self):
        first = evaluate_m3.build_test_manifest(
            self.training_protocol,
            self.evaluation_protocol,
            'Fold1',
        )
        second = evaluate_m3.build_test_manifest(
            self.training_protocol,
            self.evaluation_protocol,
            'Fold1',
        )
        self.assertEqual(first, second)

    def test_manifest_case_keys_are_unique_and_perturbations_nonidentity(self):
        manifest = evaluate_m3.build_test_manifest(
            self.training_protocol,
            self.evaluation_protocol,
            'Fold1',
        )
        keys = [case['case_key'] for case in manifest['cases']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(case['perturbation_nonidentity'] for case in manifest['cases']))
        self.assertTrue(all(case['purpose'] == 'test' for case in manifest['cases']))
        self.assertTrue(all(case['epoch'] is None for case in manifest['cases']))

    def test_manifest_test_split_is_disjoint_from_training_and_validation(self):
        for fold_id in self.training_protocol['folds']:
            fold = resolve_fold(self.training_protocol, fold_id)
            manifest = evaluate_m3.build_test_manifest(
                self.training_protocol,
                self.evaluation_protocol,
                fold_id,
            )
            observed = {case['subject_id'] for case in manifest['cases']}
            self.assertEqual(observed, set(fold['test_subject_ids']))
            self.assertFalse(observed & set(fold['train_subject_ids']))
            self.assertFalse(observed & set(fold['val_subject_ids']))

    def test_manifest_only_does_not_read_dataset_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                manifest_only=True,
                execute_test=False,
                protocol_manifest=TRAINING_PROTOCOL_PATH,
                evaluation_protocol=EVALUATION_PROTOCOL_PATH,
                fold_id='Fold1',
                output_dir=Path(directory),
                data_root=Path('must-not-be-read'),
                checkpoint=Path('must-not-be-loaded.pt'),
                device='cuda',
            )
            with mock.patch.object(
                evaluate_m3,
                '_create_dataset_for_execution',
                side_effect=AssertionError('dataset read'),
            ), mock.patch.object(
                evaluate_m3,
                '_load_checkpoint_for_execution',
                side_effect=AssertionError('checkpoint read'),
            ):
                result = evaluate_m3.run_evaluation(args)
            self.assertEqual(result['mode'], 'manifest-only')
            output = Path(directory) / 'test_manifest.json'
            self.assertTrue(output.is_file())
            manifest = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(manifest['total_cases'], 45)

    def test_no_explicit_mode_fails_before_any_real_execution(self):
        args = argparse.Namespace(manifest_only=False, execute_test=False)
        with mock.patch.object(evaluate_m3, 'run_execute_test') as execute:
            with self.assertRaisesRegex(M3EvaluationContractError, 'Explicitly choose'):
                evaluate_m3.run_evaluation(args)
        execute.assert_not_called()


class M3EvaluateCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_training_protocol(TRAINING_PROTOCOL_PATH)
        cls.evaluation_protocol = load_evaluation_protocol(EVALUATION_PROTOCOL_PATH)

    def _formal_payload(self, fold_id='Fold1'):
        fold = resolve_fold(self.protocol, fold_id)
        frozen_best = self.evaluation_protocol['formal_best_checkpoints'][fold_id]
        required_training = self.evaluation_protocol['required_formal_training']
        return {
            'checkpoint_version': CHECKPOINT_VERSION,
            'epoch': frozen_best['epoch'],
            'global_step': 100,
            'point_encoder_state_dict': {},
            'ct_encoder_state_dict': {},
            'matcher_state_dict': {},
            'optimizer_state_dict': {},
            'train_subject_ids': list(fold['train_subject_ids']),
            'val_subject_ids': list(fold['val_subject_ids']),
            'seed': required_training['seed'],
            'training_config': {
                field: required_training[field]
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
            },
            'best_val_loss': frozen_best['best_val_loss'],
            'formal_protocol': True,
            'protocol_version': self.protocol['protocol_version'],
            'protocol_hash': self.protocol['protocol_hash'],
            'fold_id': fold_id,
            'test_subject_ids': list(fold['test_subject_ids']),
            'perturbation_root_seed': self.protocol['perturbation_root_seed'],
            'perturbation_seed_scheme_version': self.protocol['seed_scheme_version'],
        }

    def _smoke_payload(self):
        payload = self._formal_payload()
        payload['checkpoint_version'] = SMOKE_CHECKPOINT_VERSION
        for field in (
            'formal_protocol',
            'protocol_version',
            'protocol_hash',
            'fold_id',
            'test_subject_ids',
            'perturbation_root_seed',
            'perturbation_seed_scheme_version',
        ):
            payload.pop(field)
        return payload

    def _validate(self, payload, path='best_val_loss.pt', fold_id='Fold1'):
        return evaluate_m3.validate_evaluation_checkpoint_metadata(
            Path(path),
            payload,
            self.protocol,
            self.evaluation_protocol,
            fold_id,
        )

    def test_checkpoint_fold_mismatch_fails(self):
        payload = self._formal_payload('Fold2')
        with self.assertRaisesRegex(M3EvaluationContractError, 'fold_id'):
            self._validate(payload, fold_id='Fold1')

    def test_checkpoint_protocol_hash_mismatch_fails(self):
        payload = self._formal_payload()
        payload['protocol_hash'] = 'f' * 64
        with self.assertRaisesRegex(M3EvaluationContractError, 'protocol_hash'):
            self._validate(payload)

    def test_checkpoint_formal_protocol_false_fails(self):
        with self.assertRaisesRegex(M3EvaluationContractError, 'formal_protocol'):
            self._validate(self._smoke_payload())

    def test_last_checkpoint_is_rejected_without_fallback(self):
        with self.assertRaisesRegex(M3EvaluationContractError, 'last.pt'):
            self._validate(self._formal_payload(), path='last.pt')

    def test_exact_frozen_fold1_formal_metadata_passes(self):
        metadata = self._validate(self._formal_payload())
        self.assertEqual(metadata['fold_id'], 'Fold1')
        self.assertEqual(metadata['epoch'], 11)
        self.assertEqual(metadata['best_val_loss'], 2.187397321065267)
        self.assertEqual(
            metadata['test_subject_ids'],
            resolve_fold(self.protocol, 'Fold1')['test_subject_ids'],
        )

    def test_checkpoint_other_fold_split_subjects_fail(self):
        payload = self._formal_payload()
        other_fold = resolve_fold(self.protocol, 'Fold2')
        payload['train_subject_ids'] = list(other_fold['train_subject_ids'])
        with self.assertRaisesRegex(M3EvaluationContractError, 'another Fold'):
            self._validate(payload)

    def test_wrong_best_epoch_fails(self):
        payload = self._formal_payload()
        payload['epoch'] += 1
        with self.assertRaisesRegex(M3EvaluationContractError, 'epoch'):
            self._validate(payload)

    def test_pilot_like_best_checkpoint_fails(self):
        payload = self._formal_payload()
        payload['epoch'] = 4
        with self.assertRaisesRegex(M3EvaluationContractError, 'frozen formal best'):
            self._validate(payload)

    def test_wrong_best_val_loss_fails_with_strict_tolerance(self):
        payload = self._formal_payload()
        payload['best_val_loss'] += 1e-9
        with self.assertRaisesRegex(M3EvaluationContractError, 'best_val_loss'):
            self._validate(payload)

    def test_wrong_training_seed_fails(self):
        payload = self._formal_payload()
        payload['seed'] += 1
        payload['training_config']['seed'] = payload['seed']
        with self.assertRaisesRegex(M3EvaluationContractError, 'checkpoint seed'):
            self._validate(payload)

    def test_training_config_seed_must_match_payload_seed(self):
        payload = self._formal_payload()
        payload['training_config']['seed'] += 1
        with self.assertRaisesRegex(M3EvaluationContractError, 'does not match checkpoint payload'):
            self._validate(payload)

    def test_wrong_learning_rate_fails(self):
        payload = self._formal_payload()
        payload['training_config']['learning_rate'] = 0.0004
        with self.assertRaisesRegex(M3EvaluationContractError, 'learning_rate'):
            self._validate(payload)

    def test_wrong_temperature_fails(self):
        payload = self._formal_payload()
        payload['training_config']['temperature'] = 0.2
        with self.assertRaisesRegex(M3EvaluationContractError, 'temperature'):
            self._validate(payload)

    def test_wrong_batch_size_and_precision_fail(self):
        for field, value in (('batch_size', 2), ('precision', 'fp16')):
            with self.subTest(field=field):
                payload = self._formal_payload()
                payload['training_config'][field] = value
                with self.assertRaisesRegex(M3EvaluationContractError, field):
                    self._validate(payload)

    def test_each_fold_best_epoch_and_loss_match_frozen_contract(self):
        self.assertEqual(
            self.evaluation_protocol['formal_best_checkpoints'],
            EXPECTED_FORMAL_BESTS,
        )

    def test_metadata_mismatch_precedes_model_and_dataset_creation(self):
        payload = self._formal_payload()
        payload['epoch'] = 4
        checkpoint_contract = {'formal_protocol': True}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'best_val_loss.pt'
            checkpoint_path.touch()
            with mock.patch.object(
                evaluate_m3,
                '_load_checkpoint_for_execution',
                return_value=(payload, checkpoint_contract),
            ), mock.patch.object(
                evaluate_m3,
                '_build_models_from_checkpoint',
            ) as build_models, mock.patch.object(
                evaluate_m3,
                '_create_dataset_for_execution',
            ) as create_dataset:
                with self.assertRaisesRegex(M3EvaluationContractError, 'epoch'):
                    evaluate_m3.run_execute_test(
                        data_root=Path('must-not-be-read'),
                        checkpoint_path=checkpoint_path,
                        output_dir=Path(directory) / 'output',
                        device_name='cuda',
                        training_protocol=self.protocol,
                        evaluation_protocol=self.evaluation_protocol,
                        fold_id='Fold1',
                    )
            build_models.assert_not_called()
            create_dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
