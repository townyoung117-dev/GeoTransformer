"""Synthetic/mock CPU contracts only. Never read real samples or formal outputs."""

import ast
import builtins
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[2] / 'experiments/geotransformer.pointct.baseline_v1'
sys.path.insert(0, str(EXPERIMENT))
import m4_osseous_formal_protocol as c
import bind_m4_osseous_formal_checkpoints as b
import evaluate_m4_osseous_formal_test as e
import aggregate_m4_osseous_formal_test as a
import run_m4_osseous_formal_test as runner


def fixture_checkpoint(sources, fold='Fold1'):
    config = b.expected_config(sources, fold)
    clean = sources['clean10']
    payload = {'checkpoint_version': 2, 'epoch': 0, 'global_step': 10, 'formal_protocol': True,
               'protocol_version': clean['protocol_version'], 'protocol_hash': clean['protocol_hash'],
               'fold_id': fold, 'seed': config['seed'], 'perturbation_root_seed': clean['perturbation_root_seed'],
               'perturbation_seed_scheme_version': clean['seed_scheme_version'],
               'train_subject_ids': clean['folds'][fold]['train_subject_ids'],
               'val_subject_ids': clean['folds'][fold]['val_subject_ids'], 'training_config': config,
               'best_val_loss': 1.0, 'point_encoder_state_dict': {'fixture': 1},
               'ct_encoder_state_dict': {'fixture': 1}, 'matcher_state_dict': {'fixture': 1},
               'optimizer_state_dict': {'param_groups': [{'lr': config['learning_rate'], 'weight_decay': config['weight_decay']}]}}
    logs = []
    for epoch in range(sources['selection_v1']['training_settings']['epochs']):
        logs.append({**config, 'epoch': epoch, 'formal_protocol': True,
                     'protocol_version': clean['protocol_version'], 'protocol_hash': clean['protocol_hash'],
                     'fold_id': fold, 'num_val_cases': 30, 'num_val_subjects': 2, 'num_val_perturbations': 30,
                     'val_mean_loss': 1.0, 'best_val_loss': 1.0, 'best_checkpoint_updated': epoch == 0})
    return payload, logs


def fixture_rows(sources, fold=None):
    return [{**{key: case[key] for key in (*c.IDENTITY, 'case_key')},
             **{key: 1.0 for key in c.ERROR_METRICS},
             'artifact_scope': 'formal_test', 'protocol_version': c.VERSION, 'protocol_sha256': c.PROTOCOL_SHA,
             'checkpoint_binding_sha256': 'a'*64, 'checkpoint_sha256': 'b'*64, 'lambda_oss': .5,
             'solver_success': True, 'solver_status': 'success', 'registration_recall_hit': True,
             'num_correspondences': 3, 'num_inliers': 3, 'inlier_ratio': 1.0, 'inference_runtime_ms': 1.0}
            for case in c.manifest(sources, fold)]


def mark_failure(row):
    row.update({key: None for key in c.ERROR_METRICS})
    row.update(solver_success=False, solver_status='insufficient_correspondences',
               registration_recall_hit=False, num_correspondences=0, num_inliers=0, inlier_ratio=0.0)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.sources = c.load_protocol()

    def test_v2_amendment_sha_binding(self):
        self.assertEqual(self.protocol['source_protocols']['amendment_v2']['sha256'],
                         'fe59f307eca6355d835fad5a9a5d18f8b7f24b940747700149b7a5668413e13c')

    def test_selected_lambda_exact_half(self):
        self.assertEqual(self.protocol['model']['lambda_oss'], .5)

    def reject_model(self, key, value):
        protocol = copy.deepcopy(self.protocol)
        protocol['model'][key] = value
        with self.assertRaises(c.ContractError): c.validate_protocol(protocol, self.sources)
        # Prove semantic enforcement even if a caller rehashes a changed config.
        protocol['protocol_sha256'] = c.digest(protocol, 'protocol_sha256')
        with mock.patch.object(c, 'PROTOCOL_SHA', protocol['protocol_sha256']):
            with self.assertRaises(c.ContractError): c.validate_protocol(protocol, self.sources)

    def test_wrong_lambda_rejected(self): self.reject_model('lambda_oss', 1.)
    def test_hard_disabled_rejected(self): self.reject_model('hard_enabled', False)
    def test_soft_disabled_rejected(self): self.reject_model('soft_enabled', False)
    def test_wrong_sigma_rejected(self): self.reject_model('soft_sigma_mm', 30.)
    def test_wrong_soft_strength_rejected(self): self.reject_model('soft_strength', 1.)
    def test_osseous_disabled_rejected(self): self.reject_model('osseous_enabled', False)

    def test_wrong_center_tau_radius_rejected(self):
        for field in ('osseous_center_hu', 'osseous_tau_hu', 'osseous_radius_mm'):
            with self.subTest(field=field): self.reject_model(field, 0.)

    def test_exact_five_folds_and_test_mapping(self):
        self.assertEqual(self.protocol['folds'], self.sources['clean10']['folds'])
        self.assertEqual(list(self.protocol['folds']), ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'])
        self.assertEqual([v['test_subject_ids'] for v in self.protocol['folds'].values()],
                         [['Pat12', 'Pat5'], ['Pat7', 'Pat4'], ['Pat11', 'Pat3'], ['Pat8', 'Pat9'], ['Pat1', 'Pat2']])

    def test_pat6_excluded(self):
        self.assertEqual(self.protocol['excluded_patient_ids'], ['Pat6'])
        self.assertNotIn('Pat6', {r['subject_id'] for r in c.manifest(self.sources)})

    def test_defect_conditions_inherited(self):
        self.assertEqual(self.protocol['defect_conditions'], self.sources['evaluation']['required_defect_ids'])

    def test_exact_perturbation_seeds_match_original_m3(self):
        from perturbation import derive_perturbation_seed
        clean = self.sources['clean10']
        for row in c.manifest(self.sources):
            self.assertEqual(row['perturbation_seed'], derive_perturbation_seed(
                scheme_version=clean['seed_scheme_version'], protocol_hash=clean['protocol_hash'],
                root_seed=clean['perturbation_root_seed'], purpose='test', epoch=None,
                fold_id=row['fold_id'], subject_id=row['subject_id'], severity=row['severity'], variant_id=row['variant_id']))

    def test_full_manifest_matches_existing_m3_generator(self):
        from defect_evaluation import build_defect_test_manifest
        for fold in self.sources['clean10']['folds']:
            original = build_defect_test_manifest(self.sources['clean10'], self.sources['evaluation'], fold)['cases']
            current = c.manifest(self.sources, fold)
            self.assertEqual(current, [{key: row[key] for key in current[0]} for row in original])

    def test_grid_count_inherited(self):
        self.assertEqual(len(c.manifest(self.sources)), self.sources['diagnostic']['expected_test_case_count'])
        self.assertEqual(len(c.manifest(self.sources)), 750)
        self.assertTrue(all(len(c.manifest(self.sources, f)) == 150 for f in self.protocol['folds']))

    def test_metric_threshold_inheritance(self):
        self.assertEqual(self.protocol['source_m3_evaluation_definition'], self.sources['evaluation'])
        evaluation = self.sources['evaluation']
        self.assertEqual((evaluation['registration_rre_threshold_deg'], evaluation['registration_rte_threshold_mm'],
                          evaluation['correspondence_inlier_threshold_mm']), (5., 10., 15.))

    def test_failure_policy_inherited(self):
        self.assertEqual(self.protocol['solver_failure_policy']['source_formal'], 'null_rre_rte')
        self.assertEqual(self.protocol['solver_failure_policy']['source_tre'], 'null_diagnostic_metrics')
        rows = fixture_rows(self.sources, 'Fold1')
        mark_failure(rows[0])
        c.validate_rows(rows, self.sources, 'Fold1')
        summary = a.summarize(rows)['overall_descriptive']
        self.assertEqual(summary['solver_failure_count'], 1)
        self.assertEqual(summary['case_count'], 150)
        self.assertEqual(summary['metrics']['rre_deg']['count'], 149)

    def test_deterministic_manifest(self):
        self.assertEqual(c.canonical(c.manifest(self.sources)), c.canonical(c.manifest(self.sources)))
        self.assertEqual(c.digest(c.manifest(self.sources)), self.protocol['manifest_sha256'])

    def test_wrong_source_sha_rejected(self):
        original = c.read_file
        def corrupt(relative):
            raw = original(relative)
            if relative.endswith('m3_6b_5fold_clean10_v2.json'):
                value = c.parse(raw)
                value['protocol_hash'] = '0'*64
                return c.canonical(value).encode()
            return raw
        with mock.patch.object(c, 'read_file', corrupt):
            with self.assertRaises(c.ContractError): c.load_sources()

    def test_wrong_formal_sha_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['protocol_sha256'] = '0'*64
        with self.assertRaises(c.ContractError): c.validate_protocol(protocol, self.sources)


class CaseAndComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.protocol, cls.sources = c.load_protocol()

    def setUp(self): self.rows = fixture_rows(self.sources)
    def validate(self): return c.validate_rows(self.rows, self.sources)

    def test_duplicate_identity_rejected(self):
        self.rows[1] = copy.deepcopy(self.rows[0])
        with self.assertRaises(c.ContractError): self.validate()

    def test_missing_identity_rejected(self):
        self.rows.pop()
        with self.assertRaises(c.ContractError): self.validate()

    def test_unexpected_patient_rejected(self):
        self.rows[0]['subject_id'] = 'Pat6'
        with self.assertRaises(c.ContractError): self.validate()

    def test_nonfinite_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                self.rows[0]['rre_deg'] = value
                with self.assertRaises(c.ContractError): self.validate()

    def test_json_nonfinite_and_duplicate_fields_rejected(self):
        for raw in ('{"x":1e999}', '{"x":NaN}', '{"x":1,"x":2}'):
            with self.assertRaises(c.ContractError): c.parse(raw)

    def test_unexpected_output_field_rejected(self):
        self.rows[0]['retuned_lambda'] = .25
        with self.assertRaises(c.ContractError): self.validate()

    def test_recall_threshold_boundary_and_failure_nulls(self):
        self.rows[0].update(rre_deg=5., rte_mm=10.)
        self.validate()
        self.rows[0]['rre_deg'] = 5.0001
        with self.assertRaises(c.ContractError): self.validate()
        mark_failure(self.rows[0])
        self.rows[0]['centroid_tre_mm'] = 0.
        with self.assertRaises(c.ContractError): self.validate()

    def test_pair_identity_mismatch_rejected(self):
        other = copy.deepcopy(self.rows)
        other[0]['perturbation_seed'] += 1
        with self.assertRaises(c.ContractError): a.paired_comparison(self.rows, other, self.sources)

    def test_pair_duplicate_rejected(self):
        other = copy.deepcopy(self.rows)
        other[1] = other[0]
        with self.assertRaises(c.ContractError): a.paired_comparison(self.rows, other, self.sources)

    def test_patient_level_preserved_with_zero_success(self):
        for row in self.rows:
            if row['subject_id'] == 'Pat12': mark_failure(row)
        self.validate()
        result = a.summarize(self.rows)
        self.assertEqual(len(result['patient_level']), 10)
        p = result['patient_level']['Pat12']
        self.assertEqual(p['solver_failure_count'], 75)
        self.assertEqual(p['solver_success_rate'], 0.)
        self.assertIsNone(p['metrics']['centroid_tre_mm']['median'])

    def test_paired_missingness_and_delta_direction(self):
        baseline = copy.deepcopy(self.rows)
        mark_failure(baseline[0])
        mark_failure(self.rows[1])
        mark_failure(baseline[2]); mark_failure(self.rows[2])
        self.rows[3]['centroid_tre_mm'] = 3.
        result = a.paired_comparison(baseline, self.rows, self.sources)
        overall = result['overall_paired_descriptive']
        self.assertEqual(overall['both_solver_success_count'], 747)
        self.assertEqual(overall['m3_only_success_count'], 1)
        self.assertEqual(overall['m4_only_success_count'], 1)
        self.assertEqual(overall['neither_success_count'], 1)
        self.assertEqual(overall['paired_error_delta']['centroid_tre_mm']['mean'], 2/747)

    def test_m3_legacy_diagnostic_adapter(self):
        clean, ev, diag = (self.sources[key] for key in ('clean10', 'evaluation', 'diagnostic'))
        legacy, diagnostics = [], []
        for row in self.rows:
            legacy.append({**row, 'protocol_version': clean['protocol_version'], 'protocol_hash': clean['protocol_hash'],
                           'evaluation_protocol_version': ev['evaluation_protocol_version'], 'evaluation_protocol_hash': ev['evaluation_protocol_hash']})
            diagnostics.append({**row, 'diagnostic_protocol_version': diag['diagnostic_protocol_version'],
                                'diagnostic_protocol_hash': diag['diagnostic_protocol_hash'],
                                'source_training_protocol_version': clean['protocol_version'], 'source_training_protocol_hash': clean['protocol_hash'],
                                'source_defect_evaluation_protocol_version': ev['evaluation_protocol_version'],
                                'source_defect_evaluation_protocol_hash': ev['evaluation_protocol_hash'], 'legacy_parameter_rte_mm': row['rte_mm']})
        normalized = a.normalize_m3(legacy, diagnostics, self.sources)
        self.assertEqual(len(normalized), 750)
        diagnostics[0]['legacy_parameter_rte_mm'] = 20.
        with self.assertRaises(c.ContractError): a.normalize_m3(legacy, diagnostics, self.sources)

    def test_synthetic_inference_uses_m3_metrics(self):
        import numpy as np
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        inference = {'point_physical': points, 'ct_physical': points,
                     'filter_output': {'point_indices': np.arange(3), 'ct_indices': np.arange(3)},
                     'registration_output': {'success': True, 'rotation': np.eye(3), 'translation': np.zeros(3)},
                     'inference_runtime_ms': 1.0}
        metrics = e.metrics_from_inference(inference, np.eye(4), self.sources['evaluation'])
        self.assertTrue(metrics['registration_recall_hit'])
        self.assertEqual(metrics['centroid_tre_mm'], 0.)
        self.assertEqual(metrics['point_tre_mean_mm'], 0.)
        inference['registration_output'] = {'success': False, 'failure_reason': 'insufficient_correspondences'}
        metrics = e.metrics_from_inference(inference, np.eye(4), self.sources['evaluation'])
        self.assertTrue(all(metrics[key] is None for key in c.ERROR_METRICS))

    def test_invalid_success_transform_follows_m3_failure_policy(self):
        import numpy as np
        inference = {'point_physical': np.eye(3), 'ct_physical': np.eye(3),
                     'filter_output': {'point_indices': np.arange(3), 'ct_indices': np.arange(3)},
                     'registration_output': {'success': True, 'rotation': np.zeros((3, 3)), 'translation': np.zeros(3)},
                     'inference_runtime_ms': 1.}
        result = e.metrics_from_inference(inference, np.eye(4), self.sources['evaluation'])
        self.assertEqual(result['solver_status'], 'invalid_registration_output')
        self.assertTrue(all(result[key] is None for key in c.ERROR_METRICS))


class CheckpointAndFirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.protocol, cls.sources = c.load_protocol()

    def setUp(self): self.payload, self.logs = fixture_checkpoint(self.sources)

    def audit(self, path=None, fold='Fold1'):
        return b.audit_metadata(self.payload, self.logs, self.sources, fold,
                                path or c.checkpoint_paths(self.sources)['Fold1'], 'b'*64, 'c'*64)

    def test_checkpoint_config_mismatch_rejected(self):
        self.payload['training_config']['m4_osseous_strength'] = 1.
        with self.assertRaises(c.ContractError): self.audit()

    def test_checkpoint_wrong_fold_rejected(self):
        with self.assertRaises(c.ContractError): self.audit(fold='Fold2')

    def test_checkpoint_basename_enforced(self):
        with self.assertRaises(c.ContractError): self.audit(path=c.checkpoint_paths(self.sources)['Fold1'].replace('best_val_loss.pt', 'last.pt'))

    def test_validation_earliest_tied_epoch_enforced(self):
        self.assertEqual(self.audit()['best_epoch'], 0)
        self.payload['epoch'] = 1
        with self.assertRaises(c.ContractError): self.audit()

    def test_incomplete_training_budget_rejected(self):
        self.logs.pop()
        with self.assertRaises(c.ContractError): self.audit()

    def test_checkpoint_sha_mismatch_before_deserialization(self):
        binding = {'folds': {'Fold1': {'checkpoint_sha256': '0'*64}}}
        with mock.patch.object(c, 'read_file', return_value=b'synthetic'), mock.patch.object(b, 'decode_checkpoint') as decode:
            with self.assertRaisesRegex(c.ContractError, 'checkpoint SHA'): b.audit_all(self.sources, binding)
            decode.assert_not_called()

    def test_formal_output_no_clobber(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(c, 'ROOT', Path(folder)):
            c.write_json_once('synthetic/output.json', {'x': 1})
            with self.assertRaises(c.ContractError): c.write_json_once('synthetic/output.json', {'x': 2})
            self.assertEqual(json.loads((Path(folder)/'synthetic/output.json').read_bytes()), {'x': 1})

    def test_concurrent_output_no_clobber(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(c, 'ROOT', Path(folder)):
            original = os.link
            def race(source, destination):
                Path(destination).write_bytes(b'other writer')
                original(source, destination)
            with mock.patch.object(c.os, 'link', race):
                with self.assertRaises(c.ContractError): c.write_json_once('synthetic/output.json', {'x': 1})
            self.assertEqual((Path(folder)/'synthetic/output.json').read_bytes(), b'other writer')

    def forbid_runtime_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'dataset', 'torch', 'train_m3', 'train_m3_defect', 'evaluate_m4_osseous_prior'}:
                raise AssertionError('runtime import forbidden in CPU audit: ' + name)
            return original(name, *args, **kwargs)
        return mock.patch('builtins.__import__', guarded)

    def test_contract_audit_loads_no_dataset_or_results(self):
        original = c.read_file
        def read(relative):
            self.assertFalse(relative.startswith('checkpoints/'))
            self.assertFalse(relative.startswith('local_data/'))
            return original(relative)
        with self.forbid_runtime_imports(), mock.patch.object(c, 'read_file', read):
            result = c.contract_audit()
        self.assertFalse(result['FORMAL_TEST_ACCESSED'])
        self.assertFalse(result['GPU_USED'])

    def test_checkpoint_binding_loads_no_dataset(self):
        files = {}
        for fold, relative in c.checkpoint_paths(self.sources).items():
            payload, logs = fixture_checkpoint(self.sources, fold)
            files[relative] = c.canonical(payload).encode()
            files[relative.rsplit('/checkpoints/', 1)[0] + '/train.jsonl'] = ''.join(c.canonical(row)+'\n' for row in logs).encode()
        with self.forbid_runtime_imports(), mock.patch.object(c, 'load_protocol', return_value=(self.protocol, self.sources)), \
                mock.patch.object(c, 'read_file', side_effect=lambda relative: files[relative]), \
                mock.patch.object(b, 'decode_checkpoint', side_effect=c.parse), \
                mock.patch.object(c, 'write_json_once') as write:
            binding = b.bind()
        self.assertEqual(len(binding['folds']), 5)
        self.assertFalse(binding['formal_test_accessed'])
        write.assert_called_once()

    def test_uncommitted_binding_rejected_before_execution(self):
        with mock.patch.object(c, 'load_protocol', return_value=(self.protocol, self.sources)), \
                mock.patch.object(c, 'require_committed', side_effect=c.ContractError('uncommitted binding')), \
                mock.patch.object(e, 'execute_fold') as execute, mock.patch.object(b, 'audit_all') as audit:
            with self.assertRaises(c.ContractError): e.evaluate('Fold1', 'NEVER_READ', 'cpu')
            execute.assert_not_called(); audit.assert_not_called()

    def test_all_five_checkpoints_audited_before_dataset(self):
        with mock.patch.object(c, 'load_protocol', return_value=(self.protocol, self.sources)), \
                mock.patch.object(b, 'read_binding', return_value={}), \
                mock.patch.object(b, 'audit_all', side_effect=c.ContractError('Fold5 mismatch')), \
                mock.patch.object(e, 'execute_fold') as execute:
            with self.assertRaises(c.ContractError): e.evaluate('Fold1', 'NEVER_READ', 'cpu')
            execute.assert_not_called()

    def test_mock_formal_execution_contract_no_real_sample(self):
        binding = {'binding_sha256': 'a'*64, 'folds': {'Fold1': {'checkpoint_sha256': 'b'*64}}}
        events = []
        def audit(*args, **kwargs): events.append('all_checkpoint_audit'); return {}, {}
        def execute(*args, **kwargs): events.append('synthetic_execution'); return fixture_rows(self.sources, 'Fold1')
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(c, 'ROOT', Path(folder)), \
                mock.patch.object(c, 'load_protocol', return_value=(self.protocol, self.sources)), \
                mock.patch.object(b, 'read_binding', return_value=binding), mock.patch.object(b, 'audit_all', audit), \
                mock.patch.object(e, 'execute_fold', execute):
            result = e.evaluate('Fold1', 'SYNTHETIC_ONLY', 'cpu')
            self.assertEqual(result['case_count'], 150)
            self.assertEqual(events, ['all_checkpoint_audit', 'synthetic_execution'])
            with self.assertRaises(c.ContractError): e.evaluate('Fold1', 'SYNTHETIC_ONLY', 'cpu')

    def test_runtime_dataset_selector_only_current_fold(self):
        requested, accessed = [], []
        class FakeDataset:
            def __init__(self, variants): self.variants = variants
            def __len__(self): return len(self.variants)
            def get_record(self, index):
                patient, defect = self.variants[index]
                return {'subject_id': patient, 'defect_id': defect}
            def __getitem__(self, index):
                accessed.append(self.variants[index])
                return self.get_record(index)
        def create(root, *, defect_variants):
            self.assertEqual(root, 'SYNTHETIC_ONLY')
            requested.extend(defect_variants)
            return FakeDataset(defect_variants)
        fake_torch = types.SimpleNamespace(device=lambda name: types.SimpleNamespace(type='cpu'),
                                          no_grad=contextlib.nullcontext)
        modules = {'torch': fake_torch, 'dataset': types.SimpleNamespace(create_dataset=create),
                   'evaluate_m3': types.SimpleNamespace(_build_models_from_checkpoint=lambda *args: ({}, {}, {}))}
        sample_metrics = {key: value for key, value in fixture_rows(self.sources, 'Fold1')[0].items()
                          if key not in (*c.IDENTITY, 'case_key')}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(e, 'evaluate_case', return_value=sample_metrics):
            rows = e.execute_fold(self.sources, 'Fold1', {}, {}, 'cpu', 'SYNTHETIC_ONLY')
        expected = [(p, d) for p in self.sources['clean10']['folds']['Fold1']['test_subject_ids']
                    for d in self.sources['evaluation']['required_defect_ids']]
        self.assertEqual(requested, expected)
        self.assertEqual(set(accessed), set(expected))
        self.assertEqual(len(rows), 150)

    def test_git_binding_requires_head_bytes(self):
        with mock.patch.object(c, 'read_file', return_value=b'new binding'), \
                mock.patch.object(c.subprocess, 'check_output', return_value=b'old binding'):
            with self.assertRaises(c.ContractError): c.require_committed(c.BINDING_PATH)
        with mock.patch.object(c, 'read_file', return_value=b'same\r\n'), \
                mock.patch.object(c.subprocess, 'check_output', return_value=b'same\n'):
            c.require_committed(c.BINDING_PATH)

    def test_no_training_entrypoint_reachable(self):
        for module in (c, b, e, a, runner):
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, ('train_m3', 'train_m3_defect'))
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
                    self.assertNotIn(name, ('train', 'run_training_step', 'run_defect_training', 'save_checkpoint'))

    def test_evaluator_cannot_write_protocol(self):
        tree = ast.parse(Path(e.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', '')
                self.assertNotIn(name, ('write_text', 'write_bytes', 'open', 'replace', 'unlink'))
        before = c.sha(c.read_file(c.PROTOCOL_PATH))
        c.contract_audit()
        self.assertEqual(before, c.sha(c.read_file(c.PROTOCOL_PATH)))

    def test_runner_audit_has_no_execution_import(self):
        with self.forbid_runtime_imports(), mock.patch.object(e, 'evaluate') as evaluate, contextlib.redirect_stdout(io.StringIO()):
            runner.main(['--contract-audit'])
            evaluate.assert_not_called()

    def test_cli_requires_explicit_mode(self):
        for entry, args in [(e.main, []), (runner.main, []), (c.main, ['--evaluate']),
                            (b.main, ['--evaluate']), (runner.main, ['--contract-audit', '--data-root', 'forbidden'])]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): entry(args)


if __name__ == '__main__':
    unittest.main()
