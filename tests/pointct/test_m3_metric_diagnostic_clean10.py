import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_DIR = EXPERIMENT_DIR / 'protocols'
OLD_DIAGNOSTIC_PATH = PROTOCOL_DIR / 'm3_metric_diagnostic_v1.json'
CLEAN_DIAGNOSTIC_PATH = (
    PROTOCOL_DIR / 'm3_metric_diagnostic_clean10_v2.json'
)
CLEAN_TRAINING_PATH = PROTOCOL_DIR / 'm3_6b_5fold_clean10_v2.json'
CLEAN_EVALUATION_PATH = PROTOCOL_DIR / 'm3_defect_eval_clean10_v2.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_m3_defect_metric_diagnostic as diagnostic_cli
import audit_m3_metric_diagnostic as diagnostic_audit
from audit_m3_metric_diagnostic import run_audit
from defect_evaluation import (
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
)
from m3_metric_diagnostic import (
    CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
    CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION,
    CLEAN10_EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD,
    DIAGNOSTIC_PROTOCOL_HASH,
    EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD,
    M3MetricDiagnosticContractError,
    aggregate_diagnostic_cases,
    compute_diagnostic_protocol_hash,
    cross_check_legacy_cases,
    get_diagnostic_protocol_contract,
    load_diagnostic_protocol,
    load_jsonl,
    validate_diagnostic_protocol,
    validate_legacy_result_tree,
    validate_source_protocols,
)
from training_protocol import load_training_protocol


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
        encoding='utf-8',
    )


def _write_legacy_tree(root, cases_by_fold, root_rows=None):
    for fold_id, rows in cases_by_fold.items():
        _write_jsonl(Path(root) / fold_id / 'cases.jsonl', rows)
    if root_rows is not None:
        _write_jsonl(Path(root) / 'cases.jsonl', root_rows)


def _diagnostic_row(manifest_case):
    row = dict(manifest_case)
    row.update(
        {
            'checkpoint_path': 'best_val_loss.pt',
            'solver_success': True,
            'solver_status': 'success',
            'registration_recall_hit': True,
            'rre_deg': 1.0,
            'legacy_parameter_rte_mm': 2.0,
            'centroid_tre_mm': 2.5,
            'point_tre_mean_mm': 3.0,
            'point_tre_median_mm': 3.0,
            'point_tre_rmse_mm': 3.5,
            'point_tre_p95_mm': 4.0,
            'point_tre_max_mm': 4.5,
            'inlier_ratio': 0.75,
            'num_correspondences': 20,
            'num_inliers': 15,
            'confidence_mean': 0.8,
            'confidence_median': 0.8,
            'confidence_p05': 0.6,
            'confidence_p95': 0.9,
            'confidence_min': 0.5,
            'confidence_max': 0.95,
            'source_centroid_norm_mm': 800.0,
            'origin_rotation_lever_proxy_mm': 14.0,
        }
    )
    return row


def _legacy_row(manifest_case):
    row = dict(manifest_case)
    row.update(
        {
            'solver_success': True,
            'solver_status': 'success',
            'registration_recall_hit': True,
            'rre_deg': 1.0,
            'rte_mm': 2.0,
            'inlier_ratio': 0.75,
            'num_correspondences': 20,
            'num_inliers': 15,
        }
    )
    return row


class M3MetricDiagnosticClean10Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_training_protocol(CLEAN_TRAINING_PATH)
        cls.evaluation = load_defect_evaluation_protocol(CLEAN_EVALUATION_PATH)
        cls.diagnostic = load_diagnostic_protocol(CLEAN_DIAGNOSTIC_PATH)
        cls.manifests = {
            fold_id: build_defect_test_manifest(
                cls.training,
                cls.evaluation,
                fold_id,
            )
            for fold_id in cls.training['folds']
        }
        cls.manifest_cases = {
            fold_id: manifest['cases']
            for fold_id, manifest in cls.manifests.items()
        }
        cls.legacy_cases = {
            fold_id: [_legacy_row(case) for case in rows]
            for fold_id, rows in cls.manifest_cases.items()
        }
        cls.diagnostic_cases = {
            fold_id: [_diagnostic_row(case) for case in rows]
            for fold_id, rows in cls.manifest_cases.items()
        }

    def test_canonical_hash_and_frozen_version_are_exact(self):
        self.assertEqual(
            self.diagnostic['diagnostic_protocol_version'],
            CLEAN10_DIAGNOSTIC_PROTOCOL_VERSION,
        )
        self.assertEqual(
            self.diagnostic['diagnostic_protocol_hash'],
            CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
        )
        self.assertEqual(
            compute_diagnostic_protocol_hash(self.diagnostic),
            CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
        )

    def test_clean10_contract_is_10_by_5_50_and_750(self):
        contract = get_diagnostic_protocol_contract(self.diagnostic)
        self.assertEqual(
            (
                contract['expected_patient_count'],
                contract['expected_defect_condition_count'],
                contract['expected_defect_instance_count'],
                contract['expected_test_case_count'],
            ),
            (10, 5, 50, 750),
        )
        self.assertFalse(contract['pat6_forensic_required'])
        self.assertEqual(contract['expected_pat6_case_count'], 0)

    def test_each_fold_is_150_and_pat6_is_absent(self):
        counts = {
            fold_id: len(rows)
            for fold_id, rows in self.manifest_cases.items()
        }
        self.assertEqual(counts, CLEAN10_EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD)
        union = [
            case
            for fold_id in self.training['folds']
            for case in self.manifest_cases[fold_id]
        ]
        self.assertEqual(len(union), 750)
        self.assertEqual(len({case['case_key'] for case in union}), 750)
        self.assertFalse(any(case['subject_id'] == 'Pat6' for case in union))

    def test_source_training_and_evaluation_bindings_are_fail_closed(self):
        validate_source_protocols(
            self.diagnostic,
            self.training,
            self.evaluation,
        )
        old_training = load_training_protocol(
            diagnostic_cli.DEFAULT_TRAINING_PROTOCOL
        )
        with self.assertRaises(M3MetricDiagnosticContractError):
            validate_source_protocols(
                self.diagnostic,
                old_training,
                self.evaluation,
            )
        tampered_training = copy.deepcopy(self.training)
        tampered_training['ready_subject_ids'][0] = 'Pat6'
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'canonical validation',
        ):
            validate_source_protocols(
                self.diagnostic,
                tampered_training,
                self.evaluation,
            )

    def test_wrong_protocol_hash_and_rehashed_contract_tamper_fail(self):
        wrong_hash = copy.deepcopy(self.diagnostic)
        wrong_hash['diagnostic_protocol_hash'] = '0' * 64
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'diagnostic_protocol_hash mismatch',
        ):
            validate_diagnostic_protocol(wrong_hash)

        rehashed = copy.deepcopy(self.diagnostic)
        rehashed['expected_test_case_count'] = 751
        rehashed['diagnostic_protocol_hash'] = compute_diagnostic_protocol_hash(
            rehashed
        )
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'not the frozen hash',
        ):
            validate_diagnostic_protocol(rehashed)

    def test_wrong_rehashed_source_bindings_fail_frozen_diagnostic_hash(self):
        changed = copy.deepcopy(self.diagnostic)
        changed['source_training_protocol_hash'] = '0' * 64
        changed['diagnostic_protocol_hash'] = compute_diagnostic_protocol_hash(
            changed
        )
        with self.assertRaisesRegex(
            M3MetricDiagnosticContractError,
            'not the frozen hash',
        ):
            validate_diagnostic_protocol(changed)

    def test_old_v1_public_contract_aliases_are_unchanged(self):
        old = load_diagnostic_protocol(OLD_DIAGNOSTIC_PATH)
        self.assertEqual(old['diagnostic_protocol_hash'], DIAGNOSTIC_PROTOCOL_HASH)
        self.assertEqual(
            get_diagnostic_protocol_contract(old),
            {
                'diagnostic_protocol_version': 'm3_metric_diagnostic_v1',
                'diagnostic_protocol_hash': DIAGNOSTIC_PROTOCOL_HASH,
                'expected_patient_count': 11,
                'expected_defect_condition_count': 5,
                'expected_defect_instance_count': 55,
                'expected_test_case_count': 825,
                'pat6_forensic_required': True,
                'expected_pat6_case_count': 75,
                'legacy_complete_union_status': 'complete_union_verified',
            },
        )
        self.assertEqual(
            EXPECTED_LEGACY_CASE_COUNTS_BY_FOLD,
            {
                'Fold1': 225,
                'Fold2': 150,
                'Fold3': 150,
                'Fold4': 150,
                'Fold5': 150,
            },
        )

    def test_clean10_aggregate_has_no_pat6_forensic_requirement_or_member(self):
        summary = aggregate_diagnostic_cases(
            [self.diagnostic_cases['Fold1'][0]],
            expected_case_count=1,
            diagnostic_protocol=self.diagnostic,
        )
        self.assertNotIn('pat6_forensic', summary)

    def test_clean10_legacy_identity_cross_check_is_exact_750(self):
        legacy = [
            row
            for fold_id in self.training['folds']
            for row in self.legacy_cases[fold_id]
        ]
        diagnostic = [
            row
            for fold_id in self.training['folds']
            for row in self.diagnostic_cases[fold_id]
        ]
        result = cross_check_legacy_cases(
            legacy,
            diagnostic,
            expected_count=750,
        )
        self.assertTrue(result['cross_check_pass'])
        self.assertEqual(result['case_count'], 750)

    def test_duplicate_and_missing_legacy_cases_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = {
                **self.manifest_cases,
                'Fold3': self.manifest_cases['Fold3'][:-1],
            }
            _write_legacy_tree(directory, missing)
            with self.assertRaisesRegex(
                M3MetricDiagnosticContractError,
                'legacy Fold3 case count mismatch',
            ):
                validate_legacy_result_tree(
                    directory,
                    self.manifest_cases,
                    diagnostic_protocol=self.diagnostic,
                )

        duplicate = list(self.manifest_cases['Fold2'])
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(
                directory,
                {**self.manifest_cases, 'Fold2': duplicate},
            )
            with self.assertRaisesRegex(
                M3MetricDiagnosticContractError,
                'duplicate legacy Fold2 case identity',
            ):
                validate_legacy_result_tree(
                    directory,
                    self.manifest_cases,
                    diagnostic_protocol=self.diagnostic,
                )

    def test_complete_clean10_legacy_root_is_recognized_as_complete_union(self):
        union = [
            row
            for fold_id in self.training['folds']
            for row in self.manifest_cases[fold_id]
        ]
        with tempfile.TemporaryDirectory() as directory:
            _write_legacy_tree(directory, self.manifest_cases, root_rows=union)
            result = validate_legacy_result_tree(
                directory,
                self.manifest_cases,
                diagnostic_protocol=self.diagnostic,
            )
        self.assertEqual(result['per_fold_counts'], {
            fold_id: 150 for fold_id in self.training['folds']
        })
        self.assertEqual(result['per_fold_union_count'], 750)
        self.assertEqual(result['legacy_root_cases_status'], 'complete_union')

    def test_audit_only_cpu_path_never_executes_model_or_gpu(self):
        union = [
            row
            for fold_id in self.training['folds']
            for row in self.manifest_cases[fold_id]
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root = root / 'legacy'
            _write_legacy_tree(
                legacy_root,
                self.manifest_cases,
                root_rows=union,
            )
            args = argparse.Namespace(
                audit_only=True,
                execute_diagnostic=False,
                all_folds=True,
                fold_id=None,
                protocol_manifest=CLEAN_TRAINING_PATH,
                evaluation_protocol=CLEAN_EVALUATION_PATH,
                diagnostic_protocol=CLEAN_DIAGNOSTIC_PATH,
                legacy_results_root=legacy_root,
                output_root=root / 'output',
                checkpoint_root=None,
                json_log_root=None,
                data_root=Path('must-not-be-read'),
                device='cuda',
            )
            with mock.patch.object(
                diagnostic_cli,
                '_execute_fold',
                side_effect=AssertionError('model/GPU execution'),
            ):
                result = diagnostic_cli.run_evaluation(args)
        self.assertEqual(result['selected_expected_case_count'], 750)
        self.assertEqual(result['global_expected_case_count'], 750)
        self.assertEqual(result['pat6_case_count'], 0)
        self.assertEqual(
            result['per_fold_expected_case_counts'],
            {fold_id: 150 for fold_id in self.training['folds']},
        )
        self.assertFalse(result['model_loaded'])
        self.assertFalse(result['gpu_used'])

    def test_execute_all_folds_writes_750_unique_union_without_pat6_file(self):
        legacy_results = {
            'authoritative_union': [
                row
                for fold_id in self.training['folds']
                for row in self.legacy_cases[fold_id]
            ],
            'per_fold': self.legacy_cases,
        }

        def fake_execute_fold(**kwargs):
            fold_id = kwargs['manifest']['fold_id']
            return self.diagnostic_cases[fold_id], {
                'epoch': 20,
                'best_val_loss': 0.1,
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            diagnostic_cli,
            'resolve_fold_artifacts',
            return_value=(Path('best_val_loss.pt'), Path('train.jsonl')),
        ), mock.patch.object(
            diagnostic_cli,
            '_execute_fold',
            side_effect=fake_execute_fold,
        ):
            result = diagnostic_cli.run_execute_diagnostic(
                data_root=Path('must-not-be-read'),
                checkpoint_root=Path('checkpoint-root'),
                json_log_root=None,
                output_root=directory,
                legacy_results=legacy_results,
                device_name='cuda',
                fold_ids=tuple(self.training['folds']),
                training_protocol=self.training,
                evaluation_protocol=self.evaluation,
                diagnostic_protocol=self.diagnostic,
                manifests=self.manifests,
            )
            root = Path(directory)
            rows = load_jsonl(root / 'diagnostic_cases.jsonl')
            self.assertFalse((root / 'pat6_forensic.json').exists())
            for fold_id in self.training['folds']:
                self.assertEqual(
                    len(load_jsonl(root / fold_id / 'diagnostic_cases.jsonl')),
                    150,
                )
        self.assertEqual(len(rows), 750)
        identities = {
            (
                row['fold_id'],
                row['subject_id'],
                row['defect_id'],
                row['severity'],
                row['variant_id'],
                row['perturbation_seed'],
            )
            for row in rows
        }
        self.assertEqual(len(identities), 750)
        self.assertNotIn('pat6_forensic', result)

    def test_standalone_clean10_audit_reports_dynamic_cpu_contract(self):
        result = run_audit(
            CLEAN_TRAINING_PATH,
            CLEAN_EVALUATION_PATH,
            CLEAN_DIAGNOSTIC_PATH,
        )
        self.assertEqual(result['EXPECTED_PATIENT_COUNT'], 10)
        self.assertEqual(result['EXPECTED_DEFECT_INSTANCE_COUNT'], 50)
        self.assertEqual(result['EXPECTED_TEST_CASE_COUNT'], 750)
        self.assertEqual(result['PAT6_CASE_COUNT'], 0)
        self.assertEqual(
            result['PER_FOLD_CASE_COUNTS'],
            {fold_id: 150 for fold_id in self.training['folds']},
        )
        self.assertFalse(result['model_loaded'])
        self.assertFalse(result['gpu_used'])

    def test_standalone_audit_validates_legacy_union_and_five_checkpoints_cpu(self):
        union = [
            row
            for fold_id in self.training['folds']
            for row in self.manifest_cases[fold_id]
        ]
        metadata = {
            'epoch': 20,
            'best_val_loss': 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root = root / 'legacy'
            _write_legacy_tree(
                legacy_root,
                self.manifest_cases,
                root_rows=union,
            )
            with mock.patch.object(
                diagnostic_audit,
                'resolve_fold_artifacts',
                return_value=(Path('best_val_loss.pt'), Path('train.jsonl')),
            ), mock.patch.object(
                diagnostic_audit,
                'read_and_validate_defect_checkpoint',
                return_value=({}, metadata),
            ):
                result = run_audit(
                    CLEAN_TRAINING_PATH,
                    CLEAN_EVALUATION_PATH,
                    CLEAN_DIAGNOSTIC_PATH,
                    legacy_results_root=legacy_root,
                    checkpoint_root=root / 'checkpoints',
                )
        self.assertEqual(result['LEGACY_PER_FOLD_UNION_COUNT'], 750)
        self.assertEqual(result['LEGACY_ROOT_CASES_STATUS'], 'complete_union')
        self.assertEqual(set(result['CHECKPOINT_AUDIT']), set(self.training['folds']))
        self.assertTrue(
            all(
                fold['cpu_metadata_validation_pass']
                for fold in result['CHECKPOINT_AUDIT'].values()
            )
        )
        self.assertFalse(result['model_loaded'])
        self.assertFalse(result['gpu_used'])


if __name__ == '__main__':
    unittest.main()
