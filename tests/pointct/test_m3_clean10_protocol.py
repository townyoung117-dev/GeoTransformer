import argparse
import copy
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_DIR = EXPERIMENT_DIR / 'protocols'
OLD_TRAINING_PROTOCOL_PATH = PROTOCOL_DIR / 'm3_6b_5fold_v1.json'
CLEAN10_TRAINING_PROTOCOL_PATH = PROTOCOL_DIR / 'm3_6b_5fold_clean10_v2.json'
OLD_EVALUATION_PROTOCOL_PATH = PROTOCOL_DIR / 'm3_defect_eval_v1.json'
CLEAN10_EVALUATION_PROTOCOL_PATH = PROTOCOL_DIR / 'm3_defect_eval_clean10_v2.json'
QC_EXCLUSION_PATH = PROTOCOL_DIR / 'm3_clean10_qc_exclusion_v1.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_defect_training_adapter import (
    CLEAN10_EXPECTED_FOLD_INSTANCE_COUNTS,
    EXPECTED_FOLD_INSTANCE_COUNTS,
    run_static_protocol_audit,
)
import evaluate_m3_defect
from defect_evaluation import (
    CLEAN10_DEFECT_EVALUATION_PROTOCOL_HASH,
    EXPECTED_TEST_CASE_COUNT,
    DefectEvaluationContractError,
    build_defect_test_manifest,
    compute_defect_evaluation_protocol_hash,
    load_defect_evaluation_protocol,
    run_protocol_audit,
    validate_defect_evaluation_protocol,
    validate_protocol_pair,
)
from defect_training import (
    DEFECT_IDS,
    EXPECTED_DEFECT_INSTANCE_COUNT,
    READY_PATIENT_COUNT,
    build_defect_variants,
    build_formal_defect_split,
    get_defect_training_contract,
)
from perturbation import derive_perturbation_seed
from training_protocol import (
    CLEAN10_FOLD_SUBJECTS,
    CLEAN10_PROTOCOL_HASH,
    CLEAN10_PROTOCOL_VERSION,
    CLEAN10_READY_SUBJECT_IDS,
    CLEAN10_SUBJECT_GROUPS,
    M3TrainingProtocolError,
    PROTOCOL_HASH,
    compute_protocol_hash,
    load_training_protocol,
    resolve_fold,
    validate_training_protocol,
)


class FakeDataset:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)


class M3Clean10TrainingProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_training_protocol(OLD_TRAINING_PROTOCOL_PATH)
        cls.clean = load_training_protocol(CLEAN10_TRAINING_PROTOCOL_PATH)

    def test_old_v1_and_clean10_v2_both_pass_with_exact_hashes(self):
        self.assertEqual(
            self.old['protocol_hash'],
            'c0c635c1cb897ce33d15ba3ed58abdb12a8c5a8feb6a82fe51910146e8a99fac',
        )
        self.assertEqual(self.old['protocol_hash'], PROTOCOL_HASH)
        self.assertEqual(compute_protocol_hash(self.old), PROTOCOL_HASH)
        self.assertEqual(self.clean['protocol_version'], CLEAN10_PROTOCOL_VERSION)
        self.assertEqual(
            self.clean['protocol_hash'],
            '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c',
        )
        self.assertEqual(self.clean['protocol_hash'], CLEAN10_PROTOCOL_HASH)
        self.assertEqual(compute_protocol_hash(self.clean), CLEAN10_PROTOCOL_HASH)

    def test_legacy_count_constants_remain_frozen_v1_aliases(self):
        self.assertEqual(READY_PATIENT_COUNT, 11)
        self.assertEqual(EXPECTED_DEFECT_INSTANCE_COUNT, 55)
        self.assertEqual(EXPECTED_TEST_CASE_COUNT, 825)
        self.assertEqual(
            EXPECTED_FOLD_INSTANCE_COUNTS,
            {
                'Fold1': (30, 10, 15),
                'Fold2': (35, 10, 10),
                'Fold3': (35, 10, 10),
                'Fold4': (35, 10, 10),
                'Fold5': (30, 15, 10),
            },
        )

    def test_clean10_patients_groups_and_folds_are_exact_and_exclude_pat6(self):
        self.assertEqual(tuple(self.clean['ready_subject_ids']), CLEAN10_READY_SUBJECT_IDS)
        self.assertEqual(len(CLEAN10_READY_SUBJECT_IDS), 10)
        self.assertEqual(
            {key: tuple(value) for key, value in self.clean['groups'].items()},
            CLEAN10_SUBJECT_GROUPS,
        )
        for fold_id, expected in CLEAN10_FOLD_SUBJECTS.items():
            actual = resolve_fold(self.clean, fold_id)
            for field in ('train_subject_ids', 'val_subject_ids', 'test_subject_ids'):
                self.assertEqual(actual[field], expected[field])
        self.assertNotIn('Pat6', json.dumps(self.clean, sort_keys=True))

    def test_each_clean_patient_tests_once_validates_once_and_trains_three_times(self):
        counts = {
            split: Counter(
                subject_id
                for fold in self.clean['folds'].values()
                for subject_id in fold[split]
            )
            for split in ('train_subject_ids', 'val_subject_ids', 'test_subject_ids')
        }
        self.assertEqual(
            counts['train_subject_ids'],
            Counter({subject_id: 3 for subject_id in CLEAN10_READY_SUBJECT_IDS}),
        )
        for split in ('val_subject_ids', 'test_subject_ids'):
            self.assertEqual(
                counts[split],
                Counter({subject_id: 1 for subject_id in CLEAN10_READY_SUBJECT_IDS}),
            )

    def test_clean10_perturbation_and_seed_scheme_definitions_equal_old_v1(self):
        for field in (
            'seed_scheme_version',
            'perturbation_root_seed',
            'train_perturbation',
            'validation_perturbations',
            'test_perturbations',
        ):
            self.assertEqual(self.clean[field], self.old[field])

    def test_both_versions_remain_strictly_fail_closed_after_rehashing_tamper(self):
        for protocol in (self.old, self.clean):
            with self.subTest(protocol_version=protocol['protocol_version']):
                changed = copy.deepcopy(protocol)
                changed['train_perturbation']['max_rotation_deg'] = 19.0
                changed['protocol_hash'] = compute_protocol_hash(changed)
                with self.assertRaises(M3TrainingProtocolError):
                    validate_training_protocol(changed)

    def test_unknown_training_protocol_version_fails_closed(self):
        changed = copy.deepcopy(self.clean)
        changed['protocol_version'] = 'm3_unknown'
        changed['protocol_hash'] = compute_protocol_hash(changed)
        with self.assertRaisesRegex(M3TrainingProtocolError, 'unsupported protocol_version'):
            validate_training_protocol(changed)


class M3Clean10DefectAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_training_protocol(CLEAN10_TRAINING_PROTOCOL_PATH)
        cls.variants = build_defect_variants(tuple(cls.protocol['ready_subject_ids']))
        cls.dataset = FakeDataset(
            {'subject_id': subject_id, 'defect_id': defect_id}
            for subject_id, defect_id in cls.variants
        )

    def test_clean10_contract_is_10_by_5_and_50_unique_instances(self):
        contract = get_defect_training_contract(self.protocol)
        self.assertEqual(contract['ready_patient_count'], 10)
        self.assertEqual(contract['defect_condition_count'], 5)
        self.assertEqual(contract['expected_defect_instance_count'], 50)
        self.assertEqual(len(self.variants), 50)
        self.assertEqual(len(set(self.variants)), 50)
        self.assertEqual(
            Counter(subject_id for subject_id, _ in self.variants),
            Counter({subject_id: 5 for subject_id in CLEAN10_READY_SUBJECT_IDS}),
        )
        self.assertEqual(
            Counter(defect_id for _, defect_id in self.variants),
            Counter({defect_id: 10 for defect_id in DEFECT_IDS}),
        )
        self.assertNotIn('Pat6', {subject_id for subject_id, _ in self.variants})

    def test_every_fold_is_30_10_10_with_no_patient_or_instance_leakage(self):
        for fold_id in self.protocol['folds']:
            with self.subTest(fold_id=fold_id):
                split = build_formal_defect_split(
                    self.dataset,
                    self.protocol,
                    fold_id,
                )
                self.assertEqual(
                    (
                        len(split.train_instance_ids),
                        len(split.val_instance_ids),
                        len(split.test_instance_ids),
                    ),
                    (30, 10, 10),
                )
                patient_sets = tuple(
                    map(
                        set,
                        (
                            split.train_subject_ids,
                            split.val_subject_ids,
                            split.test_subject_ids,
                        ),
                    )
                )
                instance_sets = tuple(
                    map(
                        set,
                        (
                            split.train_instance_ids,
                            split.val_instance_ids,
                            split.test_instance_ids,
                        ),
                    )
                )
                for partition_sets in (patient_sets, instance_sets):
                    self.assertFalse(partition_sets[0] & partition_sets[1])
                    self.assertFalse(partition_sets[0] & partition_sets[2])
                    self.assertFalse(partition_sets[1] & partition_sets[2])
                self.assertEqual(set().union(*instance_sets), set(self.variants))

    def test_static_cpu_audit_selects_clean10_counts(self):
        result = run_static_protocol_audit(self.protocol)
        self.assertEqual(result['ready_patient_count'], 10)
        self.assertEqual(result['expected_instance_count'], 50)
        self.assertEqual(result['fold_counts'], CLEAN10_EXPECTED_FOLD_INSTANCE_COUNTS)
        self.assertTrue(result['pair_uniqueness_pass'])
        self.assertTrue(result['per_patient_pass'])
        self.assertTrue(result['per_condition_pass'])


class M3Clean10EvaluationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_training = load_training_protocol(OLD_TRAINING_PROTOCOL_PATH)
        cls.clean_training = load_training_protocol(CLEAN10_TRAINING_PROTOCOL_PATH)
        cls.old_evaluation = load_defect_evaluation_protocol(
            OLD_EVALUATION_PROTOCOL_PATH
        )
        cls.clean_evaluation = load_defect_evaluation_protocol(
            CLEAN10_EVALUATION_PROTOCOL_PATH
        )

    def test_clean10_evaluation_hash_is_real_canonical_sha256(self):
        self.assertEqual(
            self.clean_evaluation['evaluation_protocol_hash'],
            CLEAN10_DEFECT_EVALUATION_PROTOCOL_HASH,
        )
        self.assertEqual(
            compute_defect_evaluation_protocol_hash(self.clean_evaluation),
            CLEAN10_DEFECT_EVALUATION_PROTOCOL_HASH,
        )

    def test_clean10_evaluation_only_changes_versioned_training_binding(self):
        changed_fields = {
            'evaluation_protocol_version',
            'evaluation_protocol_hash',
            'required_training_protocol_version',
            'required_training_protocol_hash',
            'test_perturbation_source',
        }
        shared_fields = set(self.old_evaluation).difference(changed_fields)
        self.assertTrue(shared_fields)
        for field in shared_fields:
            self.assertEqual(
                self.clean_evaluation[field],
                self.old_evaluation[field],
                field,
            )
        self.assertFalse(
            self.clean_evaluation['required_enable_m4_defect_mapping']
        )

    def test_only_matching_training_evaluation_pairs_pass(self):
        validate_protocol_pair(self.old_training, self.old_evaluation)
        validate_protocol_pair(self.clean_training, self.clean_evaluation)
        for training, evaluation in (
            (self.old_training, self.clean_evaluation),
            (self.clean_training, self.old_evaluation),
        ):
            with self.subTest(
                training=training['protocol_version'],
                evaluation=evaluation['evaluation_protocol_version'],
            ):
                with self.assertRaises(DefectEvaluationContractError):
                    validate_protocol_pair(training, evaluation)

    def test_clean10_evaluation_rehashed_tamper_fails_closed(self):
        changed = copy.deepcopy(self.clean_evaluation)
        changed['registration_rre_threshold_deg'] = 6.0
        changed['evaluation_protocol_hash'] = compute_defect_evaluation_protocol_hash(
            changed
        )
        with self.assertRaises(DefectEvaluationContractError):
            validate_defect_evaluation_protocol(changed)

    def test_every_fold_has_150_cases_and_all_five_folds_have_750(self):
        case_keys = []
        for fold_id in self.clean_training['folds']:
            manifest = build_defect_test_manifest(
                self.clean_training,
                self.clean_evaluation,
                fold_id,
            )
            self.assertEqual(manifest['total_test_instances'], 10)
            self.assertEqual(manifest['total_cases'], 150)
            self.assertEqual(len(manifest['test_subject_ids']), 2)
            case_keys.extend(case['case_key'] for case in manifest['cases'])
        self.assertEqual(len(case_keys), 750)
        self.assertEqual(len(set(case_keys)), 750)
        audit = run_protocol_audit(
            self.clean_training,
            self.clean_evaluation,
        )
        self.assertEqual(audit['expected_test_instance_count'], 50)
        self.assertEqual(audit['expected_test_case_count'], 750)

    def test_five_defects_share_subject_seed_and_seed_uses_clean10_hash(self):
        manifest = build_defect_test_manifest(
            self.clean_training,
            self.clean_evaluation,
            'Fold1',
        )
        groups = defaultdict(list)
        for case in manifest['cases']:
            groups[
                (case['subject_id'], case['severity'], case['variant_index'])
            ].append(case)
        self.assertEqual(len(groups), 2 * 3 * 5)
        for identity, cases in groups.items():
            self.assertEqual(len(cases), 5)
            self.assertEqual({case['defect_id'] for case in cases}, set(DEFECT_IDS))
            self.assertEqual(len({case['perturbation_seed'] for case in cases}), 1)
            subject_id, severity, variant_index = identity
            expected = derive_perturbation_seed(
                scheme_version=self.clean_training['seed_scheme_version'],
                protocol_hash=CLEAN10_PROTOCOL_HASH,
                fold_id='Fold1',
                root_seed=self.clean_training['perturbation_root_seed'],
                purpose='test',
                epoch=None,
                subject_id=subject_id,
                severity=severity,
                variant_id=variant_index,
            )
            old_hash_seed = derive_perturbation_seed(
                scheme_version=self.clean_training['seed_scheme_version'],
                protocol_hash=PROTOCOL_HASH,
                fold_id='Fold1',
                root_seed=self.clean_training['perturbation_root_seed'],
                purpose='test',
                epoch=None,
                subject_id=subject_id,
                severity=severity,
                variant_id=variant_index,
            )
            self.assertEqual(cases[0]['perturbation_seed'], expected)
            self.assertNotEqual(cases[0]['perturbation_seed'], old_hash_seed)

    def test_clean10_manifest_only_cli_writes_five_folds_without_data_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                manifest_only=True,
                execute_test=False,
                all_folds=True,
                fold_id=None,
                protocol_manifest=CLEAN10_TRAINING_PROTOCOL_PATH,
                evaluation_protocol=CLEAN10_EVALUATION_PROTOCOL_PATH,
                output_dir=Path(directory),
                data_root=Path('must-not-be-read'),
                checkpoint_root=None,
                json_log_root=None,
                device='cuda',
            )
            with mock.patch.object(
                evaluate_m3_defect,
                '_create_defect_dataset_for_execution',
                side_effect=AssertionError('dataset read'),
            ):
                result = evaluate_m3_defect.run_evaluation(args)
            self.assertEqual(result['mode'], 'manifest-only')
            self.assertEqual(result['selected_folds'], list(self.clean_training['folds']))
            self.assertEqual(result['protocol_audit']['expected_test_case_count'], 750)
            for fold_id in self.clean_training['folds']:
                self.assertEqual(result['folds'][fold_id]['test_cases'], 150)
                self.assertTrue(
                    (Path(directory) / fold_id / 'test_manifest.json').is_file()
                )


class M3Clean10QCExclusionTest(unittest.TestCase):
    def test_qc_exclusion_provenance_is_exact_and_does_not_invent_anomaly(self):
        provenance = json.loads(QC_EXCLUSION_PATH.read_text(encoding='utf-8'))
        self.assertEqual(
            provenance,
            {
                'qc_exclusion_version': 'm3_clean10_qc_exclusion_v1',
                'applies_to_training_protocol_version': CLEAN10_PROTOCOL_VERSION,
                'applies_to_training_protocol_hash': CLEAN10_PROTOCOL_HASH,
                'excluded_subject_ids': ['Pat6'],
                'exclusion_stage': 'before_formal_clean10_training',
                'exclusion_basis': (
                    'manual QC confirmed a source-data quality problem'
                ),
                'specific_anomaly_type': None,
            },
        )


if __name__ == '__main__':
    unittest.main()
