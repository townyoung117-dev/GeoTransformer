import ast
import inspect
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_defect_training_adapter import (
    EXPECTED_FOLD_INSTANCE_COUNTS,
    run_static_protocol_audit,
)
from dataset import m2_ct_collate_fn, m2_point_collate_fn
from defect_training import (
    DEFECT_CONDITION_COUNT,
    DEFECT_IDS,
    EXPECTED_DEFECT_INSTANCE_COUNT,
    READY_PATIENT_COUNT,
    DefectTrainingContractError,
    build_defect_variants,
    build_formal_defect_split,
    build_split_provenance,
    create_formal_defect_dataset,
)
from training_protocol import FOLD_SUBJECTS, READY_SUBJECT_IDS, load_training_protocol


class FakeDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)


class DefectTrainingAdapterTest(unittest.TestCase):
    def setUp(self):
        self.protocol = load_training_protocol(PROTOCOL_PATH)
        self.variants = build_defect_variants(tuple(self.protocol['ready_subject_ids']))
        self.records = [
            {'subject_id': subject_id, 'defect_id': defect_id}
            for subject_id, defect_id in self.variants
        ]

    def test_frozen_pair_expansion_is_11_by_5_and_unique(self):
        self.assertEqual(READY_PATIENT_COUNT, 11)
        self.assertEqual(DEFECT_CONDITION_COUNT, 5)
        self.assertEqual(EXPECTED_DEFECT_INSTANCE_COUNT, 55)
        self.assertEqual(len(self.variants), 55)
        self.assertEqual(len(set(self.variants)), 55)
        self.assertEqual(
            Counter(subject_id for subject_id, _ in self.variants),
            Counter({subject_id: 5 for subject_id in READY_SUBJECT_IDS}),
        )
        self.assertEqual(
            Counter(defect_id for _, defect_id in self.variants),
            Counter({defect_id: 11 for defect_id in DEFECT_IDS}),
        )
        self.assertEqual(
            DEFECT_IDS,
            (
                'defect_001_left_maxilla_cheek_small',
                'defect_001_left_maxilla_cheek_medium',
                'defect_001_left_maxilla_cheek_large',
                'defect_001_right_maxilla_cheek_small',
                'defect_001_right_maxilla_cheek_medium',
            ),
        )
        self.assertNotIn('defect_001_right_maxilla_cheek_large', DEFECT_IDS)

    def test_all_folds_expand_to_frozen_regression_counts_without_leakage(self):
        dataset = FakeDataset(self.records)
        validation_counts = Counter()
        test_counts = Counter()
        for fold_id, expected_counts in EXPECTED_FOLD_INSTANCE_COUNTS.items():
            split = build_formal_defect_split(dataset, self.protocol, fold_id)
            self.assertEqual(
                (len(split.train_indices), len(split.val_indices), len(split.test_indices)),
                expected_counts,
            )
            patient_sets = tuple(
                map(set, (split.train_subject_ids, split.val_subject_ids, split.test_subject_ids))
            )
            self.assertFalse(patient_sets[0] & patient_sets[1])
            self.assertFalse(patient_sets[0] & patient_sets[2])
            self.assertFalse(patient_sets[1] & patient_sets[2])
            instance_sets = tuple(
                map(
                    set,
                    (split.train_instance_ids, split.val_instance_ids, split.test_instance_ids),
                )
            )
            self.assertFalse(instance_sets[0] & instance_sets[1])
            self.assertFalse(instance_sets[0] & instance_sets[2])
            self.assertFalse(instance_sets[1] & instance_sets[2])
            self.assertEqual(set().union(*instance_sets), set(self.variants))
            validation_counts.update(split.val_subject_ids)
            test_counts.update(split.test_subject_ids)
        expected_once = Counter({subject_id: 1 for subject_id in READY_SUBJECT_IDS})
        self.assertEqual(validation_counts, expected_once)
        self.assertEqual(test_counts, expected_once)

    def test_fold1_patient_order_expands_each_patient_over_frozen_defect_order(self):
        split = build_formal_defect_split(FakeDataset(self.records), self.protocol, 'Fold1')
        expected_train = tuple(
            (subject_id, defect_id)
            for subject_id in FOLD_SUBJECTS['Fold1']['train_subject_ids']
            for defect_id in DEFECT_IDS
        )
        self.assertEqual(split.train_instance_ids, expected_train)
        self.assertEqual(split.train_instance_ids[:5], tuple(('Pat11', value) for value in DEFECT_IDS))

    def test_missing_one_patient_defect_fails_closed(self):
        with self.assertRaisesRegex(DefectTrainingContractError, 'exactly once|exactly 55'):
            build_formal_defect_split(FakeDataset(self.records[:-1]), self.protocol, 'Fold1')

    def test_unknown_defect_id_fails_closed(self):
        records = list(self.records)
        records[0] = {
            'subject_id': records[0]['subject_id'],
            'defect_id': 'defect_001_right_maxilla_cheek_large',
        }
        with self.assertRaisesRegex(DefectTrainingContractError, 'unknown defect_id'):
            build_formal_defect_split(FakeDataset(records), self.protocol, 'Fold1')

    def test_duplicate_patient_defect_pair_fails_closed(self):
        records = list(self.records) + [dict(self.records[0])]
        with self.assertRaisesRegex(DefectTrainingContractError, 'duplicate instance'):
            build_formal_defect_split(FakeDataset(records), self.protocol, 'Fold1')

    def test_pat10_fails_closed(self):
        records = list(self.records)
        records[0] = {'subject_id': 'Pat10', 'defect_id': records[0]['defect_id']}
        with self.assertRaisesRegex(DefectTrainingContractError, 'unknown patient'):
            build_formal_defect_split(FakeDataset(records), self.protocol, 'Fold1')

    def test_dataset_factory_receives_all_explicit_pairs(self):
        captured = {}

        def factory(data_root, **kwargs):
            captured['data_root'] = data_root
            captured.update(kwargs)
            return FakeDataset(list(self.records))

        dataset = create_formal_defect_dataset('formal-root', self.protocol, dataset_factory=factory)
        self.assertEqual(len(dataset), 55)
        self.assertEqual(captured['data_root'], 'formal-root')
        self.assertEqual(captured['defect_variants'], self.variants)

    def test_split_provenance_keeps_subject_defect_and_fold_for_test_too(self):
        split = build_formal_defect_split(FakeDataset(self.records), self.protocol, 'Fold3')
        provenance = build_split_provenance(split)
        self.assertEqual(len(provenance['train_instances']), 35)
        self.assertEqual(len(provenance['val_instances']), 10)
        self.assertEqual(len(provenance['test_instances']), 10)
        for partition in ('train_instances', 'val_instances', 'test_instances'):
            for record in provenance[partition]:
                self.assertEqual(set(record), {'subject_id', 'defect_id', 'fold_id', 'partition'})
                self.assertEqual(record['fold_id'], 'Fold3')

    def test_static_cpu_audit_covers_all_folds(self):
        result = run_static_protocol_audit(self.protocol)
        self.assertEqual(result['ready_patient_count'], 11)
        self.assertEqual(result['defect_condition_count'], 5)
        self.assertEqual(result['expected_instance_count'], 55)
        self.assertTrue(result['pair_uniqueness_pass'])
        self.assertTrue(result['per_patient_pass'])
        self.assertTrue(result['per_condition_pass'])
        self.assertTrue(result['patient_leakage_pass'])
        self.assertTrue(result['instance_leakage_pass'])
        self.assertEqual(result['fold_counts'], EXPECTED_FOLD_INSTANCE_COUNTS)


class FrozenBaselineRegressionTest(unittest.TestCase):
    def setUp(self):
        self.protocol = load_training_protocol(PROTOCOL_PATH)

    def test_original_formal_split_still_rejects_duplicate_subject_records(self):
        source = (EXPERIMENT_DIR / 'training.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == 'build_formal_subject_split'
        )
        function_source = ast.get_source_segment(source, function)
        self.assertIn("subject_to_index = {}", function_source)
        self.assertIn("if subject_id in subject_to_index", function_source)
        self.assertIn("formal dataset contains duplicate subject record", function_source)
        self.assertNotIn("subject_to_indices.setdefault", function_source)

    def test_complete_train_entry_still_constructs_dataset_without_defect_variants(self):
        source = (EXPERIMENT_DIR / 'train_m3.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'create_dataset'
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 1)
        self.assertEqual(calls[0].keywords, [])
        self.assertEqual(ast.unparse(calls[0].args[0]), 'args.data_root')

    def test_m3_preprocessing_defaults_do_not_enable_m4_derived_fields(self):
        point_default = inspect.signature(m2_point_collate_fn).parameters[
            'enable_m4_defect_mapping'
        ].default
        ct_default = inspect.signature(m2_ct_collate_fn).parameters[
            'enable_m4_defect_mapping'
        ].default
        self.assertIs(point_default, False)
        self.assertIs(ct_default, False)


if __name__ == '__main__':
    unittest.main()
