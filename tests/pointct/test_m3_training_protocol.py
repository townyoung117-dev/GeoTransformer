import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_PATH = EXPERIMENT_DIR / 'protocols' / 'm3_6b_5fold_v1.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from perturbation import (
    PROVENANCE_FIELD,
    augment_point_sample,
    derive_perturbation_seed,
    sample_rigid_perturbation,
)
from training_protocol import (
    FOLD_SUBJECTS,
    M3TrainingProtocolError,
    PERTURBATION_ROOT_SEED,
    PROTOCOL_VERSION,
    READY_SUBJECT_IDS,
    SEED_SCHEME_VERSION,
    compute_protocol_hash,
    load_training_protocol,
    resolve_fold,
    test_perturbation_specs,
    train_perturbation_spec,
    validate_dataset_ready_subjects,
    validate_training_protocol,
    validation_perturbation_specs,
)


class M3TrainingProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = load_training_protocol(PROTOCOL_PATH)

    @staticmethod
    def _manual_hash(protocol):
        value = dict(protocol)
        value.pop('protocol_hash', None)
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _seed(self, *, purpose, subject_id, severity, variant_id, epoch):
        return derive_perturbation_seed(
            scheme_version=self.protocol['seed_scheme_version'],
            protocol_hash=self.protocol['protocol_hash'],
            fold_id='Fold1',
            root_seed=self.protocol['perturbation_root_seed'],
            purpose=purpose,
            epoch=epoch,
            subject_id=subject_id,
            severity=severity,
            variant_id=variant_id,
        )

    def test_formal_manifest_loads_and_has_expected_versions(self):
        self.assertEqual(self.protocol['protocol_version'], PROTOCOL_VERSION)
        self.assertEqual(self.protocol['seed_scheme_version'], SEED_SCHEME_VERSION)
        self.assertEqual(self.protocol['perturbation_root_seed'], PERTURBATION_ROOT_SEED)

    def test_protocol_hash_is_deterministic(self):
        self.assertEqual(compute_protocol_hash(self.protocol), compute_protocol_hash(self.protocol))

    def test_protocol_hash_matches_independent_canonical_implementation(self):
        self.assertEqual(self.protocol['protocol_hash'], self._manual_hash(self.protocol))

    def test_protocol_hash_field_is_excluded_from_hash_input(self):
        changed = copy.deepcopy(self.protocol)
        changed['protocol_hash'] = 'f' * 64
        self.assertEqual(compute_protocol_hash(changed), compute_protocol_hash(self.protocol))

    def test_tampered_manifest_fails_hash_verification(self):
        tampered = copy.deepcopy(self.protocol)
        tampered['train_perturbation']['max_rotation_deg'] = 19.0
        with self.assertRaisesRegex(M3TrainingProtocolError, 'protocol_hash mismatch'):
            validate_training_protocol(tampered)

    def test_tampered_manifest_file_fails_closed(self):
        tampered = copy.deepcopy(self.protocol)
        tampered['coordinate_system'] = 'RAS'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tampered.json'
            path.write_text(json.dumps(tampered), encoding='utf-8')
            with self.assertRaisesRegex(M3TrainingProtocolError, 'protocol_hash mismatch'):
                load_training_protocol(path)

    def test_duplicate_json_field_fails_closed(self):
        duplicate = '{"protocol_version":"a","protocol_version":"b"}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.json'
            path.write_text(duplicate, encoding='utf-8')
            with self.assertRaisesRegex(M3TrainingProtocolError, 'duplicate JSON'):
                load_training_protocol(path)

    def test_ready_subject_ids_match_exact_formal_set(self):
        self.assertEqual(tuple(self.protocol['ready_subject_ids']), READY_SUBJECT_IDS)
        self.assertEqual(len(READY_SUBJECT_IDS), 11)

    def test_pat10_is_absent_everywhere(self):
        self.assertNotIn('Pat10', json.dumps(self.protocol, sort_keys=True))

    def test_subject_groups_partition_ready_subjects_exactly_once(self):
        grouped = [subject for group in self.protocol['groups'].values() for subject in group]
        self.assertEqual(Counter(grouped), Counter(READY_SUBJECT_IDS))

    def test_manifest_has_exactly_five_folds(self):
        self.assertEqual(tuple(self.protocol['folds']), tuple(FOLD_SUBJECTS))

    def test_every_fold_has_nonempty_disjoint_complete_splits(self):
        ready = set(READY_SUBJECT_IDS)
        for fold_id in FOLD_SUBJECTS:
            with self.subTest(fold_id=fold_id):
                fold = resolve_fold(self.protocol, fold_id)
                train = set(fold['train_subject_ids'])
                val = set(fold['val_subject_ids'])
                test = set(fold['test_subject_ids'])
                self.assertTrue(train)
                self.assertTrue(val)
                self.assertTrue(test)
                self.assertFalse(train & val)
                self.assertFalse(train & test)
                self.assertFalse(val & test)
                self.assertEqual(train | val | test, ready)

    def test_every_subject_is_test_exactly_once(self):
        counts = Counter(
            subject
            for fold in self.protocol['folds'].values()
            for subject in fold['test_subject_ids']
        )
        self.assertEqual(counts, Counter({subject: 1 for subject in READY_SUBJECT_IDS}))

    def test_every_subject_is_validation_exactly_once(self):
        counts = Counter(
            subject
            for fold in self.protocol['folds'].values()
            for subject in fold['val_subject_ids']
        )
        self.assertEqual(counts, Counter({subject: 1 for subject in READY_SUBJECT_IDS}))

    def test_unknown_fold_fails_closed(self):
        with self.assertRaisesRegex(M3TrainingProtocolError, 'unknown fold_id'):
            resolve_fold(self.protocol, 'Fold6')

    def test_duplicate_protocol_subject_fails_closed(self):
        changed = copy.deepcopy(self.protocol)
        changed['ready_subject_ids'][1] = changed['ready_subject_ids'][0]
        with self.assertRaisesRegex(M3TrainingProtocolError, 'duplicate subjects'):
            validate_training_protocol(changed, verify_hash=False)

    def test_unknown_fold_subject_fails_closed(self):
        changed = copy.deepcopy(self.protocol)
        changed['folds']['Fold1']['train_subject_ids'][0] = 'Unknown'
        with self.assertRaisesRegex(M3TrainingProtocolError, 'formal protocol'):
            validate_training_protocol(changed, verify_hash=False)

    def test_dataset_ready_subject_match_is_order_independent(self):
        actual = list(reversed(READY_SUBJECT_IDS))
        self.assertEqual(validate_dataset_ready_subjects(self.protocol, actual), READY_SUBJECT_IDS)

    def test_dataset_missing_ready_subject_fails_closed(self):
        with self.assertRaisesRegex(M3TrainingProtocolError, 'missing'):
            validate_dataset_ready_subjects(self.protocol, READY_SUBJECT_IDS[:-1])

    def test_dataset_extra_subject_fails_closed(self):
        with self.assertRaisesRegex(M3TrainingProtocolError, 'unexpected'):
            validate_dataset_ready_subjects(self.protocol, READY_SUBJECT_IDS + ('Pat10',))

    def test_dataset_duplicate_subject_fails_closed(self):
        with self.assertRaisesRegex(M3TrainingProtocolError, 'duplicate'):
            validate_dataset_ready_subjects(self.protocol, READY_SUBJECT_IDS + ('Pat1',))

    def test_train_perturbation_bounds_are_20_mm_and_degrees(self):
        spec = train_perturbation_spec(self.protocol)
        self.assertEqual(spec['purpose'], 'train')
        self.assertEqual(spec['severity'], 'train')
        self.assertEqual(spec['max_rotation_deg'], 20.0)
        self.assertEqual(spec['max_translation_mm'], 20.0)
        self.assertEqual(spec['variant_count'], 1)

    def test_validation_has_three_fixed_severity_bounds(self):
        specs = validation_perturbation_specs(self.protocol)
        observed = [
            (spec['severity'], spec['max_rotation_deg'], spec['max_translation_mm'])
            for spec in specs
        ]
        self.assertEqual(
            observed,
            [('mild', 5.0, 5.0), ('moderate', 10.0, 10.0), ('hard', 20.0, 20.0)],
        )
        self.assertTrue(all(spec['epoch_policy'] == 'fixed_none' for spec in specs))

    def test_test_protocol_has_five_variants_for_each_severity(self):
        specs = test_perturbation_specs(self.protocol)
        self.assertEqual([spec['severity'] for spec in specs], ['mild', 'moderate', 'hard'])
        self.assertTrue(all(spec['variant_count'] == 5 for spec in specs))
        self.assertEqual(sum(spec['variant_count'] for spec in specs), 15)

    def test_manifest_has_no_machine_or_runtime_fields(self):
        keys = set()

        def visit(value):
            if isinstance(value, dict):
                keys.update(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.protocol)
        self.assertTrue(
            {'data_root', 'checkpoint_path', 'gpu_model', 'runtime_seconds'}.isdisjoint(keys)
        )

    def test_manifest_does_not_store_derived_seeds(self):
        keys = set()

        def visit(value):
            if isinstance(value, dict):
                keys.update(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.protocol)
        self.assertNotIn('seed', keys)
        self.assertNotIn('derived_seeds', keys)

    def test_same_train_fold_epoch_subject_has_same_seed(self):
        first = self._seed(
            purpose='train', subject_id='Pat11', severity='train', variant_id=0, epoch=3
        )
        second = self._seed(
            purpose='train', subject_id='Pat11', severity='train', variant_id=0, epoch=3
        )
        self.assertEqual(first, second)

    def test_changed_train_epoch_changes_seed(self):
        first = self._seed(
            purpose='train', subject_id='Pat11', severity='train', variant_id=0, epoch=3
        )
        second = self._seed(
            purpose='train', subject_id='Pat11', severity='train', variant_id=0, epoch=4
        )
        self.assertNotEqual(first, second)

    def test_shuffle_order_does_not_change_subject_train_seed(self):
        subjects = list(resolve_fold(self.protocol, 'Fold1')['train_subject_ids'])
        first = {
            subject: self._seed(
                purpose='train', subject_id=subject, severity='train', variant_id=0, epoch=2
            )
            for subject in subjects
        }
        random.Random(77).shuffle(subjects)
        second = {
            subject: self._seed(
                purpose='train', subject_id=subject, severity='train', variant_id=0, epoch=2
            )
            for subject in subjects
        }
        self.assertEqual(first, second)

    def test_validation_seed_is_fixed_across_training_epochs(self):
        seeds = [
            self._seed(
                purpose='val', subject_id='Pat7', severity='mild', variant_id=0, epoch=None
            )
            for _ in range(4)
        ]
        self.assertEqual(len(set(seeds)), 1)

    def test_test_variants_produce_fifteen_distinct_seeds_per_subject(self):
        seeds = {
            self._seed(
                purpose='test',
                subject_id='Pat12',
                severity=spec['severity'],
                variant_id=variant_id,
                epoch=None,
            )
            for spec in test_perturbation_specs(self.protocol)
            for variant_id in range(spec['variant_count'])
        }
        self.assertEqual(len(seeds), 15)

    def test_all_formal_test_variants_are_non_identity(self):
        for fold_id in FOLD_SUBJECTS:
            fold = resolve_fold(self.protocol, fold_id)
            for subject_id in fold['test_subject_ids']:
                for spec in test_perturbation_specs(self.protocol):
                    for variant_id in range(spec['variant_count']):
                        seed = derive_perturbation_seed(
                            scheme_version=self.protocol['seed_scheme_version'],
                            protocol_hash=self.protocol['protocol_hash'],
                            fold_id=fold_id,
                            root_seed=self.protocol['perturbation_root_seed'],
                            purpose='test',
                            epoch=None,
                            subject_id=subject_id,
                            severity=spec['severity'],
                            variant_id=variant_id,
                        )
                        sampled = sample_rigid_perturbation(
                            seed,
                            spec['max_rotation_deg'],
                            spec['max_translation_mm'],
                        )
                        self.assertTrue(
                            abs(sampled['angle_deg']) > 0.0
                            or np.any(sampled['translation_mm'] != 0.0)
                        )

    def test_augmented_sample_keeps_effective_gt_and_provenance(self):
        raw = {
            'subject_id': 'Pat11',
            'point_xyz_phys': np.asarray(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 1.0]],
                dtype=np.float32,
            ),
            'point_normal': np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            'gt_transform': np.eye(4, dtype=np.float64),
            'gt_transform_direction': 'Point Cloud -> CT',
        }
        seed = self._seed(
            purpose='train', subject_id='Pat11', severity='train', variant_id=0, epoch=0
        )
        augmented = augment_point_sample(
            raw,
            seed=seed,
            max_rotation_deg=20.0,
            max_translation_mm=20.0,
        )
        self.assertEqual(augmented['gt_transform'].shape, (4, 4))
        self.assertEqual(augmented['gt_transform'].dtype, np.float64)
        self.assertIn(PROVENANCE_FIELD, augmented)
        self.assertFalse(np.array_equal(augmented['gt_transform'], raw['gt_transform']))


if __name__ == '__main__':
    unittest.main()
