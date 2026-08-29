import copy
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_DIR = EXPERIMENT_DIR / 'protocols'
TRAINING_PATH = PROTOCOL_DIR / 'm3_6b_5fold_clean10_v2.json'
EVALUATION_PATH = PROTOCOL_DIR / 'm3_defect_eval_clean10_v2.json'
OBSERVATION_PATH = PROTOCOL_DIR / 'm3_metric_observation_clean10_v3.json'
PAIRED_PATH = PROTOCOL_DIR / 'm3_complete_vs_defect_paired_clean10_v1.json'
V1_DIAGNOSTIC_PATH = PROTOCOL_DIR / 'm3_metric_diagnostic_v1.json'
V2_DIAGNOSTIC_PATH = PROTOCOL_DIR / 'm3_metric_diagnostic_clean10_v2.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_m3_complete_vs_defect_paired as paired_cli
from defect_evaluation import (
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
)
from m3_complete_vs_defect_paired import (
    ALLOWED_SUBJECT_IDS,
    COMPLETE_CASES_PER_FOLD,
    COMPLETE_INFERENCE_CASE_COUNT,
    COMPLETE_INSTANCE_COUNT,
    M3CompleteVsDefectPairedContractError,
    PAIRED_COMPARISON_COUNT,
    PAIRED_PROTOCOL_HASH,
    PAIRED_PROTOCOL_VERSION,
    audit_complete_dataset,
    build_clean10_complete_subject_index,
    build_complete_manifest,
    build_paired_comparison,
    compute_paired_protocol_hash,
    load_paired_protocol,
    validate_paired_protocol,
)
from m3_metric_diagnostic import (
    CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
    DIAGNOSTIC_PROTOCOL_HASH,
    load_diagnostic_protocol,
    load_jsonl,
)
from m3_metric_observation import (
    OBSERVATION_PROTOCOL_HASH,
    build_observation_case,
    load_observation_protocol,
)
from training_protocol import load_training_protocol


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
        encoding='utf-8',
    )


class M3CompleteVsDefectPairedClean10Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_training_protocol(TRAINING_PATH)
        cls.evaluation = load_defect_evaluation_protocol(EVALUATION_PATH)
        cls.observation = load_observation_protocol(OBSERVATION_PATH)
        cls.paired = load_paired_protocol(PAIRED_PATH)
        cls.defect_manifests = OrderedDict(
            (
                fold_id,
                build_defect_test_manifest(
                    cls.training,
                    cls.evaluation,
                    fold_id,
                ),
            )
            for fold_id in cls.training['folds']
        )
        cls.complete_manifest = build_complete_manifest(
            cls.defect_manifests,
            cls.paired,
        )

    def _dataset(self, subject_ids=ALLOWED_SUBJECT_IDS):
        return SimpleNamespace(
            defect_enabled=False,
            records=[
                {
                    'subject_id': subject_id,
                    'pointcloud_path': f'{subject_id}/complete_points.npz',
                    'ct_path': f'{subject_id}/complete_ct.nrrd',
                }
                for subject_id in subject_ids
            ],
        )

    def _checkpoint(self, fold_id):
        return str((Path('checkpoints') / fold_id / 'best_val_loss.pt').resolve())

    def _complete_row(self, manifest_case):
        fold_id = manifest_case['fold_id']
        variant = float(manifest_case['variant_id'])
        row = dict(manifest_case)
        row.update(
            {
                'paired_protocol_version': PAIRED_PROTOCOL_VERSION,
                'paired_protocol_hash': PAIRED_PROTOCOL_HASH,
                'source_training_protocol_version': self.training[
                    'protocol_version'
                ],
                'source_training_protocol_hash': self.training['protocol_hash'],
                'source_defect_evaluation_protocol_version': self.evaluation[
                    'evaluation_protocol_version'
                ],
                'source_defect_evaluation_protocol_hash': self.evaluation[
                    'evaluation_protocol_hash'
                ],
                'source_reference_points': self.paired[
                    'complete_source_reference_points'
                ],
                'input_kind': 'complete',
                'checkpoint_path': self._checkpoint(fold_id),
                'solver_success': True,
                'solver_status': 'success',
                'registration_recall_hit': True,
                'centroid_tre_mm': 1.0 + variant * 0.01,
                'point_tre_mean_mm': 2.0 + variant * 0.01,
                'point_tre_median_mm': 1.8 + variant * 0.01,
                'point_tre_rmse_mm': 2.2 + variant * 0.01,
                'point_tre_p95_mm': 3.0 + variant * 0.01,
                'point_tre_max_mm': 4.0 + variant * 0.01,
            }
        )
        return row

    def _formal_defect_row(self, manifest_case):
        row = dict(manifest_case)
        row.update(
            {
                'protocol_version': self.training['protocol_version'],
                'protocol_hash': self.training['protocol_hash'],
                'evaluation_protocol_version': self.evaluation[
                    'evaluation_protocol_version'
                ],
                'evaluation_protocol_hash': self.evaluation[
                    'evaluation_protocol_hash'
                ],
                'checkpoint_path': self._checkpoint(manifest_case['fold_id']),
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

    def _defect_row(self, manifest_case, complete_by_base):
        base = (
            manifest_case['fold_id'],
            manifest_case['subject_id'],
            manifest_case['severity'],
            manifest_case['variant_id'],
        )
        complete = complete_by_base[base]
        replay = dict(manifest_case)
        replay.update(
            {
                'observation_protocol_version': self.observation[
                    'observation_protocol_version'
                ],
                'observation_protocol_hash': self.observation[
                    'observation_protocol_hash'
                ],
                'source_training_protocol_version': self.training[
                    'protocol_version'
                ],
                'source_training_protocol_hash': self.training['protocol_hash'],
                'source_defect_evaluation_protocol_version': self.evaluation[
                    'evaluation_protocol_version'
                ],
                'source_defect_evaluation_protocol_hash': self.evaluation[
                    'evaluation_protocol_hash'
                ],
                'source_reference_points': self.observation[
                    'source_reference_points'
                ],
                'checkpoint_path': self._checkpoint(manifest_case['fold_id']),
                'solver_success': True,
                'solver_status': 'success',
                'registration_recall_hit': True,
                'rre_deg': 1.0,
                'legacy_parameter_rte_mm': 2.0,
                'inlier_ratio': 0.75,
                'num_correspondences': 20,
                'num_inliers': 15,
                'centroid_tre_mm': complete['centroid_tre_mm'] + 0.5,
                'point_tre_mean_mm': complete['point_tre_mean_mm'] + 1.0,
                'point_tre_median_mm': complete['point_tre_median_mm'] + 0.8,
                'point_tre_rmse_mm': complete['point_tre_rmse_mm'] + 1.1,
                'point_tre_p95_mm': complete['point_tre_p95_mm'] + 1.5,
                'point_tre_max_mm': complete['point_tre_max_mm'] + 2.0,
            }
        )
        return build_observation_case(
            self._formal_defect_row(manifest_case),
            replay,
            self.observation,
        )

    def _inputs(self):
        complete = [
            self._complete_row(case)
            for case in self.complete_manifest['cases']
        ]
        complete_by_base = {
            (
                row['fold_id'],
                row['subject_id'],
                row['severity'],
                row['variant_id'],
            ): row
            for row in complete
        }
        defects = [
            self._defect_row(case, complete_by_base)
            for fold_id in self.training['folds']
            for case in self.defect_manifests[fold_id]['cases']
        ]
        return complete, defects

    def test_canonical_protocol_hash_and_sources_are_frozen(self):
        self.assertEqual(
            self.paired['paired_protocol_version'], PAIRED_PROTOCOL_VERSION
        )
        self.assertEqual(self.paired['paired_protocol_hash'], PAIRED_PROTOCOL_HASH)
        self.assertEqual(
            compute_paired_protocol_hash(self.paired), PAIRED_PROTOCOL_HASH
        )
        self.assertEqual(
            self.paired['source_tre_observation_protocol_hash'],
            OBSERVATION_PROTOCOL_HASH,
        )

    def test_raw_exact_clean10_selects_exactly_ten_instances(self):
        audit = audit_complete_dataset(self._dataset(), self.training)
        self.assertEqual(audit['raw_ready_patient_count'], 10)
        self.assertEqual(audit['selected_patient_count'], 10)
        self.assertEqual(audit['selected_complete_instance_count'], 10)
        self.assertEqual(audit['patient_count'], 10)
        self.assertEqual(audit['complete_instance_count'], COMPLETE_INSTANCE_COUNT)
        self.assertEqual(set(audit['subject_ids']), set(ALLOWED_SUBJECT_IDS))
        self.assertEqual(set(audit['instances_per_patient'].values()), {1})
        self.assertFalse(audit['defect_masks_consumed'])

    def test_raw_clean10_plus_excluded_pat6_passes_but_never_selects_pat6(self):
        raw_subjects = list(ALLOWED_SUBJECT_IDS) + ['Pat6']
        dataset = self._dataset(raw_subjects)
        audit = audit_complete_dataset(dataset, self.training)
        selected = build_clean10_complete_subject_index(dataset, self.training)
        self.assertEqual(audit['raw_ready_patient_count'], 11)
        self.assertEqual(audit['raw_ready_subject_ids'], raw_subjects)
        self.assertEqual(audit['selected_patient_count'], 10)
        self.assertEqual(audit['selected_complete_instance_count'], 10)
        self.assertEqual(audit['raw_excluded_subject_ids_present'], ['Pat6'])
        self.assertEqual(audit['pat6_raw_count'], 1)
        self.assertEqual(audit['pat6_selected_count'], 0)
        self.assertEqual(audit['pat10_selected_count'], 0)
        self.assertNotIn('Pat6', selected)
        self.assertNotIn(raw_subjects.index('Pat6'), selected.values())

    def test_missing_clean10_patient_fails_closed(self):
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'missing selected clean10 patients',
        ):
            audit_complete_dataset(
                self._dataset(ALLOWED_SUBJECT_IDS[:-1]),
                self.training,
            )

    def test_duplicate_clean10_patient_fails_closed(self):
        duplicate = list(ALLOWED_SUBJECT_IDS) + [ALLOWED_SUBJECT_IDS[0]]
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'duplicate patients',
        ):
            audit_complete_dataset(self._dataset(duplicate), self.training)

    def test_unknown_raw_extra_patient_fails_closed(self):
        unknown = list(ALLOWED_SUBJECT_IDS) + ['Pat10']
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'not selected or explicitly excluded',
        ):
            audit_complete_dataset(self._dataset(unknown), self.training)

    def test_selected_subject_index_keys_and_values_are_exact_and_unique(self):
        dataset = self._dataset(list(ALLOWED_SUBJECT_IDS) + ['Pat6'])
        selected = build_clean10_complete_subject_index(dataset, self.training)
        self.assertEqual(tuple(selected), ALLOWED_SUBJECT_IDS)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(set(selected.values())), 10)
        self.assertTrue(
            all(
                dataset.records[index]['subject_id'] == subject_id
                for subject_id, index in selected.items()
            )
        )
        self.assertNotIn('Pat6', selected)
        self.assertNotIn('Pat10', selected)

    def test_complete_dataset_defect_mode_fails_closed(self):
        invalid = self._dataset()
        invalid.defect_enabled = True
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'without defect_variants',
        ):
            audit_complete_dataset(invalid, self.training)

    def test_complete_manifest_is_150_and_30_per_fold(self):
        self.assertEqual(
            self.complete_manifest['complete_inference_case_count'],
            COMPLETE_INFERENCE_CASE_COUNT,
        )
        self.assertEqual(len(self.complete_manifest['cases']), 150)
        self.assertEqual(
            self.complete_manifest['per_fold_case_counts'],
            {fold_id: COMPLETE_CASES_PER_FOLD for fold_id in self.training['folds']},
        )
        identities = {
            (
                row['fold_id'],
                row['subject_id'],
                row['severity'],
                row['variant_id'],
                row['perturbation_seed'],
            )
            for row in self.complete_manifest['cases']
        }
        self.assertEqual(len(identities), 150)

    def test_five_defects_share_exactly_one_seed(self):
        for complete in self.complete_manifest['cases']:
            matching = [
                row
                for row in self.defect_manifests[complete['fold_id']]['cases']
                if row['subject_id'] == complete['subject_id']
                and row['severity'] == complete['severity']
                and row['variant_id'] == complete['variant_id']
            ]
            self.assertEqual(len(matching), 5)
            self.assertEqual(
                {row['perturbation_seed'] for row in matching},
                {complete['perturbation_seed']},
            )

    def test_inconsistent_defect_seed_fails_manifest_closed(self):
        changed = copy.deepcopy(self.defect_manifests)
        changed['Fold1']['cases'][0]['perturbation_seed'] += 1
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'seeds are inconsistent',
        ):
            build_complete_manifest(changed, self.paired)

    def test_paired_rows_are_750_and_150_per_fold(self):
        complete, defects = self._inputs()
        result = build_paired_comparison(
            complete,
            defects,
            self.paired,
            self.observation,
            expected_complete_manifest=self.complete_manifest['cases'],
        )
        self.assertEqual(len(result['paired_rows']), PAIRED_COMPARISON_COUNT)
        self.assertEqual(result['paired_identity_unique_count'], 750)
        self.assertEqual(
            {key: value['case_count'] for key, value in result['per_fold'].items()},
            {fold_id: 150 for fold_id in self.training['folds']},
        )

    def test_every_complete_case_has_exactly_five_defects(self):
        complete, defects = self._inputs()
        result = build_paired_comparison(
            complete, defects, self.paired, self.observation
        )
        counts = {}
        for row in result['paired_rows']:
            key = (
                row['fold_id'],
                row['subject_id'],
                row['severity'],
                row['variant_id'],
                row['perturbation_seed'],
            )
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 150)
        self.assertEqual(set(counts.values()), {5})

    def test_delta_direction_is_strictly_defect_minus_complete(self):
        complete, defects = self._inputs()
        result = build_paired_comparison(
            complete, defects, self.paired, self.observation
        )
        first = result['paired_rows'][0]
        self.assertAlmostEqual(first['delta_point_tre_mean_mm'], 1.0)
        self.assertAlmostEqual(first['delta_centroid_tre_mm'], 0.5)
        self.assertAlmostEqual(
            first['delta_point_tre_mean_mm'],
            first['defect']['point_tre_mean_mm']
            - first['complete']['point_tre_mean_mm'],
        )

    def test_patient_level_aggregation_has_ten_independent_patients(self):
        complete, defects = self._inputs()
        result = build_paired_comparison(
            complete, defects, self.paired, self.observation
        )
        patient = result['summary']['patient_level']
        self.assertEqual(patient['patient_count'], 10)
        self.assertEqual(patient['independent_patient_count'], 10)
        point = patient['delta_metrics']['delta_point_tre_mean_mm']
        self.assertEqual(point['sample_count'], 10)
        self.assertEqual(len(point['patient_means']), 10)
        self.assertAlmostEqual(point['mean'], 1.0)
        self.assertEqual(point['positive_patient_count'], 10)
        self.assertTrue(
            all(value['case_count'] == 75 for value in result['per_subject'].values())
        )
        self.assertFalse(
            result['summary']['independent_sample_t_test_performed']
        )

    def test_duplicate_and_missing_defect_pair_fail_closed(self):
        complete, defects = self._inputs()
        duplicate = list(defects)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'duplicate defect pair identity',
        ):
            build_paired_comparison(
                complete, duplicate, self.paired, self.observation
            )
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'exactly 750',
        ):
            build_paired_comparison(
                complete, defects[:-1], self.paired, self.observation
            )

    def test_protocol_hash_and_checkpoint_mismatch_fail_closed(self):
        changed = copy.deepcopy(self.paired)
        changed['paired_comparison_count'] = 751
        changed['paired_protocol_hash'] = compute_paired_protocol_hash(changed)
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'not the frozen hash',
        ):
            validate_paired_protocol(changed)
        complete, defects = self._inputs()
        defects[0]['checkpoint_path'] = 'different_checkpoint.pt'
        with self.assertRaisesRegex(
            M3CompleteVsDefectPairedContractError,
            'checkpoint mismatch',
        ):
            build_paired_comparison(
                complete, defects, self.paired, self.observation
            )

    def test_cpu_pair_aggregation_writes_bundle_without_model_or_gpu(self):
        complete, defects = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_path = root / 'complete_cases_150.jsonl'
            defect_path = root / 'cases_750.jsonl'
            output = root / 'output'
            _write_jsonl(complete_path, complete)
            _write_jsonl(defect_path, defects)
            with mock.patch.object(
                paired_cli,
                '_execute_complete_fold',
                side_effect=AssertionError('model/GPU execution'),
            ):
                result = paired_cli.run_build_paired_comparison(
                    complete_cases_path=complete_path,
                    defect_cases_path=defect_path,
                    output_root=output,
                    paired_protocol=self.paired,
                    observation_protocol=self.observation,
                    complete_manifest=self.complete_manifest,
                )
            self.assertEqual(len(load_jsonl(output / 'paired_cases_750.jsonl')), 750)
            for filename in (
                'paired_summary.json',
                'per_subject.json',
                'per_defect.json',
                'per_severity.json',
                'per_fold.json',
                'provenance.json',
                'SHA256SUMS',
            ):
                self.assertTrue((output / filename).is_file())
        self.assertFalse(result['model_loaded'])
        self.assertFalse(result['gpu_used'])
        self.assertFalse(result['training_performed'])

    def test_execute_uses_default_complete_dataset_call_without_defect_variants(self):
        complete, _ = self._inputs()
        raw_subjects = list(ALLOWED_SUBJECT_IDS) + ['Pat6']
        raw_dataset = self._dataset(raw_subjects)
        by_fold = {
            fold_id: [row for row in complete if row['fold_id'] == fold_id]
            for fold_id in self.training['folds']
        }

        def fake_fold(**kwargs):
            fold_id = kwargs['fold_id']
            selected = kwargs['selected_subject_to_index']
            self.assertEqual(tuple(selected), ALLOWED_SUBJECT_IDS)
            self.assertNotIn('Pat6', selected)
            self.assertNotIn(raw_subjects.index('Pat6'), selected.values())
            return by_fold[fold_id], {
                'epoch': 20,
                'best_val_loss': 0.1,
                'jsonl_path': f'{fold_id}.jsonl',
                'jsonl_validation': {'pass': True},
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            paired_cli,
            'create_dataset',
            return_value=raw_dataset,
        ) as create_mock, mock.patch.object(
            paired_cli,
            'resolve_fold_artifacts',
            side_effect=lambda root, fold_id, protocol, json_log_root=None: (
                Path(root) / fold_id / 'best_val_loss.pt',
                Path(root) / f'{fold_id}.jsonl',
            ),
        ), mock.patch.object(
            paired_cli,
            '_execute_complete_fold',
            side_effect=fake_fold,
        ):
            summary = paired_cli.run_execute_complete(
                data_root=Path('complete-data'),
                checkpoint_root=Path('checkpoints'),
                json_log_root=None,
                output_root=directory,
                device_name='cuda',
                training_protocol=self.training,
                evaluation_protocol=self.evaluation,
                paired_protocol=self.paired,
                complete_manifest=self.complete_manifest,
            )
            create_mock.assert_called_once_with(Path('complete-data'))
            self.assertTrue((Path(directory) / 'complete_cases_150.jsonl').is_file())
            self.assertTrue((Path(directory) / 'complete_manifest.json').is_file())
            provenance = json.loads(
                (Path(directory) / 'provenance.json').read_text(encoding='utf-8')
            )
            self.assertEqual(provenance['raw_ready_patient_count'], 11)
            self.assertEqual(provenance['selected_patient_count'], 10)
            self.assertEqual(
                provenance['raw_excluded_subject_ids_present'], ['Pat6']
            )
            self.assertEqual(
                provenance['selection_source'],
                'frozen_clean10_training_protocol',
            )
        self.assertEqual(summary['case_count'], 150)

    def test_old_v1_v2_v3_protocols_are_zero_regression(self):
        self.assertEqual(
            load_diagnostic_protocol(V1_DIAGNOSTIC_PATH)[
                'diagnostic_protocol_hash'
            ],
            DIAGNOSTIC_PROTOCOL_HASH,
        )
        self.assertEqual(
            load_diagnostic_protocol(V2_DIAGNOSTIC_PATH)[
                'diagnostic_protocol_hash'
            ],
            CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
        )
        self.assertEqual(
            load_observation_protocol(OBSERVATION_PATH)[
                'observation_protocol_hash'
            ],
            OBSERVATION_PROTOCOL_HASH,
        )


if __name__ == '__main__':
    unittest.main()
