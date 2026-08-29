import argparse
import copy
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_DIR = EXPERIMENT_DIR / 'protocols'
OBSERVATION_PATH = PROTOCOL_DIR / 'm3_metric_observation_clean10_v3.json'
TRAINING_PATH = PROTOCOL_DIR / 'm3_6b_5fold_clean10_v2.json'
EVALUATION_PATH = PROTOCOL_DIR / 'm3_defect_eval_clean10_v2.json'
V1_DIAGNOSTIC_PATH = PROTOCOL_DIR / 'm3_metric_diagnostic_v1.json'
V2_DIAGNOSTIC_PATH = PROTOCOL_DIR / 'm3_metric_diagnostic_clean10_v2.json'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_m3_defect_metric_observation as observation_cli
from defect_evaluation import (
    build_defect_test_manifest,
    load_defect_evaluation_protocol,
)
from m3_metric_diagnostic import (
    CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
    DIAGNOSTIC_PROTOCOL_HASH,
    LEGACY_CONTINUOUS_ABS_TOLERANCE,
    LEGACY_CONTINUOUS_REL_TOLERANCE,
    load_diagnostic_protocol,
    load_jsonl,
)
from m3_metric_observation import (
    EXPECTED_CASE_COUNTS_BY_FOLD,
    M3MetricObservationContractError,
    OBSERVATION_PROTOCOL_HASH,
    OBSERVATION_PROTOCOL_VERSION,
    STRICT_REPRODUCIBILITY_ABS_TOLERANCE,
    STRICT_REPRODUCIBILITY_REL_TOLERANCE,
    aggregate_mixed_clean10_750,
    aggregate_observation_cases,
    build_observation_case,
    compute_observation_protocol_hash,
    load_observation_protocol,
    validate_observation_protocol,
)
from training_protocol import load_training_protocol


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
        encoding='utf-8',
    )


class M3MetricObservationClean10Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_training_protocol(TRAINING_PATH)
        cls.evaluation = load_defect_evaluation_protocol(EVALUATION_PATH)
        cls.observation = load_observation_protocol(OBSERVATION_PATH)
        cls.manifests = OrderedDict(
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

    def _formal(self, manifest_case):
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
                'checkpoint_path': 'best_val_loss.pt',
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

    def _replay(self, manifest_case, *, rre_deg=1.0):
        row = dict(manifest_case)
        row.update(
            {
                'observation_protocol_version': OBSERVATION_PROTOCOL_VERSION,
                'observation_protocol_hash': OBSERVATION_PROTOCOL_HASH,
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
                'checkpoint_path': 'best_val_loss.pt',
                'solver_success': True,
                'solver_status': 'success',
                'registration_recall_hit': True,
                'rre_deg': rre_deg,
                'legacy_parameter_rte_mm': 2.0,
                'inlier_ratio': 0.75,
                'num_correspondences': 20,
                'num_inliers': 15,
                'centroid_tre_mm': 2.5,
                'point_tre_mean_mm': 3.0,
                'point_tre_median_mm': 3.0,
                'point_tre_rmse_mm': 3.5,
                'point_tre_p95_mm': 4.0,
                'point_tre_max_mm': 4.5,
            }
        )
        return row

    def _v2(self, manifest_case):
        row = self._replay(manifest_case)
        row['diagnostic_protocol_version'] = (
            'm3_metric_diagnostic_clean10_v2'
        )
        row['diagnostic_protocol_hash'] = CLEAN10_DIAGNOSTIC_PROTOCOL_HASH
        row.pop('observation_protocol_version')
        row.pop('observation_protocol_hash')
        return row

    def _v3(self, manifest_case, *, rre_deg=1.0):
        return build_observation_case(
            self._formal(manifest_case),
            self._replay(manifest_case, rre_deg=rre_deg),
            self.observation,
        )

    def _mixed_inputs(self, *, fold5_first_rre=1.0):
        v2 = OrderedDict(
            (
                fold_id,
                [self._v2(case) for case in self.manifests[fold_id]['cases']],
            )
            for fold_id in ('Fold1', 'Fold2', 'Fold3', 'Fold4')
        )
        fold5 = []
        for index, case in enumerate(self.manifests['Fold5']['cases']):
            rre = fold5_first_rre if index == 0 else 1.0
            fold5.append(self._v3(case, rre_deg=rre))
        return v2, fold5

    def test_v1_and_clean10_v2_protocols_are_zero_regression(self):
        self.assertEqual(
            load_diagnostic_protocol(V1_DIAGNOSTIC_PATH)[
                'diagnostic_protocol_hash'
            ],
            DIAGNOSTIC_PROTOCOL_HASH,
        )
        v2 = load_diagnostic_protocol(V2_DIAGNOSTIC_PATH)
        self.assertEqual(
            v2['diagnostic_protocol_hash'],
            CLEAN10_DIAGNOSTIC_PROTOCOL_HASH,
        )
        self.assertEqual(v2['legacy_continuous_abs_tolerance'], 1e-6)
        self.assertEqual(v2['legacy_continuous_rel_tolerance'], 1e-7)

    def test_v3_canonical_hash_and_policy_are_frozen(self):
        self.assertEqual(
            self.observation['observation_protocol_version'],
            OBSERVATION_PROTOCOL_VERSION,
        )
        self.assertEqual(
            self.observation['observation_protocol_hash'],
            OBSERVATION_PROTOCOL_HASH,
        )
        self.assertEqual(
            compute_observation_protocol_hash(self.observation),
            OBSERVATION_PROTOCOL_HASH,
        )
        self.assertEqual(
            self.observation['formal_metric_policy'],
            'frozen_formal_metrics_are_authoritative',
        )

    def test_v3_contract_is_10_50_750_and_pat6_zero(self):
        rows = [
            case
            for fold_id in self.training['folds']
            for case in self.manifests[fold_id]['cases']
        ]
        self.assertEqual(self.observation['expected_patient_count'], 10)
        self.assertEqual(self.observation['expected_defect_instance_count'], 50)
        self.assertEqual(self.observation['expected_test_case_count'], 750)
        self.assertEqual(self.observation['expected_pat6_case_count'], 0)
        self.assertEqual(
            {fold: self.manifests[fold]['total_cases'] for fold in self.manifests},
            EXPECTED_CASE_COUNTS_BY_FOLD,
        )
        self.assertFalse(any(row['subject_id'] == 'Pat6' for row in rows))

    def test_exact_identity_match_is_fail_closed(self):
        case = self.manifests['Fold5']['cases'][0]
        replay = self._replay(case)
        replay['perturbation_seed'] += 1
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'identity mismatch',
        ):
            build_observation_case(
                self._formal(case), replay, self.observation
            )

    def test_replay_pass_case(self):
        case = self.manifests['Fold5']['cases'][0]
        row = self._v3(case)
        self.assertEqual(row['replay_reproducibility']['status'], 'PASS')
        self.assertEqual(row['replay_reproducibility']['mismatched_fields'], [])

    def test_replay_fail_preserves_tre_and_frozen_formal(self):
        case = self.manifests['Fold5']['cases'][0]
        formal = self._formal(case)
        formal['rre_deg'] = 3.4837944052005345
        replay_rre = 3.4845223565532577
        row = build_observation_case(
            formal,
            self._replay(case, rre_deg=replay_rre),
            self.observation,
        )
        self.assertEqual(row['replay_reproducibility']['status'], 'FAIL')
        self.assertIn(
            'rre_deg', row['replay_reproducibility']['mismatched_fields']
        )
        self.assertEqual(
            row['frozen_formal']['rre_deg'], 3.4837944052005345
        )
        self.assertEqual(
            row['replay_observation']['replay_rre_deg'], replay_rre
        )
        self.assertEqual(row['replay_observation']['point_tre_mean_mm'], 3.0)

    def test_replay_mismatch_cannot_overwrite_formal_aggregation(self):
        case = self.manifests['Fold5']['cases'][0]
        row = self._v3(case, rre_deg=99.0)
        summary = aggregate_observation_cases(
            [row], self.observation, expected_case_count=1
        )
        self.assertEqual(summary['formal_metrics']['rre_deg']['mean'], 1.0)
        self.assertEqual(summary['reproducibility_fail_count'], 1)
        self.assertEqual(summary['tre_observations']['point_tre_mean_mm']['mean'], 3.0)

    def test_protocol_hash_and_structure_mismatch_fail_closed(self):
        changed = copy.deepcopy(self.observation)
        changed['formal_metric_policy'] = 'replay_is_authoritative'
        changed['observation_protocol_hash'] = (
            compute_observation_protocol_hash(changed)
        )
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'not the frozen hash',
        ):
            validate_observation_protocol(changed)
        case = self.manifests['Fold5']['cases'][0]
        replay = self._replay(case)
        replay['source_training_protocol_hash'] = '0' * 64
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'source_training_protocol_hash mismatch',
        ):
            build_observation_case(
                self._formal(case), replay, self.observation
            )

    def test_serialized_reproducibility_tamper_fails_closed(self):
        case = self.manifests['Fold5']['cases'][0]
        row = self._v3(case, rre_deg=1.000727951)
        row['replay_reproducibility']['status'] = 'PASS'
        row['replay_reproducibility']['mismatched_fields'] = []
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'does not match the preserved formal/replay values',
        ):
            aggregate_observation_cases(
                [row], self.observation, expected_case_count=1
            )

    def test_duplicate_and_missing_observations_fail_closed(self):
        first = self._v3(self.manifests['Fold5']['cases'][0])
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'duplicate observation case identity',
        ):
            aggregate_observation_cases(
                [first, copy.deepcopy(first)],
                self.observation,
                expected_case_count=2,
            )
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'case count mismatch',
        ):
            aggregate_observation_cases(
                [first], self.observation, expected_case_count=2
            )

    def test_mixed_600_v2_plus_150_v3_is_750_unique(self):
        v2, fold5 = self._mixed_inputs(fold5_first_rre=1.000727951)
        result = aggregate_mixed_clean10_750(
            v2,
            fold5,
            self.observation,
            expected_cases_by_fold={
                fold: self.manifests[fold]['cases'] for fold in self.manifests
            },
        )
        self.assertEqual(len(result['cases']), 750)
        self.assertEqual(result['summary']['identity_unique_count'], 750)
        self.assertEqual(result['summary']['reproducibility_pass_count'], 749)
        self.assertEqual(result['summary']['reproducibility_fail_count'], 1)
        self.assertTrue(result['summary']['TRE_OBSERVATION_COMPLETE'])
        self.assertTrue(
            result['summary']['STRICT_REPLAY_REPRODUCIBILITY_COMPLETE']
        )
        self.assertFalse(
            result['summary']['STRICT_REPLAY_REPRODUCIBILITY_ALL_PASS']
        )
        self.assertEqual(
            result['summary']['tre_observations']['point_tre_mean_mm'][
                'sample_count'
            ],
            750,
        )

    def test_mixed_fold_counts_and_identity_are_fail_closed(self):
        v2, fold5 = self._mixed_inputs()
        v2['Fold2'] = v2['Fold2'][:-1]
        with self.assertRaisesRegex(
            M3MetricObservationContractError,
            'Fold2 v2 case count mismatch',
        ):
            aggregate_mixed_clean10_750(v2, fold5, self.observation)

    def test_cpu_aggregation_writes_required_bundle_without_model(self):
        v2, fold5 = self._mixed_inputs(fold5_first_rre=1.000727951)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2_root = root / 'v2'
            v3_root = root / 'v3'
            output = root / 'output'
            for fold_id, rows in v2.items():
                _write_jsonl(
                    v2_root / fold_id / 'diagnostic_cases.jsonl', rows
                )
            _write_jsonl(
                v3_root / 'Fold5' / 'observation_cases.jsonl', fold5
            )
            with mock.patch.object(
                observation_cli.diagnostic_cli,
                '_execute_fold',
                side_effect=AssertionError('model/GPU execution'),
            ):
                result = observation_cli.run_cpu_aggregate_750(
                    v2_results_root=v2_root,
                    v3_results_root=v3_root,
                    output_root=output,
                    training_protocol=self.training,
                    evaluation_protocol=self.evaluation,
                    observation_protocol=self.observation,
                    manifests=self.manifests,
                )
            self.assertEqual(len(load_jsonl(output / 'cases_750.jsonl')), 750)
            for filename in (
                'summary_750.json',
                'reproducibility_summary.json',
                'provenance.json',
                'SHA256SUMS',
            ):
                self.assertTrue((output / filename).is_file())
        self.assertFalse(result['model_loaded'])
        self.assertFalse(result['gpu_used'])

    def test_execute_fold_writes_all_artifacts_when_replay_fails(self):
        fold_id = 'Fold5'
        formal_rows = [
            self._formal(case) for case in self.manifests[fold_id]['cases']
        ]
        replay_rows = [
            self._replay(case) for case in self.manifests[fold_id]['cases']
        ]
        replay_rows[0]['rre_deg'] = 1.000727951
        adapter_rows = []
        for row in replay_rows:
            adapted = dict(row)
            adapted['diagnostic_protocol_version'] = adapted.pop(
                'observation_protocol_version'
            )
            adapted['diagnostic_protocol_hash'] = adapted.pop(
                'observation_protocol_hash'
            )
            adapter_rows.append(adapted)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            observation_cli,
            'resolve_fold_artifacts',
            return_value=(Path('best_val_loss.pt'), Path('train.jsonl')),
        ), mock.patch.object(
            observation_cli.diagnostic_cli,
            '_execute_fold',
            return_value=(
                adapter_rows,
                {'epoch': 20, 'best_val_loss': 0.1},
            ),
        ):
            summary = observation_cli.run_execute_observation(
                data_root=Path('must-not-be-read'),
                checkpoint_root=Path('checkpoints'),
                json_log_root=None,
                output_root=directory,
                formal_results={'per_fold': {fold_id: formal_rows}},
                device_name='cuda',
                fold_ids=(fold_id,),
                training_protocol=self.training,
                evaluation_protocol=self.evaluation,
                observation_protocol=self.observation,
                manifests=self.manifests,
            )
            fold_dir = Path(directory) / fold_id
            for filename in (
                'observation_cases.jsonl',
                'observation_summary.json',
                'observation_manifest.json',
                'reproducibility_report.json',
            ):
                self.assertTrue((fold_dir / filename).is_file())
            persisted = load_jsonl(fold_dir / 'observation_cases.jsonl')
        self.assertEqual(summary['reproducibility_fail_count'], 1)
        self.assertEqual(
            persisted[0]['replay_observation']['point_tre_mean_mm'], 3.0
        )

    def test_v3_does_not_change_v2_tolerances(self):
        self.assertEqual(
            STRICT_REPRODUCIBILITY_ABS_TOLERANCE,
            LEGACY_CONTINUOUS_ABS_TOLERANCE,
        )
        self.assertEqual(
            STRICT_REPRODUCIBILITY_REL_TOLERANCE,
            LEGACY_CONTINUOUS_REL_TOLERANCE,
        )
        self.assertEqual(
            (STRICT_REPRODUCIBILITY_ABS_TOLERANCE,
             STRICT_REPRODUCIBILITY_REL_TOLERANCE),
            (1e-6, 1e-7),
        )


if __name__ == '__main__':
    unittest.main()
