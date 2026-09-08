"""CPU unittest coverage; fixtures are synthetic, integration reads validation only."""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[2] / 'experiments/geotransformer.pointct.baseline_v1'
sys.path.insert(0, str(EXPERIMENT))
import aggregate_m4_osseous_strength_selection as v1
import aggregate_m4_osseous_strength_selection_v2_amendment as v2


def successful_grid(protocol, values=None):
    values = values or {value: (1., 1., 1.) for value in v1._FROZEN_LAMBDAS}
    grid = {}
    for value, fold in v2.ARTIFACT_PATHS:
        grid[value, fold] = [{
            **identity, 'artifact_scope': 'validation_only',
            'protocol_version': v1.FROZEN_PROTOCOL_VERSION,
            'protocol_sha256': v1.FROZEN_PROTOCOL_SHA256,
            'solver_success': True, 'solver_status': 'success',
            'registration_recall_hit': False,
            **dict(zip(v1._METRICS, values[value])),
        } for identity in v1.build_expected_case_identities(protocol, fold, value)]
    return grid


def fail(row):
    row.update(solver_success=False, solver_status='insufficient_correspondences',
               registration_recall_hit=False,
               centroid_tre_mm=None, point_tre_mean_mm=None, rre_deg=None)


def current_pattern(protocol):
    """Synthetic metrics, exact observed patient success counts; not formal evidence."""
    counts = {
        .25: [15, 15, 15, 15, 15, 15, 8, 0, 15, 15],
        .5: [15, 15, 15, 15, 15, 15, 9, 3, 15, 15],
        1.: [10, 9, 15, 15, 15, 15, 0, 0, 15, 15],
        2.: [0, 0, 0, 0, 0, 0, 0, 0, 4, 3],
    }
    grid = successful_grid(protocol)
    for value in counts:
        for i, fold in enumerate(v1._FROZEN_FOLDS):
            for j in range(2):
                rows = grid[value, fold][15*j:15*(j+1)]
                for row in rows[counts[value][2*i+j]:]:
                    fail(row)
    return grid


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.amendment = v2.load_contracts()

    def setUp(self):
        self.grid = current_pattern(self.protocol)

    def aggregate(self):
        return v2.aggregate_grid_rows(self.grid, self.protocol)

    def candidate(self, value):
        return next(row for row in self.aggregate()['candidate_completeness_trace']
                    if row['lambda_oss'] == value)

    def test_current_pattern_selects_half(self):
        result = self.aggregate()
        self.assertTrue(result['V2_AMENDMENT_SELECTION_ELIGIBLE'])
        self.assertEqual(result['V2_SELECTED_LAMBDA_OSS'], .5)
        self.assertEqual(result['V2_SELECTION_BASIS'], 'POST_HOC_VALIDATION_ONLY')

    def test_v1_remains_failed_none(self):
        result = self.aggregate()
        self.assertFalse(result['V1_SELECTION_ELIGIBLE'])
        self.assertIsNone(result['V1_SELECTED_LAMBDA_OSS'])
        self.assertEqual(result['V1_STATUS'], 'FAILED_CLEANLY')

    def test_quarter_pat2_zero(self):
        row = self.candidate(.25)
        self.assertFalse(row['candidate_complete'])
        self.assertEqual(row['zero_success_patient_ids'], ['Pat2'])
        self.assertEqual(row['solver_success_count'], 128)
        self.assertEqual(row['solver_failure_count'], 22)
        self.assertEqual(row['solver_success_rate'], 128/150)

    def test_half_all_ten_nonzero(self):
        row = self.candidate(.5)
        self.assertTrue(row['candidate_complete'])
        self.assertEqual(len(row['patient_rows']), 10)
        self.assertTrue(all(p['solver_success_count'] >= 1 for p in row['patient_rows']))
        self.assertEqual(row['solver_success_count'], 132)

    def test_one_pat1_pat2_zero(self):
        row = self.candidate(1.)
        self.assertFalse(row['candidate_complete'])
        self.assertEqual(row['zero_success_patient_ids'], ['Pat1', 'Pat2'])
        self.assertEqual(row['solver_success_count'], 109)

    def test_two_known_eight_zero(self):
        row = self.candidate(2.)
        self.assertFalse(row['candidate_complete'])
        self.assertEqual(row['zero_success_patient_ids'],
                         ['Pat7', 'Pat4', 'Pat11', 'Pat3', 'Pat8', 'Pat9', 'Pat1', 'Pat2'])
        self.assertEqual(row['solver_success_count'], 7)

    def test_zero_candidates_fail_closed(self):
        for rows in self.grid.values():
            for row in rows:
                fail(row)
        result = self.aggregate()
        self.assertFalse(result['V2_AMENDMENT_SELECTION_ELIGIBLE'])
        self.assertIsNone(result['V2_SELECTED_LAMBDA_OSS'])
        self.assertEqual(result['selection_trace'][-1]['reason'], 'no_complete_candidates')

    def test_multiple_candidates_original_three_stages(self):
        # Each stage eliminates one candidate; boundary values must survive.
        self.grid = successful_grid(self.protocol, {
            .25: (1., 1., 2.), .5: (1.25, 1.25, 1.),
            1.: (1.125, 1.5, 0.), 2.: (1.5, 0., 0.),
        })
        result = self.aggregate()
        self.assertEqual(result['V2_SELECTED_LAMBDA_OSS'], .5)
        self.assertEqual([r['survivors'] for r in result['selection_trace'][1:4]],
                         [[.25, .5, 1.], [.25, .5], [.5]])
        original = v1.aggregate_grid_rows(self.grid, self.protocol)
        self.assertEqual(result['V2_SELECTED_LAMBDA_OSS'], original['SELECTED_LAMBDA_OSS'])
        for stage, name in enumerate(('primary', 'secondary', 'tertiary'), 1):
            self.assertEqual(result['selection_trace'][stage]['stage_min'],
                             original['selection_trace'][name + '_stage_min'])

    def test_multiple_complete_subset_excludes_incomplete_best_metric(self):
        self.grid = successful_grid(self.protocol, {
            .25: (0., 0., 0.), .5: (1., 1., 1.),
            1.: (1.125, 1., 1.), 2.: (4., 4., 4.),
        })
        for row in self.grid[.25, 'Fold4'][15:]:
            fail(row)
        result = self.aggregate()
        self.assertEqual(result['selection_trace'][0]['survivors'], [.5, 1., 2.])
        self.assertEqual(result['V2_SELECTED_LAMBDA_OSS'], .5)
        self.assertFalse(result['V1_SELECTION_ELIGIBLE'])

    def test_final_tie_smallest_lambda(self):
        self.grid = successful_grid(self.protocol)
        self.assertEqual(self.aggregate()['V2_SELECTED_LAMBDA_OSS'], .25)

    def test_missing_case_rejected(self):
        self.grid[.25, 'Fold1'].pop()
        with self.assertRaises(v2.Error): self.aggregate()

    def test_duplicate_case_rejected(self):
        self.grid[.25, 'Fold1'][1] = copy.deepcopy(self.grid[.25, 'Fold1'][0])
        with self.assertRaises(v2.Error): self.aggregate()

    def test_unexpected_patient_rejected(self):
        self.grid[.25, 'Fold1'][0]['subject_id'] = 'Pat99'
        with self.assertRaises(v2.Error): self.aggregate()

    def test_pat6_rejected(self):
        self.grid[.25, 'Fold1'][0]['subject_id'] = 'Pat6'
        with self.assertRaises(v2.Error): self.aggregate()

    def test_scope_keys_values_rejected(self):
        for key, value in [('test_metric', 1.), ('solver_status', 'sealed_test'),
                           ('artifact_scope', 'test'), ('artifact_scope', 'training')]:
            with self.subTest(key=key, value=value):
                self.grid = current_pattern(self.protocol)
                self.grid[.25, 'Fold1'][0][key] = value
                with self.assertRaises(v2.Error): self.aggregate()

    def test_unexpected_field_rejected(self):
        self.grid[.25, 'Fold1'][0]['extra'] = 0
        with self.assertRaises(v2.Error): self.aggregate()

    def test_nan_inf_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                self.grid[.25, 'Fold1'][0]['centroid_tre_mm'] = value
                with self.assertRaises(v2.Error): self.aggregate()

    def test_wrong_v1_sha_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['protocol_sha256'] = '0'*64
        with self.assertRaises(v2.Error): v2.aggregate_grid_rows(self.grid, protocol)
        self.grid[.25, 'Fold1'][0]['protocol_sha256'] = '0'*64
        with self.assertRaises(v2.Error): self.aggregate()

    def test_wrong_v2_sha_rejected(self):
        for field, value in [('protocol_sha256', '0'*64), ('expected_cases_per_patient', 14),
                             ('source_v1_summary_sha256', '0'*64), ('extra', True)]:
            with self.subTest(field=field):
                amendment = copy.deepcopy(self.amendment)
                amendment[field] = value
                with self.assertRaises(v2.Error):
                    v2.validate_amendment(amendment, self.protocol)

    def test_grid_wrong_lambda_or_fold_rejected(self):
        for old, new in [((.25, 'Fold1'), (.75, 'Fold1')),
                         ((.25, 'Fold1'), (.25, 'Fold6'))]:
            with self.subTest(new=new):
                grid = copy.deepcopy(self.grid)
                grid[new] = grid.pop(old)
                with self.assertRaises(v2.Error): v2.aggregate_grid_rows(grid, self.protocol)

    def test_wrong_frozen_seed_rejected(self):
        self.grid[.25, 'Fold1'][0]['perturbation_seed'] += 1
        with self.assertRaises(v2.Error): self.aggregate()

    def test_deterministic_in_memory_no_mutation(self):
        before = copy.deepcopy(self.grid)
        first = v1._canonical_json(self.aggregate())
        second = v1._canonical_json(self.aggregate())
        self.assertEqual(first, second)
        self.assertEqual(before, self.grid)

    def test_patient_median_success_only_denominator_15(self):
        row = self.candidate(.5)
        patient = next(p for p in row['patient_rows'] if p['subject_id'] == 'Pat2')
        self.assertEqual(patient['solver_success_rate'], 3/15)
        self.assertEqual(patient['median_centroid_tre_mm'], 1.)


class FirewallAndDiskTests(unittest.TestCase):
    def test_cli_accepts_no_paths_or_abbreviations(self):
        for args in [[], ['--aggregate', '--test-path', '/test'],
                     ['--aggregate', '--result-root', '/external'],
                     ['--aggregate', '--output-json', '/external'],
                     ['--aggregate', '--protocol-path', '/external'],
                     ['--aggregate', '/test'], ['--agg'], ['--verify', '--aggregate']]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): v2.build_argument_parser().parse_args(args)

    def test_forbidden_paths_never_opened(self):
        for path in ['test/results.json', v2.RESULT_ROOT + 'train.jsonl',
                     v2.RESULT_ROOT + 'lambda_0p25/fold1/checkpoints/best_val_loss.pt',
                     '../validation_cases.jsonl', '/external/validation_cases.jsonl']:
            with self.subTest(path=path), mock.patch.object(Path, 'read_bytes') as read:
                with self.assertRaises(v2.Error): v2._read_fixed(path)
                read.assert_not_called()

    def test_real_grid_selects_half_and_only_reads_allowlist(self):
        # Deliberately required locally: absence of formal validation inputs is a
        # failure, never a skip and never replaced with fabricated formal data.
        reads = []
        original = Path.read_bytes
        def guarded(path):
            relative = path.relative_to(v2._REPO_ROOT).as_posix()
            self.assertIn(relative, v2._READ_ALLOWLIST)
            reads.append(relative)
            return original(path)
        with mock.patch.object(Path, 'read_bytes', guarded):
            result = v2.aggregate_verified_inputs()
        self.assertEqual(set(reads), v2._READ_ALLOWLIST)
        self.assertEqual(len(reads), 25)
        self.assertEqual(result['V2_SELECTED_LAMBDA_OSS'], .5)
        self.assertFalse(result['V1_SELECTION_ELIGIBLE'])
        self.assertIsNone(result['V1_SELECTED_LAMBDA_OSS'])
        self.assertFalse(result['formal_test_accessed'])
        self.assertEqual(len(result['validation_artifact_sha256s']), 20)
        self.assertEqual([r['solver_success_count'] for r in result['candidate_completeness_trace']],
                         [128, 132, 109, 7])

    def test_source_summary_hash_mismatch_rejected(self):
        original = v2._read_fixed
        def changed(relative):
            raw = original(relative)
            return raw + b' ' if relative == v2.SOURCE_SUMMARY else raw
        with mock.patch.object(v2, '_read_fixed', changed):
            with self.assertRaisesRegex(v2.Error, 'source V1 summary SHA'):
                v2.load_verified_inputs()

    def test_artifact_hash_mismatch_rejected(self):
        original = v2._read_fixed
        def changed(relative):
            raw = original(relative)
            return raw + b' ' if relative in v2.ARTIFACT_PATHS.values() else raw
        with mock.patch.object(v2, '_read_fixed', changed):
            with self.assertRaisesRegex(v2.Error, 'validation artifact SHA'):
                v2.load_verified_inputs()

    def test_wrong_sidecar_rejected(self):
        with self.assertRaises(v2.Error):
            v2._sidecar(b'0'*64, v2.V2_PROTOCOL, v2.AMENDMENT_SHA256)

    def test_json_duplicate_fields_and_nonfinite_rejected(self):
        for raw in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
                    b'{"a":-Infinity}', b'{"a":1e999}', b'{"a":-1e999}']:
            with self.subTest(raw=raw), self.assertRaises(v2.Error): v2._json(raw)

    def test_source_summary_structure_and_float_tolerance(self):
        v2._equivalent_summary({'mean': 7.954210859143403}, {'mean': 7.954210859143404})
        for actual in [{'mean': 8.}, {'mean': float('nan')}, {'mean': 1., 'extra': 0},
                       {'mean': 1}, {'mean': True}]:
            with self.subTest(actual=actual), self.assertRaises(v2.Error):
                v2._equivalent_summary(actual, {'mean': 1.})

    def test_existing_output_refuses_overwrite_before_reading(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(v2, '_REPO_ROOT', Path(directory)):
            target = Path(directory) / v2.OUTPUT_SUMMARY
            target.parent.mkdir(parents=True)
            target.write_bytes(b'original')
            with mock.patch.object(v2, 'aggregate_verified_inputs') as aggregate:
                with self.assertRaisesRegex(v2.Error, 'refusing overwrite'):
                    v2.main(['--aggregate'])
                aggregate.assert_not_called()
            self.assertEqual(target.read_bytes(), b'original')

    def test_atomic_publication_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(v2, '_REPO_ROOT', Path(directory)):
            target = Path(directory) / v2.OUTPUT_SUMMARY
            target.parent.mkdir(parents=True)
            v2._write_summary({'scope': 'validation_only'})
            initial = target.read_bytes()
            with self.assertRaises(v2.Error): v2._write_summary({'other': 0})
            self.assertEqual(target.read_bytes(), initial)

    def test_concurrent_output_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(v2, '_REPO_ROOT', Path(directory)):
            target = Path(directory) / v2.OUTPUT_SUMMARY
            target.parent.mkdir(parents=True)
            original = os.link
            def race(source, destination):
                Path(destination).write_bytes(b'concurrent writer')
                return original(source, destination)
            with mock.patch.object(v2.os, 'link', race):
                with self.assertRaises(v2.Error): v2._write_summary({'new': 1})
            self.assertEqual(target.read_bytes(), b'concurrent writer')
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_reparse_component_rejected_before_read(self):
        path = v2._REPO_ROOT / v2.SOURCE_SUMMARY
        with mock.patch.object(v1, '_is_linklike', side_effect=lambda p: Path(p) == path.parent):
            with mock.patch.object(Path, 'read_bytes') as read:
                with self.assertRaises(v2.Error): v2._read_fixed(v2.SOURCE_SUMMARY)
                read.assert_not_called()

    def test_contract_audit_does_not_read_results(self):
        original = v2._read_fixed
        def guarded(relative):
            self.assertTrue(relative.startswith(v2._PROTOCOL_DIR))
            return original(relative)
        with mock.patch.object(v2, '_read_fixed', guarded):
            result = v2.run_contract_audit()
        self.assertEqual(result['contract_audit'], 'PASS')
        self.assertEqual(result['expected_case_count'], 600)
        self.assertFalse(result['result_artifacts_read'])


if __name__ == '__main__':
    unittest.main()
