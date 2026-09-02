import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
PROTOCOL_PATH = (
    EXPERIMENT_DIR
    / 'protocols'
    / 'm4_osseous_strength_selection_clean10_v1.json'
)
SIDECAR_PATH = PROTOCOL_PATH.with_suffix('.sha256')
RUN_SCRIPT_PATH = PROJECT_ROOT / 'run_m4_osseous_strength_selection_clean10.sh'
GIT_PATH = shutil.which('git')
BASH_PATH = shutil.which('bash')
if BASH_PATH is None and os.name == 'nt':
    candidate = Path(r'C:\Program Files\Git\bin\bash.exe')
    if candidate.is_file():
        BASH_PATH = str(candidate)

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aggregate_m4_osseous_strength_selection as aggregator
import evaluate_m4_osseous_strength_validation as producer


FROZEN_SHA256 = (
    'a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38'
)
FROZEN_LAMBDAS = (0.25, 0.5, 1.0, 2.0)
FROZEN_FOLDS = ('Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5')
FROZEN_VAL_SUBJECTS = {
    'Fold1': ('Pat7', 'Pat4'),
    'Fold2': ('Pat11', 'Pat3'),
    'Fold3': ('Pat8', 'Pat9'),
    'Fold4': ('Pat1', 'Pat2'),
    'Fold5': ('Pat12', 'Pat5'),
}


def _canonical_sha256(protocol):
    payload = dict(protocol)
    payload.pop('protocol_sha256', None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _successful_rows(protocol, fold_id, lambda_oss, metric_factory=None):
    rows = []
    identities = aggregator.build_expected_case_identities(
        protocol, fold_id, float(lambda_oss)
    )
    for index, identity in enumerate(identities):
        if metric_factory is None:
            metrics = (1.0, 2.0, 3.0)
        else:
            metrics = metric_factory(identity, index)
        rows.append(
            {
                'artifact_scope': 'validation_only',
                'protocol_version': aggregator.FROZEN_PROTOCOL_VERSION,
                'protocol_sha256': aggregator.FROZEN_PROTOCOL_SHA256,
                **identity,
                'solver_success': True,
                'solver_status': 'success',
                'registration_recall_hit': True,
                'centroid_tre_mm': float(metrics[0]),
                'point_tre_mean_mm': float(metrics[1]),
                'rre_deg': float(metrics[2]),
            }
        )
    return rows


def _mark_failure(row, status='solver_failed'):
    row['solver_success'] = False
    row['solver_status'] = status
    row['registration_recall_hit'] = False
    row['centroid_tre_mm'] = None
    row['point_tre_mean_mm'] = None
    row['rre_deg'] = None


def _full_grid(protocol, metrics_by_lambda=None):
    metrics_by_lambda = metrics_by_lambda or {
        0.25: (1.0, 1.0, 1.0),
        0.5: (1.1, 1.1, 1.1),
        1.0: (1.2, 1.2, 1.2),
        2.0: (2.0, 2.0, 2.0),
    }
    grid = {}
    for lambda_oss in FROZEN_LAMBDAS:
        values = metrics_by_lambda[lambda_oss]
        for fold_id in FROZEN_FOLDS:
            grid[(lambda_oss, fold_id)] = _successful_rows(
                protocol,
                fold_id,
                lambda_oss,
                metric_factory=lambda _identity, _index, values=values: values,
            )
    return grid


def _selection_summaries(values_by_lambda, incomplete=()):
    summaries = []
    for lambda_oss in FROZEN_LAMBDAS:
        centroid, point_mean, rre = values_by_lambda[lambda_oss]
        summaries.append(
            {
                'lambda_oss': float(lambda_oss),
                'candidate_complete': lambda_oss not in incomplete,
                'selection_metrics': {
                    'centroid_tre_mm': centroid,
                    'point_tre_mean_mm': point_mean,
                    'rre_deg': rre,
                },
            }
        )
    return summaries


class FrozenProtocolAndInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = aggregator.load_frozen_protocol(PROTOCOL_PATH)
        (
            cls.producer_protocol,
            cls.training_protocol,
            cls.evaluation_protocol,
            cls.diagnostic_protocol,
        ) = producer.load_selection_protocol(PROTOCOL_PATH)

    def test_protocol_hash_is_reproducible_and_sidecar_is_exact(self):
        self.assertEqual(_canonical_sha256(self.protocol), FROZEN_SHA256)
        self.assertEqual(self.protocol['protocol_sha256'], FROZEN_SHA256)
        self.assertEqual(aggregator.FROZEN_PROTOCOL_SHA256, FROZEN_SHA256)
        self.assertEqual(producer.FROZEN_PROTOCOL_SHA256, FROZEN_SHA256)
        self.assertEqual(
            SIDECAR_PATH.read_text(encoding='utf-8').strip(),
            f'{FROZEN_SHA256}  {PROTOCOL_PATH.name} '
            '(canonical JSON excluding protocol_sha256)',
        )
        self.assertEqual(self.protocol, self.producer_protocol)

    def test_lambda_grid_is_exact_unique_finite_and_positive(self):
        grid = self.protocol['lambda_oss_grid']
        self.assertEqual(tuple(grid), FROZEN_LAMBDAS)
        self.assertEqual(len(grid), len(set(grid)))
        self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in grid))
        for invalid in (
            [0.25, 0.5, 1.0, 1.0],
            [0.0, 0.5, 1.0, 2.0],
            [float('nan'), 0.5, 1.0, 2.0],
            [0.25, 0.5, 1.0, 3.0],
        ):
            mutated = copy.deepcopy(self.protocol)
            mutated['lambda_oss_grid'] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                    aggregator.validate_frozen_protocol(mutated)

    def test_clean10_fold_mapping_is_exact_disjoint_and_excludes_pat6(self):
        ready = set(self.protocol['clean10']['ready_subject_ids'])
        self.assertEqual(len(ready), 10)
        self.assertNotIn('Pat6', ready)
        self.assertEqual(self.protocol['clean10']['excluded_subject_ids'], ['Pat6'])
        validation_counts = {subject_id: 0 for subject_id in ready}
        for fold_id in FROZEN_FOLDS:
            fold = self.protocol['clean10']['folds'][fold_id]
            train = set(fold['train_subject_ids'])
            validation = set(fold['val_subject_ids'])
            test = set(fold['test_subject_ids'])
            self.assertEqual(tuple(fold['val_subject_ids']), FROZEN_VAL_SUBJECTS[fold_id])
            self.assertEqual(len(train), 6)
            self.assertEqual(len(validation), 2)
            self.assertEqual(len(test), 2)
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, ready)
            self.assertNotIn('Pat6', train | validation | test)
            for subject_id in validation:
                validation_counts[subject_id] += 1
        self.assertEqual(set(validation_counts.values()), {1})

    def test_validation_manifests_use_only_val_patients_and_exact_600_identities(self):
        all_identities = set()
        for lambda_oss in FROZEN_LAMBDAS:
            for fold_id in FROZEN_FOLDS:
                rows = producer.build_validation_manifest(
                    self.producer_protocol,
                    self.training_protocol,
                    fold_id,
                    lambda_oss,
                )
                expected = aggregator.build_expected_case_identities(
                    self.protocol, fold_id, lambda_oss
                )
                self.assertEqual(len(rows), 30)
                self.assertEqual({row['subject_id'] for row in rows}, set(FROZEN_VAL_SUBJECTS[fold_id]))
                self.assertFalse(
                    {row['subject_id'] for row in rows}
                    & set(self.protocol['clean10']['folds'][fold_id]['test_subject_ids'])
                )
                projected = tuple(
                    {
                        field: row[field]
                        for field in self.protocol['validation_protocol']['case_identity']
                    }
                    for row in rows
                )
                self.assertEqual(projected, expected)
                all_identities.update(
                    tuple(row[field] for field in self.protocol['validation_protocol']['case_identity'])
                    for row in rows
                )
        self.assertEqual(len(all_identities), 600)

    def test_same_seed_and_non_lambda_contract_across_candidates(self):
        settings = self.protocol['training_settings']
        self.assertEqual(settings['only_variable_across_candidates'], 'lambda_oss')
        self.assertEqual(settings['epochs'], 20)
        self.assertEqual(settings['seed'], 20260815)
        self.assertEqual(settings['optimizer']['name'], 'AdamW')
        self.assertEqual(settings['optimizer']['learning_rate'], 0.0003)
        self.assertEqual(settings['optimizer']['weight_decay'], 0.0001)
        self.assertEqual(settings['scheduler']['name'], 'none')
        self.assertEqual(settings['batch_size'], 1)
        self.assertEqual(settings['m4_soft']['sigma_mm'], 60.0)
        self.assertEqual(settings['m4_soft']['strength'], 2.0)
        for fold_id in FROZEN_FOLDS:
            baseline = producer.build_validation_manifest(
                self.producer_protocol, self.training_protocol, fold_id, 0.25
            )
            for lambda_oss in FROZEN_LAMBDAS[1:]:
                candidate = producer.build_validation_manifest(
                    self.producer_protocol,
                    self.training_protocol,
                    fold_id,
                    lambda_oss,
                )
                for first, second in zip(baseline, candidate):
                    first = dict(first)
                    second = dict(second)
                    first.pop('lambda_oss')
                    second.pop('lambda_oss')
                    self.assertEqual(first, second)

    def test_run_script_and_producer_have_validation_only_interfaces(self):
        script = RUN_SCRIPT_PATH.read_text(encoding='utf-8')
        producer_source = (
            EXPERIMENT_DIR / 'evaluate_m4_osseous_strength_validation.py'
        ).read_text(encoding='utf-8')
        aggregator_source = (
            EXPERIMENT_DIR / 'aggregate_m4_osseous_strength_selection.py'
        ).read_text(encoding='utf-8')
        self.assertIn("readonly -a LAMBDAS=('0.25' '0.5' '1.0' '2.0')", script)
        self.assertIn("readonly -a FOLDS=('Fold1' 'Fold2' 'Fold3' 'Fold4' 'Fold5')", script)
        self.assertIn('combination_count}" -eq 20', script)
        self.assertIn('checkpoints/m4_osseous_strength_selection_v1', script)
        self.assertIn('--epochs 20', script)
        self.assertIn('--seed 20260815', script)
        self.assertIn('--m4-soft-sigma-mm 60', script)
        self.assertIn('--m4-soft-strength 2', script)
        self.assertIn('--m4-osseous-strength "${lambda_oss}"', script)
        self.assertIn('--execute-validation', script)
        for forbidden in (
            'evaluate_m3_defect.py',
            '--execute-test',
            '--test-root',
            '--test-subject',
        ):
            self.assertNotIn(forbidden, script)
        for forbidden in ('test_subject_ids', 'test_indices', 'test_instances'):
            self.assertNotIn(forbidden, producer_source)
            self.assertNotIn(forbidden, aggregator_source)
        self.assertNotIn('SELECTED_LAMBDA_OSS', RUN_SCRIPT_PATH.name)

    def test_dry_run_and_live_validation_use_the_same_indexed_cuda_device(self):
        script = RUN_SCRIPT_PATH.read_text(encoding='utf-8')
        validation_blocks = re.findall(
            r'^\s*validation_command=\(\n(.*?)^\s*\)$',
            script,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertEqual(len(validation_blocks), 2)
        normalized_blocks = [' '.join(block.split()) for block in validation_blocks]
        self.assertEqual(normalized_blocks[0], normalized_blocks[1])
        for block in validation_blocks:
            self.assertIn('--execute-validation', block)
            self.assertEqual(block.count('--device cuda:0'), 1)
            self.assertNotRegex(block, r'--device\s+cuda(?:\s|$)')

    def test_all_public_protocol_driven_entries_reject_rehashed_mutation(self):
        mutated = copy.deepcopy(self.protocol)
        mutated['selection_rule']['primary']['practical_tie_threshold_mm'] = 0.5
        mutated['protocol_sha256'] = _canonical_sha256(mutated)
        valid_rows = _successful_rows(self.protocol, 'Fold1', 0.25)
        valid_summaries = _selection_summaries(
            {value: (1.0, 1.0, 1.0) for value in FROZEN_LAMBDAS}
        )
        public_calls = (
            lambda: aggregator.derive_expected_seed(mutated, 'Fold1', 'Pat7', 'mild'),
            lambda: aggregator.build_expected_case_identities(mutated, 'Fold1', 0.25),
            lambda: aggregator.validate_validation_cases(valid_rows, mutated, 'Fold1', 0.25),
            lambda: aggregator.collapse_patient_rows(valid_rows, mutated, 'Fold1', 0.25),
            lambda: aggregator.apply_selection_rule(valid_summaries, mutated),
            lambda: aggregator.aggregate_grid_rows({}, mutated),
            lambda: aggregator.frozen_output_root(mutated),
            lambda: aggregator.load_validation_grid(mutated, PROJECT_ROOT),
        )
        for call in public_calls:
            with self.subTest(call=call):
                with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                    call()


@unittest.skipUnless(
    GIT_PATH is not None and BASH_PATH is not None,
    'Git and bash are required for runner provenance tests.',
)
class RunnerProvenanceRegressionTest(unittest.TestCase):
    _EXPECTED_BRANCH = 'm4_osseous_strength_selection_v1'
    _FROZEN_BASE = 'd6a10cf092089426307b5f93766b8c5a5a2636f9'

    @staticmethod
    def _write(root, relative_path, content):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8', newline='\n')
        return path

    @staticmethod
    def _git(root, *arguments):
        return subprocess.run(
            [GIT_PATH, *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
        )

    def _make_repository(
        self,
        *,
        branch=None,
        unrelated_base=False,
        unexpected_tracked_path=False,
    ):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        branch = branch or self._EXPECTED_BRANCH
        self._git(root, 'init', '-b', branch)
        self._git(root, 'config', 'user.name', 'M4 Provenance Test')
        self._git(root, 'config', 'user.email', 'm4-provenance@example.invalid')
        self._git(root, 'config', 'core.autocrlf', 'false')
        self._git(root, 'config', 'core.filemode', 'false')

        experiment = Path('experiments/geotransformer.pointct.baseline_v1')
        self._write(
            root,
            experiment / 'protocols/m3_6b_5fold_clean10_v2.json',
            '{}\n',
        )
        self._write(root, experiment / 'train_m3_defect.py', '# base fixture\n')
        self._git(root, 'add', '--all')
        self._git(root, 'commit', '-m', 'fixture base')
        base_commit = self._git(root, 'rev-parse', 'HEAD').stdout.strip()

        expected_base = base_commit
        if unrelated_base:
            base_tree = self._git(
                root, 'rev-parse', f'{base_commit}^{{tree}}'
            ).stdout.strip()
            expected_base = self._git(
                root, 'commit-tree', base_tree, '-m', 'unrelated fixture base'
            ).stdout.strip()

        runner = RUN_SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertEqual(runner.count(self._FROZEN_BASE), 1)
        runner = runner.replace(self._FROZEN_BASE, expected_base, 1)
        allowlisted_files = {
            Path('M4_OSSEOUS_STRENGTH_SELECTION_CLEAN10_V1_PROTOCOL.md'):
                '# fixture protocol\n',
            experiment / 'aggregate_m4_osseous_strength_selection.py':
                '# fixture aggregator\n',
            experiment / 'evaluate_m4_osseous_strength_validation.py':
                '# fixture producer\n',
            experiment / 'protocols/m4_osseous_strength_selection_clean10_v1.json':
                '{}\n',
            experiment / 'protocols/m4_osseous_strength_selection_clean10_v1.sha256':
                'fixture\n',
            Path('run_m4_osseous_strength_selection_clean10.sh'): runner,
            Path('tests/pointct/test_m4_osseous_strength_selection.py'):
                '# fixture tests\n',
        }
        for relative_path, content in allowlisted_files.items():
            self._write(root, relative_path, content)
        if unexpected_tracked_path:
            self._write(root, Path('geotransformer/unexpected_source.py'), '# forbidden\n')
        self._git(root, 'add', '--all')
        self._git(root, 'commit', '-m', 'fixture post-commit runner')
        head_commit = self._git(root, 'rev-parse', 'HEAD').stdout.strip()
        return temporary, root, base_commit, head_commit

    @staticmethod
    def _run_runner(root):
        environment = os.environ.copy()
        environment['DRY_RUN'] = '1'
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment.pop('PYTHON_BIN', None)
        return subprocess.run(
            [BASH_PATH, RUN_SCRIPT_PATH.name],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=environment,
        )

    def test_post_commit_descendant_allowlist_passes_provenance_and_reaches_sha_check(self):
        temporary, root, base_commit, head_commit = self._make_repository()
        try:
            self.assertNotEqual(head_commit, base_commit)
            result = self._run_runner(root)
            combined = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('embedded protocol_sha256', combined)
            self.assertNotIn('HEAD differs from frozen protocol', combined)
            self.assertNotIn('not an ancestor of HEAD', combined)
            self.assertNotIn('outside the frozen M4-2D allowlist', combined)
        finally:
            temporary.cleanup()

    def test_non_ancestor_base_fails_closed(self):
        temporary, root, _, _ = self._make_repository(unrelated_base=True)
        try:
            result = self._run_runner(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                'Frozen base commit is not an ancestor of HEAD',
                result.stdout + result.stderr,
            )
        finally:
            temporary.cleanup()

    def test_tracked_change_outside_allowlist_fails_closed(self):
        temporary, root, _, _ = self._make_repository(
            unexpected_tracked_path=True
        )
        try:
            result = self._run_runner(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                'Tracked path changed outside the frozen M4-2D allowlist',
                result.stdout + result.stderr,
            )
        finally:
            temporary.cleanup()

    def test_dirty_tracked_worktree_fails_closed(self):
        temporary, root, _, _ = self._make_repository()
        try:
            training_protocol = (
                root
                / 'experiments/geotransformer.pointct.baseline_v1'
                / 'protocols/m3_6b_5fold_clean10_v2.json'
            )
            training_protocol.write_text('{"dirty":true}\n', encoding='utf-8')
            result = self._run_runner(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                'Tracked worktree is dirty',
                result.stdout + result.stderr,
            )
        finally:
            temporary.cleanup()

    def test_dirty_staged_index_fails_closed(self):
        temporary, root, _, _ = self._make_repository()
        try:
            training_protocol = (
                root
                / 'experiments/geotransformer.pointct.baseline_v1'
                / 'protocols/m3_6b_5fold_clean10_v2.json'
            )
            training_protocol.write_text('{"staged":true}\n', encoding='utf-8')
            self._git(root, 'add', str(training_protocol.relative_to(root)))
            result = self._run_runner(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                'Staged index is dirty',
                result.stdout + result.stderr,
            )
        finally:
            temporary.cleanup()

    def test_wrong_branch_fails_closed(self):
        temporary, root, _, _ = self._make_repository(branch='wrong_branch')
        try:
            result = self._run_runner(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                'Branch differs from frozen protocol',
                result.stdout + result.stderr,
            )
        finally:
            temporary.cleanup()


class AggregatorCaseContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = aggregator.load_frozen_protocol(PROTOCOL_PATH)

    def test_exact_case_contract_and_patient_level_collapse(self):
        def metrics(identity, index):
            local_index = index if identity['subject_id'] == 'Pat7' else index - 15
            base = 1.0 if identity['subject_id'] == 'Pat7' else 101.0
            value = base + local_index
            return value, value + 0.5, value + 1.0

        rows = _successful_rows(self.protocol, 'Fold1', 0.25, metrics)
        validated = aggregator.validate_validation_cases(
            rows, self.protocol, 'Fold1', 0.25
        )
        self.assertEqual(len(validated), 30)
        patients = aggregator.collapse_patient_rows(
            rows, self.protocol, 'Fold1', 0.25
        )
        self.assertEqual(len(patients), 2)
        self.assertEqual([row['subject_id'] for row in patients], ['Pat7', 'Pat4'])
        self.assertEqual([row['case_count'] for row in patients], [15, 15])
        self.assertEqual(patients[0]['median_centroid_tre_mm'], 8.0)
        self.assertEqual(patients[1]['median_centroid_tre_mm'], 108.0)
        self.assertEqual(patients[0]['median_point_tre_mean_mm'], 8.5)
        self.assertEqual(patients[1]['median_rre_deg'], 109.0)

    def test_failures_are_null_and_remain_in_descriptive_denominator(self):
        rows = _successful_rows(self.protocol, 'Fold1', 0.25)
        _mark_failure(rows[0], status='no_correspondences')
        patients = aggregator.collapse_patient_rows(
            rows, self.protocol, 'Fold1', 0.25
        )
        pat7 = patients[0]
        self.assertEqual(pat7['case_count'], 15)
        self.assertEqual(pat7['solver_success_count'], 14)
        self.assertEqual(pat7['solver_failure_count'], 1)
        self.assertAlmostEqual(pat7['solver_success_rate'], 14.0 / 15.0)
        self.assertEqual(pat7['registration_recall_hits'], 14)
        self.assertTrue(pat7['candidate_complete'])
        self.assertEqual(pat7['median_centroid_tre_mm'], 1.0)

    def test_zero_success_patient_is_incomplete_with_null_metrics(self):
        rows = _successful_rows(self.protocol, 'Fold1', 0.25)
        for row in rows:
            if row['subject_id'] == 'Pat7':
                _mark_failure(row)
        patients = aggregator.collapse_patient_rows(
            rows, self.protocol, 'Fold1', 0.25
        )
        pat7 = patients[0]
        self.assertFalse(pat7['candidate_complete'])
        self.assertEqual(pat7['solver_success_count'], 0)
        self.assertIsNone(pat7['median_centroid_tre_mm'])
        self.assertIsNone(pat7['median_point_tre_mean_mm'])
        self.assertIsNone(pat7['median_rre_deg'])

    def test_rejects_duplicate_missing_and_unexpected_identities(self):
        baseline = _successful_rows(self.protocol, 'Fold1', 0.25)
        cases = []
        duplicate = copy.deepcopy(baseline)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        cases.append(('duplicate', duplicate))
        cases.append(('missing', copy.deepcopy(baseline[:-1])))
        unexpected = copy.deepcopy(baseline)
        unexpected[0]['perturbation_seed'] += 1
        cases.append(('wrong seed', unexpected))
        wrong_subject = copy.deepcopy(baseline)
        wrong_subject[0]['subject_id'] = 'Pat6'
        cases.append(('Pat6', wrong_subject))
        wrong_fold = copy.deepcopy(baseline)
        wrong_fold[0]['fold_id'] = 'Fold2'
        cases.append(('wrong fold', wrong_fold))
        wrong_lambda = copy.deepcopy(baseline)
        wrong_lambda[0]['lambda_oss'] = 0.5
        cases.append(('wrong lambda', wrong_lambda))
        wrong_variant = copy.deepcopy(baseline)
        wrong_variant[0]['variant_id'] = 1
        cases.append(('wrong variant', wrong_variant))
        for name, rows in cases:
            with self.subTest(name=name):
                with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                    aggregator.validate_validation_cases(
                        rows, self.protocol, 'Fold1', 0.25
                    )

    def test_rejects_test_scope_fields_values_and_unexpected_fields(self):
        baseline = _successful_rows(self.protocol, 'Fold1', 0.25)
        mutations = []
        test_key = copy.deepcopy(baseline)
        test_key[0]['test_metric'] = 1.0
        mutations.append(('test key', test_key))
        test_value = copy.deepcopy(baseline)
        test_value[0]['solver_status'] = 'holdout_test'
        mutations.append(('test value', test_value))
        test_scope = copy.deepcopy(baseline)
        test_scope[0]['artifact_scope'] = 'testing'
        mutations.append(('test scope', test_scope))
        benign_extra = copy.deepcopy(baseline)
        benign_extra[0]['extra_metric'] = 1.0
        mutations.append(('unexpected field', benign_extra))
        for name, rows in mutations:
            with self.subTest(name=name):
                with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                    aggregator.validate_validation_cases(
                        rows, self.protocol, 'Fold1', 0.25
                    )

    def test_success_and_failure_numeric_policies_fail_closed(self):
        baseline = _successful_rows(self.protocol, 'Fold1', 0.25)
        mutations = []
        for invalid in (float('nan'), float('inf'), -float('inf'), -1.0, True, None):
            rows = copy.deepcopy(baseline)
            rows[0]['centroid_tre_mm'] = invalid
            mutations.append((f'invalid success metric {invalid!r}', rows))
        rows = copy.deepcopy(baseline)
        rows[0]['solver_status'] = 'failed'
        mutations.append(('success wrong status', rows))
        rows = copy.deepcopy(baseline)
        _mark_failure(rows[0])
        rows[0]['centroid_tre_mm'] = 1.0
        mutations.append(('failure non-null metric', rows))
        rows = copy.deepcopy(baseline)
        _mark_failure(rows[0], status='success')
        mutations.append(('failure success status', rows))
        rows = copy.deepcopy(baseline)
        _mark_failure(rows[0])
        rows[0]['registration_recall_hit'] = True
        mutations.append(('failure recall hit', rows))
        for name, candidate in mutations:
            with self.subTest(name=name):
                with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                    aggregator.validate_validation_cases(
                        candidate, self.protocol, 'Fold1', 0.25
                    )

    def test_jsonl_loader_rejects_duplicate_keys_blank_lines_and_nan(self):
        fixtures = {
            'duplicate': '{"artifact_scope":"validation_only","artifact_scope":"validation_only"}\n',
            'blank': '{}\n\n',
            'nan': '{"value":NaN}\n',
            'non_object': '[]\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, content in fixtures.items():
                path = Path(directory) / f'{name}.jsonl'
                path.write_text(content, encoding='utf-8')
                with self.subTest(name=name):
                    with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                        aggregator._load_jsonl(path, name)


class AggregatorSelectionRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = aggregator.load_frozen_protocol(PROTOCOL_PATH)

    def test_primary_tie_is_anchored_to_stage_min_not_pairwise_chained(self):
        summaries = _selection_summaries(
            {
                0.25: (1.00, 1.30, 1.0),
                0.5: (1.24, 1.00, 1.0),
                1.0: (1.48, 0.50, 0.5),
                2.0: (2.00, 0.25, 0.25),
            }
        )
        decision = aggregator.apply_selection_rule(summaries, self.protocol)
        self.assertEqual(decision['trace']['primary_survivors'], [0.25, 0.5])
        self.assertEqual(decision['trace']['secondary_survivors'], [0.5])
        self.assertEqual(decision['trace']['tertiary_survivors'], [0.5])
        self.assertEqual(decision['selected_lambda_oss'], 0.5)

    def test_exact_threshold_is_kept_and_value_above_threshold_is_rejected(self):
        summaries = _selection_summaries(
            {
                0.25: (1.0, 1.0, 1.0),
                0.5: (1.25, 1.0, 1.0),
                1.0: (1.2500001, 0.0, 0.0),
                2.0: (3.0, 0.0, 0.0),
            }
        )
        decision = aggregator.apply_selection_rule(summaries, self.protocol)
        self.assertEqual(decision['trace']['primary_survivors'], [0.25, 0.5])
        self.assertEqual(decision['selected_lambda_oss'], 0.25)

    def test_secondary_then_tertiary_filters_only_prior_survivors(self):
        summaries = _selection_summaries(
            {
                0.25: (1.0, 1.26, 0.0),
                0.5: (1.1, 1.00, 1.26),
                1.0: (1.2, 1.25, 1.00),
                2.0: (2.0, 0.0, 0.0),
            }
        )
        decision = aggregator.apply_selection_rule(summaries, self.protocol)
        self.assertEqual(decision['trace']['primary_survivors'], [0.25, 0.5, 1.0])
        self.assertEqual(decision['trace']['secondary_survivors'], [0.5, 1.0])
        self.assertEqual(decision['trace']['tertiary_survivors'], [1.0])
        self.assertEqual(decision['selected_lambda_oss'], 1.0)

    def test_all_tied_selects_smallest_lambda_independent_of_input_order(self):
        summaries = _selection_summaries(
            {value: (1.0, 2.0, 3.0) for value in FROZEN_LAMBDAS}
        )
        summaries.reverse()
        decision = aggregator.apply_selection_rule(summaries, self.protocol)
        self.assertEqual(decision['selected_lambda_oss'], 0.25)
        self.assertEqual(
            decision['trace']['tertiary_survivors'], list(FROZEN_LAMBDAS)
        )

    def test_incomplete_candidate_forbids_selection(self):
        summaries = _selection_summaries(
            {value: (1.0, 1.0, 1.0) for value in FROZEN_LAMBDAS},
            incomplete=(1.0,),
        )
        decision = aggregator.apply_selection_rule(summaries, self.protocol)
        self.assertFalse(decision['eligible'])
        self.assertIsNone(decision['selected_lambda_oss'])
        self.assertIsNone(decision['trace'])
        self.assertTrue(
            all('practical_tie_status' in summary for summary in summaries)
        )

    def test_selection_rejects_duplicate_or_non_finite_candidate_metrics(self):
        duplicate = _selection_summaries(
            {value: (1.0, 1.0, 1.0) for value in FROZEN_LAMBDAS}
        )
        duplicate[-1]['lambda_oss'] = 1.0
        invalid = _selection_summaries(
            {value: (1.0, 1.0, 1.0) for value in FROZEN_LAMBDAS}
        )
        invalid[0]['selection_metrics']['centroid_tre_mm'] = float('nan')
        for name, summaries in (('duplicate', duplicate), ('nan', invalid)):
            with self.subTest(name=name):
                with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                    aggregator.apply_selection_rule(summaries, self.protocol)


class AggregatorGridAndOutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = aggregator.load_frozen_protocol(PROTOCOL_PATH)

    def test_complete_grid_reports_patient_fold_and_descriptive_statistics(self):
        result = aggregator.aggregate_grid_rows(
            _full_grid(self.protocol), self.protocol
        )
        self.assertEqual(result['artifact_scope'], 'validation_only')
        self.assertEqual(result['grid_shape'], {'lambda_count': 4, 'fold_count': 5})
        self.assertEqual(result['case_count'], 600)
        self.assertTrue(result['selection_eligible'])
        self.assertIn('SELECTED_LAMBDA_OSS', result)
        self.assertEqual(len(result['lambda_summaries']), 4)
        for summary in result['lambda_summaries']:
            self.assertEqual(summary['unique_patient_count'], 10)
            self.assertEqual(summary['case_count'], 150)
            self.assertEqual(len(summary['patient_rows']), 10)
            self.assertEqual(len(summary['fold_summaries']), 5)
            self.assertEqual(
                {row['subject_id'] for row in summary['patient_rows']},
                set(self.protocol['clean10']['ready_subject_ids']),
            )
            self.assertIn('practical_tie_status', summary)
            for fold_summary in summary['fold_summaries']:
                self.assertEqual(fold_summary['patient_count'], 2)
                self.assertEqual(fold_summary['case_count'], 30)
            for metric in ('centroid_tre_mm', 'point_tre_mean_mm', 'rre_deg'):
                statistics = summary['metrics'][metric]
                self.assertEqual(
                    set(statistics),
                    {'mean', 'median', 'p95_linear', 'best_patient', 'worst_patient'},
                )

    def test_linear_p95_is_explicit_interpolation(self):
        self.assertAlmostEqual(aggregator._p95_linear(range(10)), 8.55)
        self.assertEqual(aggregator._p95_linear([4.0]), 4.0)

    def test_missing_or_extra_grid_entry_fails_closed(self):
        grid = _full_grid(self.protocol)
        grid.pop((2.0, 'Fold5'))
        with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
            aggregator.aggregate_grid_rows(grid, self.protocol)
        grid = _full_grid(self.protocol)
        grid[(3.0, 'Fold1')] = grid[(2.0, 'Fold1')]
        with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
            aggregator.aggregate_grid_rows(grid, self.protocol)

    def test_zero_success_patient_omits_uppercase_selected_key(self):
        grid = _full_grid(self.protocol)
        for row in grid[(0.25, 'Fold1')]:
            if row['subject_id'] == 'Pat7':
                _mark_failure(row)
        result = aggregator.aggregate_grid_rows(grid, self.protocol)
        self.assertFalse(result['selection_eligible'])
        self.assertEqual(result['selection_reason'], 'one_or_more_zero_success_patients')
        self.assertNotIn('SELECTED_LAMBDA_OSS', result)
        self.assertIsNone(result['selection_trace'])

    def test_result_root_and_summary_output_are_isolated_and_non_overwriting(self):
        expected_root = PROJECT_ROOT / 'checkpoints' / 'm4_osseous_strength_selection_v1'
        self.assertEqual(aggregator.frozen_output_root(self.protocol), expected_root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                aggregator.load_validation_grid(self.protocol, root)
            with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                aggregator._atomic_write_summary(root / 'wrong.json', {}, root)
            output = root / aggregator.SUMMARY_BASENAME
            aggregator._atomic_write_summary(output, {'artifact_scope': 'validation_only'}, root)
            self.assertTrue(output.is_file())
            with self.assertRaises(aggregator.M4OsseousStrengthAggregationError):
                aggregator._atomic_write_summary(output, {'artifact_scope': 'validation_only'}, root)

    def test_producer_fold_path_requires_exact_frozen_isolation(self):
        expected_root = PROJECT_ROOT / 'checkpoints' / 'm4_osseous_strength_selection_v1'
        lambda_dirs = self.protocol['output_layout']['lambda_directories']
        fold_dirs = self.protocol['output_layout']['fold_directories']
        for lambda_oss in FROZEN_LAMBDAS:
            lambda_dir = next(
                value
                for key, value in lambda_dirs.items()
                if float(key) == lambda_oss
            )
            for index, fold_id in enumerate(FROZEN_FOLDS):
                expected = expected_root / lambda_dir / fold_dirs[index]
                paths = producer._fold_paths(
                    self.protocol, fold_id, lambda_oss, expected
                )
                self.assertEqual(paths['fold_dir'], expected)
                self.assertEqual(paths['checkpoint'].name, 'best_val_loss.pt')
                self.assertEqual(paths['validation_cases'].name, 'validation_cases.jsonl')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(producer.M4OsseousStrengthValidationContractError):
                producer._fold_paths(
                    self.protocol, 'Fold1', 0.25, Path(directory) / 'fold1'
                )


if __name__ == '__main__':
    unittest.main()
