"""CPU-only, fixed-path post-hoc amendment of the frozen V1 validation grid.

No dataset, model, training/evaluation entrypoint, or caller-supplied path is
accepted. V1's immutable identity validation and patient collapse are reused;
only the candidate eligibility policy changes. Synthetic rows can be evaluated
in memory, but only verified, byte-bound disk inputs can produce the artifact.
"""

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path

import aggregate_m4_osseous_strength_selection as v1


AMENDMENT_VERSION = 'm4_osseous_strength_selection_clean10_v2_amendment'
AMENDMENT_STATUS = 'POST_HOC_VALIDATION_ONLY_AMENDMENT_FROZEN_BEFORE_FORMAL_TEST'
AMENDMENT_SHA256 = 'fe59f307eca6355d835fad5a9a5d18f8b7f24b940747700149b7a5668413e13c'
SOURCE_SUMMARY_SHA256 = 'dc8ef14428481bea6dc2addaf6b555b90f999b023252dcc0bb75b33063762b42'
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_DIR = 'experiments/geotransformer.pointct.baseline_v1/protocols/'
V1_PROTOCOL = _PROTOCOL_DIR + v1.FROZEN_PROTOCOL_VERSION + '.json'
V2_PROTOCOL = _PROTOCOL_DIR + AMENDMENT_VERSION + '.json'
RESULT_ROOT = 'checkpoints/m4_osseous_strength_selection_v1/'
SOURCE_SUMMARY = RESULT_ROOT + 'validation_selection_summary.json'
OUTPUT_SUMMARY = RESULT_ROOT + 'validation_selection_v2_amendment_summary.json'
ARTIFACT_PATHS = {
    (value, fold): RESULT_ROOT + directory + '/' + fold.lower()
    + '/validation/validation_cases.jsonl'
    for value, directory in v1._FROZEN_LAMBDA_DIRECTORIES.items()
    for fold in v1._FROZEN_FOLDS
}
_READ_ALLOWLIST = frozenset([
    V1_PROTOCOL, V1_PROTOCOL.replace('.json', '.sha256'),
    V2_PROTOCOL, V2_PROTOCOL.replace('.json', '.sha256'),
    SOURCE_SUMMARY, *ARTIFACT_PATHS.values(),
])
Error = v1.M4OsseousStrengthAggregationError


def _read_fixed(relative):
    if relative not in _READ_ALLOWLIST:
        raise Error('path is outside the exact validation/protocol read allowlist')
    path = _REPO_ROOT / relative
    v1._assert_no_linklike_components(path, _REPO_ROOT, 'fixed input')
    v1._assert_contained_regular_file(path, _REPO_ROOT, 'fixed input')
    if not stat.S_ISREG(path.stat().st_mode):
        raise Error('fixed input is not a regular file')
    return path.read_bytes()


def _json(raw):
    try:
        value = json.loads(
            raw.decode('utf-8'),
            object_pairs_hook=v1._reject_duplicate_json_fields,
            parse_constant=v1._reject_nonstandard_json_number,
        )
        # Also rejects numeric overflow such as 1e999, including nested fields.
        v1._canonical_json(value)
        return value
    except (ValueError, UnicodeError) as exc:
        raise Error('invalid finite UTF-8 JSON') from exc


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(actual, expected, label):
    if v1._canonical_json(actual) != v1._canonical_json(expected):
        raise Error(f'{label} mismatch')


def _sidecar(raw, relative, digest):
    expected = (f'{digest}  {Path(relative).name} '
                '(canonical JSON excluding protocol_sha256)')
    _require(raw.decode('utf-8').strip(), expected, 'canonical sidecar')


def validate_amendment(amendment, source_protocol):
    v1.validate_frozen_protocol(source_protocol)
    if not isinstance(amendment, dict):
        raise Error('amendment must be an object')
    payload = dict(amendment)
    stored = payload.pop('protocol_sha256', None)
    _require(stored, AMENDMENT_SHA256, 'V2 trusted SHA')
    _require(v1._canonical_sha256(payload), AMENDMENT_SHA256, 'V2 computed SHA')
    for key, value in {
        'protocol_version': AMENDMENT_VERSION,
        'protocol_status': AMENDMENT_STATUS,
        'artifact_scope': 'validation_only',
        'selection_basis': 'POST_HOC_VALIDATION_ONLY',
        'source_v1_protocol_version': v1.FROZEN_PROTOCOL_VERSION,
        'source_v1_protocol_sha256': v1.FROZEN_PROTOCOL_SHA256,
        'source_v1_summary_sha256': SOURCE_SUMMARY_SHA256,
        'lambda_oss_grid': list(v1._FROZEN_LAMBDAS),
        'validation_folds': {k: list(v) for k, v in v1._FROZEN_VAL_SUBJECTS.items()},
        'expected_cases_per_fold': 30,
        'expected_cases_per_lambda': 150,
        'expected_cases_per_patient': 15,
        'clean10_patient_ids': list(v1._FROZEN_READY_SUBJECTS),
        'excluded_patient_ids': ['Pat6'],
        'inherited_selection_rule': source_protocol['selection_rule'],
    }.items():
        _require(amendment.get(key), value, key)
    _require(sorted(amendment['validation_artifact_sha256s']),
             sorted(ARTIFACT_PATHS.values()), 'exact artifact paths')
    return amendment


def load_contracts():
    source = v1.validate_frozen_protocol(_json(_read_fixed(V1_PROTOCOL)))
    _sidecar(_read_fixed(V1_PROTOCOL.replace('.json', '.sha256')),
             V1_PROTOCOL, v1.FROZEN_PROTOCOL_SHA256)
    amendment = validate_amendment(_json(_read_fixed(V2_PROTOCOL)), source)
    _sidecar(_read_fixed(V2_PROTOCOL.replace('.json', '.sha256')),
             V2_PROTOCOL, AMENDMENT_SHA256)
    return source, amendment


def _equivalent_summary(actual, expected, path='summary'):
    """Exact structure/types/counts; finite floats allow cross-runtime rounding."""
    if type(actual) is not type(expected):
        raise Error(f'{path}: type mismatch')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise Error(f'{path}: unexpected or missing fields')
        for key in expected:
            _equivalent_summary(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise Error(f'{path}: length mismatch')
        for index, (a, b) in enumerate(zip(actual, expected)):
            _equivalent_summary(a, b, f'{path}[{index}]')
    elif isinstance(expected, float):
        if not (math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise Error(f'{path}: numeric mismatch')
    elif actual != expected:
        raise Error(f'{path}: value mismatch')


def aggregate_grid_rows(grid, protocol):
    """Pure deterministic computation; synthetic inputs carry no disk provenance."""
    original = v1.aggregate_grid_rows(grid, protocol)
    candidates = original['lambda_summaries']
    complete = [row for row in candidates if row['candidate_complete']]
    trace = [{
        'stage': 0,
        'rule': 'all_10_patients_have_at_least_one_solver_success_in_15_cases',
        'survivors': [row['lambda_oss'] for row in complete],
    }]
    selected = None
    if not complete:
        reason = 'no_complete_candidates'
    elif len(complete) == 1:
        reason = 'single_complete_candidate'
        selected = complete[0]['lambda_oss']
    else:
        reason = 'original_v1_three_stage_rule_on_complete_candidates'
        survivors = complete
        for stage, metric, rule_key, threshold_key in (
            (1, 'centroid_tre_mm', 'primary', 'practical_tie_threshold_mm'),
            (2, 'point_tre_mean_mm', 'secondary', 'practical_tie_threshold_mm'),
            (3, 'rre_deg', 'tertiary', 'practical_tie_threshold_deg'),
        ):
            threshold = protocol['selection_rule'][rule_key][threshold_key]
            minimum = min(row['selection_metrics'][metric] for row in survivors)
            limit = minimum + threshold
            survivors = [row for row in survivors
                         if row['selection_metrics'][metric] <= limit]
            trace.append({'stage': stage, 'metric': metric,
                          'stage_min': minimum, 'threshold': threshold,
                          'stage_limit': limit,
                          'survivors': [row['lambda_oss'] for row in survivors]})
        selected = min(row['lambda_oss'] for row in survivors)
    trace.append({'stage': 'decision', 'reason': reason,
                  'final_tie_policy': 'smallest_lambda_oss',
                  'selected_lambda_oss': selected})
    result = {
        'artifact_scope': 'validation_only',
        'formal_test_accessed': False,
        'amendment_version': AMENDMENT_VERSION,
        'V1_SELECTION_ELIGIBLE': original['selection_eligible'],
        'V1_SELECTED_LAMBDA_OSS': original.get('SELECTED_LAMBDA_OSS'),
        'V1_SELECTION_REASON': original['selection_reason'],
        'V1_STATUS': ('PASS' if original['selection_eligible'] else 'FAILED_CLEANLY'),
        'V2_AMENDMENT_SELECTION_ELIGIBLE': selected is not None,
        'V2_SELECTED_LAMBDA_OSS': selected,
        'V2_AMENDMENT_STATUS': 'PASS' if selected is not None else 'FAILED_CLEANLY',
        'V2_SELECTION_BASIS': 'POST_HOC_VALIDATION_ONLY',
        'case_count': original['case_count'],
        'grid_shape': original['grid_shape'],
        'candidate_completeness_trace': [{
            'lambda_oss': row['lambda_oss'],
            'case_count': row['case_count'],
            'solver_success_count': row['solver_success_count'],
            'solver_failure_count': row['solver_failure_count'],
            'solver_success_rate': row['solver_success_rate'],
            'zero_success_patient_ids': row['incomplete_patient_ids'],
            'candidate_complete': row['candidate_complete'],
            'patient_rows': row['patient_rows'],
            'selection_metrics': row['selection_metrics'],
        } for row in candidates],
        'selection_trace': trace,
    }
    v1._canonical_json(result)
    return result


def load_verified_inputs():
    source, amendment = load_contracts()
    raw_summary = _read_fixed(SOURCE_SUMMARY)
    _require(_sha(raw_summary), SOURCE_SUMMARY_SHA256, 'source V1 summary SHA')
    summary = _json(raw_summary)
    v1._reject_result_firewall_tokens(summary)
    grid, hashes = {}, {}
    for key, relative in ARTIFACT_PATHS.items():
        raw = _read_fixed(relative)
        hashes[relative] = _sha(raw)
        _require(hashes[relative], amendment['validation_artifact_sha256s'][relative],
                 'validation artifact SHA')
        lines = raw.splitlines()
        if len(lines) != 30 or any(not line.strip() for line in lines):
            raise Error('each fixed artifact requires exactly 30 nonblank cases')
        grid[key] = [_json(line) for line in lines]
    recomputed = v1.aggregate_grid_rows(grid, source)
    _equivalent_summary(summary, recomputed)
    _require(summary['selection_eligible'], False, 'official V1 failed eligibility')
    _require(summary.get('SELECTED_LAMBDA_OSS'), None, 'official V1 no selection')
    _require(summary['selection_reason'], 'one_or_more_zero_success_patients',
             'official V1 failure reason')
    return source, amendment, grid, hashes


def aggregate_verified_inputs():
    source, amendment, grid, hashes = load_verified_inputs()
    result = aggregate_grid_rows(grid, source)
    result.update({
        'amendment_protocol_sha256': amendment['protocol_sha256'],
        'source_v1_protocol_version': v1.FROZEN_PROTOCOL_VERSION,
        'source_v1_protocol_sha256': v1.FROZEN_PROTOCOL_SHA256,
        'source_v1_summary_sha256': SOURCE_SUMMARY_SHA256,
        'validation_artifact_sha256s': hashes,
        'source_v1_summary_verification': 'PASS',
        'summary_float_comparison': {'relative_tolerance': 1e-12,
                                     'absolute_tolerance': 1e-12},
        'source_read_policy': 'fixed_validation_and_protocol_allowlist_only',
    })
    return result


def _check_output_absent():
    path = _REPO_ROOT / OUTPUT_SUMMARY
    v1._assert_no_linklike_components(path, _REPO_ROOT, 'V2 output')
    if path.exists():
        raise Error('V2 output already exists; refusing overwrite')
    return path


def _write_summary(result):
    """Publish a fully flushed file with an atomic no-clobber hard link.

    os.replace is intentionally avoided: it could overwrite a concurrent writer.
    A filesystem without hard-link support fails closed without a fallback.
    """
    path = _check_output_absent()
    payload = (v1._canonical_json(result) + '\n').encode('utf-8')
    descriptor, name = tempfile.mkstemp(prefix='.m4_v2_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _check_output_absent()
        os.link(name, path)
    except OSError as exc:
        raise Error('cannot publish V2 summary without overwriting') from exc
    finally:
        Path(name).unlink()


def run_contract_audit():
    source, amendment = load_contracts()
    identities = [v1.build_expected_case_identities(source, fold, value)
                  for value, fold in ARTIFACT_PATHS]
    return {'contract_audit': 'PASS', 'artifact_scope': 'validation_only',
            'formal_test_accessed': False, 'gpu_used': False,
            'result_artifacts_read': False, 'files_written': False,
            'lambda_fold_pairs': len(identities),
            'expected_case_count': sum(map(len, identities)),
            'amendment_protocol_sha256': amendment['protocol_sha256']}


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--contract-audit', action='store_true')
    modes.add_argument('--aggregate', action='store_true')
    modes.add_argument('--verify', action='store_true',
                       help='Recompute and verify the fixed V2 output without writing.')
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.contract_audit:
        result = run_contract_audit()
    else:
        if args.aggregate:
            _check_output_absent()
        result = aggregate_verified_inputs()
        if args.aggregate:
            _write_summary(result)
        else:
            path = _REPO_ROOT / OUTPUT_SUMMARY
            v1._assert_no_linklike_components(path, _REPO_ROOT, 'V2 output')
            v1._assert_contained_regular_file(path, _REPO_ROOT, 'V2 output')
            _require(_json(path.read_bytes()), result, 'V2 output recomputation')
    print(v1._canonical_json(result))
    return 0 if result.get('V2_AMENDMENT_SELECTION_ELIGIBLE', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
