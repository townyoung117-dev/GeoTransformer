"""Standard-library-only frozen formal contract. Importing never accesses data."""

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'experiments/geotransformer.pointct.baseline_v1/'
PROTOCOL_DIR = EXPERIMENT + 'protocols/'
VERSION = 'm4_osseous_formal_test_clean10_v1'
PROTOCOL_PATH = PROTOCOL_DIR + VERSION + '.json'
PROTOCOL_SHA = 'e33c0c4721e1dc303b9f5211b3f7ca2854f15e7933ae6ebd7a14303cd6c67f22'
IMPLEMENTATION_COMMIT = 'f5bc5ab74a5d095ff2f26a8e35b914057731cda4'
BINDING_PATH = PROTOCOL_DIR + VERSION + '_checkpoint_binding.json'
OUTPUT_ROOT = 'checkpoints/' + VERSION + '/'
SOURCE_SPECS = {
    'clean10': ('m3_6b_5fold_clean10_v2', 'protocol_hash', 'protocol_version',
                '34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'),
    'evaluation': ('m3_defect_eval_clean10_v2', 'evaluation_protocol_hash', 'evaluation_protocol_version',
                   'cfd519b923be3b623cffec8c8f5830cb5b1160859461180eddeb7134a0adb6b0'),
    'diagnostic': ('m3_metric_diagnostic_clean10_v2', 'diagnostic_protocol_hash', 'diagnostic_protocol_version',
                   '5f40c322433c2274d356213969f9041d2931ff0e058efb0c541bcf5e1a9624bc'),
    'selection_v1': ('m4_osseous_strength_selection_clean10_v1', 'protocol_sha256', 'protocol_version',
                     'a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38'),
    'amendment_v2': ('m4_osseous_strength_selection_clean10_v2_amendment', 'protocol_sha256', 'protocol_version',
                     'fe59f307eca6355d835fad5a9a5d18f8b7f24b940747700149b7a5668413e13c'),
}
IDENTITY = ('fold_id', 'subject_id', 'defect_id', 'severity', 'variant_id', 'perturbation_seed')
ERROR_METRICS = ('rre_deg', 'rte_mm', 'centroid_tre_mm', 'point_tre_mean_mm',
                 'point_tre_median_mm', 'point_tre_rmse_mm', 'point_tre_p95_mm', 'point_tre_max_mm')
DESCRIPTIVE = ('num_correspondences', 'num_inliers', 'inlier_ratio', 'inference_runtime_ms')
ROW_FIELDS = set(IDENTITY) | set(ERROR_METRICS) | set(DESCRIPTIVE) | {
    'case_key', 'artifact_scope', 'protocol_version', 'protocol_sha256',
    'checkpoint_binding_sha256', 'checkpoint_sha256', 'lambda_oss',
    'solver_success', 'solver_status', 'registration_recall_hit',
}


class ContractError(RuntimeError):
    pass


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ContractError('nonfinite or non-JSON value') from exc


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value, field=None):
    value = dict(value) if field else value
    if field:
        value.pop(field, None)
    return sha(canonical(value).encode())


def equal(actual, expected, name):
    if canonical(actual) != canonical(expected):
        raise ContractError(f'{name} mismatch')


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError('duplicate JSON field')
            result[key] = value
        return result
    def constant(token):
        raise ContractError(f'nonfinite JSON token: {token}')
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)
        return value
    except (ValueError, UnicodeError) as exc:
        raise ContractError('invalid JSON') from exc


def safe_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ContractError('path must be repository-relative without traversal')
    path = ROOT
    for part in ('', *relative.parts):
        path = path / part
        if path.is_symlink():
            raise ContractError('symlink forbidden')
        if path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ContractError('reparse point forbidden')
    return path


def read_file(relative):
    path = safe_path(relative)
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ContractError(f'required regular file missing: {relative}')
    return path.read_bytes()


def write_once(relative, raw):
    path = safe_path(relative)
    if path.exists():
        raise ContractError(f'output exists; refusing overwrite: {relative}')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.m4_formal_', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        safe_path(relative)
        os.link(temporary, path)
    except OSError as exc:
        raise ContractError('atomic no-clobber publication failed') from exc
    finally:
        Path(temporary).unlink()


def write_json_once(relative, value):
    write_once(relative, (canonical(value) + '\n').encode())


def load_sources():
    result = {}
    for name, (version, field, version_field, anchor) in SOURCE_SPECS.items():
        value = parse(read_file(PROTOCOL_DIR + version + '.json'))
        equal(value.get(version_field), version, name + ' version')
        equal(value.get(field), anchor, name + ' stored SHA')
        equal(digest(value, field), anchor, name + ' computed SHA')
        result[name] = value
    return result


def expected_model(sources):
    settings = sources['selection_v1']['training_settings']
    return {'hard_enabled': True, 'soft_enabled': True, 'osseous_enabled': True,
            'soft_sigma_mm': settings['m4_soft']['sigma_mm'],
            'soft_strength': settings['m4_soft']['strength'],
            'osseous_center_hu': settings['osseous_score']['center_hu'],
            'osseous_tau_hu': settings['osseous_score']['tau_hu'],
            'osseous_radius_mm': settings['osseous_score']['radius_mm'], 'lambda_oss': 0.5}


def manifest(sources, fold=None):
    """Exact M3 identity/seed payload; no PRNG sampling, dataset or sample I/O."""
    clean = sources['clean10']
    rows = []
    if fold is not None and fold not in clean['folds']:
        raise ContractError('unknown fold')
    for fold_id, mapping in clean['folds'].items():
        if fold is not None and fold_id != fold:
            continue
        for patient in mapping['test_subject_ids']:
            for defect in sources['evaluation']['required_defect_ids']:
                for spec in clean['test_perturbations']:
                    for variant in range(spec['variant_count']):
                        seed_payload = {
                            'scheme_version': clean['seed_scheme_version'],
                            'protocol_hash': clean['protocol_hash'], 'fold_id': fold_id,
                            'root_seed': clean['perturbation_root_seed'],
                            'purpose': spec['purpose'], 'epoch': None,
                            'subject_id': patient, 'severity': spec['severity'], 'variant_id': variant,
                        }
                        seed = int.from_bytes(hashlib.sha256(canonical(seed_payload).encode()).digest()[:8], 'big')
                        rows.append({'fold_id': fold_id, 'subject_id': patient, 'defect_id': defect,
                                     'severity': spec['severity'], 'variant_id': variant,
                                     'perturbation_seed': seed,
                                     'case_key': f'{fold_id}|{patient}|{defect}|{spec["severity"]}|{variant}',
                                     'max_rotation_deg': spec['max_rotation_deg'],
                                     'max_translation_mm': spec['max_translation_mm']})
    return rows


def checkpoint_paths(sources):
    return {fold: 'checkpoints/m4_osseous_strength_selection_v1/lambda_0p50/'
            + fold.lower() + '/checkpoints/best_val_loss.pt' for fold in sources['clean10']['folds']}


def validate_protocol(protocol, sources):
    equal(protocol.get('protocol_sha256'), PROTOCOL_SHA, 'formal stored SHA')
    equal(digest(protocol, 'protocol_sha256'), PROTOCOL_SHA, 'formal computed SHA')
    equal(protocol['protocol_status'], 'FROZEN_BEFORE_FIRST_M4_FORMAL_TEST', 'freeze status')
    equal(protocol['FORMAL_TEST_ACCESSED_AT_FREEZE'], False, 'freeze firewall')
    equal(protocol['model'], expected_model(sources), 'fixed model')
    equal(protocol['folds'], sources['clean10']['folds'], 'inherited folds')
    equal(protocol['defect_conditions'], sources['evaluation']['required_defect_ids'], 'inherited defects')
    equal(protocol['test_perturbations'], sources['clean10']['test_perturbations'], 'inherited perturbations')
    equal(protocol['test_case_count'], sources['diagnostic']['expected_test_case_count'], 'inherited count')
    equal(protocol['test_case_count'], len(manifest(sources)), 'derived count')
    equal(protocol['manifest_sha256'], digest(manifest(sources)), 'manifest identities/seeds')
    equal(protocol['checkpoint_paths'], checkpoint_paths(sources), 'checkpoint paths')
    equal(protocol['source_protocols'], {name: {'version': spec[0], 'sha256': spec[3]}
                                        for name, spec in SOURCE_SPECS.items()}, 'source bindings')
    return protocol


def load_protocol():
    sources = load_sources()
    protocol = validate_protocol(parse(read_file(PROTOCOL_PATH)), sources)
    sidecar = read_file(PROTOCOL_PATH.replace('.json', '.sha256')).decode().strip()
    equal(sidecar, f'{PROTOCOL_SHA}  {VERSION}.json (canonical JSON excluding protocol_sha256)', 'sidecar')
    for relative, expected in protocol['implementation_source_sha256s'].items():
        equal(sha(read_file(relative).replace(b'\r\n', b'\n')), expected, 'inherited implementation source')
    return protocol, sources


def require_committed(relative):
    """Require the current bytes, after Git EOL normalization, in HEAD (not just index)."""
    raw = read_file(relative)
    try:
        stored = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT,
                                         stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ContractError(f'must be frozen in Git HEAD before execution: {relative}') from exc
    equal(sha(raw.replace(b'\r\n', b'\n')), sha(stored.replace(b'\r\n', b'\n')),
          'Git-tracked frozen bytes')


def identity(row):
    return canonical([row[key] for key in IDENTITY])


def finite(value, name, *, nonnegative=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ContractError(f'{name} must be a finite number')


def validate_rows(rows, sources, fold=None, binding=None):
    expected = {identity(row): row for row in manifest(sources, fold)}
    if len(rows) != len(expected):
        raise ContractError('missing or extra formal identities')
    seen = set()
    for row in rows:
        canonical(row)
        if set(row) != ROW_FIELDS:
            raise ContractError('unexpected/missing formal case fields')
        key = identity(row)
        if key not in expected or key in seen:
            raise ContractError('unexpected/duplicate formal identity')
        seen.add(key)
        equal(row['case_key'], expected[key]['case_key'], 'case key')
        for name, value in {'artifact_scope': 'formal_test', 'protocol_version': VERSION,
                            'protocol_sha256': PROTOCOL_SHA, 'lambda_oss': 0.5}.items():
            equal(row[name], value, name)
        for key_name in ('checkpoint_binding_sha256', 'checkpoint_sha256'):
            if not isinstance(row[key_name], str) or len(row[key_name]) != 64 or any(c not in '0123456789abcdef' for c in row[key_name]):
                raise ContractError('invalid checkpoint provenance hash')
        if binding:
            equal(row['checkpoint_binding_sha256'], binding['binding_sha256'], 'case binding')
            equal(row['checkpoint_sha256'], binding['folds'][row['fold_id']]['checkpoint_sha256'], 'case checkpoint')
        if type(row['solver_success']) is not bool or type(row['registration_recall_hit']) is not bool:
            raise ContractError('case success/recall must be boolean')
        if not isinstance(row['solver_status'], str) or not row['solver_status'].strip():
            raise ContractError('missing solver status')
        for metric in DESCRIPTIVE:
            finite(row[metric], metric)
        n, hits = row['num_correspondences'], row['num_inliers']
        if type(n) is not int or type(hits) is not int or hits > n:
            raise ContractError('invalid correspondence counts')
        equal(row['inlier_ratio'], hits/n if n else 0.0, 'inlier ratio')
        if row['solver_success']:
            if n < 3 or row['solver_status'] != 'success':
                raise ContractError('invalid solver success')
            for metric in ERROR_METRICS:
                finite(row[metric], metric)
            evaluation = sources['evaluation']
            recall = (row['rre_deg'] <= evaluation['registration_rre_threshold_deg']
                      and row['rte_mm'] <= evaluation['registration_rte_threshold_mm'])
            equal(row['registration_recall_hit'], recall, 'frozen Recall thresholds')
        else:
            if row['solver_status'] == 'success' or row['registration_recall_hit'] or any(row[m] is not None for m in ERROR_METRICS):
                raise ContractError('failed solver must retain null error metrics and false Recall')
    return rows


def contract_audit():
    protocol, sources = load_protocol()
    return {'CPU_CONTRACT_AUDIT': 'PASS', 'FORMAL_PROTOCOL_SHA': PROTOCOL_SHA,
            'SOURCE_V2_AMENDMENT_SHA': SOURCE_SPECS['amendment_v2'][3],
            'SELECTED_LAMBDA_OSS': protocol['model']['lambda_oss'],
            'TEST_GRID_CASE_COUNT': len(manifest(sources)),
            'CHECKPOINT_BINDING': 'PENDING_SERVER_CPU_AUDIT',
            'FORMAL_TEST_ACCESSED': False, 'GPU_USED': False,
            'reads': 'frozen protocols and implementation source only'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--contract-audit', action='store_true', required=True)
    parser.parse_args(argv)
    print(canonical(contract_audit()))


if __name__ == '__main__':
    main()
