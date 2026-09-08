"""Future CPU aggregation and paired comparison. No result reads at import time."""

import argparse
import math
import statistics
from pathlib import Path

import m4_osseous_formal_protocol as contract
import bind_m4_osseous_formal_checkpoints as checkpoints


def stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p95_linear': None}
    for value in values:
        contract.finite(value, 'summary value', nonnegative=False)
    position = .95 * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95_linear': values[lower] + (position - lower) * (values[upper] - values[lower])}


def group_summary(rows):
    n = len(rows)
    success = sum(row['solver_success'] for row in rows)
    recall = sum(row['registration_recall_hit'] for row in rows)
    return {'case_count': n, 'solver_success_count': success, 'solver_failure_count': n-success,
            'solver_success_rate': success/n, 'registration_recall_count': recall, 'registration_recall': recall/n,
            'metrics': {metric: stats(row[metric] for row in rows)
                        for metric in (*contract.ERROR_METRICS, *contract.DESCRIPTIVE)}}


def summarize(rows):
    if not rows:
        raise contract.ContractError('cannot summarize empty grid')
    patients = {key: group_summary([row for row in rows if row['subject_id'] == key])
                for key in sorted({row['subject_id'] for row in rows})}
    return {'artifact_scope': 'formal_test', 'protocol_sha256': contract.PROTOCOL_SHA,
            'patient_level': patients,
            'fold_level': {key: group_summary([row for row in rows if row['fold_id'] == key])
                           for key in sorted({row['fold_id'] for row in rows})},
            'overall_descriptive': group_summary(rows),
            'patient_macro_medians': {metric: stats(p['metrics'][metric]['median'] for p in patients.values())
                                      for metric in contract.ERROR_METRICS},
            'metric_missingness': 'error metrics use solver-success cases only; count retained; zero-success patients remain null',
            'legacy_rte_caveat': 'translation parameter error is origin-sensitive; retain historical Recall comparability'}


def paired_comparison(m3, m4, sources):
    expected = {contract.identity(row) for row in contract.manifest(sources)}
    def unique(rows):
        mapped = {contract.identity(row): row for row in rows}
        if len(mapped) != len(rows) or set(mapped) != expected:
            raise contract.ContractError('paired comparison requires exact complete M3/M4 identity match')
        return mapped
    baseline, proposed = unique(m3), unique(m4)
    def group(keys):
        pairs = [(baseline[key], proposed[key]) for key in sorted(keys)]
        both = [(a, b) for a, b in pairs if a['solver_success'] and b['solver_success']]
        n = len(pairs)
        return {'case_count': n, 'both_solver_success_count': len(both),
                'm3_only_success_count': sum(a['solver_success'] and not b['solver_success'] for a, b in pairs),
                'm4_only_success_count': sum(b['solver_success'] and not a['solver_success'] for a, b in pairs),
                'neither_success_count': sum(not a['solver_success'] and not b['solver_success'] for a, b in pairs),
                'solver_success_rate_delta': sum(int(b['solver_success'])-int(a['solver_success']) for a, b in pairs)/n,
                'registration_recall_delta': sum(int(b['registration_recall_hit'])-int(a['registration_recall_hit']) for a, b in pairs)/n,
                'paired_error_delta': {metric: stats(b[metric]-a[metric] for a, b in both)
                                       for metric in contract.ERROR_METRICS}}
    patients = {patient: group([key for key in expected if baseline[key]['subject_id'] == patient])
                for patient in sources['clean10']['ready_subject_ids']}
    return {'artifact_scope': 'formal_test_paired_comparison', 'protocol_sha256': contract.PROTOCOL_SHA,
            'delta_direction': 'M4 minus M3',
            'paired_error_policy': 'both-success exact pairs only; all missingness counts retained; no error imputation',
            'M3': {**summarize(m3), 'method': 'M3_baseline',
                   'source_evaluation_protocol_sha256': sources['evaluation']['evaluation_protocol_hash']},
            'M4': {**summarize(m4), 'method': 'M4_fixed_half',
                   'source_formal_protocol_sha256': contract.PROTOCOL_SHA},
            'patient_level_paired': patients,
            'fold_level_paired': {fold: group([key for key in expected if baseline[key]['fold_id'] == fold])
                                  for fold in sources['clean10']['folds']},
            'overall_paired_descriptive': group(expected),
            'patient_macro_paired_medians': {metric: stats(p['paired_error_delta'][metric]['median'] for p in patients.values())
                                             for metric in contract.ERROR_METRICS}}


def _read_jsonl(path):
    path = Path(path)
    # For future external M3 roots, reject linklike ancestors too.
    for component in (path, *path.parents):
        if component.is_symlink() or (component.exists() and getattr(component.lstat(), 'st_file_attributes', 0) & 0x400):
            raise contract.ContractError('linklike result input')
    raw = path.read_bytes()
    rows = [contract.parse(line) for line in raw.splitlines()]
    return rows, contract.sha(raw)


def read_m4(sources, binding):
    rows, hashes = [], {}
    for fold in sources['clean10']['folds']:
        relative = contract.OUTPUT_ROOT + fold.lower() + '/formal_test_cases.jsonl'
        current, hashes[relative] = _read_jsonl(contract.safe_path(relative))
        contract.validate_rows(current, sources, fold, binding)
        rows.extend(current)
    contract.validate_rows(rows, sources, binding=binding)
    return rows, hashes


def normalize_m3(legacy_rows, diagnostic_rows, sources):
    """Validate original M3 provenance and join independent TRE diagnostics."""
    expected = {contract.identity(row) for row in contract.manifest(sources)}
    def unique(rows):
        mapped = {contract.identity(row): row for row in rows}
        if len(mapped) != len(rows) or set(mapped) != expected:
            raise contract.ContractError('M3 baseline/diagnostic must match exact frozen identity grid')
        return mapped
    legacy, diagnostic = unique(legacy_rows), unique(diagnostic_rows)
    clean, evaluation, tre = sources['clean10'], sources['evaluation'], sources['diagnostic']
    normalized = []
    for key in sorted(expected):
        a, b = legacy[key], diagnostic[key]
        case_key = '|'.join(str(a[field]) for field in contract.IDENTITY[:-1])
        contract.equal(a.get('case_key'), case_key, 'M3 case key')
        contract.equal(b.get('case_key'), case_key, 'M3 diagnostic case key')
        contract.canonical(a)
        contract.canonical(b)
        for field, value in {'protocol_version': clean['protocol_version'], 'protocol_hash': clean['protocol_hash'],
                             'evaluation_protocol_version': evaluation['evaluation_protocol_version'],
                             'evaluation_protocol_hash': evaluation['evaluation_protocol_hash']}.items():
            contract.equal(a.get(field), value, 'M3 legacy provenance')
        for field, value in {'diagnostic_protocol_version': tre['diagnostic_protocol_version'],
                             'diagnostic_protocol_hash': tre['diagnostic_protocol_hash'],
                             'source_training_protocol_version': clean['protocol_version'],
                             'source_training_protocol_hash': clean['protocol_hash'],
                             'source_defect_evaluation_protocol_version': evaluation['evaluation_protocol_version'],
                             'source_defect_evaluation_protocol_hash': evaluation['evaluation_protocol_hash']}.items():
            contract.equal(b.get(field), value, 'M3 diagnostic provenance')
        for field in ('solver_success', 'solver_status', 'registration_recall_hit', 'num_correspondences', 'num_inliers'):
            contract.equal(a[field], b[field], 'M3 legacy/diagnostic ' + field)
        for field, diagnostic_field in [('rre_deg', 'rre_deg'), ('rte_mm', 'legacy_parameter_rte_mm'), ('inlier_ratio', 'inlier_ratio')]:
            x, y = a[field], b[diagnostic_field]
            if x is None or y is None:
                contract.equal(x, y, 'M3 null metric')
            elif not math.isclose(x, y, rel_tol=tre['legacy_continuous_rel_tolerance'], abs_tol=tre['legacy_continuous_abs_tolerance']):
                raise contract.ContractError('M3 original vs diagnostic metric disagreement')
        row = {field: a[field] for field in (*contract.IDENTITY, 'solver_success', 'solver_status',
                                            'registration_recall_hit', *contract.DESCRIPTIVE, 'rre_deg', 'rte_mm')}
        row.update({field: b[field] for field in contract.ERROR_METRICS if field not in ('rre_deg', 'rte_mm')})
        # Shared numeric/failure checks without assigning fictitious M4 provenance to M3.
        if type(row['solver_success']) is not bool or type(row['registration_recall_hit']) is not bool:
            raise contract.ContractError('invalid M3 booleans')
        for field in contract.DESCRIPTIVE:
            contract.finite(row[field], field)
        n, hits = row['num_correspondences'], row['num_inliers']
        if type(n) is not int or type(hits) is not int or hits > n:
            raise contract.ContractError('invalid M3 correspondence counts')
        contract.equal(row['inlier_ratio'], hits/n if n else 0., 'M3 ratio')
        if row['solver_success']:
            if n < 3 or row['solver_status'] != 'success':
                raise contract.ContractError('invalid M3 solver success')
            for field in contract.ERROR_METRICS:
                contract.finite(row[field], field)
            contract.equal(row['registration_recall_hit'],
                           row['rre_deg'] <= evaluation['registration_rre_threshold_deg'] and row['rte_mm'] <= evaluation['registration_rte_threshold_mm'], 'M3 Recall')
        elif row['registration_recall_hit'] or row['solver_status'] == 'success' or any(row[field] is not None for field in contract.ERROR_METRICS):
            raise contract.ContractError('invalid M3 solver failure metrics')
        normalized.append(row)
    return normalized


def aggregate(compare=False, legacy_root=None, diagnostic_root=None):
    _, sources = contract.load_protocol()
    binding = checkpoints.read_binding(sources)
    output = contract.OUTPUT_ROOT + ('m3_vs_m4_paired_comparison.json' if compare else 'formal_test_summary.json')
    if contract.safe_path(output).exists():
        raise contract.ContractError('formal output exists; refusing overwrite')
    rows, hashes = read_m4(sources, binding)
    if compare:
        legacy, diagnostic = [], []
        for fold in sources['clean10']['folds']:
            for root, basename, destination in ((legacy_root, 'cases.jsonl', legacy),
                                                 (diagnostic_root, 'diagnostic_cases.jsonl', diagnostic)):
                path = Path(root) / fold / basename
                current, checksum = _read_jsonl(path)
                expected = {contract.identity(row) for row in contract.manifest(sources, fold)}
                if len(current) != len(expected) or {contract.identity(row) for row in current} != expected:
                    raise contract.ContractError('M3 file does not match its exact frozen fold')
                hashes[str(path)] = checksum
                destination.extend(current)
        result = paired_comparison(normalize_m3(legacy, diagnostic, sources), rows, sources)
    else:
        result = summarize(rows)
    result.update(checkpoint_binding_sha256=binding['binding_sha256'], input_artifact_sha256s=hashes)
    contract.write_json_once(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--aggregate', action='store_true')
    mode.add_argument('--compare', action='store_true')
    parser.add_argument('--m3-legacy-root')
    parser.add_argument('--m3-diagnostic-root')
    args = parser.parse_args(argv)
    if args.compare != bool(args.m3_legacy_root and args.m3_diagnostic_root) or (args.aggregate and (args.m3_legacy_root or args.m3_diagnostic_root)):
        parser.error('--compare requires both M3 roots; --aggregate permits neither')
    print(contract.canonical(aggregate(args.compare, args.m3_legacy_root, args.m3_diagnostic_root)))


if __name__ == '__main__':
    main()
