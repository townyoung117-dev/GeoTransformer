"""Validate completed human labels and score the frozen M4-2B3 gate."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_m4_anatomy_verification import (  # noqa: E402
    ALLOWED_CATEGORIES,
    CLEAN10_SUBJECT_IDS,
    CONTROL_COUNT_PER_PATIENT,
    DISTRACTOR_CATEGORIES,
    HIGH_COUNT_PER_PATIENT,
    POINTS_PER_PATIENT,
    REVIEWER_FIELDS,
    TARGET_CATEGORIES,
    AnatomyPacketError,
    validate_hidden_records,
)


STATUS_PASS = 'PASS'
STATUS_FAIL = 'FAIL'
STATUS_PENDING = 'PENDING_HUMAN_REVIEW'


class AnatomyScoringError(RuntimeError):
    """Raised when the hidden key or human form fails closed validation."""


def load_hidden_key(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise AnatomyScoringError(f'Cannot read hidden key {path}: {error}') from error
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise AnatomyScoringError('Hidden key schema_version must be 1.')
    records = payload.get('records')
    if not isinstance(records, list):
        raise AnatomyScoringError('Hidden key records must be a list.')
    try:
        validate_hidden_records(records)
    except AnatomyPacketError as error:
        raise AnatomyScoringError(f'Hidden key violates protocol: {error}') from error
    return payload


def load_reviewer_rows(path: Path) -> list:
    try:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REVIEWER_FIELDS:
                raise AnatomyScoringError(
                    f'Reviewer form columns must be exactly {REVIEWER_FIELDS!r}.'
                )
            return list(reader)
    except AnatomyScoringError:
        raise
    except OSError as error:
        raise AnatomyScoringError(f'Cannot read reviewer form {path}: {error}') from error


def _hidden_index(hidden_key: Mapping) -> dict:
    records = hidden_key.get('records')
    if not isinstance(records, list):
        raise AnatomyScoringError('Hidden key records must be a list.')
    try:
        validate_hidden_records(records)
    except AnatomyPacketError as error:
        raise AnatomyScoringError(f'Hidden key violates protocol: {error}') from error
    index = {}
    for record in records:
        key = (str(record['anonymous_case']), str(record['anonymous_point']))
        if key in index:
            raise AnatomyScoringError(f'Duplicate hidden point: {key!r}.')
        index[key] = record
    return index


def validate_reviewer_rows(rows: Sequence[Mapping], hidden_key: Mapping) -> dict:
    hidden = _hidden_index(hidden_key)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise AnatomyScoringError('Reviewer rows must be a sequence.')
    seen = set()
    labels = {}
    incomplete = []
    for row_number, row in enumerate(rows, start=2):
        if not isinstance(row, Mapping) or set(row) != set(REVIEWER_FIELDS):
            raise AnatomyScoringError(f'Reviewer row {row_number} has invalid columns.')
        anonymous_case = str(row['anonymous_case']).strip()
        anonymous_point = str(row['anonymous_point']).strip()
        key = (anonymous_case, anonymous_point)
        if key in seen:
            raise AnatomyScoringError(f'Duplicate reviewer point: {anonymous_point!r}.')
        seen.add(key)
        if key not in hidden:
            raise AnatomyScoringError(f'Unknown reviewer point: {key!r}.')

        category = str(row['anatomical_category']).strip()
        confidence_text = str(row['reviewer_confidence']).strip()
        comments = str(row['comments'])
        if not category and not confidence_text:
            incomplete.append(key)
            labels[key] = {'category': None, 'confidence': None, 'comments': comments}
            continue
        if not category or not confidence_text:
            raise AnatomyScoringError(
                f'{anonymous_point} must provide category and confidence together.'
            )
        if category not in ALLOWED_CATEGORIES:
            raise AnatomyScoringError(
                f'{anonymous_point} has invalid anatomical category {category!r}.'
            )
        try:
            confidence = int(confidence_text)
        except ValueError as error:
            raise AnatomyScoringError(
                f'{anonymous_point} confidence must be an integer from 1 to 5.'
            ) from error
        if str(confidence) != confidence_text or not 1 <= confidence <= 5:
            raise AnatomyScoringError(
                f'{anonymous_point} confidence must be an integer from 1 to 5.'
            )
        labels[key] = {
            'category': category,
            'confidence': confidence,
            'comments': comments,
        }

    missing = set(hidden) - seen
    extra = seen - set(hidden)
    if missing:
        example = sorted(missing)[0]
        raise AnatomyScoringError(
            f'Reviewer form is missing {len(missing)} points; first missing point is {example!r}.'
        )
    if extra:
        raise AnatomyScoringError(f'Reviewer form contains {len(extra)} extra points.')
    if len(rows) != len(hidden):
        raise AnatomyScoringError(
            f'Reviewer form must contain exactly {len(hidden)} unique point rows.'
        )
    return {
        'labels': labels,
        'complete_point_count': len(hidden) - len(incomplete),
        'expected_point_count': len(hidden),
        'incomplete_points': [point for _, point in incomplete],
        'is_complete': not incomplete,
    }


def evaluate_frozen_gate(patient_metrics: Sequence[Mapping], complete_patient_count: int) -> dict:
    if len(patient_metrics) != len(CLEAN10_SUBJECT_IDS):
        raise AnatomyScoringError('Frozen gate requires exactly ten patient metric rows.')
    required_fields = {
        'target_fraction_high',
        'target_fraction_control',
        'target_enrichment',
        'distractor_high_fraction',
    }
    for metric in patient_metrics:
        if not required_fields.issubset(metric):
            raise AnatomyScoringError('Patient metric row is missing a frozen metric.')
        for field in required_fields:
            value = float(metric[field])
            if not math.isfinite(value):
                raise AnatomyScoringError(f'Patient metric {field} must be finite.')
    positive_count = sum(
        float(metric['target_fraction_high'])
        > float(metric['target_fraction_control'])
        for metric in patient_metrics
    )
    median_high = float(median(float(row['target_fraction_high']) for row in patient_metrics))
    median_enrichment = float(median(float(row['target_enrichment']) for row in patient_metrics))
    median_distractor = float(
        median(float(row['distractor_high_fraction']) for row in patient_metrics)
    )
    clauses = {
        'at_least_8_of_10_high_greater_than_control': positive_count >= 8,
        'median_target_fraction_high_at_least_0p50': median_high >= 0.50,
        'median_target_enrichment_at_least_0p20': median_enrichment >= 0.20,
        'median_distractor_high_fraction_at_most_0p30': median_distractor <= 0.30,
        'all_10_patients_complete_20_labels': complete_patient_count == 10,
    }
    return {
        'positive_patient_count': int(positive_count),
        'median_target_fraction_high': median_high,
        'median_target_enrichment': median_enrichment,
        'median_distractor_high_fraction': median_distractor,
        'complete_patient_count': int(complete_patient_count),
        'clauses': clauses,
        'all_clauses_pass': all(clauses.values()),
    }


def score_reviewer_rows(rows: Sequence[Mapping], hidden_key: Mapping) -> dict:
    validated = validate_reviewer_rows(rows, hidden_key)
    if not validated['is_complete']:
        return {
            'status': STATUS_PENDING,
            'human_review_required': True,
            'complete_point_count': validated['complete_point_count'],
            'expected_point_count': validated['expected_point_count'],
            'incomplete_point_count': len(validated['incomplete_points']),
            'patient_metrics': [],
            'gate': None,
        }

    hidden = _hidden_index(hidden_key)
    labels = validated['labels']
    patient_metrics = []
    for subject_id in CLEAN10_SUBJECT_IDS:
        patient = [record for record in hidden.values() if record['subject_id'] == subject_id]
        high = [record for record in patient if record['group'] == 'HIGH']
        control = [record for record in patient if record['group'] == 'CONTROL']
        if len(high) != HIGH_COUNT_PER_PATIENT or len(control) != CONTROL_COUNT_PER_PATIENT:
            raise AnatomyScoringError(f'{subject_id} hidden group counts are invalid.')

        def category(record):
            key = (str(record['anonymous_case']), str(record['anonymous_point']))
            return labels[key]['category']

        target_high = sum(category(record) in TARGET_CATEGORIES for record in high)
        target_control = sum(category(record) in TARGET_CATEGORIES for record in control)
        distractor_high = sum(category(record) in DISTRACTOR_CATEGORIES for record in high)
        fraction_high = float(target_high / HIGH_COUNT_PER_PATIENT)
        fraction_control = float(target_control / CONTROL_COUNT_PER_PATIENT)
        patient_metrics.append(
            {
                'subject_id': subject_id,
                'complete_point_count': POINTS_PER_PATIENT,
                'target_fraction_high': fraction_high,
                'target_fraction_control': fraction_control,
                'target_enrichment': float(fraction_high - fraction_control),
                'distractor_high_fraction': float(
                    distractor_high / HIGH_COUNT_PER_PATIENT
                ),
            }
        )
    gate = evaluate_frozen_gate(patient_metrics, complete_patient_count=10)
    return {
        'status': STATUS_PASS if gate['all_clauses_pass'] else STATUS_FAIL,
        'human_review_required': False,
        'complete_point_count': validated['complete_point_count'],
        'expected_point_count': validated['expected_point_count'],
        'incomplete_point_count': 0,
        'patient_metrics': patient_metrics,
        'gate': gate,
    }


def cohens_kappa(primary_rows: Sequence[Mapping], second_rows: Sequence[Mapping], hidden_key: Mapping):
    first = validate_reviewer_rows(primary_rows, hidden_key)
    second = validate_reviewer_rows(second_rows, hidden_key)
    if not first['is_complete'] or not second['is_complete']:
        return {
            'status': STATUS_PENDING,
            'cohens_kappa': None,
            'observed_agreement': None,
            'expected_agreement': None,
        }
    keys = sorted(first['labels'])
    first_values = [first['labels'][key]['category'] for key in keys]
    second_values = [second['labels'][key]['category'] for key in keys]
    count = len(keys)
    observed = sum(a == b for a, b in zip(first_values, second_values)) / count
    first_counts = Counter(first_values)
    second_counts = Counter(second_values)
    expected = sum(
        (first_counts[category] / count) * (second_counts[category] / count)
        for category in ALLOWED_CATEGORIES
    )
    if math.isclose(expected, 1.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else None
    else:
        kappa = float((observed - expected) / (1.0 - expected))
    return {
        'status': 'COMPLETE',
        'cohens_kappa': kappa,
        'observed_agreement': float(observed),
        'expected_agreement': float(expected),
        'point_count': count,
    }


def score_verification(
    primary_rows: Sequence[Mapping],
    hidden_key: Mapping,
    second_rows: Sequence[Mapping] = None,
) -> dict:
    primary = score_reviewer_rows(primary_rows, hidden_key)
    report = {
        'schema_version': 1,
        'primary_reviewer': primary,
        'structure_level_anatomy_verification': primary['status'],
        'human_review_required': primary['status'] == STATUS_PENDING,
        'target_categories': list(TARGET_CATEGORIES),
        'distractor_categories': list(DISTRACTOR_CATEGORIES),
        'primary_gate_unchanged_by_agreement': True,
    }
    if second_rows is not None:
        report['second_reviewer'] = score_reviewer_rows(second_rows, hidden_key)
        report['inter_rater_agreement'] = cohens_kappa(
            primary_rows,
            second_rows,
            hidden_key,
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-key', required=True, type=Path)
    parser.add_argument('--reviewer-form', required=True, type=Path)
    parser.add_argument('--second-reviewer-form', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        hidden_key = load_hidden_key(args.hidden_key)
        primary_rows = load_reviewer_rows(args.reviewer_form)
        second_rows = (
            load_reviewer_rows(args.second_reviewer_form)
            if args.second_reviewer_form is not None
            else None
        )
        report = score_verification(primary_rows, hidden_key, second_rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    except AnatomyScoringError as error:
        print(f'SCORING = FAIL_CLOSED\n{error}', file=sys.stderr)
        return 1
    status = report['structure_level_anatomy_verification']
    print(json.dumps(report, indent=2))
    print(f'STRUCTURE_LEVEL_ANATOMY_VERIFICATION = {status}')
    print(
        'HUMAN_REVIEW_REQUIRED = '
        + ('YES' if report['human_review_required'] else 'NO')
    )
    return 2 if status == STATUS_PENDING else 0


if __name__ == '__main__':
    raise SystemExit(main())
