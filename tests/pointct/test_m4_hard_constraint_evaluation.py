import functools
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import evaluate_m3
import evaluate_m4_hard_constraint
from evaluation import M3EvaluationContractError


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch

    import dataset
    import m4_hard_constraint
    import matching_filter
    import registration
    import training
else:
    torch = None


class _StaticEncoder:
    def __init__(self, output):
        self.output = output
        self.grad_modes = []

    def __call__(self, _inputs):
        if torch is not None:
            self.grad_modes.append(torch.is_grad_enabled())
        return self.output


class _RecordingMatcher:
    def __init__(self, log_assignment, *, replace_masks=False):
        self.log_assignment = log_assignment
        self.replace_masks = replace_masks
        self.calls = []

    def __call__(self, *, q, k, point_valid_mask, ct_valid_mask):
        grad_enabled = torch.is_grad_enabled()
        self.calls.append(
            {
                'q': q,
                'k': k,
                'point_valid_mask': point_valid_mask,
                'ct_valid_mask': ct_valid_mask,
                'grad_enabled': grad_enabled,
            }
        )
        if self.replace_masks:
            returned_point_mask = torch.ones_like(point_valid_mask)
            returned_ct_mask = torch.ones_like(ct_valid_mask)
        else:
            returned_point_mask = point_valid_mask
            returned_ct_mask = ct_valid_mask
        return {
            'log_assignment': self.log_assignment,
            'point_valid_mask': returned_point_mask,
            'ct_valid_mask': returned_ct_mask,
        }


def _encoder_outputs(*, with_mapping):
    q = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256) + 1.0
    k = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256) + 2.0
    point_phys = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ct_phys = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [-50.0, -50.0, -50.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    point_output = {'Q': q, 'Xp_phys_coarse': point_phys}
    ct_output = {'K': k, 'Xv_phys_coarse': ct_phys}
    if with_mapping:
        point_output.update(
            {
                'point_intact_coarse': torch.tensor(
                    [True, False, True, True]
                ),
                'point_raw_total_count_coarse': torch.tensor(
                    [2, 2, 2, 2], dtype=torch.int64
                ),
                'point_raw_defect_count_coarse': torch.tensor(
                    [0, 1, 0, 0], dtype=torch.int64
                ),
            }
        )
        ct_output.update(
            {
                'ct_intact_coarse': torch.tensor([True, True, False, True]),
                'ct_raw_total_count_coarse': torch.tensor(
                    [3, 3, 3, 3], dtype=torch.int64
                ),
                'ct_raw_defect_count_coarse': torch.tensor(
                    [0, 0, 2, 0], dtype=torch.int64
                ),
            }
        )
    return point_output, ct_output


def _log_assignment():
    # Three intact pairs are mutual.  The excluded Point/CT pair deliberately
    # has the highest ordinary score and must still never reach registration.
    scores = torch.full((5, 5), -20.0, dtype=torch.float32)
    scores[0, 0] = -0.1
    scores[2, 1] = -0.2
    scores[3, 3] = -0.3
    scores[1, 2] = 10.0
    scores[:4, 4] = -10.0
    scores[4, :4] = -10.0
    scores[4, 4] = 0.0
    return scores


def _prepare_recorder(storage):
    def prepare(sample, point_collate_fn, ct_collate_fn, device):
        storage.update(
            {
                'sample': sample,
                'point_collate_fn': point_collate_fn,
                'ct_collate_fn': ct_collate_fn,
                'device': device,
            }
        )
        return object(), object()

    return prepare


class M4HardConstraintEvaluationEntryTest(unittest.TestCase):
    def test_independent_entry_does_not_delegate_to_frozen_m3_inference(self):
        self.assertFalse(hasattr(evaluate_m4_hard_constraint, '_run_inference_case'))


@unittest.skipUnless(
    TORCH_AVAILABLE,
    'PyTorch is required for M4 hard-constraint evaluation numerical tests.',
)
class M4HardConstraintEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.evaluation_protocol = {'matching_filter_min_confidence': None}

    def test_frozen_m3_default_keeps_original_collates_masks_and_result_shape(self):
        point_output, ct_output = _encoder_outputs(with_mapping=False)
        matcher = _RecordingMatcher(_log_assignment())
        prepared = {}
        filter_call = {}

        def fake_filter(
            log_assignment,
            point_valid_mask=None,
            ct_valid_mask=None,
            min_confidence=None,
        ):
            filter_call.update(
                {
                    'log_assignment': log_assignment,
                    'point_valid_mask': point_valid_mask,
                    'ct_valid_mask': ct_valid_mask,
                    'min_confidence': min_confidence,
                }
            )
            empty = torch.empty((0,), dtype=torch.long)
            return {
                'point_indices': empty,
                'ct_indices': empty.clone(),
                'confidence': torch.empty((0,), dtype=torch.float32),
                'num_correspondences': 0,
            }

        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(prepared),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            side_effect=fake_filter,
        ), mock.patch.object(
            registration,
            'estimate_weighted_point_to_ct_transform',
            return_value={'num_correspondences': 0},
        ), mock.patch.object(
            m4_hard_constraint,
            'resolve_m4_matching_masks',
            side_effect=AssertionError('M3 default called M4 resolver'),
        ):
            result = evaluate_m3._run_inference_case(
                {'subject_id': 'Pat1'},
                _StaticEncoder(point_output),
                _StaticEncoder(ct_output),
                matcher,
                self.evaluation_protocol,
                self.device,
            )

        self.assertIs(prepared['point_collate_fn'], dataset.m2_point_collate_fn)
        self.assertIs(prepared['ct_collate_fn'], dataset.m2_ct_collate_fn)
        self.assertTrue(torch.all(matcher.calls[0]['point_valid_mask']))
        self.assertTrue(torch.all(matcher.calls[0]['ct_valid_mask']))
        self.assertTrue(torch.all(filter_call['point_valid_mask']))
        self.assertTrue(torch.all(filter_call['ct_valid_mask']))
        self.assertEqual(
            set(result),
            {
                'point_physical',
                'ct_physical',
                'filter_output',
                'registration_output',
                'inference_runtime_ms',
            },
        )

    def test_explicit_m4_chain_uses_intact_masks_through_registration(self):
        point_output, ct_output = _encoder_outputs(with_mapping=True)
        point_encoder = _StaticEncoder(point_output)
        ct_encoder = _StaticEncoder(ct_output)
        matcher = _RecordingMatcher(_log_assignment())
        prepared = {}
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(prepared),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            wraps=matching_filter.extract_dustbin_aware_mutual_correspondences,
        ) as mutual_filter, mock.patch.object(
            registration,
            'estimate_weighted_point_to_ct_transform',
            wraps=registration.estimate_weighted_point_to_ct_transform,
        ) as weighted_registration:
            result = evaluate_m4_hard_constraint.run_m4_hard_constraint_inference(
                sample={'subject_id': 'Pat1'},
                point_encoder=point_encoder,
                ct_encoder=ct_encoder,
                matcher=matcher,
                evaluation_protocol=self.evaluation_protocol,
                device=self.device,
            )

        point_collate_fn = prepared['point_collate_fn']
        ct_collate_fn = prepared['ct_collate_fn']
        self.assertIsInstance(point_collate_fn, functools.partial)
        self.assertIsInstance(ct_collate_fn, functools.partial)
        self.assertIs(point_collate_fn.func, dataset.m2_point_collate_fn)
        self.assertIs(ct_collate_fn.func, dataset.m2_ct_collate_fn)
        self.assertIs(point_collate_fn.keywords['enable_m4_defect_mapping'], True)
        self.assertIs(ct_collate_fn.keywords['enable_m4_defect_mapping'], True)

        expected_point_mask = point_output['point_intact_coarse']
        expected_ct_mask = ct_output['ct_intact_coarse']
        torch.testing.assert_close(
            matcher.calls[0]['point_valid_mask'], expected_point_mask
        )
        torch.testing.assert_close(
            matcher.calls[0]['ct_valid_mask'], expected_ct_mask
        )
        self.assertIs(
            mutual_filter.call_args.kwargs['point_valid_mask'],
            expected_point_mask,
        )
        self.assertIs(
            mutual_filter.call_args.kwargs['ct_valid_mask'],
            expected_ct_mask,
        )
        self.assertEqual(point_encoder.grad_modes, [False])
        self.assertEqual(ct_encoder.grad_modes, [False])
        self.assertIs(matcher.calls[0]['grad_enabled'], False)

        filter_output = result['filter_output']
        torch.testing.assert_close(
            filter_output['point_indices'], torch.tensor([0, 2, 3])
        )
        torch.testing.assert_close(
            filter_output['ct_indices'], torch.tensor([0, 1, 3])
        )
        registration_args = weighted_registration.call_args.args
        self.assertTrue(torch.all(expected_point_mask[registration_args[2]]))
        self.assertTrue(torch.all(expected_ct_mask[registration_args[3]]))
        self.assertTrue(result['registration_output']['success'])
        self.assertIs(result['m4_hard_constraint_enabled'], True)
        self.assertIs(result['m4_hard_constraint_active'], True)
        self.assertEqual(result['point_total_tokens'], 4)
        self.assertEqual(result['point_intact_tokens'], 3)
        self.assertEqual(result['point_excluded_tokens'], 1)
        self.assertEqual(result['ct_total_tokens'], 4)
        self.assertEqual(result['ct_intact_tokens'], 3)
        self.assertEqual(result['ct_excluded_tokens'], 1)
        self.assertIs(result['point_valid_mask'], expected_point_mask)
        self.assertIs(result['ct_valid_mask'], expected_ct_mask)
        self.assertEqual(
            result['m4_mask_provenance'],
            {
                'point_source': 'point_encoder_output.point_intact_coarse',
                'ct_source': 'ct_encoder_output.ct_intact_coarse',
                'matcher_masks_preserved': True,
                'mutual_filter_uses_resolved_masks': True,
                'registration_correspondences_verified_intact': True,
            },
        )

    def test_explicit_m4_rejects_missing_mapping_instead_of_falling_back(self):
        point_output, ct_output = _encoder_outputs(with_mapping=False)
        matcher = _RecordingMatcher(_log_assignment())
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            return_value=(object(), object()),
        ):
            with self.assertRaisesRegex(
                M3EvaluationContractError,
                'requires complete Point and CT',
            ):
                evaluate_m4_hard_constraint.run_m4_hard_constraint_inference(
                    sample={'subject_id': 'Pat1'},
                    point_encoder=_StaticEncoder(point_output),
                    ct_encoder=_StaticEncoder(ct_output),
                    matcher=matcher,
                    evaluation_protocol=self.evaluation_protocol,
                    device=self.device,
                )
        self.assertEqual(matcher.calls, [])

    def test_matcher_cannot_replace_resolved_m4_masks(self):
        point_output, ct_output = _encoder_outputs(with_mapping=True)
        matcher = _RecordingMatcher(_log_assignment(), replace_masks=True)
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            return_value=(object(), object()),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
        ) as mutual_filter:
            with self.assertRaisesRegex(
                M3EvaluationContractError,
                'did not preserve',
            ):
                evaluate_m4_hard_constraint.run_m4_hard_constraint_inference(
                    sample={'subject_id': 'Pat1'},
                    point_encoder=_StaticEncoder(point_output),
                    ct_encoder=_StaticEncoder(ct_output),
                    matcher=matcher,
                    evaluation_protocol=self.evaluation_protocol,
                    device=self.device,
                )
        mutual_filter.assert_not_called()

    def test_defect_correspondence_is_rejected_before_registration(self):
        point_output, ct_output = _encoder_outputs(with_mapping=True)
        matcher = _RecordingMatcher(_log_assignment())
        bad_filter_output = {
            'point_indices': torch.tensor([1], dtype=torch.long),
            'ct_indices': torch.tensor([0], dtype=torch.long),
            'confidence': torch.tensor([0.9], dtype=torch.float32),
            'num_correspondences': 1,
        }
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            return_value=(object(), object()),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            return_value=bad_filter_output,
        ), mock.patch.object(
            registration,
            'estimate_weighted_point_to_ct_transform',
        ) as weighted_registration:
            with self.assertRaisesRegex(
                M3EvaluationContractError,
                'excluded defect token',
            ):
                evaluate_m4_hard_constraint.run_m4_hard_constraint_inference(
                    sample={'subject_id': 'Pat1'},
                    point_encoder=_StaticEncoder(point_output),
                    ct_encoder=_StaticEncoder(ct_output),
                    matcher=matcher,
                    evaluation_protocol=self.evaluation_protocol,
                    device=self.device,
                )
        weighted_registration.assert_not_called()

if __name__ == '__main__':
    unittest.main()
