import functools
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch

    import dataset
    import evaluate_m4_osseous_prior
    import evaluate_m4_soft_modulation
    import m4_osseous_integration
    import matching_filter
    import registration
    import training
    from evaluation import M3EvaluationContractError
    from matching import PointCTMatcher
else:
    torch = None


class M4OsseousEvaluationSourceContractTest(unittest.TestCase):
    def test_evaluator_is_independent_and_contains_the_complete_chain(self):
        source = (
            EXPERIMENT_DIR / 'evaluate_m4_osseous_prior.py'
        ).read_text(encoding='utf-8')
        self.assertNotIn('_run_inference_case', source)
        self.assertIn('resolve_m4_matching_masks', source)
        self.assertIn('run_m4_osseous_integrated_matching', source)
        self.assertIn('extract_dustbin_aware_mutual_correspondences', source)
        self.assertIn('_require_intact_filter_output', source)
        self.assertIn('estimate_weighted_point_to_ct_transform', source)
        self.assertNotIn('from evaluate_m3 import', source)

    def test_frozen_m3_evaluator_has_no_osseous_edits(self):
        source = (EXPERIMENT_DIR / 'evaluate_m3.py').read_text(encoding='utf-8')
        self.assertNotIn('osseous', source.lower())


if TORCH_AVAILABLE:
    class StaticEncoder(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, _input):
            return self.output


def _encoder_outputs(*, reverse_ct=False, with_mapping=True):
    q = torch.zeros((4, 256), dtype=torch.float32)
    k = torch.zeros((4, 256), dtype=torch.float32)
    q[:, :4] = torch.eye(4, dtype=torch.float32)
    k[:, :4] = torch.tensor(
        [
            [0.8, 0.6, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.0, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [150.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    point_output = {'Q': q, 'Xp_phys_coarse': coordinates}
    ct_locations = coordinates.flip(0) if reverse_ct else coordinates.clone()
    ct_output = {'K': k, 'Xv_phys_coarse': ct_locations}
    if with_mapping:
        valid = torch.tensor([True, False, True, True])
        point_output.update(
            {
                'point_intact_coarse': valid,
                'point_raw_total_count_coarse': torch.ones(4, dtype=torch.int64),
                'point_raw_defect_count_coarse': (~valid).to(torch.int64),
            }
        )
        ct_valid = valid.clone()
        ct_output.update(
            {
                'ct_intact_coarse': ct_valid,
                'ct_raw_total_count_coarse': torch.ones(4, dtype=torch.int64),
                'ct_raw_defect_count_coarse': (~ct_valid).to(torch.int64),
            }
        )
    return point_output, ct_output, coordinates


def _sample():
    return {
        'subject_id': 'ToyOsseousEvaluation',
        'ct_volume': np.asarray(
            [[[-1000, -1000, 1000, 1000]]], dtype=np.int16
        ),
        'ct_spacing': np.asarray([50.0, 1.0, 1.0]),
        'ct_origin': np.zeros(3, dtype=np.float64),
        'ct_direction': np.eye(3, dtype=np.float64),
    }


def _prepare_recorder(storage, support):
    def prepare(sample, point_collate_fn, ct_collate_fn, device):
        storage.update(
            {
                'sample': sample,
                'point_collate_fn': point_collate_fn,
                'ct_collate_fn': ct_collate_fn,
                'device': device,
            }
        )
        return object(), {'ct_support_phys_20mm': support.clone()}

    return prepare


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for evaluator tests.')
class M4OsseousEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.protocol = {'matching_filter_min_confidence': None}
        self.matcher = PointCTMatcher(
            projected_dim=256,
            temperature=0.2,
            sinkhorn_iterations=60,
            alpha_init=1.0,
        )

    def _run(self, strength, *, reverse_ct=False, with_mapping=True, sample=None):
        point_output, ct_output, support = _encoder_outputs(
            reverse_ct=reverse_ct,
            with_mapping=with_mapping,
        )
        prepared = {}
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(prepared, support),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            wraps=matching_filter.extract_dustbin_aware_mutual_correspondences,
        ) as mutual_filter, mock.patch.object(
            registration,
            'estimate_weighted_point_to_ct_transform',
            wraps=registration.estimate_weighted_point_to_ct_transform,
        ) as weighted_registration, mock.patch.object(
            m4_osseous_integration,
            'run_m4_osseous_integrated_matching',
            wraps=m4_osseous_integration.run_m4_osseous_integrated_matching,
        ) as core:
            result = evaluate_m4_osseous_prior.run_m4_osseous_prior_inference(
                sample=_sample() if sample is None else sample,
                point_encoder=StaticEncoder(point_output),
                ct_encoder=StaticEncoder(ct_output),
                matcher=self.matcher,
                evaluation_protocol=self.protocol,
                device=self.device,
                sigma_mm=60.0,
                soft_strength=2.0,
                osseous_strength=strength,
            )
        return (
            result,
            prepared,
            mutual_filter,
            weighted_registration,
            core,
            point_output,
            ct_output,
        )

    def test_complete_chain_masks_diagnostics_and_provenance(self):
        (
            result,
            prepared,
            mutual_filter,
            weighted_registration,
            core,
            point_output,
            ct_output,
        ) = self._run(1.0)
        self.assertIsInstance(prepared['point_collate_fn'], functools.partial)
        self.assertIsInstance(prepared['ct_collate_fn'], functools.partial)
        self.assertIs(prepared['point_collate_fn'].func, dataset.m2_point_collate_fn)
        self.assertIs(prepared['ct_collate_fn'].func, dataset.m2_ct_collate_fn)
        self.assertTrue(
            prepared['point_collate_fn'].keywords['enable_m4_defect_mapping']
        )
        self.assertTrue(
            prepared['ct_collate_fn'].keywords['enable_m4_defect_mapping']
        )
        point_mask = point_output['point_intact_coarse']
        ct_mask = ct_output['ct_intact_coarse']
        self.assertIs(result['point_valid_mask'], point_mask)
        self.assertIs(result['ct_valid_mask'], ct_mask)
        self.assertIs(mutual_filter.call_args.kwargs['point_valid_mask'], point_mask)
        self.assertIs(mutual_filter.call_args.kwargs['ct_valid_mask'], ct_mask)
        registration_args = weighted_registration.call_args.args
        self.assertTrue(torch.all(point_mask[registration_args[2]]))
        self.assertTrue(torch.all(ct_mask[registration_args[3]]))
        self.assertTrue(result['m4_hard_constraint_active'])
        self.assertTrue(result['m4_soft_modulation_active'])
        self.assertTrue(result['m4_osseous_prior_active'])
        self.assertEqual(result['osseous_strength'], 1.0)
        self.assertEqual(result['osseous_center_hu'], 300.0)
        self.assertEqual(result['osseous_tau_hu'], 100.0)
        self.assertEqual(result['osseous_radius_mm'], 20.0)
        self.assertLess(result['osseous_score_min'], result['osseous_score_max'])
        self.assertTrue(
            result['m4_osseous_provenance'][
                'support_to_matcher_token_alignment_verified'
            ]
        )
        self.assertFalse(
            result['m4_osseous_provenance']['complete_counterpart_used']
        )
        self.assertFalse(result['m4_osseous_provenance']['ground_truth_used'])
        self.assertFalse(result['m4_osseous_provenance']['identity_used'])
        kwargs = core.call_args.kwargs
        self.assertNotIn('sample', kwargs)
        self.assertNotIn('gt_transform', kwargs)
        self.assertNotIn('subject_id', kwargs)
        self.assertNotIn('defect_id', kwargs)

    def test_zero_strength_matches_m4_2a_assignment_filter_and_registration(self):
        point_output, ct_output, support = _encoder_outputs()
        soft_prepared = {}
        with mock.patch.object(
            training,
            '_prepare_model_inputs',
            side_effect=_prepare_recorder(soft_prepared, support),
        ), mock.patch.object(
            matching_filter,
            'extract_dustbin_aware_mutual_correspondences',
            wraps=matching_filter.extract_dustbin_aware_mutual_correspondences,
        ) as soft_filter, mock.patch.object(
            registration,
            'estimate_weighted_point_to_ct_transform',
            wraps=registration.estimate_weighted_point_to_ct_transform,
        ) as soft_registration:
            soft = evaluate_m4_soft_modulation.run_m4_soft_modulation_inference(
                sample={'subject_id': 'ToySoftReference'},
                point_encoder=StaticEncoder(point_output),
                ct_encoder=StaticEncoder(ct_output),
                matcher=self.matcher,
                evaluation_protocol=self.protocol,
                device=self.device,
                sigma_mm=60.0,
                strength=2.0,
            )
        integrated, _, integrated_filter, integrated_registration, _, _, _ = self._run(
            0.0
        )
        self.assertTrue(torch.equal(soft['base_similarity'], integrated['base_similarity']))
        self.assertTrue(
            torch.equal(soft['modulated_similarity'], integrated['soft_similarity'])
        )
        self.assertTrue(torch.equal(soft['log_assignment'], integrated['log_assignment']))
        self.assertTrue(
            torch.equal(soft_filter.call_args.args[0], integrated_filter.call_args.args[0])
        )
        for soft_arg, integrated_arg in zip(
            soft_registration.call_args.args,
            integrated_registration.call_args.args,
        ):
            self.assertTrue(torch.equal(soft_arg, integrated_arg))
        self.assertEqual(
            set(soft['filter_output']), set(integrated['filter_output'])
        )
        for name in soft['filter_output']:
            soft_value = soft['filter_output'][name]
            integrated_value = integrated['filter_output'][name]
            if torch.is_tensor(soft_value):
                self.assertTrue(torch.equal(soft_value, integrated_value), name)
            else:
                self.assertEqual(soft_value, integrated_value, name)
        self.assertEqual(
            soft['registration_output']['num_correspondences'],
            integrated['registration_output']['num_correspondences'],
        )
        self.assertFalse(integrated['osseous_similarity_changed'])

    def test_nonzero_strength_changes_sinkhorn_without_mutating_q_k(self):
        zero, _, _, _, _, point_output, ct_output = self._run(0.0)
        q_before = point_output['Q'].clone()
        k_before = ct_output['K'].clone()
        changed, _, _, _, _, _, _ = self._run(3.0)
        self.assertTrue(changed['osseous_similarity_changed'])
        self.assertFalse(
            torch.allclose(
                zero['log_assignment'], changed['log_assignment'], rtol=1e-7, atol=1e-7
            )
        )
        self.assertTrue(torch.equal(point_output['Q'], q_before))
        self.assertTrue(torch.equal(ct_output['K'], k_before))

    def test_missing_ct_mapping_and_wrong_token_order_fail_closed(self):
        for field in ('ct_volume', 'ct_spacing', 'ct_origin', 'ct_direction'):
            sample = _sample()
            del sample[field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    M3EvaluationContractError, 'missing defective CT fields'
                ):
                    self._run(1.0, sample=sample)
        with self.assertRaisesRegex(
            M3EvaluationContractError, 'requires complete Point and CT'
        ):
            self._run(1.0, with_mapping=False)
        with self.assertRaisesRegex(
            M3EvaluationContractError, 'including token order'
        ):
            self._run(1.0, reverse_ct=True)

    def test_invalid_strength_fails_closed(self):
        for value in (-1.0, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    M3EvaluationContractError,
                    'osseous-integrated matching failed',
                ):
                    self._run(value)


if __name__ == '__main__':
    unittest.main()
