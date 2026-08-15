import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments' / 'geotransformer.pointct.baseline_v1'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    import training
    from gt_correspondence import build_coarse_gt_correspondence
    from matching_loss import (
        CollisionAwareMatchLossError,
        compute_collision_aware_match_loss,
    )
    from training import (
        BATCH_SIZE,
        PRECISION,
        SMOKE_SPLIT_STATUS,
        TRAINING_DEFAULTS_STATUS,
        M3TrainingContractError,
        TrainingConfig,
        aggregate_step_results,
        build_subject_split,
        create_optimizer,
        load_checkpoint,
        run_training_step,
        run_validation_step,
        save_checkpoint,
        save_epoch_checkpoints,
        set_random_seed,
        validate_training_config,
    )


    class ToyPointEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 4)
            self.received = None
            self.forward_training_modes = []

        def forward(self, point_dict):
            self.received = point_dict
            self.forward_training_modes.append(self.training)
            q = self.projection(point_dict['features'])
            point_phys = point_dict['points'][3] / point_dict['point_network_scale_mm_to_m']
            return {'Q': q, 'Xp_phys_coarse': point_phys}


    class ToyCTEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 4)
            self.received = None
            self.forward_training_modes = []

        def forward(self, ct_dict):
            self.received = ct_dict
            self.forward_training_modes.append(self.training)
            count = ct_dict['ct_support_phys_20mm'].shape[0]
            k = self.projection(ct_dict['ct_context_features'][:count])
            return {'K': k, 'Xv_phys_coarse': ct_dict['ct_support_phys_20mm']}


    class ToyMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            self.received = None
            self.forward_training_modes = []

        def forward(self, q, k, point_valid_mask=None, ct_valid_mask=None):
            self.received = {
                'q': q,
                'k': k,
                'point_valid_mask': point_valid_mask,
                'ct_valid_mask': ct_valid_mask,
            }
            self.forward_training_modes.append(self.training)
            ordinary = self.scale * torch.matmul(q, k.transpose(0, 1))
            dustbin_column = torch.zeros(
                (ordinary.shape[0], 1),
                dtype=torch.float32,
                device=ordinary.device,
            )
            dustbin_row = torch.zeros(
                (1, ordinary.shape[1] + 1),
                dtype=torch.float32,
                device=ordinary.device,
            )
            assignment = torch.cat(
                (torch.cat((ordinary, dustbin_column), dim=1), dustbin_row),
                dim=0,
            )
            return {
                'log_assignment': assignment,
                'point_valid_mask': point_valid_mask,
                'ct_valid_mask': ct_valid_mask,
            }


    class EmptyModule(torch.nn.Module):
        pass


class M3TrainingSourceContractTest(unittest.TestCase):
    def test_training_sources_keep_the_m3_6a_boundary(self):
        prohibited_imports = {
            'matching_filter',
            'registration',
            'local_global_registration',
        }
        prohibited_calls = {
            'weighted_procrustes',
            'ransac',
            'icp',
        }
        prohibited_source_terms = {
            'matching_filter',
            'registration loss',
            'rre loss',
            'rte loss',
            'contrastive loss',
            'triplet loss',
            'chamfer loss',
            'fusion loss',
            'synthetic rigid',
            'random rigid',
        }
        for filename in ('training.py', 'train_m3.py'):
            path = EXPERIMENT_DIR / filename
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            imported = set()
            called = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.lower() for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported.add(node.module.lower())
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called.add(node.func.id.lower())
                    elif isinstance(node.func, ast.Attribute):
                        called.add(node.func.attr.lower())
            self.assertTrue(prohibited_imports.isdisjoint(imported), filename)
            self.assertTrue(prohibited_calls.isdisjoint(called), filename)
            for term in prohibited_source_terms:
                self.assertNotIn(term, source.lower(), filename)

    def test_training_source_reuses_frozen_gt_and_loss_helpers(self):
        tree = ast.parse((EXPERIMENT_DIR / 'training.py').read_text(encoding='utf-8'))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn('build_coarse_gt_correspondence', imports)
        self.assertIn('compute_collision_aware_match_loss', imports)

    def test_smoke_only_labels_are_explicit_in_source(self):
        source = (EXPERIMENT_DIR / 'training.py').read_text(encoding='utf-8')
        self.assertIn('TRAINING SMOKE DEFAULTS - NOT FROZEN PAPER HYPERPARAMETERS', source)
        self.assertIn('SMOKE SPLIT ONLY - NOT FROZEN EVALUATION SPLIT', source)
        self.assertIn("BATCH_SIZE = 1", source)
        self.assertIn("PRECISION = 'fp32'", source)


@unittest.skipUnless(torch is not None, 'PyTorch is not installed locally.')
class M3TrainingContractTest(unittest.TestCase):
    def setUp(self):
        set_random_seed(7)
        self.point_encoder = ToyPointEncoder()
        self.ct_encoder = ToyCTEncoder()
        self.matcher = ToyMatcher()
        self.optimizer = create_optimizer(
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            learning_rate=1e-2,
            weight_decay=0.0,
        )
        self.sample = self._sample()

    @staticmethod
    def _sample():
        point_phys = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        ct_phys = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        point_branch = {
            'features': torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            'points': [point_phys * 0.001] * 4,
            'neighbors': [torch.zeros((1, 1), dtype=torch.int64)],
            'subsampling': [torch.zeros((1, 1), dtype=torch.int64)],
            'upsampling': [torch.zeros((1, 1), dtype=torch.int64)],
            'point_network_scale_mm_to_m': 0.001,
        }
        ct_branch = {
            'ct_context_features': torch.tensor([[1.5], [2.5]], dtype=torch.float32),
            'ct_context_indices': torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int32),
            'ct_context_spatial_shape': np.asarray([1, 1, 2], dtype=np.int64),
            'ct_support_indices_20mm': torch.tensor(
                [[0, 0, 0], [0, 0, 1]], dtype=torch.int32
            ),
            'ct_support_linear_20mm': torch.tensor([0, 1], dtype=torch.int64),
            'ct_support_spatial_shape_20mm': np.asarray([1, 1, 2], dtype=np.int64),
            'ct_support_phys_20mm': ct_phys,
            'physical_unit': 'mm',
            'coordinate_system': 'left-posterior-superior (LPS)',
        }
        return {
            'subject_id': 'SubjectA',
            'gt_transform': np.eye(4, dtype=np.float64),
            'gt_transform_direction': 'Point Cloud -> CT',
            '_point_branch': point_branch,
            '_ct_branch': ct_branch,
        }

    @staticmethod
    def _point_collate(samples):
        sample = samples[0]
        return {'subject_id': sample['subject_id'], 'point': sample['_point_branch']}

    @staticmethod
    def _ct_collate(samples):
        sample = samples[0]
        return {'subject_id': sample['subject_id'], 'ct': sample['_ct_branch']}

    def _step_kwargs(self):
        return {
            'point_collate_fn': self._point_collate,
            'ct_collate_fn': self._ct_collate,
            'primary_max_distance_mm': 0.25,
            'high_confidence_distance_mm': 0.1,
            'device': torch.device('cpu'),
        }

    @staticmethod
    def _module_parameters(module):
        return [parameter.detach().clone() for parameter in module.parameters()]

    @staticmethod
    def _any_parameter_changed(module, before):
        return any(
            not torch.equal(parameter.detach(), old)
            for parameter, old in zip(module.parameters(), before)
        )

    def test_explicit_subject_split_resolves_manifest_indices(self):
        class Dataset:
            records = [
                {'subject_id': 'A'},
                {'subject_id': 'B'},
                {'subject_id': 'A'},
                {'subject_id': 'C'},
            ]

        split = build_subject_split(Dataset(), ['A', 'B'], ['C'])
        self.assertEqual(split.train_subject_ids, ('A', 'B'))
        self.assertEqual(split.val_subject_ids, ('C',))
        self.assertEqual(split.train_indices, (0, 2, 1))
        self.assertEqual(split.val_indices, (3,))

    def test_split_leakage_fails_closed(self):
        class Dataset:
            records = [{'subject_id': 'A'}, {'subject_id': 'B'}]

        with self.assertRaisesRegex(M3TrainingContractError, 'leakage'):
            build_subject_split(Dataset(), ['A'], ['A'])

    def test_unknown_subject_fails_closed(self):
        class Dataset:
            records = [{'subject_id': 'A'}, {'subject_id': 'B'}]

        with self.assertRaisesRegex(M3TrainingContractError, 'unknown'):
            build_subject_split(Dataset(), ['A'], ['C'])

    def test_empty_split_fails_closed(self):
        class Dataset:
            records = [{'subject_id': 'A'}, {'subject_id': 'B'}]

        for train_ids, val_ids in (([], ['B']), (['A'], [])):
            with self.subTest(train_ids=train_ids, val_ids=val_ids):
                with self.assertRaisesRegex(M3TrainingContractError, 'must not be empty'):
                    build_subject_split(Dataset(), train_ids, val_ids)

    def test_duplicate_subject_in_explicit_split_fails_closed(self):
        class Dataset:
            records = [{'subject_id': 'A'}, {'subject_id': 'B'}]

        with self.assertRaisesRegex(M3TrainingContractError, 'duplicate'):
            build_subject_split(Dataset(), ['A', 'A'], ['B'])

    def test_training_step_updates_all_three_modules_with_finite_gradients(self):
        point_before = self._module_parameters(self.point_encoder)
        ct_before = self._module_parameters(self.ct_encoder)
        matcher_before = self._module_parameters(self.matcher)
        for parameter in (
            list(self.point_encoder.parameters())
            + list(self.ct_encoder.parameters())
            + list(self.matcher.parameters())
        ):
            parameter.grad = torch.full_like(parameter, float('inf'))

        result = run_training_step(
            self.sample,
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            self.optimizer,
            **self._step_kwargs(),
        )
        self.assertEqual(result['loss'].ndim, 0)
        self.assertEqual(result['loss'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertTrue(result['loss'].requires_grad)
        self.assertTrue(self._any_parameter_changed(self.point_encoder, point_before))
        self.assertTrue(self._any_parameter_changed(self.ct_encoder, ct_before))
        self.assertTrue(self._any_parameter_changed(self.matcher, matcher_before))
        for module in (self.point_encoder, self.ct_encoder, self.matcher):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_gt_is_not_passed_to_encoder_or_matcher_arguments(self):
        run_validation_step(
            self.sample,
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            **self._step_kwargs(),
        )
        for received in (self.point_encoder.received, self.ct_encoder.received, self.matcher.received):
            self.assertTrue(received)
            self.assertFalse(any(str(key).startswith('gt_') for key in received))
        self.assertEqual(
            set(self.matcher.received),
            {'q', 'k', 'point_valid_mask', 'ct_valid_mask'},
        )

    def test_gt_helper_receives_encoder_physical_coarse_coordinates(self):
        captured = {}

        def gt_spy(**kwargs):
            captured.update(kwargs)
            return build_coarse_gt_correspondence(**kwargs)

        with mock.patch.object(training, 'build_coarse_gt_correspondence', gt_spy):
            run_validation_step(
                self.sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                **self._step_kwargs(),
            )
        np.testing.assert_allclose(
            captured['Xp_phys_coarse'],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(
            captured['Xv_phys_coarse'],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )

    def test_formal_collision_aware_loss_is_called(self):
        calls = []

        def loss_spy(*args, **kwargs):
            calls.append((args, kwargs))
            return compute_collision_aware_match_loss(*args, **kwargs)

        with mock.patch.object(training, 'compute_collision_aware_match_loss', loss_spy):
            result = run_validation_step(
                self.sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                **self._step_kwargs(),
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['num_supervised_points'], 2)
        self.assertEqual(result['num_supervised_groups'], 2)
        self.assertEqual(result['num_collision_groups'], 0)
        self.assertEqual(len(calls[0][0]), 3)
        self.assertEqual(set(calls[0][1]), {'point_valid_mask', 'ct_valid_mask'})

    def test_high_confidence_is_diagnostic_only(self):
        def gt_with_high_confidence(value):
            def builder(**kwargs):
                output = build_coarse_gt_correspondence(**kwargs)
                output['gt_high_confidence'] = np.full(
                    output['gt_primary_valid'].shape,
                    value,
                    dtype=bool,
                )
                return output

            return builder

        with mock.patch.object(
            training,
            'build_coarse_gt_correspondence',
            gt_with_high_confidence(True),
        ):
            first = run_validation_step(
                self.sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                **self._step_kwargs(),
            )
        with mock.patch.object(
            training,
            'build_coarse_gt_correspondence',
            gt_with_high_confidence(False),
        ):
            second = run_validation_step(
                self.sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                **self._step_kwargs(),
            )
        torch.testing.assert_close(first['loss'], second['loss'])
        self.assertEqual(first['num_high_confidence_points'], 2)
        self.assertEqual(second['num_high_confidence_points'], 0)

    def test_validation_has_no_backward_or_parameter_update(self):
        modules = (self.point_encoder, self.ct_encoder, self.matcher)
        before = [self._module_parameters(module) for module in modules]
        for module in modules:
            for parameter in module.parameters():
                parameter.grad = None
        result = run_validation_step(
            self.sample,
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            **self._step_kwargs(),
        )
        self.assertFalse(result['loss'].requires_grad)
        for module, old_parameters in zip(modules, before):
            self.assertFalse(self._any_parameter_changed(module, old_parameters))
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_validation_uses_eval_and_restores_previous_modes(self):
        self.point_encoder.train(True)
        self.ct_encoder.train(False)
        self.matcher.train(True)
        run_validation_step(
            self.sample,
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            **self._step_kwargs(),
        )
        self.assertFalse(self.point_encoder.forward_training_modes[-1])
        self.assertFalse(self.ct_encoder.forward_training_modes[-1])
        self.assertFalse(self.matcher.forward_training_modes[-1])
        self.assertTrue(self.point_encoder.training)
        self.assertFalse(self.ct_encoder.training)
        self.assertTrue(self.matcher.training)

    def test_batch_size_and_fp32_contract_fail_closed(self):
        for config, message in (
            (TrainingConfig(batch_size=2), 'batch_size=1'),
            (TrainingConfig(precision='float16'), 'fp32'),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(M3TrainingContractError, message):
                    validate_training_config(config)
        self.assertEqual(BATCH_SIZE, 1)
        self.assertEqual(PRECISION, 'fp32')

    def test_non_fp32_trainable_module_fails_closed(self):
        self.point_encoder.to(dtype=torch.float64)
        with self.assertRaisesRegex(M3TrainingContractError, 'float32'):
            run_validation_step(
                self.sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                **self._step_kwargs(),
            )

    def test_optimizer_requires_all_formal_trainable_modules(self):
        with self.assertRaisesRegex(M3TrainingContractError, 'no trainable parameters'):
            create_optimizer(EmptyModule(), self.ct_encoder, self.matcher)

    def test_zero_gt_supervision_fails_closed(self):
        def no_supervision(**kwargs):
            output = build_coarse_gt_correspondence(**kwargs)
            output['gt_primary_valid'] = np.zeros_like(output['gt_primary_valid'])
            return output

        with mock.patch.object(training, 'build_coarse_gt_correspondence', no_supervision):
            with self.assertRaisesRegex(CollisionAwareMatchLossError, 'zero valid Points'):
                run_validation_step(
                    self.sample,
                    self.point_encoder,
                    self.ct_encoder,
                    self.matcher,
                    **self._step_kwargs(),
                )

    def test_out_of_range_gt_index_fails_closed(self):
        def invalid_index(**kwargs):
            output = build_coarse_gt_correspondence(**kwargs)
            output['gt_primary_ct_index'][0] = 2
            return output

        with mock.patch.object(training, 'build_coarse_gt_correspondence', invalid_index):
            with self.assertRaisesRegex(CollisionAwareMatchLossError, 'out-of-range'):
                run_validation_step(
                    self.sample,
                    self.point_encoder,
                    self.ct_encoder,
                    self.matcher,
                    **self._step_kwargs(),
                )

    def test_device_mismatch_fails_closed(self):
        with self.assertRaisesRegex(M3TrainingContractError, 'requested device'):
            run_validation_step(
                self.sample,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                **{**self._step_kwargs(), 'device': torch.device('meta')},
            )

    def _checkpoint_fields(self, **overrides):
        fields = {
            'epoch': 3,
            'global_step': 17,
            'point_encoder': self.point_encoder,
            'ct_encoder': self.ct_encoder,
            'matcher': self.matcher,
            'optimizer': self.optimizer,
            'train_subject_ids': ['SubjectA'],
            'val_subject_ids': ['SubjectB'],
            'seed': 7,
            'training_config': TrainingConfig(),
            'best_val_loss': 0.75,
        }
        fields.update(overrides)
        return fields

    def test_checkpoint_roundtrip_restores_models_optimizer_and_progress(self):
        run_training_step(
            self.sample,
            self.point_encoder,
            self.ct_encoder,
            self.matcher,
            self.optimizer,
            **self._step_kwargs(),
        )
        expected = [
            self._module_parameters(module)
            for module in (self.point_encoder, self.ct_encoder, self.matcher)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roundtrip.pt'
            save_checkpoint(path, **self._checkpoint_fields())
            with torch.no_grad():
                for module in (self.point_encoder, self.ct_encoder, self.matcher):
                    for parameter in module.parameters():
                        parameter.add_(10.0)
            state = load_checkpoint(
                path,
                self.point_encoder,
                self.ct_encoder,
                self.matcher,
                self.optimizer,
                train_subject_ids=['SubjectA'],
                val_subject_ids=['SubjectB'],
                map_location='cpu',
                expected_training_config=TrainingConfig(),
            )
        self.assertEqual(state['epoch'], 3)
        self.assertEqual(state['global_step'], 17)
        self.assertEqual(state['best_val_loss'], 0.75)
        for module, old_parameters in zip(
            (self.point_encoder, self.ct_encoder, self.matcher), expected
        ):
            for parameter, old in zip(module.parameters(), old_parameters):
                torch.testing.assert_close(parameter, old)
        self.assertTrue(self.optimizer.state)

    def test_checkpoint_split_mismatch_fails_before_state_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.pt'
            save_checkpoint(path, **self._checkpoint_fields())
            before = self._module_parameters(self.point_encoder)
            with self.assertRaisesRegex(M3TrainingContractError, 'split does not match'):
                load_checkpoint(
                    path,
                    self.point_encoder,
                    self.ct_encoder,
                    self.matcher,
                    self.optimizer,
                    train_subject_ids=['SubjectB'],
                    val_subject_ids=['SubjectA'],
                    map_location='cpu',
                )
        self.assertFalse(self._any_parameter_changed(self.point_encoder, before))

    def test_malformed_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'malformed.pt'
            torch.save({'epoch': 1}, path)
            with self.assertRaisesRegex(M3TrainingContractError, 'malformed'):
                load_checkpoint(
                    path,
                    self.point_encoder,
                    self.ct_encoder,
                    self.matcher,
                    self.optimizer,
                    train_subject_ids=['SubjectA'],
                    val_subject_ids=['SubjectB'],
                    map_location='cpu',
                )

    def test_best_checkpoint_is_selected_only_by_validation_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            fields = self._checkpoint_fields()
            fields.pop('best_val_loss')
            first = save_epoch_checkpoints(
                directory,
                val_loss=0.8,
                best_val_loss=float('inf'),
                **fields,
            )
            second = save_epoch_checkpoints(
                directory,
                val_loss=0.9,
                best_val_loss=first['best_val_loss'],
                **fields,
            )
            third = save_epoch_checkpoints(
                directory,
                val_loss=0.7,
                best_val_loss=second['best_val_loss'],
                **fields,
            )
            try:
                payload = torch.load(
                    Path(directory) / 'best_val_loss.pt',
                    map_location='cpu',
                    weights_only=True,
                )
            except TypeError:
                payload = torch.load(
                    Path(directory) / 'best_val_loss.pt',
                    map_location='cpu',
                )
        self.assertTrue(first['improved'])
        self.assertFalse(second['improved'])
        self.assertIsNone(second['best_path'])
        self.assertTrue(third['improved'])
        self.assertEqual(payload['best_val_loss'], 0.7)

    def test_seed_controls_python_numpy_and_torch(self):
        import random

        set_random_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(1))
        set_random_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])

    def test_epoch_aggregation_reports_required_loss_and_collision_totals(self):
        first = {
            'loss': torch.tensor(1.0),
            'num_supervised_points': 2,
            'num_supervised_groups': 2,
            'num_collision_groups': 0,
        }
        second = {
            'loss': 3.0,
            'num_supervised_points': 4,
            'num_supervised_groups': 3,
            'num_collision_groups': 1,
        }
        output = aggregate_step_results([first, second])
        self.assertEqual(output['mean_loss'], 2.0)
        self.assertEqual(output['num_cases'], 2)
        self.assertEqual(output['num_supervised_points'], 6)
        self.assertEqual(output['num_supervised_groups'], 5)
        self.assertEqual(output['num_collision_groups'], 1)

    def test_smoke_defaults_are_explicitly_not_frozen(self):
        self.assertIn('NOT FROZEN PAPER HYPERPARAMETERS', TRAINING_DEFAULTS_STATUS)
        self.assertIn('NOT FROZEN EVALUATION SPLIT', SMOKE_SPLIT_STATUS)
        config = validate_training_config(TrainingConfig())
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.weight_decay, 1e-4)


if __name__ == '__main__':
    unittest.main()
