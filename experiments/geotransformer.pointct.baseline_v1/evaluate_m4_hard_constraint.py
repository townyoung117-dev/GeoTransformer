"""Independent M4 defect-aware hard-constraint inference entry.

The M3 evaluator is byte-frozen. This module owns the M4-only preprocessing,
mask resolution, matching, filtering, and registration path while reusing
stable lower-level Point/CT components.
"""

import math
import time
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial

from evaluation import M3EvaluationContractError


def _require_mapping(value, name):
    if not isinstance(value, Mapping):
        raise M3EvaluationContractError(f"{name} must be a mapping.")
    return value


@contextmanager
def _temporary_evaluation_mode(torch, *modules):
    """Evaluate torch modules without permanently changing caller-owned state."""
    states = []
    for module in modules:
        if isinstance(module, torch.nn.Module):
            states.append((module, bool(module.training)))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.train(was_training)


def _synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _require_matcher_output(
    torch,
    matcher_output,
    *,
    point_valid_mask,
    ct_valid_mask,
    point_count,
    ct_count,
    device,
):
    matcher_output = _require_mapping(matcher_output, "PointCTMatcher output")
    required = ("log_assignment", "point_valid_mask", "ct_valid_mask")
    missing = [field for field in required if field not in matcher_output]
    if missing:
        raise M3EvaluationContractError(
            f"PointCTMatcher output is missing fields: {missing}."
        )

    matcher_point_mask = matcher_output["point_valid_mask"]
    matcher_ct_mask = matcher_output["ct_valid_mask"]
    if not torch.is_tensor(matcher_point_mask) or not torch.equal(
        matcher_point_mask,
        point_valid_mask,
    ):
        raise M3EvaluationContractError(
            "PointCTMatcher did not preserve the resolved Point hard mask."
        )
    if not torch.is_tensor(matcher_ct_mask) or not torch.equal(
        matcher_ct_mask,
        ct_valid_mask,
    ):
        raise M3EvaluationContractError(
            "PointCTMatcher did not preserve the resolved CT hard mask."
        )

    log_assignment = matcher_output["log_assignment"]
    if not torch.is_tensor(log_assignment):
        raise M3EvaluationContractError(
            "PointCTMatcher log_assignment must be a torch.Tensor."
        )
    if log_assignment.device != device or log_assignment.dtype != torch.float32:
        raise M3EvaluationContractError(
            "PointCTMatcher log_assignment must be float32 on the requested device."
        )
    if tuple(log_assignment.shape) != (point_count + 1, ct_count + 1):
        raise M3EvaluationContractError(
            "PointCTMatcher log_assignment shape is inconsistent with Q and K."
        )
    if bool(torch.isnan(log_assignment).any()) or bool(
        (log_assignment == float("inf")).any()
    ):
        raise M3EvaluationContractError(
            "PointCTMatcher log_assignment contains NaN or positive infinity."
        )
    return log_assignment


def _require_intact_filter_output(
    torch,
    filter_output,
    *,
    point_valid_mask,
    ct_valid_mask,
    device,
):
    filter_output = _require_mapping(
        filter_output,
        "mutual correspondence filter output",
    )
    required = ("point_indices", "ct_indices", "confidence", "num_correspondences")
    missing = [field for field in required if field not in filter_output]
    if missing:
        raise M3EvaluationContractError(
            f"mutual correspondence filter output is missing fields: {missing}."
        )

    point_indices = filter_output["point_indices"]
    ct_indices = filter_output["ct_indices"]
    confidence = filter_output["confidence"]
    for name, value, dtype in (
        ("point_indices", point_indices, torch.long),
        ("ct_indices", ct_indices, torch.long),
        ("confidence", confidence, torch.float32),
    ):
        if not torch.is_tensor(value) or value.ndim != 1:
            raise M3EvaluationContractError(
                f"mutual correspondence filter {name} must be a rank-one tensor."
            )
        if value.dtype != dtype or value.device != device:
            raise M3EvaluationContractError(
                f"mutual correspondence filter {name} has the wrong dtype or device."
            )

    count = int(point_indices.numel())
    if ct_indices.numel() != count or confidence.numel() != count:
        raise M3EvaluationContractError(
            "mutual correspondence filter index and confidence counts differ."
        )
    reported_count = filter_output["num_correspondences"]
    if isinstance(reported_count, bool):
        raise M3EvaluationContractError(
            "mutual correspondence filter num_correspondences must be an integer."
        )
    try:
        reported_count = int(reported_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise M3EvaluationContractError(
            "mutual correspondence filter num_correspondences must be an integer."
        ) from error
    if reported_count != count:
        raise M3EvaluationContractError(
            "mutual correspondence filter reported an inconsistent count."
        )
    if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()):
        raise M3EvaluationContractError(
            "mutual correspondence filter confidence must be finite and nonnegative."
        )

    if count:
        point_out_of_range = bool((point_indices < 0).any()) or bool(
            (point_indices >= point_valid_mask.numel()).any()
        )
        ct_out_of_range = bool((ct_indices < 0).any()) or bool(
            (ct_indices >= ct_valid_mask.numel()).any()
        )
        if point_out_of_range or ct_out_of_range:
            raise M3EvaluationContractError(
                "mutual correspondence filter returned an out-of-range token index."
            )
        if not bool(point_valid_mask[point_indices].all()) or not bool(
            ct_valid_mask[ct_indices].all()
        ):
            raise M3EvaluationContractError(
                "mutual correspondence filter selected an excluded defect token."
            )
    return point_indices, ct_indices, confidence, count


def run_m4_hard_constraint_inference(
    *,
    sample,
    point_encoder,
    ct_encoder,
    matcher,
    evaluation_protocol,
    device,
) -> dict:
    """Run one strict M4 inference case without modifying the frozen M3 path."""
    import torch

    from dataset import m2_ct_collate_fn, m2_point_collate_fn
    from m4_hard_constraint import M4HardConstraintError, resolve_m4_matching_masks
    from matching_filter import extract_dustbin_aware_mutual_correspondences
    from registration import estimate_weighted_point_to_ct_transform
    from training import _prepare_model_inputs, _require_encoder_outputs

    evaluation_protocol = _require_mapping(
        evaluation_protocol,
        "evaluation_protocol",
    )
    if "matching_filter_min_confidence" not in evaluation_protocol:
        raise M3EvaluationContractError(
            "evaluation_protocol is missing matching_filter_min_confidence."
        )
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise M3EvaluationContractError(
            "device must identify a valid torch device."
        ) from error

    point_collate_fn = partial(
        m2_point_collate_fn,
        enable_m4_defect_mapping=True,
    )
    ct_collate_fn = partial(
        m2_ct_collate_fn,
        enable_m4_defect_mapping=True,
    )
    point_input, ct_input = _prepare_model_inputs(
        sample,
        point_collate_fn,
        ct_collate_fn,
        device,
    )

    with torch.no_grad(), _temporary_evaluation_mode(
        torch,
        point_encoder,
        ct_encoder,
        matcher,
    ):
        _synchronize(torch, device)
        start = time.perf_counter()
        point_output = point_encoder(point_input)
        ct_output = ct_encoder(ct_input)
        q, k, point_physical, ct_physical = _require_encoder_outputs(
            point_output,
            ct_output,
            device,
        )
        try:
            hard_constraint = resolve_m4_matching_masks(
                point_output,
                ct_output,
                device,
            )
        except M4HardConstraintError as error:
            raise M3EvaluationContractError(
                f"M4 hard-constraint mask resolution failed: {error}"
            ) from error
        hard_constraint = _require_mapping(
            hard_constraint,
            "M4 hard-constraint mask resolution",
        )
        if hard_constraint.get("enabled") is not True:
            raise M3EvaluationContractError(
                "M4 hard-constraint evaluation requires complete Point and CT "
                "coarse mapping artifacts."
            )

        point_valid_mask = hard_constraint["point_valid_mask"]
        ct_valid_mask = hard_constraint["ct_valid_mask"]
        matcher_output = matcher(
            q=q,
            k=k,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
        )
        log_assignment = _require_matcher_output(
            torch,
            matcher_output,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            point_count=int(q.shape[0]),
            ct_count=int(k.shape[0]),
            device=device,
        )

        # Pass the original resolved masks, not reconstructed or
        # matcher-substituted tensors, into the mutual filter.
        filter_output = extract_dustbin_aware_mutual_correspondences(
            log_assignment,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            min_confidence=evaluation_protocol["matching_filter_min_confidence"],
        )
        (
            point_indices,
            ct_indices,
            confidence,
            correspondence_count,
        ) = _require_intact_filter_output(
            torch,
            filter_output,
            point_valid_mask=point_valid_mask,
            ct_valid_mask=ct_valid_mask,
            device=device,
        )

        # Registration has no mask arguments. Fail closed immediately above
        # unless every supplied correspondence is intact under these masks.
        registration_output = estimate_weighted_point_to_ct_transform(
            point_physical,
            ct_physical,
            point_indices,
            ct_indices,
            confidence,
        )
        registration_output = _require_mapping(
            registration_output,
            "weighted registration output",
        )
        if "num_correspondences" not in registration_output or int(
            registration_output["num_correspondences"]
        ) != correspondence_count:
            raise M3EvaluationContractError(
                "weighted registration correspondence count does not match "
                "mutual filtering."
            )
        _synchronize(torch, device)
        runtime_ms = (time.perf_counter() - start) * 1000.0

    if not math.isfinite(runtime_ms) or runtime_ms < 0.0:
        raise M3EvaluationContractError("inference runtime is invalid.")
    return {
        "point_physical": point_physical,
        "ct_physical": ct_physical,
        "filter_output": filter_output,
        "registration_output": registration_output,
        "inference_runtime_ms": runtime_ms,
        "m4_hard_constraint_enabled": True,
        "m4_hard_constraint_active": True,
        "point_valid_mask": point_valid_mask,
        "ct_valid_mask": ct_valid_mask,
        "point_total_tokens": hard_constraint["point_total_count"],
        "point_intact_tokens": hard_constraint["point_intact_count"],
        "point_excluded_tokens": hard_constraint["point_excluded_count"],
        "ct_total_tokens": hard_constraint["ct_total_count"],
        "ct_intact_tokens": hard_constraint["ct_intact_count"],
        "ct_excluded_tokens": hard_constraint["ct_excluded_count"],
        "m4_mask_provenance": {
            "point_source": "point_encoder_output.point_intact_coarse",
            "ct_source": "ct_encoder_output.ct_intact_coarse",
            "matcher_masks_preserved": True,
            "mutual_filter_uses_resolved_masks": True,
            "registration_correspondences_verified_intact": True,
        },
    }


__all__ = ["run_m4_hard_constraint_inference"]
