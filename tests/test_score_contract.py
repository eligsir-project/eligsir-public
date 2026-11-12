import dataclasses

import pytest
import torch

from eligsir.utils.image_provenance import (
    CanonicalScoreDomain,
    ChannelLayout,
    ColorOrder,
    EncodingFormat,
    ImageProvenance,
    ImageSourceDescriptor,
    TransformStep,
    TransformType,
    provenance_for_canonical_float,
)
from eligsir.utils.score_contract import (
    apply_declared_score_clamp,
    apply_declared_score_clamp_pair,
    declare_canonical_score_tensor,
    validate_score_pair,
)


def _provenance(shape=(3, 4, 5)):
    return provenance_for_canonical_float("float32", shape, shape[-2:])


def test_unattributed_float_pair_is_not_promoted_to_canonical_provenance():
    image = torch.zeros(3, 4, 5)

    with pytest.raises(ValueError, match="explicit score provenance"):
        validate_score_pair(image, image.clone())


@pytest.mark.parametrize(
    "bad",
    [
        torch.full((3, 4, 5), 255.0),
        torch.full((3, 4, 5), -5.0),
        torch.full((3, 4, 5), float("nan")),
    ],
)
def test_non_finite_or_out_of_unit_range_pair_is_rejected(bad):
    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        validate_score_pair(
            bad,
            torch.zeros_like(bad),
            render_provenance=_provenance(),
            gt_provenance=_provenance(),
        )


def test_hwc_is_rejected_instead_of_misread_as_chw():
    image = torch.zeros(4, 5, 3)

    with pytest.raises(ValueError, match="CHW or NCHW"):
        validate_score_pair(
            image,
            image.clone(),
            render_provenance=_provenance(),
            gt_provenance=_provenance(),
        )


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 4, 5)])
def test_explicit_canonical_chw_and_nchw_pairs_are_accepted(shape):
    image = torch.full(shape, 0.5)
    provenance = _provenance(shape[-3:])

    audit = validate_score_pair(
        image,
        image.clone(),
        render_provenance=provenance,
        gt_provenance=provenance,
    )

    assert audit.resolution == (4, 5)


def test_explicit_matching_resize_provenance_is_accepted_at_the_score_boundary():
    """The source grid may differ when both final tensors declare the same resize."""
    image = torch.full((3, 4, 5), 0.5)
    source = ImageSourceDescriptor(
        dtype="float32",
        shape=(3, 8, 10),
        layout=ChannelLayout.CHW,
        color_order=ColorOrder.RGB,
        encoding=EncodingFormat.RAW,
        resolution=(8, 10),
        domain="float32_0_1",
    )
    canonical = CanonicalScoreDomain(
        dtype="float32",
        domain="float32_0_1",
        resolution=(4, 5),
        is_declared_transform=True,
    )
    provenance = ImageProvenance(source=source, canonical_score_domain=canonical).add_step(
        TransformStep(
            transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
            description="bilinear resize",
            target_resolution=(4, 5),
        )
    )
    audit = validate_score_pair(
        image,
        image.clone(),
        render_provenance=provenance,
        gt_provenance=provenance,
        context="test_matching_resize",
    )

    assert audit.resolution == (4, 5)


def test_declared_provenance_must_be_internally_consistent():
    image = torch.full((3, 4, 5), 0.5)
    provenance = _provenance()
    wrong_dtype = dataclasses.replace(
        provenance,
        source=dataclasses.replace(provenance.source, dtype="float64"),
    )

    with pytest.raises(ValueError, match="dtype and numeric-domain"):
        validate_score_pair(
            image,
            image.clone(),
            render_provenance=wrong_dtype,
            gt_provenance=provenance,
        )


def test_explicit_score_clamp_makes_finite_renderer_overshoot_scorable():
    render = torch.full((3, 4, 5), 0.5)
    render[0, 0, 0] = 1.1
    gt, _ = declare_canonical_score_tensor(
        torch.zeros_like(render),
        context="test_ground_truth",
    )

    clamped, score_gt, transform = apply_declared_score_clamp_pair(
        render,
        gt,
        context="test_renderer_output",
    )
    audit = validate_score_pair(
        clamped,
        score_gt,
        context="test_explicit_clamp",
    )

    record = transform["render"]
    assert clamped.max().item() == 1.0
    assert score_gt is gt
    assert transform["gt"]["applied"] is False
    assert render.max().item() == pytest.approx(1.1)
    assert record["samples_changed"] == 1
    assert record["samples_above_max"] == 1
    assert record["samples_below_min"] == 0
    assert record["samples_total"] == render.numel()
    assert record["fraction_above_max"] == pytest.approx(1 / render.numel())
    assert record["mean_excess_above_max"] == pytest.approx(0.1)
    assert record["max_excess_above_max"] == pytest.approx(0.1)
    assert record["value_max"] == pytest.approx(1.1)
    assert audit.resolution == (4, 5)

    provenance_report = transform["score_provenance"]
    assert provenance_report["contract_version"] == 1
    assert provenance_report["dtype"] == "float32"
    assert provenance_report["domain"] == "float32_0_1"
    assert provenance_report["resolution"] == (4, 5)
    assert provenance_report["render"]["steps"] == [
        {
            "type": "declared_score_transform",
            "description": "clamp raw renderer output to the canonical score domain",
            "target_domain": "float32_0_1",
        }
    ]
    assert provenance_report["gt"]["steps"] == []


def test_score_pair_helper_validates_the_finished_pair_before_metrics_run():
    """SSIM/LPIPS callers receive only a pair already checked as one domain."""
    render = torch.zeros(3, 4, 5)
    gt, _ = declare_canonical_score_tensor(
        torch.zeros(3, 4, 4), context="test_ground_truth"
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        apply_declared_score_clamp_pair(render, gt, context="test_pair_preflight")


def test_score_pair_does_not_sanitize_out_of_range_ground_truth():
    render = torch.full((3, 4, 5), 0.5)
    gt, provenance = declare_canonical_score_tensor(
        torch.zeros_like(render),
        context="test_ground_truth_before_corruption",
    )
    gt[0, 0, 0] = 1.1
    # Mutation does not erase lineage; the score boundary must still reject the
    # now-invalid samples rather than trust provenance as a substitute for data.
    assert provenance is not None

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        apply_declared_score_clamp_pair(
            render,
            gt,
            context="test_invalid_gt",
        )


def test_score_pair_requires_ground_truth_provenance():
    render = torch.full((3, 4, 5), 0.5)
    gt = torch.zeros_like(render)

    with pytest.raises(ValueError, match="explicit score provenance"):
        apply_declared_score_clamp_pair(
            render,
            gt,
            context="test_unattributed_gt",
        )


def test_score_pair_rejects_serialized_visualization_artifacts():
    """JPEG/MP4/contact-sheet lineage never reaches a scientific score."""
    image = torch.full((3, 4, 5), 0.5)
    render, _ = declare_canonical_score_tensor(image.clone(), context="test_render")
    for artifact in ("jpeg", "mp4_frame", "contact_sheet"):
        bad_gt = provenance_for_canonical_float("float32", (3, 4, 5), (4, 5)).add_step(
            TransformStep(TransformType.VISUALIZATION_ONLY, artifact)
        )
        with pytest.raises(ValueError, match="VISUALIZATION_ONLY"):
            validate_score_pair(
                render,
                image,
                gt_provenance=bad_gt,
                context=f"test_{artifact}",
            )


def test_score_pair_rejects_encoded_source_and_undeclared_resize():
    image = torch.full((3, 4, 5), 0.5)
    render, _ = declare_canonical_score_tensor(image.clone(), context="test_render")
    encoded = ImageProvenance(
        source=ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 4, 5),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.JPEG,
            resolution=(4, 5),
            domain="float32_0_1",
        ),
        canonical_score_domain=CanonicalScoreDomain("float32", "float32_0_1", (4, 5)),
    )
    with pytest.raises(ValueError, match="serialized/presentation artifact"):
        validate_score_pair(render, image, gt_provenance=encoded, context="test_jpeg")

    undeclared_resize = ImageProvenance(
        source=ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 8, 10),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(8, 10),
            domain="float32_0_1",
        ),
        canonical_score_domain=CanonicalScoreDomain("float32", "float32_0_1", (4, 5)),
    )
    with pytest.raises(ValueError, match="not marked as a declared score transform"):
        validate_score_pair(
            render, image, gt_provenance=undeclared_resize, context="test_undeclared_resize"
        )


def test_score_pair_rejects_integer_quantization_before_scoring():
    quantized = torch.full((3, 4, 5), 128, dtype=torch.uint8)
    with pytest.raises(ValueError, match="must be floating point"):
        validate_score_pair(
            quantized,
            quantized,
            render_provenance=_provenance(),
            gt_provenance=_provenance(),
            context="test_integer_quantization",
        )


def test_explicit_score_clamp_tracks_below_and_above_range_separately():
    render = torch.full((3, 2, 2), 0.5)
    render[0, 0, 0] = -0.2
    render[1, 0, 0] = 1.3
    render[2, 0, 0] = 1.1

    clamped, record = apply_declared_score_clamp(
        render,
        context="test_overshoot_telemetry",
    )

    assert clamped.min().item() == 0.0
    assert clamped.max().item() == 1.0
    assert record["samples_changed"] == 3
    assert record["samples_below_min"] == 1
    assert record["samples_above_max"] == 2
    assert record["mean_excess_above_max"] == pytest.approx(0.2)
    assert record["max_excess_above_max"] == pytest.approx(0.3)


def test_explicit_score_clamp_supports_inference_tensors():
    with torch.inference_mode():
        render = torch.full((3, 4, 5), 1.1)
        gt = torch.zeros_like(render)
        render_score, _ = apply_declared_score_clamp(
            render, context="test_inference_render"
        )
        gt_score, _ = apply_declared_score_clamp(gt, context="test_inference_gt")
        audit = validate_score_pair(
            render_score, gt_score, context="test_inference_pair"
        )

    assert audit.resolution == (4, 5)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_explicit_score_clamp_does_not_hide_non_finite_renderer_output(bad_value):
    render = torch.zeros(3, 4, 5)
    render[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        apply_declared_score_clamp(
            render,
            context="test_non_finite_renderer_output",
        )
