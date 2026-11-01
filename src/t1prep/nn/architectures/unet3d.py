"""deepmriprep's Unet3d, with the StepActivation of its segmentation models.

As defined in deepmriprep-train's ``src/models.py``: double 3x3x3
convolutions with instance norm and ReLU, max-pooling on the way down, 2x2x2
transposed convolutions on the way up, each skip concatenated in front.  The
depth and the widths are read from the weights.  T1Prep runs the coarse
segmentation model (the image as one channel, at 1/1.5 resolution) and the 18
patch models (image and coarse ``p0`` as two channels, 128^3 patches).
"""

from __future__ import annotations

from . import register


def canonical(weights: dict) -> dict:
    """Name the coarse model's weights like the patch models'.

    deepmriprep-train wraps the coarse model as ``Sequential(Unet3d,
    StepActivation)``, whose weights start with ``0.`` and ``1.``, and the
    patch models as ``TwoInputsUnet3d`` (``unet.``, ``act.``).
    """
    if not any(k.startswith("0.inp.") for k in weights):
        return dict(weights)
    renamed = {"0": "unet", "1": "act"}
    out = {}
    for key, value in weights.items():
        head, rest = key.split(".", 1)
        out[f"{renamed.get(head, head)}.{rest}"] = value
    return out


def step_activation(ops, w, x):
    """deepmriprep's StepActivation: soft steps onto the label values 0..3.

    The terms are added in the order the archives traced them, which keeps
    the torch backend bit-identical to them.
    """
    factor, half = w["act.factor"], w["act.half_margin"]
    fx = factor * x
    return (
        ops.sigmoid(fx)
        + ops.sigmoid(fx - (1 - half) * factor) * 0.5
        + ops.sigmoid(fx - (half + 1) * factor) * 0.5
        + ops.sigmoid(fx - (2 - half) * factor) * 0.5
        + ops.sigmoid(fx - (half + 2) * factor) * 0.5
    )


@register("unet3d", signature="unet.inp.double_conv.0.weight", canonical=canonical)
def unet3d(ops, w, x):
    """Forward pass; the StepActivation is applied when the file holds one."""

    def double_conv(x, name):
        for i in (0, 3):
            weight = w[f"{name}.double_conv.{i}.weight"]
            x = ops.conv(
                x,
                weight,
                w[f"{name}.double_conv.{i}.bias"],
                padding=weight.shape[-1] // 2,
            )
            x = ops.relu(ops.instance_norm(x))
        return x

    depth = 0
    while f"unet.down{depth}.conv.double_conv.0.weight" in w:
        depth += 1
    x = double_conv(x, "unet.inp")
    skips = []
    for i in range(depth):
        skips.append(x)
        x = ops.max_pool(double_conv(x, f"unet.down{i}.conv"), 2, 2, 0)
    for i in range(depth):
        x = ops.conv_transpose(
            x, w[f"unet.up{i}.up_conv.weight"], w[f"unet.up{i}.up_conv.bias"], stride=2
        )
        x = double_conv(ops.cat(skips.pop(), x), f"unet.up{i}.conv")
    x = ops.conv(x, w["unet.out.weight"], w["unet.out.bias"])
    return step_activation(ops, w, x) if "act.factor" in w else x
