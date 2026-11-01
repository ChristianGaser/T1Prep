"""deepbet's LinkNet3d: a residual encoder with additive skips.

A strided 7x7x7 stem and max-pooling, four stages of two residual blocks
(the first block of a downsampling stage strided, with a 1x1x1 projection),
decoder blocks of 1x1x1 / transposed 3x3x3 / 1x1x1 convolutions that undo
their stage's stride, and two transposed convolutions back to full size,
with a softmax over the classes.  Only the final head carries biases.
T1Prep runs two of them for skull-stripping: a bounding-box model at 128^3
and the brain model at 256^3.
"""

from __future__ import annotations

from . import register


@register("linknet3d", signature="init.0.weight")
def linknet3d(ops, w, x):
    """Forward pass, the class probabilities over the channels."""

    def norm_relu(x):
        return ops.relu(ops.instance_norm(x))

    def stride(stage):  # a stage downsamples where its first block projects
        return 2 if f"down{stage}.block1.downsample.0.weight" in w else 1

    x = norm_relu(ops.conv(x, w["init.0.weight"], None, stride=2, padding=3))
    x = ops.max_pool(x, 3, 2, 1)
    skips = [x]
    for i in range(4):
        for block in ("block1", "block2"):
            name = f"down{i}.{block}"
            s = stride(i) if block == "block1" else 1
            y = norm_relu(
                ops.conv(
                    x, w[f"{name}.double_conv.0.weight"], None, stride=s, padding=1
                )
            )
            y = ops.instance_norm(
                ops.conv(y, w[f"{name}.double_conv.3.weight"], None, padding=1)
            )
            if s == 2:
                x = ops.instance_norm(
                    ops.conv(x, w[f"{name}.downsample.0.weight"], None, stride=2)
                )
            x = ops.relu(y + x)
        skips.append(x)
    x = skips.pop()
    for i in range(4):
        s = stride(3 - i)
        x = norm_relu(ops.conv(x, w[f"up{i}.conv1.0.weight"]))
        x = norm_relu(
            ops.conv_transpose(
                x,
                w[f"up{i}.tp_conv.0.weight"],
                None,
                stride=s,
                padding=1,
                output_padding=s - 1,
            )
        )
        x = norm_relu(ops.conv(x, w[f"up{i}.conv2.0.weight"]))
        x = x + skips.pop()
    x = norm_relu(
        ops.conv_transpose(
            x,
            w["tp_conv1.0.weight"],
            w["tp_conv1.0.bias"],
            stride=2,
            padding=1,
            output_padding=1,
        )
    )
    x = norm_relu(ops.conv(x, w["conv.0.weight"], w["conv.0.bias"], padding=1))
    x = ops.conv_transpose(x, w["tp_conv2.weight"], w["tp_conv2.bias"], stride=2)
    return ops.softmax(x)
