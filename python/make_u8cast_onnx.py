"""Create a tiny float32 RGB -> uint8 RGB conversion ONNX.

The RIFE / Ani4K TensorRT engines output float32 in [0,1]. Downloading that
at 4K costs 3*2160*3840*4 = 99MB per frame. Running this cast on the GPU
first cuts the download to 25MB.

    python make_u8cast_onnx.py rgb_u8cast.onnx
"""

from __future__ import annotations

import sys

import onnx
from onnx import TensorProto, helper


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "rgb_u8cast.onnx"

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, "height", "width"]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.UINT8, [1, 3, "height", "width"]
    )

    scale = helper.make_tensor(
        "scale", TensorProto.FLOAT, [], [255.0]
    )
    half = helper.make_tensor(
        "half", TensorProto.FLOAT, [], [0.5]
    )
    lo = helper.make_tensor(
        "lo", TensorProto.FLOAT, [], [0.0]
    )
    hi = helper.make_tensor(
        "hi", TensorProto.FLOAT, [], [255.0]
    )

    nodes = [
        helper.make_node("Mul", ["input", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "half"], ["rounded"]),
        helper.make_node("Clip", ["rounded", "lo", "hi"], ["clipped"]),
        helper.make_node("Cast", ["clipped"], ["output"], to=TensorProto.UINT8),
    ]

    graph = helper.make_graph(
        nodes,
        "rgb_u8cast",
        [inp],
        [out],
        initializer=[scale, half, lo, hi],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="kazumi-u8cast",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
