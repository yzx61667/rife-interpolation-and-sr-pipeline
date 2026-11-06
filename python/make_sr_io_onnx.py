"""Create the two tiny ONNX graphs used for GPU-side uint8 I/O around Ani4K.

sr_pre_u8hwc_to_f32chw:
    input  (N, H, W, 3) uint8  RGB, HWC
    output (N, 3, H, W) float32 NCHW, value = x / 255

sr_post_f32chw_to_u8hwc:
    input  (N, 3, H, W) float32 NCHW
    output (N, H, W, 3) uint8  RGB, HWC, value = round(clip(x * 255, 0, 255))

Both are shape-dynamic in N/H/W; TensorRT profiles are passed at build time.

    python make_sr_io_onnx.py [output_dir]
"""

from __future__ import annotations

import os
import sys

import onnx
from onnx import TensorProto, helper

PRE_NAME = "sr_pre_u8hwc_to_f32chw"
POST_NAME = "sr_post_f32chw_to_u8hwc"


def _save(graph, name: str, out_dir: str) -> str:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="kazumi-sr-io",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = os.path.join(out_dir, name + ".onnx")
    onnx.save(model, path)
    print(f"wrote {path}")
    return path


def make_pre(out_dir: str) -> str:
    inp = helper.make_tensor_value_info(
        "input", TensorProto.UINT8, ["N", "H", "W", 3]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["N", 3, "H", "W"]
    )
    inv255 = helper.make_tensor("inv255", TensorProto.FLOAT, [], [1.0 / 255.0])
    nodes = [
        helper.make_node("Cast", ["input"], ["f32"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["f32"], ["chw"], perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["chw", "inv255"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, PRE_NAME, [inp], [out], initializer=[inv255]
    )
    return _save(graph, PRE_NAME, out_dir)


def make_post(out_dir: str) -> str:
    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["N", 3, "H", "W"]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.UINT8, ["N", "H", "W", 3]
    )
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [255.0])
    lo = helper.make_tensor("lo", TensorProto.FLOAT, [], [0.0])
    hi = helper.make_tensor("hi", TensorProto.FLOAT, [], [255.0])
    # TensorRT only allows UINT8 on graph I/O, not on intermediate tensors:
    # transpose in float32, then cast as the very last node.
    nodes = [
        helper.make_node("Mul", ["input", "scale"], ["scaled"]),
        helper.make_node("Round", ["scaled"], ["rounded"]),
        helper.make_node("Clip", ["rounded", "lo", "hi"], ["clipped"]),
        helper.make_node("Transpose", ["clipped"], ["hwc"], perm=[0, 2, 3, 1]),
        helper.make_node("Cast", ["hwc"], ["output"], to=TensorProto.UINT8),
    ]
    graph = helper.make_graph(
        nodes, POST_NAME, [inp], [out], initializer=[scale, lo, hi]
    )
    return _save(graph, POST_NAME, out_dir)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = argv[0] if argv else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    make_pre(out_dir)
    make_post(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
