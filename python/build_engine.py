"""Build a TensorRT engine from a user-supplied ONNX model.

The model weights are not part of this repository. Obtain an ONNX model that
you are allowed to use, then build an engine for your GPU:

    python build_engine.py --model /path/to/ani4k2.onnx
    python build_engine.py --model /path/to/ani4k2.onnx --fp32 --force
    python build_engine.py --model /path/to/ani4k2.onnx --fp8 --force

FP8 is experimental here: TensorRT needs a compatible Q/DQ model or explicit
precision constraints. See docs/architecture.md for the tested result.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trt_engine as te  # noqa: E402

# Default engine output directory next to this script. It is gitignored:
# engine files are build artifacts and are never committed.
te.ENGINE_DIR = os.path.join(HERE, "engines")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build TensorRT engine for SR model")
    ap.add_argument("--model", required=True, help="path to the user-supplied ONNX model")
    ap.add_argument("--engine", default=None, help="output .engine path (default: engines/<model>_<precision>_<gpu>.engine)")
    precision = ap.add_mutually_exclusive_group()
    precision.add_argument("--fp32", action="store_true", help="build in FP32 instead of FP16")
    precision.add_argument("--fp8", action="store_true", help="build with FP8 enabled (FP16/TF32 stay available)")
    ap.add_argument(
        "--strongly-typed",
        action="store_true",
        help="create a strongly-typed network (use with an explicit Q/DQ ONNX for FP8)",
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if a valid cached engine exists")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--min", nargs=4, type=int, default=list(te.DEFAULT_PROFILE["min"]), metavar=("N", "C", "H", "W"))
    ap.add_argument("--opt", nargs=4, type=int, default=list(te.DEFAULT_PROFILE["opt"]), metavar=("N", "C", "H", "W"))
    ap.add_argument("--max", nargs=4, type=int, default=list(te.DEFAULT_PROFILE["max"]), metavar=("N", "C", "H", "W"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    profile = {"min": tuple(args.min), "opt": tuple(args.opt), "max": tuple(args.max)}
    fp16 = not args.fp32
    fp8 = args.fp8
    strongly_typed = args.strongly_typed
    engine = args.engine or te.engine_path_for(
        args.model, fp16, fp8, strongly_typed
    )

    if not args.force and te.engine_is_valid(
        engine, args.model, fp16, profile, fp8, strongly_typed
    ):
        print(f"engine up to date: {engine}", file=sys.stderr)
        return 0

    try:
        te.build_engine(
            args.model,
            engine,
            fp16=fp16,
            profile=profile,
            workspace_gb=args.workspace_gb,
            verbose=True,
            fp8=fp8,
            strongly_typed=strongly_typed,
        )
    except Exception as e:  # noqa: BLE001
        print(f"BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
