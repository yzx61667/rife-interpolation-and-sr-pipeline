"""Streaming RIFE frame interpolation host.

stdin  : rawvideo rgb24 frames, fixed W x H, back to back
stdout : rawvideo rgb24 frames at the same W x H, interpolated
         factor=2 -> for each input pair: prev, mid(0.5)
         factor=3 -> for each input pair: prev, mid(1/3), mid(2/3)
         the final input frame is emitted once at EOF
Logs go to stderr only.
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort

PAD = 32


def read_exact(stream, size):
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = stream.readinto(view[got:])
        if not n:
            return None if got == 0 else bytes(buf[:got])
        got += n
    return bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--factor", type=int, default=2, choices=[2, 3])
    ap.add_argument("--out-frames", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    w, h = args.width, args.height
    frame_bytes = w * h * 3
    pw = (w + PAD - 1) // PAD * PAD
    ph = (h + PAD - 1) // PAD * PAD
    pad_x = pw - w
    pad_y = ph - h
    timesteps = [0.5] if args.factor == 2 else [1.0 / 3.0, 2.0 / 3.0]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        args.model, so, providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    def to_input(raw):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        if pad_y or pad_x:
            arr = np.pad(arr, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

    def infer(a, b, t):
        out = session.run(
            None,
            {
                "img0": a,
                "img1": b,
                "timestep": np.array([t], dtype=np.float32),
            },
        )[0]
        img = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)[:h, :w] * 255.0
        return img.astype(np.uint8).tobytes()

    emitted = 0

    def emit(data):
        nonlocal emitted
        if args.out_frames and emitted >= args.out_frames:
            return False
        stdout.write(data)
        emitted += 1
        return True

    prev_raw = read_exact(stdin, frame_bytes)
    if prev_raw is None or len(prev_raw) < frame_bytes:
        return 0
    prev = to_input(prev_raw)

    # 预热一次，避免首帧建图时间混进吞吐
    infer(prev, prev, timesteps[0])

    while True:
        cur_raw = read_exact(stdin, frame_bytes)
        if cur_raw is None:
            break
        if len(cur_raw) < frame_bytes:
            break
        cur = to_input(cur_raw)
        if not emit(prev_raw):
            return 0
        for t in timesteps:
            if not emit(infer(prev, cur, t)):
                return 0
        prev_raw, prev = cur_raw, cur

    emit(prev_raw)
    stdout.flush()
    if not args.quiet:
        print(f"rife_host: emitted {emitted} frames", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
