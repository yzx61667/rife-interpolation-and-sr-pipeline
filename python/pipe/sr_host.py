#!/usr/bin/env python3
"""sr_host.py -- zero-disk 2x super-resolution host for ffmpeg rawvideo pipes.

  stdin  : rawvideo, pix_fmt rgb24, fixed W x H, frames back-to-back (exactly
           W*H*3 bytes each, no headers, no padding)
  stdout : rawvideo, pix_fmt rgb24, (W*scale) x (H*scale), same frame order
  stderr : logs / stats only (stdout is never touched by anything but frames)

The point: frames never touch the filesystem. ffmpeg decodes, we infer through
ONNX Runtime (DirectML by default), ffmpeg re-encodes.

Example:
    ffmpeg -i in.mp4 -vf format=rgb24 -f rawvideo - |
      python sr_host.py --width 640 --height 360 --model ani4k2.onnx |
      ffmpeg -f rawvideo -pix_fmt rgb24 -s 1280x720 -framerate 24 -i - \
             -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p out.mp4
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

FRAME_BYTES = 3  # rgb24


# --------------------------------------------------------------------------- #
# logging (stderr only)
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - never die while logging
        pass


# --------------------------------------------------------------------------- #
# pipe helpers
# --------------------------------------------------------------------------- #
def read_exact(fh, view: memoryview) -> int:
    """Fill `view` completely; return bytes read (< len at EOF only)."""
    total = len(view)
    pos = 0
    while pos < total:
        got = fh.readinto(view[pos:])
        if not got:
            break
        pos += got
    return pos


def put_wait(q: "queue.Queue", item, stop: threading.Event) -> bool:
    """Blocking put that still honours the stop flag. False => stopping."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


# --------------------------------------------------------------------------- #
# frame conversion
# --------------------------------------------------------------------------- #
INV255 = np.float32(1.0 / 255.0)


def preprocess(raw: np.ndarray, inp: np.ndarray, h: int, w: int) -> None:
    """uint8 HWC (h,w,3) -> float32 0..1 NCHW (1,3,ph,pw), edge-padded.

    `inp` is reused every frame; only the active (h,w) area is written and the
    padded border is edge-replicated so the net sees no dark seam.
    """
    dst = inp[0, :, :h, :w]  # (3,h,w) strided view into the NCHW buffer
    # transpose is a stride change only; multiply() does the uint8 -> float32
    # cast and the /255 in one pass, writing straight into NCHW layout.
    np.multiply(raw.transpose(2, 0, 1), INV255, out=dst, casting="unsafe")

    ph, pw = inp.shape[2], inp.shape[3]
    if pw != w:
        inp[0, :, :h, w:] = inp[0, :, :h, w - 1 : w]
    if ph != h:
        inp[0, :, h:, :] = inp[0, :, h - 1 : h, :]


def postprocess(out: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """float32 (1,3,ph,pw) 0..1 -> C-contiguous uint8 (oh,ow,3), cropped."""
    o = out[0][:, :oh, :ow]          # (3,oh,ow) cropped view, modified in place
    np.multiply(o, 255.0, out=o)
    np.add(o, 0.5, out=o)            # round-to-nearest on the truncating cast
    np.clip(o, 0.0, 255.0, out=o)
    return o.transpose(1, 2, 0).astype(np.uint8, order="C")


# --------------------------------------------------------------------------- #
# args / session
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sr_host.py",
        description="Zero-disk 2x super-resolution filter for rawvideo pipes.",
    )
    p.add_argument("--width", type=int, required=True, help="input frame width")
    p.add_argument("--height", type=int, required=True, help="input frame height")
    p.add_argument("--model", required=True, help="path to the .onnx model")
    p.add_argument("--scale", type=int, default=2, help="upscale factor (only 2)")
    p.add_argument("--device-id", type=int, default=0,
                   help="DirectML adapter index (default 0)")
    p.add_argument("--provider", default="dml", choices=["dml", "cpu"],
                   help="execution provider (default dml)")
    p.add_argument("--queue", type=int, default=4,
                   help="pipeline queue depth in frames (default 4)")
    p.add_argument("--no-threads", action="store_true",
                   help="single-threaded loop (for A/B measurements)")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the one-frame DirectML graph warmup")
    p.add_argument("--stats", action="store_true",
                   help="print throughput summary to stderr on exit")
    p.add_argument("--quiet", action="store_true",
                   help="suppress info logs (errors still printed)")
    return p.parse_args(argv)


def build_session(args):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # errors only, keeps stderr clean

    if args.provider == "dml":
        if args.device_id:
            providers = [
                ("DmlExecutionProvider", {"device_id": args.device_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(args.model, so, providers=providers)
    return sess, sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    if args.scale != 2:
        log(f"sr_host: --scale {args.scale} unsupported (model is fixed 2x)")
        return 2

    w, h = args.width, args.height
    if w <= 0 or h <= 0:
        log("sr_host: --width/--height must be positive")
        return 2
    if not os.path.isfile(args.model):
        log(f"sr_host: model not found: {args.model}")
        return 2

    # the anime network needs spatial dims divisible by 8: pad, infer, crop
    pw = (w + 7) // 8 * 8
    ph = (h + 7) // 8 * 8
    ow, oh = 2 * w, 2 * h

    t_load = time.perf_counter()
    sess, in_name, out_names = build_session(args)
    if not args.quiet:
        log(f"sr_host: model {os.path.basename(args.model)} "
            f"({time.perf_counter() - t_load:.2f}s load)")
        log(f"sr_host: ep    {sess.get_providers()}")
        log(f"sr_host: pipe  {w}x{h} rgb24 -> {ow}x{oh} rgb24")
        if (pw, ph) != (w, h):
            log(f"sr_host: pad   {w}x{h} -> {pw}x{ph} infer -> crop {ow}x{oh}")
        log(f"sr_host: mode  {'single-thread' if args.no_threads else 'pipelined'}")

    n_in = w * h * FRAME_BYTES
    inp = np.zeros((1, 3, ph, pw), dtype=np.float32)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    n_frames = 0
    infer_s = 0.0
    rc = 0

    def infer_one(raw: np.ndarray) -> np.ndarray:
        """raw rgb24 frame -> raw float32 NCHW model output (GPU work only)."""
        nonlocal infer_s
        preprocess(raw, inp, h, w)
        t0 = time.perf_counter()
        out = sess.run(out_names, {in_name: inp})[0]
        infer_s += time.perf_counter() - t0
        return out

    def finish(out: np.ndarray) -> np.ndarray:
        """model output -> packed rgb24 bytes (pure CPU, runs on the writer)."""
        return postprocess(out, oh, ow)

    if not args.no_warmup:
        # first DirectML dispatch builds the compiled graph (~150 ms); do it
        # before the clock starts and before the pipe fills up
        t0 = time.perf_counter()
        finish(infer_one(np.zeros((h, w, FRAME_BYTES), dtype=np.uint8)))
        infer_s = 0.0
        if not args.quiet:
            log(f"sr_host: warmup {(time.perf_counter() - t0) * 1000:.0f} ms")

    t_start = time.perf_counter()

    if args.no_threads:
        buf = bytearray(n_in)
        mv = memoryview(buf)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        try:
            while read_exact(stdin, mv) == n_in:
                img = finish(infer_one(frame))
                n_frames += 1
                stdout.write(memoryview(img))
                stdout.flush()
        except (BrokenPipeError, KeyboardInterrupt):
            pass
    else:
        in_q: "queue.Queue" = queue.Queue(maxsize=max(1, args.queue))
        out_q: "queue.Queue" = queue.Queue(maxsize=max(1, args.queue))
        stop = threading.Event()
        errors: list = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    buf = bytearray(n_in)
                    got = read_exact(stdin, memoryview(buf))
                    if got < n_in:
                        break
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                    if not put_wait(in_q, frame, stop):
                        return
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                put_wait(in_q, None, stop)

        def writer() -> None:
            try:
                while True:
                    item = out_q.get()
                    if item is None:
                        break
                    # float -> packed rgb24 happens here, overlapping the
                    # next frame's GPU dispatch on the inference thread
                    img = finish(item)
                    stdout.write(memoryview(img))
                    stdout.flush()
            except BrokenPipeError:
                stop.set()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                stop.set()

        rt = threading.Thread(target=reader, name="sr-reader", daemon=True)
        wt = threading.Thread(target=writer, name="sr-writer", daemon=True)
        rt.start()
        wt.start()

        try:
            while True:
                try:
                    frame = in_q.get(timeout=0.1)
                except queue.Empty:
                    if not rt.is_alive() and in_q.empty():
                        break
                    continue
                if frame is None:
                    break
                out = infer_one(frame)
                n_frames += 1
                if not put_wait(out_q, out, stop):
                    break
        except (BrokenPipeError, KeyboardInterrupt):
            stop.set()
        finally:
            stop.set()
            # the writer sentinel must be delivered even when stopping,
            # otherwise the join below blocks for the full timeout
            if wt.is_alive():
                while True:
                    try:
                        out_q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        if not wt.is_alive():
                            break
            wt.join(timeout=15)

        if errors:
            log(f"sr_host: I/O error: {errors[0]!r}")
            rc = 1

    elapsed = time.perf_counter() - t_start
    if args.stats or not args.quiet:
        fps = n_frames / elapsed if elapsed > 0 else 0.0
        ipf = (infer_s / n_frames * 1000.0) if n_frames else 0.0
        log(f"sr_host: {n_frames} frames in {elapsed:.2f}s -> {fps:.2f} fps "
            f"end-to-end (inference {ipf:.1f} ms/frame)")

    try:
        stdout.flush()
    except (BrokenPipeError, ValueError):
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
