#!/usr/bin/env python3
"""trt_host.py -- zero-disk 2x super-resolution host for ffmpeg rawvideo pipes.

TensorRT variant of sr_host.py. Same wire protocol, same CLI surface.

  stdin  : rawvideo, pix_fmt rgb24, fixed W x H, frames back-to-back (exactly
           W*H*3 bytes each, no headers, no padding)
  stdout : rawvideo, pix_fmt rgb24, (W*scale) x (H*scale), same frame order
  stderr : logs / stats only (stdout is never touched by anything but frames)

Frames never touch the filesystem: ffmpeg decodes, TensorRT infers, ffmpeg
re-encodes.

Example:
    ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgb24 - |
      python trt_host.py --width 640 --height 360 --engine ani4k2_fp16.engine |
      ffmpeg -f rawvideo -pix_fmt rgb24 -s 1280x720 -framerate 24 -i - \
             -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p out.mp4
"""

from __future__ import annotations

import argparse
import os
import queue
import struct
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import trt_engine as te  # noqa: E402

FRAME_BYTES = 3  # rgb24

# Repeat-aware RIFE -> SR protocol. Each unit starts with an 8-byte
# little-endian header (magic, repeat). repeat=1 carries one rgb24 frame;
# repeat>1 carries no payload and expands to ``repeat`` copies of the last 4K
# output. Magic spells "EFIR" on the wire.
REPEAT_MAGIC = 0x52494645
REPEAT_HEADER = struct.Struct("<II")
HEADER_BYTES = REPEAT_HEADER.size


def pack_repeat(repeat: int) -> bytes:
    """Encode one protocol unit header."""
    return REPEAT_HEADER.pack(REPEAT_MAGIC, int(repeat))


def read_repeat_header(fh):
    """Read a unit header; return repeat, or None on a clean EOF."""
    buf = bytearray(HEADER_BYTES)
    got = read_exact(fh, memoryview(buf))
    if got == 0:
        return None
    if got < HEADER_BYTES:
        raise ValueError("truncated repeat header")
    magic, repeat = REPEAT_HEADER.unpack(buf)
    if magic != REPEAT_MAGIC:
        raise ValueError(f"bad repeat magic 0x{magic:08x}")
    if repeat < 1:
        raise ValueError(f"invalid repeat count {repeat}")
    if repeat > 64:  # RIFE factors are 2/3; protects against a corrupt stream
        raise ValueError(f"repeat count too large: {repeat}")
    return repeat


class RepeatJob:
    """An ordered unit that expands to ``repeat`` copies of the last output."""

    __slots__ = ("repeat",)

    def __init__(self, repeat: int):
        self.repeat = repeat


class OrderedRepeatWriter:
    """Ordered protocol writer: reorders seq items and expands repeat units.

    Used by run_multi and unit tests. Normal batches update ``last_out`` with
    a dedicated copy of their last frame, so a later repeat job can never
    refer to a reused pinned-ring buffer.
    """

    def __init__(self, fh):
        self.fh = fh
        self.pending = {}
        self.expected = 0
        self.last_out = None
        self.frames = 0
        self.repeat_frames = 0
        self.repeat_units = 0

    def add_batch(self, seq: int, frames) -> None:
        last = len(frames) - 1
        for i, frame in enumerate(frames):
            self.pending[seq + i] = (frame, i == last)
        self._drain()

    def add_repeat(self, seq: int, repeat: int) -> None:
        self.pending[seq] = RepeatJob(repeat)
        self._drain()

    def _write(self, data) -> None:
        write_all(self.fh, memoryview(data).cast("B"))
        self.frames += 1

    def _drain(self) -> None:
        while self.expected in self.pending:
            item = self.pending.pop(self.expected)
            if isinstance(item, RepeatJob):
                if self.last_out is None:
                    raise RuntimeError("repeat unit before any output")
                for _ in range(item.repeat):
                    self._write(self.last_out)
                self.repeat_frames += item.repeat
                self.repeat_units += 1
            else:
                frame, is_batch_last = item
                self._write(frame)
                if is_batch_last:
                    src = (
                        frame
                        if isinstance(frame, np.ndarray)
                        else np.frombuffer(frame, dtype=np.uint8)
                    )
                    # Reuse one 25 MB buffer: allocating a fresh 4K frame per
                    # batch (thousands of times) costs ~3 ms/copy vs ~0.9 ms.
                    if self.last_out is None or self.last_out.shape != src.shape:
                        self.last_out = np.empty_like(src, dtype=np.uint8)
                    np.copyto(self.last_out, src, casting="unsafe")
            self.expected += 1

    def finish(self) -> None:
        """Flush remaining items in sequence order (normally a no-op)."""
        self._drain()
        if self.pending:
            raise RuntimeError(
                f"gap in repeat-protocol output sequence at {self.expected}"
            )


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


def write_all(fh, data) -> None:
    """Write every byte; raises BrokenPipeError if the reader went away."""
    mv = memoryview(data)
    while len(mv):
        n = fh.write(mv)
        if not n:
            raise BrokenPipeError("downstream closed")
        mv = mv[n:]
    fh.flush()


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
# args
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="trt_host.py",
        description="Zero-disk 2x super-resolution filter (TensorRT) for rawvideo pipes.",
    )
    p.add_argument("--width", type=int, required=True, help="input frame width")
    p.add_argument("--height", type=int, required=True, help="input frame height")
    p.add_argument("--engine", required=True, help="path to the TensorRT .engine")
    p.add_argument("--model", default=None,
                   help="optional .onnx path, only used to hint how to build a missing engine")
    p.add_argument("--scale", type=int, default=2, help="upscale factor (only 2)")

    # accepted for drop-in compatibility with sr_host.py
    p.add_argument("--device-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--provider", default="trt", help=argparse.SUPPRESS)

    p.add_argument("--queue", type=int, default=4,
                   help="pipeline queue depth in frames (default 4)")
    p.add_argument("--workers", type=int, default=2,
                   help="concurrent TensorRT contexts (default 2)")
    p.add_argument("--batch", type=int, default=2,
                   help="frames per inference (engine profile must allow it)")
    p.add_argument("--static-threshold", type=float, default=0.01,
                   help="reuse the previous output when the changed-pixel "
                        "fraction is at or below this percent (0 disables)")
    p.add_argument("--no-threads", action="store_true",
                   help="single-threaded loop (for A/B measurements)")
    p.add_argument("--legacy-io", action="store_true",
                   help="force NumPy float I/O instead of the optional GPU "
                        "uint8 pre/post engines (A/B and fallback testing)")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the 2-frame warmup before the first output frame")
    p.add_argument("--repeat-protocol", action="store_true",
                   help="read the 8-byte RIFE repeat header; repeat>1 reuses "
                        "the last 4K output without inference")
    p.add_argument("--stats", action="store_true",
                   help="print throughput summary to stderr on exit")
    p.add_argument("--quiet", action="store_true",
                   help="suppress info logs (errors still printed)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# multi-worker pipeline
# --------------------------------------------------------------------------- #
def _put_wait(q: "queue.Queue", item, stop: threading.Event, timeout: float = 0.1) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _get_wait(q: "queue.Queue", stop: threading.Event, timeout: float = 0.2):
    while not stop.is_set():
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            continue
    return None


def run_multi(args, w: int, h: int, pw: int, ph: int, ow: int, oh: int,
              batch: int = 1) -> int:
    """Run N independent TensorRT contexts and write frames in input order.

    A single context leaves the GPU under-filled (measured 5.4 fps); two
    contexts reach ~7.2 fps. Batch 2 on each context adds about 1.4x.
    """
    workers = max(2, args.workers)
    batch = max(1, batch)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    n_in_bytes = w * h * FRAME_BYTES
    pad_y, pad_x = ph - h, pw - w

    t_load = time.perf_counter()
    runners = []
    try:
        for _ in range(workers):
            runners.append(te.TrtRunner(
                args.engine, batch=batch, legacy_io=args.legacy_io))
    except Exception as exc:  # noqa: BLE001
        for r in runners:
            r.close()
        log(f"trt_host: cannot load engine: {exc}")
        return 1

    if not args.quiet:
        log(f"trt_host: engine {os.path.basename(args.engine)} "
            f"({time.perf_counter() - t_load:.2f}s load, {workers} contexts)")
        log(f"trt_host: pipe  {w}x{h} rgb24 -> {ow}x{oh} rgb24")
        if (pw, ph) != (w, h):
            log(f"trt_host: pad   {w}x{h} -> {pw}x{ph} infer -> crop {ow}x{oh}")
        log(f"trt_host: mode  pipelined ({workers} workers, batch {batch})")
        if args.repeat_protocol:
            log("trt_host: repeat protocol on (8-byte header, no payload on repeat)")

    if not args.no_warmup:
        t0 = time.perf_counter()
        zero = np.zeros((ph, pw, FRAME_BYTES), dtype=np.uint8)
        for runner in runners:
            if batch > 1:
                runner.process_batch([zero, zero])
            else:
                for _ in range(2):
                    runner.process(zero)
        if not args.quiet:
            log(f"trt_host: warmup {(time.perf_counter() - t0) * 1000:.0f} ms "
                f"({workers} contexts, batch {batch})")
            log(f"trt_host: io    {runners[0].io_mode}")

    use_static = (
        args.static_threshold > 0 and batch == 1 and not args.repeat_protocol
    )
    if args.static_threshold > 0 and batch > 1 and not args.quiet:
        log("trt_host: static reuse disabled with batch > 1")
    if args.repeat_protocol and args.static_threshold > 0 and not args.quiet:
        log("trt_host: repeat protocol wins; legacy static reuse disabled")
    in_q: "queue.Queue" = queue.Queue(maxsize=max(1, args.queue))
    out_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    errors: list = []
    lock = threading.Lock()
    counts = {
        "in": 0, "out": 0, "dup": 0,
        "units": 0, "repeat_units": 0, "repeat_frames": 0,
    }

    def reader() -> None:
        seq = 0
        prev_frame = None
        pending = []

        def flush_pending() -> bool:
            nonlocal pending
            if not pending:
                return True
            start = pending[0][0]
            frames = [p[1] for p in pending]
            pending = []
            return _put_wait(in_q, (start, frames), stop)

        try:
            while not stop.is_set():
                if args.repeat_protocol:
                    repeat = read_repeat_header(stdin)
                    if repeat is None:
                        break
                    with lock:
                        counts["units"] += 1
                    if repeat > 1:
                        # A repeat is a complete ordered unit: flush the
                        # partial inference batch before queueing it.
                        with lock:
                            counts["repeat_units"] += 1
                            counts["repeat_frames"] += repeat
                        if not flush_pending():
                            break
                        if not _put_wait(
                            out_q, (seq, RepeatJob(repeat)), stop
                        ):
                            break
                        seq += 1
                        continue

                buf = bytearray(n_in_bytes)
                if read_exact(stdin, memoryview(buf)) < n_in_bytes:
                    if args.repeat_protocol:
                        raise ValueError("truncated frame payload")
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                with lock:
                    counts["in"] += 1
                frame_seq = seq
                seq += 1

                if use_static:
                    is_static = False
                    if prev_frame is not None:
                        # Conservative: pixels whose any channel moved >2/255.
                        cur_s = frame[::2, ::2]
                        prev_s = prev_frame[::2, ::2]
                        diff = np.abs(
                            cur_s.astype(np.int16) - prev_s.astype(np.int16)
                        )
                        changed = np.count_nonzero(np.any(diff > 2, axis=2))
                        frac = changed / float(cur_s.shape[0] * cur_s.shape[1])
                        is_static = frac <= args.static_threshold / 100.0
                    prev_frame = frame.copy()
                    if is_static:
                        with lock:
                            counts["dup"] += 1
                        if not _put_wait(out_q, (frame_seq, None), stop):
                            break
                        continue

                if pad_y or pad_x:
                    padded = np.pad(frame, ((0, pad_y), (0, pad_x), (0, 0)),
                                    mode="edge")
                else:
                    padded = np.ascontiguousarray(frame)

                if batch == 1:
                    if not _put_wait(in_q, (frame_seq, [padded]), stop):
                        break
                else:
                    pending.append((frame_seq, padded))
                    if len(pending) >= batch:
                        if not flush_pending():
                            break
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()
        finally:
            if pending and not stop.is_set():
                flush_pending()
            if not stop.is_set():
                for _ in range(workers):
                    in_q.put(None)

    def worker(runner) -> None:
        try:
            while not stop.is_set():
                item = _get_wait(in_q, stop)
                if item is None:
                    break
                start, frames = item
                outs = runner.process_batch(frames)
                outs = [o[:oh, :ow] for o in outs]
                outs = [
                    o if o.flags["C_CONTIGUOUS"] else np.ascontiguousarray(o)
                    for o in outs
                ]
                if not _put_wait(out_q, (start, outs), stop):
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()
        finally:
            out_q.put(None)

    def writer() -> None:
        if args.repeat_protocol:
            ordered = OrderedRepeatWriter(stdout)
            done = 0
            try:
                while done < workers:
                    item = out_q.get()
                    if item is None:
                        done += 1
                        continue
                    seq, out = item
                    if isinstance(out, RepeatJob):
                        ordered.add_repeat(seq, out.repeat)
                    else:
                        ordered.add_batch(seq, out)
                ordered.finish()
                with lock:
                    counts["out"] = ordered.frames
            except BrokenPipeError:
                stop.set()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                stop.set()
            return

        pending_out = {}
        expected = 0
        done = 0
        last = None
        try:
            while done < workers:
                item = out_q.get()
                if item is None:
                    done += 1
                    continue
                seq, out = item
                if out is None:
                    pending_out[seq] = None
                else:
                    for i, frame_out in enumerate(out):
                        pending_out[seq + i] = frame_out
                while expected in pending_out:
                    data = pending_out.pop(expected)
                    if data is None:
                        if last is None:
                            raise RuntimeError("static frame before any output")
                        data = last
                    else:
                        last = data
                    write_all(stdout, memoryview(data).cast("B"))
                    with lock:
                        counts["out"] += 1
                    expected += 1
            for seq in sorted(pending_out):
                data = pending_out.pop(seq)
                if data is None:
                    if last is None:
                        raise RuntimeError("static frame before any output")
                    data = last
                else:
                    last = data
                write_all(stdout, memoryview(data).cast("B"))
                with lock:
                    counts["out"] += 1
        except BrokenPipeError:
            stop.set()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()

    threads = [threading.Thread(target=reader, name="trt-reader", daemon=True)]
    for i, runner in enumerate(runners):
        threads.append(threading.Thread(
            target=worker, args=(runner,), name=f"trt-worker-{i}", daemon=True))
    threads.append(threading.Thread(target=writer, name="trt-writer", daemon=True))

    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    for runner in runners:
        runner.close()

    if errors:
        log(f"trt_host: I/O error: {errors[0]!r}")
        return 1

    if args.stats or not args.quiet:
        if args.repeat_protocol:
            fps = counts["out"] / elapsed if elapsed > 0 else 0.0
            log(
                f"trt_host: units={counts['units']} "
                f"normal={counts['in']} "
                f"repeat_units={counts['repeat_units']} "
                f"repeated_frames={counts['repeat_frames']} "
                f"expanded={counts['out']} in {elapsed:.2f}s -> "
                f"{fps:.2f} fps end-to-end ({workers} workers)"
            )
        else:
            if counts["dup"]:
                log(f"trt_host: static {counts['dup']}/{counts['in']} frames "
                    f"({counts['dup'] / max(1, counts['in']) * 100:.1f}%) reused")
            fps = counts["out"] / elapsed if elapsed > 0 else 0.0
            log(f"trt_host: {counts['in']} frames in {elapsed:.2f}s -> {fps:.2f} fps "
                f"end-to-end ({workers} workers)")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    if args.scale != 2:
        log(f"trt_host: --scale {args.scale} unsupported (model is fixed 2x)")
        return 2

    w, h = args.width, args.height
    if w <= 0 or h <= 0:
        log("trt_host: --width/--height must be positive")
        return 2

    if not os.path.isfile(args.engine):
        log(f"trt_host: engine not found: {args.engine}")
        if args.model:
            log(f"trt_host: build it with: python build_engine.py --model \"{args.model}\" "
                f"--engine \"{args.engine}\"")
        else:
            log("trt_host: pass --engine pointing at an existing TensorRT engine, or build one "
                "with build_engine.py")
        return 2

    # the anime network needs spatial dims divisible by 8: pad, infer, crop
    meta = te.load_engine_meta(args.engine) or {}
    prof = meta.get("profile") or {}
    pmin = prof.get("min") or [1, 3, 184, 320]
    pmax = prof.get("max") or [1, 3, 1088, 1920]
    min_h, min_w = pmin[2], pmin[3]
    max_h, max_w = pmax[2], pmax[3]
    pw = max((w + 7) // 8 * 8, min_w)
    ph = max((h + 7) // 8 * 8, min_h)
    if ph > max_h or pw > max_w:
        log(f"trt_host: {w}x{h} needs a {pw}x{ph} canvas, outside the engine profile "
            f"({max_w}x{max_h} max)")
        return 2
    max_batch = 1
    try:
        max_batch = int((prof.get("max") or [1, 3, 1088, 1920])[0])
    except Exception:  # noqa: BLE001
        max_batch = 1
    if max_batch < 1:
        max_batch = 1
    batch = max(1, min(args.batch, max_batch))
    ow, oh = 2 * w, 2 * h

    if not args.no_threads and args.workers > 1:
        return run_multi(args, w, h, pw, ph, ow, oh, batch)

    t_load = time.perf_counter()
    try:
        runner = te.TrtRunner(args.engine, legacy_io=args.legacy_io)
    except Exception as exc:  # noqa: BLE001
        log(f"trt_host: cannot load engine: {exc}")
        return 1

    if not args.quiet:
        log(f"trt_host: engine {os.path.basename(args.engine)} "
            f"({time.perf_counter() - t_load:.2f}s load)")
        log(f"trt_host: pipe  {w}x{h} rgb24 -> {ow}x{oh} rgb24")
        if (pw, ph) != (w, h):
            log(f"trt_host: pad   {w}x{h} -> {pw}x{ph} infer -> crop {ow}x{oh}")
        log(f"trt_host: mode  {'single-thread' if args.no_threads else 'pipelined'}")
        if args.repeat_protocol:
            log("trt_host: repeat protocol on (single-thread path)")

    n_in = w * h * FRAME_BYTES
    infer_s = 0.0

    def infer_one(frame: np.ndarray) -> np.ndarray:
        """rgb24 (h,w,3) -> rgb24 (oh,ow,3), C-contiguous."""
        nonlocal infer_s
        if ph != h or pw != w:
            padded = np.pad(frame, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        else:
            padded = frame
        t0 = time.perf_counter()
        out = runner.process(padded)
        infer_s += time.perf_counter() - t0
        out = out[:oh, :ow]
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)
        return out

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    n_frames = 0
    units = 0
    repeat_units = 0
    repeat_frames = 0
    rc = 0

    if not args.no_warmup:
        # two full passes so buffer allocation + first kernel launch never land
        # in the measured/first-output frame
        t0 = time.perf_counter()
        zero = np.zeros((h, w, FRAME_BYTES), dtype=np.uint8)
        for _ in range(2):
            infer_one(zero)
        infer_s = 0.0
        if not args.quiet:
            log(f"trt_host: warmup {(time.perf_counter() - t0) * 1000:.0f} ms (2 frames)")
            log(f"trt_host: io    {runner.io_mode}")

    t_start = time.perf_counter()

    if args.repeat_protocol:
        # Order is trivial here (one reader/infer loop). A repeat writes the
        # previous 4K output and never runs inference.
        last_out = None
        buf = bytearray(n_in)
        mv = memoryview(buf)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        try:
            while True:
                repeat = read_repeat_header(stdin)
                if repeat is None:
                    break
                units += 1
                if repeat > 1:
                    if last_out is None:
                        raise RuntimeError("repeat unit before any output")
                    for _ in range(repeat):
                        write_all(stdout, memoryview(last_out).cast("B"))
                        n_frames += 1
                    repeat_units += 1
                    repeat_frames += repeat
                    continue
                if read_exact(stdin, mv) < n_in:
                    raise ValueError("truncated frame payload")
                img = infer_one(frame)
                n_frames += 1
                write_all(stdout, img)
                # Dedicated copy, never a view into the runner's ring.
                last_out = np.array(img, dtype=np.uint8, copy=True)
        except (BrokenPipeError, KeyboardInterrupt):
            pass
        except Exception as exc:  # noqa: BLE001
            log(f"trt_host: I/O error: {exc!r}")
            rc = 1
    elif args.no_threads:
        buf = bytearray(n_in)
        mv = memoryview(buf)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        try:
            while read_exact(stdin, mv) == n_in:
                img = infer_one(frame)
                n_frames += 1
                write_all(stdout, img)
        except (BrokenPipeError, KeyboardInterrupt):
            pass
        except Exception as exc:  # noqa: BLE001
            log(f"trt_host: I/O error: {exc!r}")
            rc = 1
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
                    write_all(stdout, item)
            except BrokenPipeError:
                stop.set()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                stop.set()

        rt = threading.Thread(target=reader, name="trt-reader", daemon=True)
        wt = threading.Thread(target=writer, name="trt-writer", daemon=True)
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
            # always deliver the writer sentinel, otherwise join blocks
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
            log(f"trt_host: I/O error: {errors[0]!r}")
            rc = 1

    elapsed = time.perf_counter() - t_start
    if args.stats or not args.quiet:
        fps = n_frames / elapsed if elapsed > 0 else 0.0
        if args.repeat_protocol:
            log(
                f"trt_host: units={units} normal={units - repeat_units} "
                f"repeat_units={repeat_units} "
                f"repeated_frames={repeat_frames} expanded={n_frames} "
                f"in {elapsed:.2f}s -> {fps:.2f} fps end-to-end"
            )
        else:
            ipf = (infer_s / n_frames * 1000.0) if n_frames else 0.0
            log(f"trt_host: {n_frames} frames in {elapsed:.2f}s -> {fps:.2f} fps "
                f"end-to-end (inference {ipf:.1f} ms/frame)")

    runner.close()

    try:
        stdout.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
