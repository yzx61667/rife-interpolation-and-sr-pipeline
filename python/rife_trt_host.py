#!/usr/bin/env python3
"""rife_trt_host.py -- streaming RIFE frame interpolation on TensorRT.

Wire protocol is identical to pipe/rife_host.py:

  stdin  : rawvideo rgb24, fixed W x H, frames back-to-back
  stdout : rawvideo rgb24, same W x H
           factor=2 -> for each pair: prev, mid(0.5)
           factor=3 -> for each pair: prev, mid(1/3), mid(2/3)
           the final input frame is emitted once at EOF
  stderr : logs / stats only

The engine is a static 1x7xHxW RIFE 4.7 network:
  input  = [frame_a (3), frame_b (3), timestep (1)]
  output = interpolated frame (3)

Frames are prepared on the CPU, uploaded, inferred and downloaded on three
separate CUDA streams. With two frame slots the upload of the next pair and
the download of the previous one overlap the inference, which matters at 4K
where a single inference is ~75 ms.
"""

from __future__ import annotations

import argparse
import ctypes
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

try:  # keep the CPU detector importable without the TensorRT runtime
    import tensorrt as trt  # noqa: E402
    import trt_engine as te  # noqa: E402
except Exception as _TRT_IMPORT_ERROR:  # noqa: BLE001
    trt = None
    te = None
else:
    _TRT_IMPORT_ERROR = None

FRAME_BYTES = 3

# Repeat-aware RIFE -> SR wire protocol. Every output unit starts with an
# 8-byte little-endian header (magic, repeat). repeat=1 is followed by one
# rgb24 frame; repeat>1 has no payload and tells the SR host to write its last
# 4K output that many times. Magic spells "EFIR" on the wire.
REPEAT_MAGIC = 0x52494645
REPEAT_HEADER = struct.Struct("<II")
HEADER_REPEAT_1 = REPEAT_HEADER.pack(REPEAT_MAGIC, 1)


def pack_repeat(repeat: int) -> bytes:
    """Encode one protocol unit header."""
    return REPEAT_HEADER.pack(REPEAT_MAGIC, int(repeat))


class PairRepeatDetector:
    """Task C held-pair detector with causal 2-pair confirmation.

    Pair ``i`` is the pair between source frames ``i`` and ``i+1``. A pair is
    raw-held when its 2x-subsampled changed-pixel fraction is <= threshold,
    2x-subsampled MAD is <= mad, and the approximate 4x gray scene score of
    this pair and the previous pair is below the scene guard. Pair ``i`` is
    confirmed when raw-held pairs ``i-1`` and ``i`` both pass, so output is
    never delayed.
    """

    def __init__(self, height: int, width: int, threshold_pct: float = 0.05,
                 scene: float = 8.0, mad: float = 0.02):
        self.h = height
        self.w = width
        self.threshold = max(0.0, float(threshold_pct)) / 100.0
        self.scene_threshold = float(scene)
        self.mad_threshold = float(mad)
        self.n2 = ((height + 1) // 2) * ((width + 1) // 2)
        self.n4 = ((height + 3) // 4) * ((width + 3) // 4)
        # The 4x grid is a subset of the 2x grid: count4 <= count2 and
        # N4 ~= N2/4, so frac4 > 4*T proves frac2 > T. The exact ratio keeps
        # the prefilter conservative on odd dimensions.
        self.coarse_bound = self.threshold * (float(self.n2) / float(self.n4))

        # Preallocated scratch: detection runs in the RIFE reader thread, so
        # there is no concurrent use and no per-pair allocation churn.
        h2 = (height + 1) // 2
        w2 = (width + 1) // 2
        h4 = (height + 3) // 4
        w4 = (width + 3) // 4
        self._d2 = np.empty((h2, w2, 3), dtype=np.int16)
        self._d4 = np.empty((h4, w4, 3), dtype=np.int16)

        self.prev_raw = False
        self.prev_scene = 0.0
        self.pairs = 0
        self.raw_pairs = 0
        self.confirmed_pairs = 0
        self.prefilter_rejects = 0
        self.fine_checks = 0
        self.elapsed = 0.0

    def update(self, prev_raw: bytes, cur_raw: bytes) -> bool:
        """Consume one source pair; return True when it is confirmed."""
        t0 = time.perf_counter()
        a = np.frombuffer(prev_raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        b = np.frombuffer(cur_raw, dtype=np.uint8).reshape(self.h, self.w, 3)

        # 4x pass: scene guard and the cheap motion prefilter.
        a4 = a[::4, ::4]
        b4 = b[::4, ::4]
        np.subtract(a4, b4, out=self._d4, dtype=np.int16)
        # Linear approximation of the 4x gray scene score, from the signed
        # channel differences (saves materializing two gray frames).
        scene_diff = self._d4[..., 0].astype(np.int32) * 77
        scene_diff += self._d4[..., 1].astype(np.int32) * 150
        scene_diff += self._d4[..., 2].astype(np.int32) * 29
        scene = float(np.abs(scene_diff).mean()) / 256.0

        np.abs(self._d4, out=self._d4)
        limit4 = self.coarse_bound * self.n4
        changed_channels4 = int(np.count_nonzero(self._d4 > 2))
        if changed_channels4 <= limit4:
            frac4_ok = True
        elif changed_channels4 > 3.0 * limit4:
            frac4_ok = False
        else:
            frac4_ok = (
                np.count_nonzero(np.any(self._d4 > 2, axis=2)) <= limit4
            )

        self.pairs += 1
        if not frac4_ok:
            # Obvious motion: the fine threshold cannot pass.
            self.prefilter_rejects += 1
            raw = False
        else:
            self.fine_checks += 1
            a2 = a[::2, ::2]
            b2 = b[::2, ::2]
            np.subtract(a2, b2, out=self._d2, dtype=np.int16)
            np.abs(self._d2, out=self._d2)

            # A changed pixel contributes 1..3 changed channels, so the
            # channel count gives exact cheap bounds; only the narrow band
            # between them needs the per-pixel reduction.
            limit = self.threshold * self.n2
            changed_channels = int(np.count_nonzero(self._d2 > 2))
            if changed_channels <= limit:
                frac_ok = True
            elif changed_channels > 3.0 * limit:
                frac_ok = False
            else:
                frac_ok = (
                    np.count_nonzero(np.any(self._d2 > 2, axis=2)) <= limit
                )

            if frac_ok:
                # 2x-subsampled MAD; matches full-res classification on the
                # Task C clip10 sample and is ~20x cheaper.
                mad = float(self._d2.mean())
                raw = (
                    mad <= self.mad_threshold
                    and scene < self.scene_threshold
                    and self.prev_scene < self.scene_threshold
                )
            else:
                raw = False

        confirmed = raw and self.prev_raw
        self.prev_raw = raw
        self.prev_scene = scene
        self.raw_pairs += int(raw)
        self.confirmed_pairs += int(confirmed)
        self.elapsed += time.perf_counter() - t0
        return confirmed


def log(msg: str) -> None:
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - never die while logging
        pass


def read_exact(fh, view: memoryview) -> int:
    pos = 0
    total = len(view)
    while pos < total:
        got = fh.readinto(view[pos:])
        if not got:
            break
        pos += got
    return pos


def write_all(fh, data) -> None:
    mv = memoryview(data)
    while len(mv):
        n = fh.write(mv)
        if not n:
            raise BrokenPipeError("downstream closed")
        mv = mv[n:]
    fh.flush()


class Slot:
    """One set of device + pinned host buffers."""

    __slots__ = ("d_in", "d_out", "h_in", "h_out", "in_view", "out_view",
                 "n_in", "n_out", "d_u8", "h_u8", "u8_view", "n_u8")

    def __init__(self, cuda, h: int, w: int, u8: bool = False):
        self.n_in = 7 * h * w * 4
        self.n_out = 3 * h * w * 4
        self.d_in = cuda.malloc(self.n_in)
        self.d_out = cuda.malloc(self.n_out)
        self.h_in = cuda.host_alloc(self.n_in)
        self.h_out = cuda.host_alloc(self.n_out)
        self.in_view = te.pinned_array(self.h_in, (7, h, w), ctypes.c_float)
        self.out_view = te.pinned_array(self.h_out, (3, h, w), ctypes.c_float)
        self.d_u8 = self.h_u8 = self.u8_view = None
        self.n_u8 = 3 * h * w if u8 else 0
        if u8:
            self.d_u8 = cuda.malloc(self.n_u8)
            self.h_u8 = cuda.host_alloc(self.n_u8)
            self.u8_view = te.pinned_array(self.h_u8, (3, h, w), ctypes.c_uint8)

    def free(self, cuda) -> None:
        for attr in ("d_in", "d_out", "d_u8"):
            ptr = getattr(self, attr, None)
            if ptr:
                cuda.free(ptr)
                setattr(self, attr, None)
        for attr in ("h_in", "h_out", "h_u8"):
            ptr = getattr(self, attr, None)
            if ptr:
                cuda.host_free(ptr)
                setattr(self, attr, None)


class RifeTrtRunner:
    def __init__(self, engine_path: str, width: int, height: int, slots: int = 2):
        if trt is None or te is None:
            raise RuntimeError(
                f"TensorRT runtime unavailable: {_TRT_IMPORT_ERROR}"
            )
        self.w = width
        self.h = height
        self.engine_path = engine_path
        self.n_slots = max(1, slots)
        self._load_engine()
        self.cuda = te.cuda()
        self.stream = self.cuda.new_stream()
        self._load_cast()
        self.slots = [
            Slot(self.cuda, height, width, u8=self.use_cast)
            for _ in range(self.n_slots)
        ]
        if self.use_cast:
            for i, slot in enumerate(self.slots):
                ctx = self.cast_contexts[i]
                ctx.set_tensor_address(self._cast_in, slot.d_out)
                ctx.set_tensor_address(self._cast_out, slot.d_u8)

    def _load_engine(self) -> None:
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(self.engine_path)
        with open(self.engine_path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f"failed to deserialize {self.engine_path} "
                "(engine is GPU / TensorRT-version bound)"
            )
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        if self.in_name is None or self.out_name is None:
            raise RuntimeError("engine must have exactly one input and one output")

        # One context per slot, so the next inference can be queued without
        # waiting for the previous one (addresses are per-context).
        self.contexts = []
        for _ in range(self.n_slots):
            ctx = self.engine.create_execution_context()
            if ctx is None:
                raise RuntimeError("failed to create execution context")
            self.contexts.append(ctx)
        self.context = self.contexts[0]

        ishape = tuple(self.engine.get_tensor_shape(self.in_name))
        if len(ishape) != 4 or int(ishape[1]) != 7:
            raise ValueError(f"not a RIFE 7-channel engine: input {ishape}")
        self._dynamic = int(ishape[2]) < 0 or int(ishape[3]) < 0
        if self._dynamic:
            for ctx in self.contexts:
                ctx.set_input_shape(self.in_name, (1, 7, self.h, self.w))
            ishape = tuple(self.context.get_tensor_shape(self.in_name))
        eh, ew = int(ishape[2]), int(ishape[3])
        if eh != self.h or ew != self.w:
            raise ValueError(
                f"engine expects {ew}x{eh}, but the pipe is {self.w}x{self.h}"
            )

        oshape = tuple(self.context.get_tensor_shape(self.out_name))
        if any(int(d) < 0 for d in oshape):
            oshape = (1, 3, self.h, self.w)
        if int(oshape[1]) != 3 or int(oshape[2]) != self.h or int(oshape[3]) != self.w:
            raise ValueError(f"unexpected engine output shape: {oshape}")

    def _load_cast(self) -> None:
        """Optional float32 -> uint8 post-process engine (one context per slot)."""
        self.use_cast = False
        self.cast_contexts = []
        self._cast_engine = None
        self._cast_in = None
        self._cast_out = None

        # Measured 2026-11-05 at 4K under identical GPU load: cast 8.5 vs
        # float 8.1 source fps (+5%). The SR host gains more because its CPU
        # conversion is on the critical path. The cast is optional: if the
        # engine file is missing, both hosts fall back to the float download.
        if self.engine.get_tensor_dtype(self.out_name) != trt.DataType.FLOAT:
            return
        cast_path = os.path.join(
            os.path.dirname(os.path.abspath(self.engine_path)), "rgb_u8cast.engine"
        )
        if not os.path.isfile(cast_path):
            return
        try:
            with open(cast_path, "rb") as f:
                blob = f.read()
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            engine = runtime.deserialize_cuda_engine(blob)
            if engine is None:
                return
            in_name = out_name = None
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    in_name = name
                else:
                    out_name = name
            if in_name is None or out_name is None:
                return
            if engine.get_tensor_dtype(in_name) != trt.DataType.FLOAT:
                return
            if engine.get_tensor_dtype(out_name) != trt.DataType.UINT8:
                return
            contexts = []
            for _ in range(self.n_slots):
                ctx = engine.create_execution_context()
                if ctx is None:
                    return
                ctx.set_input_shape(in_name, (1, 3, self.h, self.w))
                contexts.append(ctx)
            self._cast_engine = engine
            self._cast_runtime = runtime
            self._cast_in = in_name
            self._cast_out = out_name
            self.cast_contexts = contexts
            self.use_cast = True
        except Exception:  # noqa: BLE001 - cast is an optimization, not required
            self.use_cast = False
            self.cast_contexts = []

    # -- frame conversion -------------------------------------------------
    def prepare(self, slot: int, a_raw: bytes, b_raw: bytes, t: float) -> None:
        s = self.slots[slot]
        a = np.frombuffer(a_raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        b = np.frombuffer(b_raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        np.multiply(a.transpose(2, 0, 1), np.float32(1.0 / 255.0), out=s.in_view[0:3])
        np.multiply(b.transpose(2, 0, 1), np.float32(1.0 / 255.0), out=s.in_view[3:6])
        s.in_view[6].fill(t)

    def prepare_delta(self, slot: int, b_raw: bytes) -> None:
        """Only the new frame; channel a is copied from the previous slot."""
        s = self.slots[slot]
        b = np.frombuffer(b_raw, dtype=np.uint8).reshape(self.h, self.w, 3)
        np.multiply(b.transpose(2, 0, 1), np.float32(1.0 / 255.0), out=s.in_view[3:6])

    def init_t(self, stream, t: float = 0.5) -> None:
        """Upload the constant timestep plane once for every slot."""
        n_t = self.h * self.w * 4
        off = 6 * self.h * self.w * 4
        for s in self.slots:
            s.in_view[6].fill(t)
            self.cuda.h2d_async(s.d_in + off, s.h_in + off, n_t, stream)
        self.cuda.sync_stream(stream)

    def to_bytes(self, slot: int) -> bytes:
        s = self.slots[slot]
        if self.use_cast:
            out = np.ascontiguousarray(s.u8_view.transpose(1, 2, 0))
            return out.tobytes()
        np.clip(s.out_view, 0.0, 1.0, out=s.out_view)
        hwc = s.out_view.transpose(1, 2, 0)
        np.multiply(hwc, np.float32(255.0), out=hwc)
        np.add(hwc, np.float32(0.5), out=hwc)
        out = np.empty((self.h, self.w, 3), dtype=np.uint8)
        np.copyto(out, hwc, casting="unsafe")
        return out.tobytes()

    # -- CUDA pieces ------------------------------------------------------
    def upload(self, slot: int, stream, full: bool = True) -> None:
        s = self.slots[slot]
        if full:
            self.cuda.h2d_async(s.d_in, s.h_in, s.n_in, stream)
        else:
            off = 3 * self.h * self.w * 4
            self.cuda.h2d_async(s.d_in + off, s.h_in + off, off, stream)

    def copy_prev(self, slot: int, prev_slot: int, stream) -> None:
        """Copy the previous pair's second frame into this slot's first frame."""
        off = 3 * self.h * self.w * 4
        self.cuda.d2d_async(
            self.slots[slot].d_in,
            self.slots[prev_slot].d_in + off,
            off,
            stream,
        )

    def execute(self, slot: int, stream) -> None:
        s = self.slots[slot]
        ctx = self.contexts[slot]
        ctx.set_tensor_address(self.in_name, s.d_in)
        ctx.set_tensor_address(self.out_name, s.d_out)
        ctx.execute_async_v3(stream_handle=stream)
        if self.use_cast:
            self.cast_contexts[slot].execute_async_v3(stream_handle=stream)

    def download(self, slot: int, stream) -> None:
        s = self.slots[slot]
        if self.use_cast:
            self.cuda.d2h_async(s.h_u8, s.d_u8, s.n_u8, stream)
        else:
            self.cuda.d2h_async(s.h_out, s.d_out, s.n_out, stream)

    def infer_sync(self, slot: int) -> bytes:
        self.upload(slot, self.stream)
        self.execute(slot, self.stream)
        self.download(slot, self.stream)
        self.cuda.sync_stream(self.stream)
        return self.to_bytes(slot)

    def warmup(self) -> None:
        zeros = bytes(self.w * self.h * FRAME_BYTES)
        self.prepare(0, zeros, zeros, 0.5)
        self.infer_sync(0)

    def close(self) -> None:
        for s in self.slots:
            s.free(self.cuda)
        if getattr(self, "stream", None):
            self.cuda.free_stream(self.stream)
            self.stream = None


class Job:
    __slots__ = ("slot", "t", "prev_raw", "final_raw", "full", "prev_slot",
                 "static", "repeat")

    def __init__(self, slot: int, t: float = 0.0, prev_raw=None, final_raw=None,
                 full: bool = True, prev_slot=None, static: bool = False,
                 repeat: int = 0):
        self.slot = slot
        self.t = t
        self.prev_raw = prev_raw
        self.final_raw = final_raw
        self.full = full
        self.prev_slot = prev_slot
        self.static = static
        self.repeat = repeat


class Pipeline:
    def __init__(self, runner: RifeTrtRunner, factor: int, slots: int,
                 quiet: bool, stats: bool, static_threshold: float = 0.02,
                 repeat_protocol: bool = False,
                 repeat_threshold: float = 0.05,
                 repeat_scene: float = 8.0,
                 repeat_mad: float = 0.02):
        self.r = runner
        self.factor = factor
        self.timesteps = [0.5] if factor == 2 else [1.0 / 3.0, 2.0 / 3.0]
        self.quiet = quiet
        self.stats = stats
        self.repeat_protocol = repeat_protocol
        self.header_repeat_1 = HEADER_REPEAT_1
        self.detector = (
            PairRepeatDetector(
                runner.h, runner.w, repeat_threshold, repeat_scene, repeat_mad
            )
            if repeat_protocol
            else None
        )
        self.repeat_pairs = 0
        # The legacy static path owns per-pair decisions only when the repeat
        # protocol is off (main() clears static_threshold when both are set).
        self.static_threshold = static_threshold
        self.slots = max(1, slots)
        self.stop = threading.Event()
        self.errors: list = []
        self.n_in = 0
        self.n_out = 0

        self.free: "queue.Queue" = queue.Queue()
        for i in range(self.slots):
            self.free.put(i)
        self.jobs: "queue.Queue" = queue.Queue(maxsize=self.slots)
        self.out_jobs: "queue.Queue" = queue.Queue(maxsize=self.slots)

        c = runner.cuda
        self.up_stream = c.new_stream()
        self.dl_stream = c.new_stream()
        self.up_ev = [c.new_event() for _ in range(self.slots)]
        self.done_ev = [c.new_event() for _ in range(self.slots)]
        self.out_ev = [c.new_event() for _ in range(self.slots)]
        self._t_start = 0.0
        self._t_end = 0.0

        # factor=2 always uses t=0.5, so only the new frame needs uploading;
        # the previous frame is copied device-to-device from the other slot.
        self.use_delta = factor == 2
        self.submitted = 0
        self.last_slot = None
        self.static_pairs = 0
        if self.use_delta:
            runner.init_t(self.up_stream, 0.5)

    # -- queue helpers ----------------------------------------------------
    def _put(self, q: "queue.Queue", item, timeout: float = 0.2) -> bool:
        while not self.stop.is_set():
            try:
                q.put(item, timeout=timeout)
                return True
            except queue.Full:
                continue
        return False

    def _get(self, q: "queue.Queue", timeout: float = 0.2):
        while not self.stop.is_set():
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                continue
        return None

    def _acquire_slot(self):
        while not self.stop.is_set():
            try:
                return self.free.get(timeout=0.2)
            except queue.Empty:
                continue
        return None

    # -- worker threads ---------------------------------------------------
    def _gpu_loop(self) -> None:
        try:
            while not self.stop.is_set():
                job = self._get(self.jobs)
                if job is None:
                    break
                if job.static or job.slot < 0:
                    self._put(self.out_jobs, job)
                    continue

                s = job.slot
                c = self.r.cuda
                # Upload the next pair first; it overlaps the previous inference.
                if job.prev_slot is not None:
                    self.r.copy_prev(s, job.prev_slot, self.up_stream)
                self.r.upload(s, self.up_stream, full=job.full)
                c.record(self.up_ev[s], self.up_stream)

                # The inference must not start before its upload finished.
                c.wait_event(self.r.stream, self.up_ev[s])
                self.r.execute(s, self.r.stream)
                c.record(self.done_ev[s], self.r.stream)

                c.wait_event(self.dl_stream, self.done_ev[s])
                self.r.download(s, self.dl_stream)
                c.record(self.out_ev[s], self.dl_stream)
                self._put(self.out_jobs, job)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(exc)
            self.stop.set()
        finally:
            self._put(self.out_jobs, None)

    def _write_frame(self, stdout, data) -> None:
        if self.repeat_protocol:
            write_all(stdout, self.header_repeat_1)
        write_all(stdout, data)
        self.n_out += 1

    def _writer_loop(self) -> None:
        stdout = sys.stdout.buffer
        try:
            while not self.stop.is_set():
                job = self._get(self.out_jobs)
                if job is None:
                    break
                if job.repeat:
                    # No payload: the SR host writes its last 4K output
                    # ``repeat`` times. n_out stays the expanded count.
                    write_all(stdout, pack_repeat(job.repeat))
                    self.n_out += job.repeat
                    continue
                if job.static:
                    # Near-identical pair: every interpolated step is the frame.
                    for _ in range(self.factor):
                        self._write_frame(stdout, job.prev_raw)
                    continue
                if job.final_raw is not None:
                    self._write_frame(stdout, job.final_raw)
                    continue

                s = job.slot
                self.r.cuda.sync_event(self.out_ev[s])
                data = self.r.to_bytes(s)
                if job.prev_raw is not None:
                    self._write_frame(stdout, job.prev_raw)
                self._write_frame(stdout, data)
                self.free.put(s)
        except BrokenPipeError:
            self.stop.set()
        except Exception as exc:  # noqa: BLE001
            self.errors.append(exc)
            self.stop.set()
        finally:
            try:
                stdout.flush()
            except Exception:  # noqa: BLE001
                pass

    # -- reader -----------------------------------------------------------
    def _read_frame(self, fh, frame_bytes: int):
        buf = bytearray(frame_bytes)
        if read_exact(fh, memoryview(buf)) < frame_bytes:
            return None
        return bytes(buf)

    def run(self) -> int:
        gt = threading.Thread(target=self._gpu_loop, name="rife-trt-gpu", daemon=True)
        wt = threading.Thread(target=self._writer_loop, name="rife-trt-writer", daemon=True)
        gt.start()
        wt.start()

        stdin = sys.stdin.buffer
        frame_bytes = self.r.w * self.r.h * FRAME_BYTES
        prev_raw = self._read_frame(stdin, frame_bytes)
        if prev_raw is None:
            self._put(self.jobs, None)
            gt.join()
            wt.join()
            return 0

        self.n_in = 1
        self._t_start = time.perf_counter()
        force_full = False
        try:
            while not self.stop.is_set():
                cur_raw = self._read_frame(stdin, frame_bytes)
                if cur_raw is None:
                    break
                self.n_in += 1

                if self.detector is not None:
                    # Causal 2-pair confirmation: pair i is usable only when
                    # raw-held pairs i-1 and i both pass. No output is delayed.
                    if self.detector.update(prev_raw, cur_raw):
                        if not self._put(self.jobs, Job(-1, repeat=self.factor)):
                            break
                        self.repeat_pairs += 1
                        force_full = True
                        prev_raw = cur_raw
                        continue
                elif self.static_threshold > 0:
                    step = 4 if self.r.w * self.r.h > 1920 * 1080 else 2
                    a = np.frombuffer(prev_raw, np.uint8).reshape(
                        self.r.h, self.r.w, 3)[::step, ::step]
                    b = np.frombuffer(cur_raw, np.uint8).reshape(
                        self.r.h, self.r.w, 3)[::step, ::step]
                    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
                    changed = np.count_nonzero(np.any(diff > 2, axis=2))
                    frac = changed / float(a.shape[0] * a.shape[1])
                    is_static = frac <= self.static_threshold / 100.0
                else:
                    is_static = prev_raw == cur_raw

                if self.detector is None and is_static:
                    # No GPU work: prev is repeated for every interpolated step.
                    if not self._put(self.jobs, Job(-1, prev_raw=prev_raw,
                                                   static=True)):
                        break
                    self.static_pairs += 1
                    force_full = True
                    prev_raw = cur_raw
                    continue

                for idx, t in enumerate(self.timesteps):
                    slot = self._acquire_slot()
                    if slot is None:
                        break
                    full = (not self.use_delta) or self.submitted == 0 or force_full
                    if full:
                        self.r.prepare(slot, prev_raw, cur_raw, t)
                        force_full = False
                    else:
                        self.r.prepare_delta(slot, cur_raw)
                    job = Job(
                        slot,
                        t,
                        prev_raw=prev_raw if idx == 0 else None,
                        full=full,
                        prev_slot=None if full else self.last_slot,
                    )
                    if not self._put(self.jobs, job):
                        self.free.put(slot)
                        break
                    self.last_slot = slot
                    self.submitted += 1
                if self.stop.is_set():
                    break
                prev_raw = cur_raw

            if not self.stop.is_set() and prev_raw is not None:
                self._put(self.jobs, Job(-1, final_raw=prev_raw))
            self._put(self.jobs, None)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(exc)
            self.stop.set()
            self._put(self.jobs, None)

        gt.join()
        wt.join()
        self._t_end = time.perf_counter()

        if self.errors:
            log(f"rife_trt_host: error: {self.errors[0]!r}")
            return 1
        self._report()
        return 0

    def _report(self) -> None:
        elapsed = max(1e-9, self._t_end - self._t_start)
        src_fps = self.n_in / elapsed
        out_fps = self.n_out / elapsed
        if self.stats or not self.quiet:
            if self.static_pairs:
                log(f"rife_trt_host: static_pairs={self.static_pairs}")
            if self.repeat_protocol and self.detector is not None:
                det = self.detector
                ms = det.elapsed / max(1, det.pairs) * 1000.0
                log(
                    f"rife_trt_host: repeat_pairs={self.repeat_pairs} "
                    f"repeat_frames={self.repeat_pairs * self.factor} "
                    f"raw_pairs={det.raw_pairs} fine_checks={det.fine_checks} "
                    f"prefilter_rejects={det.prefilter_rejects} "
                    f"detect_ms_per_pair={ms:.3f}"
                )
            log(
                f"rife_trt_host: frames_in={self.n_in} frames_out={self.n_out} "
                f"src_fps={src_fps:.2f} out_fps={out_fps:.2f}"
            )

    def close(self) -> None:
        c = self.r.cuda
        for e in self.up_ev + self.done_ev + self.out_ev:
            c.free_event(e)
        c.free_stream(self.up_stream)
        c.free_stream(self.dl_stream)


def run_single(runner: RifeTrtRunner, factor: int, quiet: bool, stats: bool,
               repeat_protocol: bool = False,
               repeat_threshold: float = 0.05,
               repeat_scene: float = 8.0,
               repeat_mad: float = 0.02) -> int:
    timesteps = [0.5] if factor == 2 else [1.0 / 3.0, 2.0 / 3.0]
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    frame_bytes = runner.w * runner.h * FRAME_BYTES
    detector = (
        PairRepeatDetector(runner.h, runner.w, repeat_threshold,
                           repeat_scene, repeat_mad)
        if repeat_protocol
        else None
    )
    repeat_pairs = 0

    def emit(data) -> None:
        if repeat_protocol:
            write_all(stdout, HEADER_REPEAT_1)
        write_all(stdout, data)

    prev_raw = None
    buf = bytearray(frame_bytes)
    if read_exact(stdin, memoryview(buf)) == frame_bytes:
        prev_raw = bytes(buf)
    if prev_raw is None:
        return 0

    n_in = 1
    n_out = 0
    t0 = time.perf_counter()
    try:
        while True:
            buf = bytearray(frame_bytes)
            if read_exact(stdin, memoryview(buf)) < frame_bytes:
                break
            cur_raw = bytes(buf)
            n_in += 1
            if detector is not None and detector.update(prev_raw, cur_raw):
                write_all(stdout, pack_repeat(factor))
                n_out += factor
                repeat_pairs += 1
                prev_raw = cur_raw
                continue
            for idx, t in enumerate(timesteps):
                runner.prepare(0, prev_raw, cur_raw, t)
                data = runner.infer_sync(0)
                if idx == 0:
                    emit(prev_raw)
                    n_out += 1
                emit(data)
                n_out += 1
            prev_raw = cur_raw
        if prev_raw is not None:
            emit(prev_raw)
            n_out += 1
    except BrokenPipeError:
        return 0

    elapsed = max(1e-9, time.perf_counter() - t0)
    if stats or not quiet:
        if detector is not None:
            ms = detector.elapsed / max(1, detector.pairs) * 1000.0
            log(
                f"rife_trt_host: repeat_pairs={repeat_pairs} "
                f"repeat_frames={repeat_pairs * factor} "
                f"detect_ms_per_pair={ms:.3f}"
            )
        log(
            f"rife_trt_host: frames_in={n_in} frames_out={n_out} "
            f"src_fps={n_in / elapsed:.2f} out_fps={n_out / elapsed:.2f}"
        )
    return 0


def detect_stream(args: argparse.Namespace) -> int:
    """CPU-only held-pair measurement: rawvideo stdin, stats on stderr."""
    log(
        f"repeat_detect: threshold={args.repeat_threshold}% changed "
        f"mad<={args.repeat_mad} scene<{args.repeat_scene} "
        f"causal 2-pair, factor={args.factor}"
    )
    detector = PairRepeatDetector(
        args.height, args.width, args.repeat_threshold,
        args.repeat_scene, args.repeat_mad,
    )
    stdin = sys.stdin.buffer
    frame_bytes = args.width * args.height * FRAME_BYTES
    prev_raw = None
    buf = bytearray(frame_bytes)
    if read_exact(stdin, memoryview(buf)) == frame_bytes:
        prev_raw = bytes(buf)
    n_in = 1 if prev_raw is not None else 0
    while prev_raw is not None:
        buf = bytearray(frame_bytes)
        if read_exact(stdin, memoryview(buf)) < frame_bytes:
            break
        cur_raw = bytes(buf)
        n_in += 1
        detector.update(prev_raw, cur_raw)
        prev_raw = cur_raw

    saved = detector.confirmed_pairs * args.factor
    ms = detector.elapsed / max(1, detector.pairs) * 1000.0
    log(
        f"repeat_detect: frames_in={n_in} pairs={detector.pairs} "
        f"raw_pairs={detector.raw_pairs} "
        f"confirmed_pairs={detector.confirmed_pairs} "
        f"saved_frames={saved} "
        f"prefilter_rejects={detector.prefilter_rejects} "
        f"fine_checks={detector.fine_checks} "
        f"detect_ms_per_pair={ms:.3f}"
    )
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rife_trt_host.py",
        description="Streaming RIFE interpolation (TensorRT) for rawvideo pipes.",
    )
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--engine", default=None,
                   help="TensorRT engine (required unless --detect-only)")
    p.add_argument("--factor", type=int, default=2, choices=[2, 3])
    p.add_argument("--slots", type=int, default=2, help="frame buffers (default 2)")
    p.add_argument("--static-threshold", type=float, default=0.0,
                   help="treat a pair as static when the changed-pixel "
                        "fraction is at or below this percent; 0 = only "
                        "byte-identical pairs (default)")
    p.add_argument("--repeat-protocol", action="store_true",
                   help="emit the repeat-aware header protocol for the SR host")
    p.add_argument("--repeat-threshold", type=float, default=0.05,
                   help="held-pair changed-pixel fraction in percent (default 0.05)")
    p.add_argument("--repeat-scene", type=float, default=8.0,
                   help="4x gray scene guard threshold (default 8)")
    p.add_argument("--repeat-mad", type=float, default=0.02,
                   help="held-pair full-resolution MAD limit (default 0.02)")
    p.add_argument("--detect-only", action="store_true",
                   help="CPU-only: read rawvideo and report held-pair counts")
    p.add_argument("--no-pipeline", action="store_true",
                   help="single-threaded loop (for A/B measurements)")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        log("rife_trt_host: --width/--height must be positive")
        return 2
    if args.detect_only:
        return detect_stream(args)
    if not args.engine:
        log("rife_trt_host: --engine is required unless --detect-only")
        return 2

    if args.repeat_protocol and args.static_threshold > 0:
        log("rife_trt_host: --repeat-protocol wins; legacy --static-threshold disabled")
        args.static_threshold = 0.0

    t0 = time.perf_counter()
    try:
        runner = RifeTrtRunner(
            args.engine, args.width, args.height, slots=args.slots
        )
    except Exception as exc:  # noqa: BLE001
        log(f"rife_trt_host: cannot load engine: {exc}")
        return 1

    if not args.quiet:
        log(
            f"rife_trt_host: engine {os.path.basename(args.engine)} "
            f"({time.perf_counter() - t0:.2f}s load)"
        )
        log(
            f"rife_trt_host: pipe {args.width}x{args.height} rgb24, "
            f"factor {args.factor}, "
            f"{'single-thread' if args.no_pipeline else f'pipelined ({args.slots} slots)'}"
        )
        if args.repeat_protocol:
            log(
                f"rife_trt_host: repeat protocol on "
                f"(threshold={args.repeat_threshold}% changed, "
                f"mad<={args.repeat_mad}, scene<{args.repeat_scene}, "
                f"causal 2-pair)"
            )

    try:
        runner.warmup()
        if args.no_pipeline:
            return run_single(
                runner, args.factor, args.quiet, args.stats,
                repeat_protocol=args.repeat_protocol,
                repeat_threshold=args.repeat_threshold,
                repeat_scene=args.repeat_scene,
                repeat_mad=args.repeat_mad,
            )
        pipe = Pipeline(runner, args.factor, args.slots, args.quiet, args.stats,
                        args.static_threshold,
                        repeat_protocol=args.repeat_protocol,
                        repeat_threshold=args.repeat_threshold,
                        repeat_scene=args.repeat_scene,
                        repeat_mad=args.repeat_mad)
        try:
            return pipe.run()
        finally:
            pipe.close()
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())

