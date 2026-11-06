"""TensorRT engine build/load + zero-extra-dependency CUDA runtime access.

Only dependencies: tensorrt (pip wheel), nvidia-cuda-runtime-cu12 (pip wheel,
supplies cudart64_12.dll), numpy.

Everything here is Windows/ctypes based so that no CUDA toolkit, PyCUDA or
PyTorch install is required.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import weakref
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_size_t, c_void_p

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(HERE, "engines")

# Optional GPU-side uint8<->float I/O engines, looked up next to the main
# engine. Missing/shape-incompatible engines fall back to NumPy I/O.
SR_PRE_ENGINE = "sr_pre_u8hwc_to_f32chw.engine"
SR_POST_ENGINE = "sr_post_f32chw_to_u8hwc.engine"

# ---------------------------------------------------------------------------
# DLL discovery
# ---------------------------------------------------------------------------

_CTX_HANDLES = []


def _add_dll_dirs() -> None:
    """Make cudart / nvinfer findable without touching the system PATH."""
    try:
        import nvidia.cuda_runtime as _rt  # type: ignore

        bin_dir = os.path.join(os.path.dirname(_rt.__file__), "bin")
    except Exception:
        bin_dir = os.path.join(
            os.path.dirname(sys.executable),
            "..",
            "Lib",
            "site-packages",
            "nvidia",
            "cuda_runtime",
            "bin",
        )
        bin_dir = os.path.abspath(bin_dir)

    candidates = [bin_dir]
    # tensorrt_libs normally self-registers via its __init__, but add manually.
    try:
        import tensorrt_libs  # type: ignore

        candidates.append(os.path.dirname(tensorrt_libs.__file__))
    except Exception:
        pass

    for d in candidates:
        if os.path.isdir(d):
            try:
                _CTX_HANDLES.append(os.add_dll_directory(d))
            except (AttributeError, OSError):
                pass


def _cudart_path() -> str:
    try:
        import nvidia.cuda_runtime as _rt  # type: ignore

        p = os.path.join(os.path.dirname(_rt.__file__), "bin", "cudart64_12.dll")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    # bundled/embeddable runtime: cudart sits next to python.exe
    p = os.path.join(os.path.dirname(sys.executable), "cudart64_12.dll")
    if os.path.exists(p):
        return p
    return "cudart64_12.dll"


_add_dll_dirs()

# ---------------------------------------------------------------------------
# CUDA runtime (ctypes)
# ---------------------------------------------------------------------------

_CUDA_MEMCPY_H2D = 1
_CUDA_MEMCPY_D2H = 2
_CUDA_HOST_ALLOC_PORTABLE = 1


def _chk(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} failed: CUDA error {err} ({_cuda.dll.cudaGetErrorString(err).decode()})")


class _Cuda:
    def __init__(self) -> None:
        self.dll = ctypes.CDLL(_cudart_path())
        d = self.dll
        d.cudaGetErrorString.restype = c_char_p
        d.cudaGetErrorString.argtypes = [c_int]
        d.cudaSetDevice.argtypes = [c_int]
        d.cudaSetDevice.restype = c_int
        d.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        d.cudaMalloc.restype = c_int
        d.cudaFree.argtypes = [c_void_p]
        d.cudaFree.restype = c_int
        d.cudaMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_void_p]
        d.cudaMemcpyAsync.restype = c_int
        d.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        d.cudaMemcpy.restype = c_int
        d.cudaHostAlloc.argtypes = [POINTER(c_void_p), c_size_t, ctypes.c_uint]
        d.cudaHostAlloc.restype = c_int
        d.cudaFreeHost.argtypes = [c_void_p]
        d.cudaFreeHost.restype = c_int
        d.cudaStreamCreateWithFlags.argtypes = [POINTER(c_void_p), ctypes.c_uint]
        d.cudaStreamCreateWithFlags.restype = c_int
        d.cudaStreamSynchronize.argtypes = [c_void_p]
        d.cudaStreamSynchronize.restype = c_int
        d.cudaStreamDestroy.argtypes = [c_void_p]
        d.cudaStreamDestroy.restype = c_int
        d.cudaStreamQuery.argtypes = [c_void_p]
        d.cudaStreamQuery.restype = c_int
        d.cudaEventCreate.argtypes = [POINTER(c_void_p)]
        d.cudaEventCreate.restype = c_int
        d.cudaEventRecord.argtypes = [c_void_p, c_void_p]
        d.cudaEventRecord.restype = c_int
        d.cudaEventSynchronize.argtypes = [c_void_p]
        d.cudaEventSynchronize.restype = c_int
        d.cudaStreamWaitEvent.argtypes = [c_void_p, c_void_p, ctypes.c_uint]
        d.cudaStreamWaitEvent.restype = c_int
        d.cudaEventElapsedTime.argtypes = [POINTER(c_float), c_void_p, c_void_p]
        d.cudaEventElapsedTime.restype = c_int
        d.cudaEventDestroy.argtypes = [c_void_p]
        d.cudaEventDestroy.restype = c_int
        d.cudaDeviceSynchronize.argtypes = []
        d.cudaDeviceSynchronize.restype = c_int
        _chk(d.cudaSetDevice(0), "cudaSetDevice")

    def malloc(self, nbytes: int) -> int:
        p = c_void_p()
        _chk(self.dll.cudaMalloc(byref(p), c_size_t(nbytes)), "cudaMalloc")
        return p.value

    def free(self, ptr) -> None:
        if ptr:
            self.dll.cudaFree(c_void_p(ptr))

    def host_alloc(self, nbytes: int) -> int:
        p = c_void_p()
        _chk(self.dll.cudaHostAlloc(byref(p), c_size_t(nbytes), _CUDA_HOST_ALLOC_PORTABLE), "cudaHostAlloc")
        return p.value

    def host_free(self, ptr) -> None:
        if ptr:
            self.dll.cudaFreeHost(c_void_p(ptr))

    def h2d_async(self, dst, src, nbytes, stream) -> None:
        _chk(self.dll.cudaMemcpyAsync(c_void_p(dst), c_void_p(src), c_size_t(nbytes), _CUDA_MEMCPY_H2D, c_void_p(stream)), "cudaMemcpy H2D")

    def d2d_async(self, dst, src, nbytes, stream) -> None:
        _chk(self.dll.cudaMemcpyAsync(c_void_p(dst), c_void_p(src), c_size_t(nbytes), 3, c_void_p(stream)), "cudaMemcpy D2D")

    def d2h_async(self, dst, src, nbytes, stream) -> None:
        _chk(self.dll.cudaMemcpyAsync(c_void_p(dst), c_void_p(src), c_size_t(nbytes), _CUDA_MEMCPY_D2H, c_void_p(stream)), "cudaMemcpy D2H")

    def h2d(self, dst, src, nbytes) -> None:
        _chk(self.dll.cudaMemcpy(c_void_p(dst), c_void_p(src), c_size_t(nbytes), _CUDA_MEMCPY_H2D), "cudaMemcpy H2D")

    def new_stream(self):
        s = c_void_p()
        _chk(self.dll.cudaStreamCreateWithFlags(byref(s), 1), "cudaStreamCreate")
        return s.value

    def free_stream(self, stream) -> None:
        if stream:
            self.dll.cudaStreamDestroy(c_void_p(stream))

    def sync_stream(self, stream) -> None:
        _chk(self.dll.cudaStreamSynchronize(c_void_p(stream)), "cudaStreamSynchronize")

    def sync_device(self) -> None:
        _chk(self.dll.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def wait_event(self, stream, event) -> None:
        _chk(
            self.dll.cudaStreamWaitEvent(c_void_p(stream), c_void_p(event), 0),
            "cudaStreamWaitEvent",
        )

    def sync_event(self, event) -> None:
        _chk(self.dll.cudaEventSynchronize(c_void_p(event)), "cudaEventSynchronize")

    def new_event(self):
        e = c_void_p()
        _chk(self.dll.cudaEventCreate(byref(e)), "cudaEventCreate")
        return e.value

    def record(self, event, stream) -> None:
        _chk(self.dll.cudaEventRecord(c_void_p(event), c_void_p(stream)), "cudaEventRecord")

    def event_ms(self, a, b) -> float:
        # synchronize b, then elapsed time
        _chk(self.dll.cudaEventSynchronize(c_void_p(b)), "cudaEventSynchronize")
        ms = c_float()
        _chk(self.dll.cudaEventElapsedTime(byref(ms), c_void_p(a), c_void_p(b)), "cudaEventElapsedTime")
        return ms.value

    def free_event(self, e) -> None:
        if e:
            self.dll.cudaEventDestroy(c_void_p(e))


_cuda: "_Cuda | None" = None


def cuda() -> _Cuda:
    global _cuda
    if _cuda is None:
        _cuda = _Cuda()
    return _cuda


def pinned_array(ptr: int, shape, dtype) -> np.ndarray:
    return np.ctypeslib.as_array(ctypes.cast(ptr, POINTER(dtype)), shape=shape)


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name = out.stdout.strip().splitlines()[0].strip()
        if name:
            return name
    except Exception:
        pass
    return "unknown_gpu"


def gpu_tag(name: str | None = None) -> str:
    name = name or gpu_name()
    keep = [c if c.isalnum() else "_" for c in name.lower()]
    tag = "".join(keep).replace("__", "_").strip("_")
    return tag[:40] or "gpu"


def file_sha256(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def align_up(v: int, m: int = 8) -> int:
    return (v + m - 1) // m * m


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = {
    "min": (1, 3, 184, 320),       # ~320x180 (aligned to 8)
    "opt": (1, 3, 720, 1280),      # 640x360 * 2 resolution workload / 720p
    "max": (1, 3, 1088, 1920),     # 1080p + padding headroom
}


def engine_path_for(
    onnx_path: str,
    fp16: bool = True,
    fp8: bool = False,
    strongly_typed: bool = False,
) -> str:
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    if strongly_typed:
        precision = "fp8st" if fp8 else "fp16st"
    else:
        precision = "fp8" if fp8 else ("fp16" if fp16 else "fp32")
    return os.path.join(ENGINE_DIR, f"{stem}_{precision}_{gpu_tag()}.engine")


def _meta_path(engine_path: str) -> str:
    return engine_path + ".json"


def load_engine_meta(engine_path: str) -> dict | None:
    """Sidecar metadata written at build time (profile bounds, precision, GPU)."""
    try:
        with open(_meta_path(engine_path), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_engine(
    onnx_path: str,
    engine_path: str | None = None,
    fp16: bool = True,
    profile: dict | None = None,
    workspace_gb: float = 2.0,
    opt_level: int = 5,
    verbose: bool = True,
    fp8: bool = False,
    strongly_typed: bool = False,
) -> str:
    """Build a TensorRT engine from ONNX and cache it on disk.

    ``strongly_typed`` creates a strongly-typed network; use it with an
    explicit Q/DQ ONNX when TensorRT rejects FP8 tactics on a weak-typed
    network. Q/DQ supplies the precision constraints.
    """
    import tensorrt as trt

    profile = profile or DEFAULT_PROFILE
    engine_path = engine_path or engine_path_for(
        onnx_path, fp16, fp8, strongly_typed
    )
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network_flags = 0
    if strongly_typed:
        network_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{msgs}")

    in_name = network.get_input(0).name
    out_name = network.get_output(0).name
    print(f"[build] network inputs={network.num_inputs} outputs={network.num_outputs}", file=sys.stderr)
    print(f"[build] input {in_name} {network.get_input(0).shape}", file=sys.stderr)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = opt_level
    # Strongly-typed networks take their precision from the graph/Q-DQ nodes;
    # TensorRT rejects BuilderFlag::kFP8 when STRONGLY_TYPED is set. Weak-typed
    # FP8 keeps FP16/TF32 available for layers without an FP8 kernel.
    if not strongly_typed:
        if fp16 or fp8:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.TF32)
        if fp8:
            config.set_flag(trt.BuilderFlag.FP8)

    prof = builder.create_optimization_profile()
    prof.set_shape(in_name, tuple(profile["min"]), tuple(profile["opt"]), tuple(profile["max"]))
    config.add_optimization_profile(prof)

    if strongly_typed:
        precision = "fp8st" if fp8 else "fp16st"
    else:
        precision = "fp8" if fp8 else ("fp16" if fp16 else "fp32")
    print(
        f"[build] precision={precision} network={'strongly-typed' if strongly_typed else 'weak-typed'} "
        f"profile min={profile['min']} opt={profile['opt']} max={profile['max']}",
        file=sys.stderr,
    )
    print(f"[build] building engine (TRT {trt.__version__}, GPU {gpu_name()}) ...", file=sys.stderr)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None (see TensorRT log above)")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    meta = {
        "onnx": os.path.abspath(onnx_path),
        "onnx_sha256": file_sha256(onnx_path),
        "engine": os.path.abspath(engine_path),
        "fp16": fp16,
        "fp8": bool(fp8),
        "strongly_typed": bool(strongly_typed),
        "profile": {k: list(v) for k, v in profile.items()},
        "tensorrt": trt.__version__,
        "gpu": gpu_name(),
        "workspace_gb": workspace_gb,
        "builder_opt_level": opt_level,
        "input_name": in_name,
        "output_name": out_name,
    }
    with open(_meta_path(engine_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(engine_path) / (1 << 20)
    print(f"[build] engine written: {engine_path} ({size_mb:.1f} MiB)", file=sys.stderr)
    return engine_path


def engine_is_valid(
    engine_path: str,
    onnx_path: str | None = None,
    fp16: bool = True,
    profile: dict | None = None,
    fp8: bool = False,
    strongly_typed: bool = False,
) -> bool:
    """Cheap staleness check against the sidecar metadata."""
    import tensorrt as trt

    if not os.path.exists(engine_path) or not os.path.exists(_meta_path(engine_path)):
        return False
    try:
        with open(_meta_path(engine_path), encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    if meta.get("tensorrt") != trt.__version__:
        return False
    if meta.get("gpu") != gpu_name():
        return False
    if bool(meta.get("fp16")) != bool(fp16):
        return False
    # Missing key means a legacy sidecar: valid for the default fp8=False path,
    # stale for an fp8=True request.
    if bool(meta.get("fp8")) != bool(fp8):
        return False
    if bool(meta.get("strongly_typed")) != bool(strongly_typed):
        return False
    if profile is not None:
        for k, v in profile.items():
            if list(meta.get("profile", {}).get(k, [])) != list(v):
                return False
    if onnx_path:
        if os.path.abspath(onnx_path) != meta.get("onnx"):
            return False
        if file_sha256(onnx_path) != meta.get("onnx_sha256"):
            return False
    return True


def ensure_engine(
    onnx_path: str,
    engine_path: str | None = None,
    fp16: bool = True,
    profile: dict | None = None,
    rebuild: bool = False,
    verbose: bool = True,
    fp8: bool = False,
    strongly_typed: bool = False,
) -> str:
    engine_path = engine_path or engine_path_for(
        onnx_path, fp16, fp8, strongly_typed
    )
    if rebuild or not engine_is_valid(
        engine_path, onnx_path, fp16, profile, fp8, strongly_typed
    ):
        return build_engine(
            onnx_path,
            engine_path,
            fp16=fp16,
            profile=profile,
            verbose=verbose,
            fp8=fp8,
            strongly_typed=strongly_typed,
        )
    if verbose:
        print(f"[build] reusing cached engine {engine_path}", file=sys.stderr)
    return engine_path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TrtRunner:
    """Minimal dynamic-shape TensorRT runner for NCHW float32 RGB models.

    Feed HWC uint8 frames, get HWC uint8 frames back.
    """

    def __init__(self, engine_path: str, verbose: bool = False, batch: int = 2,
                 legacy_io: bool = False):
        import tensorrt as trt

        self._batch = max(1, batch)
        self._trt = trt
        self._legacy_io = bool(legacy_io)
        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path} (engine is GPU/TRT-version bound!)")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create execution context")
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("engine must have exactly one input and one output")

        self._cuda = cuda()
        self.stream = self._cuda.new_stream()
        self._ev_start = self._cuda.new_event()
        self._ev_end = self._cuda.new_event()

        # Optional float32 -> uint8 cast engine, run on the same stream right
        # after the main network. Cuts the 4K download from 99MB to 25MB.
        self._cast_engine = None
        self._cast_ctx = None
        self._cast_ctxs = []
        self._cast_in = None
        self._cast_out = None
        self._d_u8 = None
        self._h_u8 = None
        self._n_u8 = 0
        if self.engine.get_tensor_dtype(self.output_name) == trt.DataType.FLOAT:
            self._init_cast(engine_path)

        # GPU-side uint8 HWC <-> float32 NCHW I/O engines (optional).
        self._io_ready = False
        self._pre_engine = None
        self._pre_runtime = None
        self._pre_ctx = None
        self._pre_in = None
        self._pre_out = None
        self._post_engine = None
        self._post_runtime = None
        self._post_ctx = None
        self._post_in = None
        self._post_out = None
        self._h_in_u8 = None
        self._d_in_u8 = None
        self._d_pre_out = None
        self._d_post_out = None
        self._u8_slots = []         # [{"ptr": int, "ref": weakref|None}]
        self._retired_u8_slots = []
        self._u8_slot_bytes = 0
        if not self._legacy_io:
            self._init_sr_io(engine_path)

        self._shape = None          # (H, W) input
        self._out_hw = None
        self._d_in = None
        self._d_out = None
        self._h_in = None
        self._h_out = None
        self._n_in_bytes = 0
        self._n_out_bytes = 0
        self.last_gpu_ms = 0.0

    def _init_cast(self, engine_path: str) -> None:
        import tensorrt as trt

        cast_path = os.path.join(
            os.path.dirname(os.path.abspath(engine_path)), "rgb_u8cast.engine"
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
            for _ in range(2):
                ctx = engine.create_execution_context()
                if ctx is None:
                    return
                contexts.append(ctx)
            self._cast_engine = engine
            self._cast_runtime = runtime
            self._cast_ctxs = contexts
            self._cast_ctx = contexts[0]
            self._cast_in = in_name
            self._cast_out = out_name
        except Exception:  # noqa: BLE001 - cast is an optimization, not required
            self._cast_engine = None
            self._cast_ctx = None
            self._cast_ctxs = []

    def _load_aux_engine(self, path: str):
        """Deserialize a tiny helper engine; returns (engine, runtime, in, out)."""
        import tensorrt as trt

        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                blob = f.read()
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(blob)
            if engine is None:
                return None
            in_name = out_name = None
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    in_name = name
                else:
                    out_name = name
            if in_name is None or out_name is None:
                return None
            return engine, runtime, in_name, out_name
        except Exception:  # noqa: BLE001 - helper engines are optional
            return None

    def _init_sr_io(self, engine_path: str) -> None:
        """Find and validate the optional uint8 HWC -> f32 NCHW pair.

        Layouts are inferred from the built engines: pre input must be
        (N,H,W,3) uint8 and post output (N,H,W,3) uint8. Anything unexpected
        leaves `_io_ready` False so the legacy NumPy path is used.
        """
        import tensorrt as trt

        try:
            if self.engine.get_tensor_dtype(self.input_name) != trt.DataType.FLOAT:
                return
            if self.engine.get_tensor_dtype(self.output_name) != trt.DataType.FLOAT:
                return
            ishape = tuple(self.engine.get_tensor_shape(self.input_name))
            oshape = tuple(self.engine.get_tensor_shape(self.output_name))
            if len(ishape) != 4 or len(oshape) != 4:
                return
            if ishape[1] != 3 or oshape[1] != 3:
                return

            base = os.path.dirname(os.path.abspath(engine_path))
            pre = self._load_aux_engine(os.path.join(base, SR_PRE_ENGINE))
            post = self._load_aux_engine(os.path.join(base, SR_POST_ENGINE))
            if pre is None or post is None:
                return
            pre_engine, pre_runtime, pre_in, pre_out = pre
            post_engine, post_runtime, post_in, post_out = post

            pre_in_shape = tuple(pre_engine.get_tensor_shape(pre_in))
            pre_out_shape = tuple(pre_engine.get_tensor_shape(pre_out))
            post_in_shape = tuple(post_engine.get_tensor_shape(post_in))
            post_out_shape = tuple(post_engine.get_tensor_shape(post_out))
            if len(pre_in_shape) != 4 or pre_in_shape[-1] != 3:
                return
            if len(pre_out_shape) != 4:
                return
            if len(post_in_shape) != 4 or post_in_shape[1] != 3:
                return
            if len(post_out_shape) != 4 or post_out_shape[-1] != 3:
                return

            if pre_engine.get_tensor_dtype(pre_in) != trt.DataType.UINT8:
                return
            if pre_engine.get_tensor_dtype(pre_out) != trt.DataType.FLOAT:
                return
            if post_engine.get_tensor_dtype(post_in) != trt.DataType.FLOAT:
                return
            if post_engine.get_tensor_dtype(post_out) != trt.DataType.UINT8:
                return

            pre_ctx = pre_engine.create_execution_context()
            post_ctx = post_engine.create_execution_context()
            if pre_ctx is None or post_ctx is None:
                return

            self._pre_engine = pre_engine
            self._pre_runtime = pre_runtime
            self._pre_ctx = pre_ctx
            self._pre_in = pre_in
            self._pre_out = pre_out
            self._post_engine = post_engine
            self._post_runtime = post_runtime
            self._post_ctx = post_ctx
            self._post_in = post_in
            self._post_out = post_out
            self._io_ready = True
        except Exception:  # noqa: BLE001 - fall back to the NumPy path
            self._io_ready = False

    @property
    def io_mode(self) -> str:
        if self._io_ready:
            return "gpu-io"
        if getattr(self, "_cast_ctxs", None):
            return "legacy-u8cast"
        return "legacy-f32"

    # -- buffers ---------------------------------------------------------
    def _alloc_for(self, h: int, w: int) -> None:
        n = self._batch
        self.context.set_input_shape(self.input_name, (n, 3, h, w))
        oshape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(d < 0 for d in oshape):
            oshape = (n, 3, h * 2, w * 2)
        oh, ow = int(oshape[2]), int(oshape[3])

        self._free_buffers()
        if self._io_ready:
            try:
                self._alloc_io_for(n, h, w, oh, ow)
                self._shape = (h, w)
                self._out_hw = (oh, ow)
                return
            except Exception:  # noqa: BLE001 - profile mismatch, use NumPy I/O
                self._io_ready = False
                self._free_buffers()
        self._alloc_legacy_for(n, h, w, oh, ow)
        self._shape = (h, w)
        self._out_hw = (oh, ow)

    def _alloc_io_for(self, n: int, h: int, w: int, oh: int, ow: int) -> None:
        """Allocate device buffers for pre -> main -> post with shared I/O.

        The main engine reads the pre engine's device output directly; post
        reads the main engine's device output. Only HWC uint8 crosses PCIe.
        """
        c = self._cuda
        n_u8_in = n * h * w * 3
        self._h_in_u8 = c.host_alloc(n_u8_in)
        self._d_in_u8 = c.malloc(n_u8_in)
        self._d_pre_out = c.malloc(n * 3 * h * w * 4)

        ok = self._pre_ctx.set_input_shape(self._pre_in, (n, h, w, 3))
        if ok is False:
            raise RuntimeError("pre engine rejected input shape")
        pre_out_shape = tuple(self._pre_ctx.get_tensor_shape(self._pre_out))
        if pre_out_shape != (n, 3, h, w):
            raise RuntimeError(f"pre output shape {pre_out_shape} != {(n, 3, h, w)}")
        self._pre_ctx.set_tensor_address(self._pre_in, self._d_in_u8)
        self._pre_ctx.set_tensor_address(self._pre_out, self._d_pre_out)

        self._d_out = c.malloc(n * 3 * oh * ow * 4)
        self._n_in_bytes = n * 3 * h * w * 4
        self._n_out_bytes = n * 3 * oh * ow * 4
        self.context.set_tensor_address(self.input_name, self._d_pre_out)
        self.context.set_tensor_address(self.output_name, self._d_out)

        ok = self._post_ctx.set_input_shape(self._post_in, (n, 3, oh, ow))
        if ok is False:
            raise RuntimeError("post engine rejected input shape")
        post_out_shape = tuple(self._post_ctx.get_tensor_shape(self._post_out))
        if post_out_shape != (n, oh, ow, 3):
            raise RuntimeError(f"post output shape {post_out_shape} != {(n, oh, ow, 3)}")
        self._d_post_out = c.malloc(n * oh * ow * 3)
        self._post_ctx.set_tensor_address(self._post_in, self._d_out)
        self._post_ctx.set_tensor_address(self._post_out, self._d_post_out)

        # Small ring of pinned download slots. Output arrays handed to the
        # caller are views into these buffers, so a slot is only reused after
        # every previous view into it has been released (weakref).
        self._u8_slot_bytes = n * oh * ow * 3
        for _ in range(3):
            ptr = c.host_alloc(self._u8_slot_bytes)
            self._u8_slots.append({"ptr": ptr, "ref": None})

    def _alloc_legacy_for(self, n: int, h: int, w: int, oh: int, ow: int) -> None:
        n_in = n * 3 * h * w * 4
        n_out = n * 3 * oh * ow * 4
        self._d_in = self._cuda.malloc(n_in)
        self._d_out = self._cuda.malloc(n_out)
        self._h_in = self._cuda.host_alloc(n_in)
        self._h_out = self._cuda.host_alloc(n_out)
        self._n_in_bytes = n_in
        self._n_out_bytes = n_out
        self.context.set_tensor_address(self.input_name, self._d_in)
        self.context.set_tensor_address(self.output_name, self._d_out)
        if self._cast_ctxs:
            frame_out = 3 * oh * ow
            n_u8 = n * frame_out
            self._d_u8 = self._cuda.malloc(n_u8)
            self._h_u8 = self._cuda.host_alloc(n_u8)
            self._n_u8 = n_u8
            for i in range(n):
                ctx = self._cast_ctxs[i]
                ctx.set_input_shape(self._cast_in, (1, 3, oh, ow))
                ctx.set_tensor_address(
                    self._cast_in, self._d_out + i * frame_out * 4
                )
                ctx.set_tensor_address(
                    self._cast_out, self._d_u8 + i * frame_out
                )

    def _free_buffers(self) -> None:
        for attr in ("_d_in", "_d_out", "_d_u8", "_d_in_u8", "_d_pre_out",
                     "_d_post_out"):
            ptr = getattr(self, attr, None)
            if ptr:
                self._cuda.free(ptr)
                setattr(self, attr, None)
        for attr in ("_h_in", "_h_out", "_h_u8", "_h_in_u8"):
            ptr = getattr(self, attr, None)
            if ptr:
                self._cuda.host_free(ptr)
                setattr(self, attr, None)
        # Outputs are views into pinned slots; retire instead of freeing so a
        # writer still holding a frame is never pointed at freed memory.
        self._retired_u8_slots.extend(self._u8_slots)
        self._u8_slots = []
        self._u8_slot_bytes = 0
        self._reap_retired_slots()

    def _reap_retired_slots(self) -> None:
        keep = []
        for slot in self._retired_u8_slots:
            ref = slot.get("ref")
            if ref is not None and ref() is not None:
                keep.append(slot)      # a caller still holds a view
            else:
                ptr = slot.get("ptr")
                if ptr:
                    self._cuda.host_free(ptr)
        self._retired_u8_slots = keep

    def _free_slot_or_defer(self, slot) -> None:
        """Free a pinned slot now, or after the last view into it dies."""
        ptr = slot.get("ptr")
        if not ptr:
            return
        ref = slot.get("ref")
        arr = ref() if ref is not None else None
        if arr is not None:
            # Caller still holds a view (e.g. it called close() early):
            # release the pinned memory when that view is garbage collected.
            weakref.finalize(arr, self._cuda.host_free, ptr)
        else:
            self._cuda.host_free(ptr)

    def close(self) -> None:
        self._free_buffers()
        for slot in self._retired_u8_slots:
            self._free_slot_or_defer(slot)
        self._retired_u8_slots = []
        for e in (self._ev_start, self._ev_end):
            if e:
                self._cuda.free_event(e)
        self._ev_start = self._ev_end = None
        if getattr(self, "stream", None):
            self._cuda.dll.cudaStreamDestroy(c_void_p(self.stream))
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- shapes ----------------------------------------------------------
    @property
    def out_hw(self):
        return self._out_hw

    # -- inference -------------------------------------------------------
    def _upload_batch(self, frames) -> None:
        n = len(frames)
        h, w = frames[0].shape[0], frames[0].shape[1]
        dst = pinned_array(self._h_in, (n, 3, h, w), c_float)
        for i, frame in enumerate(frames):
            np.multiply(
                np.transpose(frame, (2, 0, 1)),
                np.float32(1.0 / 255.0),
                out=dst[i],
            )
        self._cuda.h2d_async(self._d_in, self._h_in, self._n_in_bytes, self.stream)

    def _download_batch(self, n: int, oh: int, ow: int):
        src = pinned_array(self._h_out, (n, 3, oh, ow), c_float)
        np.clip(src, 0.0, 1.0, out=src)
        hwc = src.transpose(0, 2, 3, 1)
        np.multiply(hwc, np.float32(255.0), out=hwc)
        np.add(hwc, np.float32(0.5), out=hwc)  # +0.5 so truncation == rounding
        out = np.empty((n, oh, ow, 3), dtype=np.uint8)
        np.copyto(out, hwc, casting="unsafe")
        return [out[i] for i in range(n)]

    def _download_u8_batch(self, n: int, oh: int, ow: int):
        src = pinned_array(self._h_u8, (n, 3, oh, ow), ctypes.c_uint8)
        out = np.ascontiguousarray(src.transpose(0, 2, 3, 1))
        return [out[i] for i in range(n)]

    # -- GPU I/O path ----------------------------------------------------
    def _upload_io_batch(self, frames) -> None:
        n = len(frames)
        h, w = frames[0].shape[0], frames[0].shape[1]
        dst = pinned_array(self._h_in_u8, (n, h, w, 3), ctypes.c_uint8)
        for i, frame in enumerate(frames):
            np.copyto(dst[i], frame)
        self._cuda.h2d_async(self._d_in_u8, self._h_in_u8, n * h * w * 3, self.stream)

    def _acquire_u8_slot(self, n: int, oh: int, ow: int) -> np.ndarray:
        need = n * oh * ow * 3
        if need != self._u8_slot_bytes:
            raise RuntimeError("uint8 download slot size mismatch")
        self._reap_retired_slots()
        while True:
            for slot in self._u8_slots:
                ref = slot["ref"]
                if ref is None or ref() is None:
                    arr = pinned_array(
                        slot["ptr"], (n, oh, ow, 3), ctypes.c_uint8
                    )
                    slot["ref"] = weakref.ref(arr)
                    return arr
            # The writer still owns a view into every slot; wait for it to
            # finish instead of overwriting a frame that is being written.
            time.sleep(0.0005)

    def _download_io_batch(self, n: int, oh: int, ow: int):
        arr = self._acquire_u8_slot(n, oh, ow)
        nbytes = n * oh * ow * 3
        self._cuda.d2h_async(arr.ctypes.data, self._d_post_out, nbytes, self.stream)
        self._cuda.sync_stream(self.stream)
        return [arr[i] for i in range(n)]

    def process_batch(self, frames):
        """frames: list of HxWx3 uint8 RGB -> list of 2x HxWx3 uint8 RGB."""
        n = len(frames)
        if n == 0:
            return []
        h, w = frames[0].shape[0], frames[0].shape[1]
        if self._shape != (h, w) or self._batch != n:
            self._batch = n
            self._alloc_for(h, w)
        oh, ow = self._out_hw
        if self._io_ready:
            self._upload_io_batch(frames)
            self._pre_ctx.execute_async_v3(stream_handle=self.stream)
            self.context.execute_async_v3(stream_handle=self.stream)
            self._post_ctx.execute_async_v3(stream_handle=self.stream)
            return self._download_io_batch(n, oh, ow)
        self._upload_batch(frames)
        self.context.execute_async_v3(stream_handle=self.stream)
        if self._cast_ctxs:
            for i in range(n):
                self._cast_ctxs[i].execute_async_v3(stream_handle=self.stream)
            self._cuda.d2h_async(self._h_u8, self._d_u8, self._n_u8, self.stream)
            self._cuda.sync_stream(self.stream)
            return self._download_u8_batch(n, oh, ow)
        self._cuda.d2h_async(self._h_out, self._d_out, self._n_out_bytes, self.stream)
        self._cuda.sync_stream(self.stream)
        return self._download_batch(n, oh, ow)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """frame: HxWx3 uint8 RGB -> 2x HxWx3 uint8 RGB."""
        return self.process_batch([frame])[0]

    def process_timed(self, frame: np.ndarray):
        """Same as process(), plus device-side elapsed ms of the kernel."""
        h, w = frame.shape[0], frame.shape[1]
        if self._shape != (h, w) or self._batch != 1:
            self._batch = 1
            self._alloc_for(h, w)
        oh, ow = self._out_hw
        if self._io_ready:
            self._upload_io_batch([frame])
            self._cuda.record(self._ev_start, self.stream)
            self._pre_ctx.execute_async_v3(stream_handle=self.stream)
            self.context.execute_async_v3(stream_handle=self.stream)
            self._post_ctx.execute_async_v3(stream_handle=self.stream)
            self._cuda.record(self._ev_end, self.stream)
            out = self._download_io_batch(1, oh, ow)[0]
            self.last_gpu_ms = self._cuda.event_ms(self._ev_start, self._ev_end)
            return out
        self._upload_batch([frame])
        self._cuda.record(self._ev_start, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream)
        if self._cast_ctxs:
            self._cast_ctxs[0].execute_async_v3(stream_handle=self.stream)
        self._cuda.record(self._ev_end, self.stream)
        if self._cast_ctxs:
            self._cuda.d2h_async(self._h_u8, self._d_u8, self._n_u8, self.stream)
            self._cuda.sync_stream(self.stream)
            self.last_gpu_ms = self._cuda.event_ms(self._ev_start, self._ev_end)
            return self._download_u8_batch(1, oh, ow)[0]
        self._cuda.d2h_async(self._h_out, self._d_out, self._n_out_bytes, self.stream)
        self._cuda.sync_stream(self.stream)
        self.last_gpu_ms = self._cuda.event_ms(self._ev_start, self._ev_end)
        return self._download_batch(1, oh, ow)[0]
