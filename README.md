# RIFE 插帧与我们使用的超分模型管线

**Source-only B2 TensorRT pipeline** for offline anime 2x super-resolution with
RIFE interpolation, built for Windows + NVIDIA TensorRT. It is the B2 order used
by the Kazumi-Rife export page:

```text
1080p source
  -> RIFE 4.7 at 1080p (x2)
  -> Ani4K v2 4K upscale
  -> NVENC HEVC 47.95
  -> mux source audio
```

The order is deliberate: interpolating **before** upscaling avoids the
sub-pixel jitter that RIFE produces when it is run on already-upscaled 4K
frames. Do not swap the order.

## What is in this repository

| Path | Contents |
|---|---|
| `python/trt_host.py` | Streaming SR host: rawvideo 1080p in, 4K out, 2 workers x batch 2, GPU pre/post I/O, optional repeat protocol |
| `python/rife_trt_host.py` | Streaming RIFE 4.7 host, 1080p ×2/×3, optional causal repeat detection |
| `python/trt_engine.py` | TensorRT engine build/load, CUDA runtime via ctypes, GPU uint8 I/O ring |
| `python/build_engine.py` | Build FP16/FP32/experimental FP8 engines from user-supplied ONNX |
| `python/make_sr_io_onnx.py` | Generate the two tiny uint8↔float GPU I/O ONNX graphs |
| `python/make_u8cast_onnx.py` | Generate the tiny float→uint8 cast used by RIFE |
| `python/sample_b2.py` | Standalone OS-pipe B2 sample (decode → RIFE → SR → NVENC → mux) |
| `python/pipe/` | DirectML/ONNX Runtime fallback hosts (no TensorRT) |
| `app/` | Reference copies of the Kazumi-Rife Dart integration and `bench_b2.dart` |
| `docs/` | Architecture, build/run, benchmark, license notes |

## What is **not** in this repository

No model weights, no `.engine` files, no Python runtime, no ffmpeg/realcugan
binaries, no Flutter build output. Those are large third-party artifacts with
their own licenses. See [THIRD_PARTY.md](THIRD_PARTY.md) and
`docs/build.md` for how to supply them.

In particular:

- **Ani4K v2 model**: published separately by its authors; the research notes
  record **CC-BY-NC-4.0**. Do not bundle or redistribute it without checking
  the current model card.
- **AnimeJaNai V3.1**: **CC BY-NC-SA 4.0**; optional/local only. It is not
  included and must not be built into a GPL installer.
- **RIFE 4.7 model/weights**: obtain from the upstream project and follow its
  license; not included.

## Measured result (RTX 4060 Laptop 8GB)

3-minute 1080p source, B2, 4K HEVC, 47.95 fps, AAC:

| Configuration | Time | SR |
|---|---:|---:|
| FP16 static 1080p + repeat (cold) | **10.9 min** | 13.26 fps |
| Phase 0 baseline (FP16 static, no repeat) | 12.5 min | 11.58 fps |
| Pure SR, no interpolation (estimate) | ~5.5–6 min | 13–14 fps |

The machine must be cooled before timing: the same configuration immediately
after sustained benchmarks took 13.8 min, while the output file was
**SHA-256 identical**. See `docs/benchmarks.md`.

## Quick start

1. Install Windows NVIDIA driver, TensorRT 10.13.x, Python 3.13, and ffmpeg.
2. Obtain legally usable ONNX models (Ani4K v2, RIFE 4.7).
3. Generate and build the small GPU-I/O engines:

   ```bat
   python python\make_sr_io_onnx.py python\engines
   python python\build_engine.py --model path\to\ani4k2.onnx ^
     --engine python\engines\ani4k2_static1080_fp16.engine ^
     --min 2 3 1080 1920 --opt 2 3 1080 1920 --max 2 3 1080 1920
   python python\make_u8cast_onnx.py rgb_u8cast.onnx
   ```

   Build the RIFE engine with the profile required by `rife_trt_host.py`
   (`1 x 7 x 1080 x 1920` for 1080p). Build `rgb_u8cast.engine` from the tiny
   cast ONNX. Put `sr_pre_*.engine` and `sr_post_*.engine` next to the SR engine;
   the host discovers them there.

4. Run a sample:

   ```bat
   python python\sample_b2.py ^
     --ffmpeg C:\tools\ffmpeg.exe ^
     --python C:\Python313\python.exe ^
     --src clip.mp4 --out out.mp4 ^
     --rife-engine python\engines\rife_1080p.engine ^
     --sr-engine python\engines\ani4k2_static1080_fp16.engine ^
     --width 1920 --height 1080 --out-fps 48000/1001
   ```

See [docs/build.md](docs/build.md) for exact profiles, model licensing, and the
app-integration patch.

## App integration

`app/lib/services/sr/sr_pipeline.dart`, `sr_engine_service.dart`, and
`app/tool/bench_b2.dart` are reference copies from the Kazumi-Rife fork. They
are **not standalone**; apply them to a Kazumi-Rife source tree (GPLv3). The
same B2 order and engine selection are documented in
[docs/architecture.md](docs/architecture.md).

Repeat protocol is opt-in: both the RIFE and SR hosts must be started with
`--repeat-protocol`; otherwise the rawvideo path is used. The app enables it
only for `TensorRT + interpolation + streaming`.

## License

The code in this repository is **GPL-3.0**, because it is derived from
[Kazumi](https://github.com/Predidit/Kazumi) (GPLv3). See [LICENSE](LICENSE) and
[THIRD_PARTY.md](THIRD_PARTY.md). Models and runtime binaries are not part of
this repository and are not covered by its GPL license.

No warranty; this is an experimental pipeline. Verify model licenses and output
quality before distributing anything built with it.
