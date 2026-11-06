# Benchmarks

All numbers are from a **single RTX 4060 Laptop 8GB**, Windows 11, TensorRT
10.13.3.9, FFmpeg NVENC. GPU work must be serialized and the GPU cooled before
timing. The author's machine is not a controlled benchmark rig; treat these as
reproducible lab notes, not guarantees.

## 3-minute source, B2 (1080p → 4K, 47.95 fps)

| Configuration | Time | SR | Output |
|---|---:|---:|---|
| Phase 1: FP16 static 1080p + repeat, cold | **10.9 min** | 13.26 fps | 411 MB, 8633 frames |
| Phase 0: FP16 static + GPU I/O, no repeat | 12.5 min | 11.58 fps | 455 MB, 8633 frames |
| batch 2 before Phase 0 | 13.4 min | 10.79 fps | 455 MB, 8635 frames (old decode) |
| pure SR, no interpolation (estimate) | ~5.5–6 min | 13–14 fps | 4317 frames |

Source decode with `-fps_mode passthrough`: **4317 frames**. RIFE ×2 gives
**8633** (`2*4317-1`). The final mux must preserve 8633; `-shortest` with
B-frames trimmed the old sample to 8629.

## Repeat detector

Full-source detector on the 3-minute sample:

- 4316 pairs; 975 raw-held; **446 confirmed pairs**; **892 skipped SR frames**
  (10.33%).
- Detection cost: **4.8–5.3 ms/pair** (~21 s CPU for the source).
- 10-second repeat vs no-repeat 4K SSIM: mean **0.9977**, minimum **0.9964**.
- 0.1% threshold is unsafe (freezes a 68×47 face/mouth region at 97–101 s);
  0.05% + causal 2-pair confirmation is the tested default.

## Stage costs and the GPU-sharing ceiling

On the same machine, with an AnimeJaNai model as a fast-SR probe:

| Probe | Result |
|---|---:|
| SR host alone, GPU-I/O, fast drain | 36.8 fps |
| SR host + concurrent NVENC | 32.0 fps |
| RIFE + SR + NVENC, full 10 s chain | 20–23 fps |
| OS-pipe chain (no Dart relay) | 23.65 fps |
| Dart-relay chain | ~22.3 fps |
| NVENC alone, 4K HEVC | ~110 fps |

The chain is GPU-sharing bound; the Dart relay costs only ~5%. During the chain
the GPU was 80–97 W, 2685 MHz, max 64°C, so this was not thermal throttling.
The earlier 15.9 fps "GPU-I/O cap" was an artifact of background load/heat; it
did not reproduce with an idle GPU.

RIFE on the Radeon 780M was measured and rejected:

| Device | Source fps |
|---|---:|
| 780M, ONNX Runtime DirectML | 2.27 |
| 4060, ONNX Runtime DirectML | 4.91 |
| 4060, TensorRT | 43 (standalone) |

B2 needs ~6.6 source fps for the current Ani4K SR and ~10–11.5 for a faster
AnimeJaNai SR.

## Thermal warning

The same Phase-1 configuration run immediately after continuous GPU benchmarks
took **13.8 min**; the cold rerun took **10.9 min**, and the two output files
were **SHA-256 identical** (`E4920F…`). Heat affects speed, not pixels. Cool the
GPU before timing; a 2–3 minute thermal spread can swamp a 10–15% optimization.

## Model experiments (not shipped)

- **AnimeJaNai V3.1 Performance/Balanced** (CC BY-NC-SA 4.0): local legacy path
  ~1.8× chain, but NC license and host-path constraints; no integration.
- **Ani4K FP8** (automatic and Q/DQ): slower than FP16 and/or visible quality
  loss; no integration. These were development-only experiments and are not
  part of this source-only repository.
