# Architecture

## Pipeline order (B2)

```text
source 1080p (passthrough decode)
  -> RIFE 4.7 at 1080p, x2
  -> Ani4K v2 at 1080p -> 4K
  -> NVENC HEVC 4K, 47.95 fps
  -> mux source audio
```

B2 means **interpolate before super-resolution**. Running RIFE on the 4K output
produces visible sub-pixel jitter in static scenes; doing it at 1080p first
reduced the measured static-scene artifact from 1.62/255 to 0.13/255 in the
source material. Do not turn this into `SR -> RIFE`.

## Components

### `trt_host.py` (SR)

- Input: rawvideo RGB24, source resolution (`1920x1080`).
- Output: rawvideo RGB24, `3840x2160`, one frame per input frame; timestamps are
  implied by the stream.
- Default: 2 TensorRT contexts (workers) × batch 2.
- GPU I/O path:
  - `sr_pre_u8hwc_to_f32chw`: uint8 HWC -> float32 NCHW, `x/255`;
  - the main Ani4K engine reads the pre engine's device buffer directly;
  - `sr_post_f32chw_to_u8hwc`: float32 NCHW -> uint8 HWC, `round(clip(x*255))`;
  - only 6.2 MB/frame goes H2D and 24.9 MB/frame D2H at 1080p→4K.
- If the pre/post engines are missing or shape-incompatible, it falls back to
  the legacy NumPy path (`rgb_u8cast` or float), so the host still works.
- `--repeat-protocol` reads the 8-byte RIFE header and reuses the previous 4K
  frame for held pairs; see below.

### `rife_trt_host.py` (interpolation)

- Input/output: rawvideo RGB24 at the source resolution.
- A static RIFE 4.7 engine with profile `1 x 7 x H x W` (1080p or 4K).
- For `N` input frames, output is `N*factor - (factor-1)` frames.
- `--repeat-protocol` runs the causal held-pair detector on source pairs:
  - 2×-subsampled changed-pixel fraction ≤ **0.05%** (any channel >2/255);
  - 2× MAD ≤ **0.02** (0–255 scale);
  - 4×-subsampled gray scene score < **8** at pair `i` and `i-1`;
  - pair `i` is confirmed only when pairs `i-1` and `i` are both raw-held
    (causal 2-pair confirmation, no lookahead, no gap filling);
  - singleton runs are implicitly excluded.
- A confirmed pair emits one 8-byte header (`magic 0x52494645`, `repeat=factor`)
  with **no payload**; SR writes its last 4K output `factor` times. All normal
  units are `header(repeat=1) + rgb24`.
- The expanded output count is unchanged; only SR inference is skipped.

### Frame counts (3-minute sample)

- Decode with `-fps_mode passthrough`: **4317** frames.
- RIFE ×2: **8633** frames (`2*4317-1`).
- Repeat detector: **446** confirmed pairs / **892** skipped SR frames
  (10.33% of 8633).
- SR host on the full sample: `units=8187 normal=7741 repeat_units=446
  expanded=8633`.
- Final mux must preserve 8633 frames; do not use `-shortest` (B-frame
  reordering trimmed the old sample to 8629).

## Static engine and app selection

The main Ani4K engine can be built with a static 1080p profile
(`min=opt=max` at `2×3×1080×1920`). It is selected only for exact
`1920x1080` input; any other size falls back to a dynamic engine. The profile is
not a range, so routing a different size to it would pad/crop incorrectly.

In the Dart app (`app/lib/services/sr/sr_engine_service.dart`):

- `trtEngineFor(width, height)` prefers `ani4k2_static1080_fp16.engine` for
  exact 1080p;
- `trtEnginePath` explicitly excludes `_static`, `sr_pre_`, `sr_post_`, `rife_`,
  and `rgb_u8cast` so directory ordering cannot select an auxiliary engine.

## Encoding and mux

- `hevc_nvenc -preset p5 -rc vbr -cq 19 -b:v 0 -pix_fmt yuv420p`.
- Decode uses `-map 0:v:0 -fps_mode passthrough`.
- Mux maps only `0:a:0?`, copies the video stream, adds `-tag:v hvc1`, and does
  **not** use `-shortest`.

## Performance model

The SR pass dominates. On an RTX 4060 Laptop 8GB, the static 1080p FP16 engine
runs at ~13.3 fps in the full B2 chain (cold), and the 3-minute sample completes
in 10.9 minutes. Pure SR without interpolation is roughly 5.5–6 minutes.

The chain is **GPU-sharing bound**, not host-I/O bound:

- SR host alone with a fast drain: 36.8 fps;
- SR + NVENC concurrently: 32.0 fps;
- RIFE + SR + NVENC full chain: 20–23 fps;
- an OS-pipe chain (no Dart relay) is 23.65 fps vs ~22.3 through the app relay,
  so the Dart relay is only ~5%.

Moving RIFE to the 780M was measured and rejected: DirectML on the iGPU gives
2.27 source fps, below the 6.6 required by Ani4K and the 10–11.5 required by
AnimeJaNai. The remaining practical levers are a smaller/faster model (with its
own license and quality gate) or accepting the current default.

## Repeat protocol caveats

- Both hosts must receive `--repeat-protocol`; flags must match.
- DirectML SR/RIFE and the file-mode path do not speak the header protocol.
- A repeat unit before any output, a bad magic, a truncated header/payload, or a
  sequence gap is a hard error, not silent padding.
- The SR writer keeps a dedicated 25 MB copy of the last 4K output; it never
  retains a reusable pinned-ring view.
- `--repeat-protocol` wins over the legacy `--static-threshold`; the legacy
  static path is disabled when repeat is on.
