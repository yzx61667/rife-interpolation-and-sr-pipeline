# Third-party components and licenses

This repository contains **source code only**. The following components are
required at runtime but are not included; obtain them from their authors and
follow their licenses.

| Component | Role | License / distribution note |
|---|---|---|
| [Kazumi](https://github.com/Predidit/Kazumi) | Upstream Flutter app; this pipeline is derived from a Kazumi-Rife fork | **GPL-3.0**; keep upstream attribution and source availability |
| [Ani4K v2](https://github.com/bloc97/Anime4K) | 1080p→4K super-resolution model | The research notes record **CC-BY-NC-4.0**; verify the current model card. Do not redistribute without permission |
| [AnimeJaNai V3.1](https://github.com/the-database/mpv-AnimeJaNai) | Optional/local 2x SR model | **CC BY-NC-SA 4.0**; not bundled, not for a GPL installer; user-supplied only |
| [RIFE / Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | Frame interpolation model | Obtain the RIFE 4.7 ONNX/engine from the upstream project and follow its license; not included |
| [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) | Engine build/runtime | NVIDIA proprietary license; user installs the matching TensorRT/CUDA runtime |
| [FFmpeg](https://ffmpeg.org/) | Decode/encode/mux | LGPL/GPL depending on build; user installs; not included |
| [ONNX Runtime](https://onnxruntime.ai/) | DirectML fallback hosts | MIT; optional; not included |
| [Real-CUGAN](https://github.com/nihui/realcugan-ncnn-vulkan) / [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | App fallback upscalers (not used by the TRT B2 path) | See upstream; binaries are not included |

## Model and binary policy

- Engine files (`.engine`), ONNX weights (`.onnx`), FFmpeg/Real-CUGAN/Real-ESRGAN
  binaries, Python runtimes, TensorRT DLLs, and Flutter build output are
  **gitignored** and must never be committed.
- The GPL-3.0 license of this repository applies to the source code, not to
  separately obtained models or proprietary runtimes.
- If you publish a build, keep non-commercial models out of the installer, or
  make them explicit user-supplied downloads with the correct license notice.
