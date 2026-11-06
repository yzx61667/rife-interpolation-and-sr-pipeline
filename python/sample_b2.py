#!/usr/bin/env python3
"""Standalone B2 sample: RIFE at source resolution -> SR 2x -> NVENC -> mux.

This script deliberately contains no model or engine files. Pass the paths to
engines you built yourself (see docs/build.md). It mirrors the export order used
by the Kazumi-Rife app: interpolate 1080p first, then upscale to 4K.

Example (PowerShell):
    python sample_b2.py `
      --ffmpeg C:\\tools\\ffmpeg.exe `
      --python C:\\Python313\\python.exe `
      --src clip.mp4 --out out.mp4 `
      --rife-engine engines/rife_1080p.engine `
      --sr-engine engines/sr_1080p.engine `
      --width 1920 --height 1080 --out-fps 48000/1001
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time


def drain(proc: subprocess.Popen, sink: list[str], prefix: str) -> None:
    assert proc.stderr is not None
    for raw in iter(proc.stderr.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            sink.append(line)
            if len(sink) > 40:
                del sink[:10]
            print(f"[{prefix}] {line}", file=sys.stderr, flush=True)


def start(args: list[str], stdin=None, stdout=None) -> tuple[subprocess.Popen, list[str], threading.Thread]:
    proc = subprocess.Popen(
        args,
        stdin=stdin,
        stdout=stdout,
        stderr=subprocess.PIPE,
    )
    lines: list[str] = []
    thread = threading.Thread(target=drain, args=(proc, lines, os.path.basename(args[0])), daemon=True)
    thread.start()
    return proc, lines, thread


def main() -> int:
    ap = argparse.ArgumentParser(description="B2: RIFE source-res -> SR 2x -> NVENC -> mux")
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rife-engine", required=True)
    ap.add_argument("--sr-engine", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--factor", type=int, default=2, choices=(2, 3))
    ap.add_argument("--out-fps", default="48000/1001")
    ap.add_argument("--sr-workers", type=int, default=2)
    ap.add_argument("--sr-batch", type=int, default=2)
    ap.add_argument("--repeat", dest="repeat", action="store_true", default=True)
    ap.add_argument("--no-repeat", dest="repeat", action="store_false")
    ap.add_argument("--cq", type=int, default=19)
    ap.add_argument("--tmp", default=None, help="temporary video-only output (default: <out>.video.mp4)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rife_host = os.path.join(here, "rife_trt_host.py")
    sr_host = os.path.join(here, "trt_host.py")
    for p in (args.ffmpeg, args.python, args.src, args.rife_engine, args.sr_engine,
              rife_host, sr_host):
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            return 2

    width, height = args.width, args.height
    out_w, out_h = width * 2, height * 2
    tmp = args.tmp or (args.out + ".video.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    started = time.perf_counter()
    procs = []
    try:
        decode, _, _ = start([
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", args.src, "-map", "0:v:0", "-vf", "format=rgb24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-",
        ], stdout=subprocess.PIPE)

        rife_args = [
            args.python, "-B", rife_host,
            "--width", str(width), "--height", str(height),
            "--engine", args.rife_engine, "--factor", str(args.factor),
        ]
        if args.repeat:
            rife_args += ["--repeat-protocol", "--repeat-threshold", "0.05",
                          "--repeat-scene", "8", "--repeat-mad", "0.02"]
        rife_args += ["--stats", "--quiet"]
        rife, _, _ = start(rife_args, stdin=decode.stdout, stdout=subprocess.PIPE)

        sr_args = [
            args.python, "-B", sr_host,
            "--width", str(width), "--height", str(height),
            "--engine", args.sr_engine,
            "--workers", str(args.sr_workers), "--batch", str(args.sr_batch),
        ]
        if args.repeat:
            sr_args.append("--repeat-protocol")
        sr_args += ["--stats", "--quiet"]
        sr, _, _ = start(sr_args, stdin=rife.stdout, stdout=subprocess.PIPE)

        enc, _, _ = start([
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{out_w}x{out_h}", "-framerate", args.out_fps,
            "-i", "-",
            "-c:v", "hevc_nvenc", "-preset", "p5",
            "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0",
            "-pix_fmt", "yuv420p", "-an", tmp,
        ], stdin=sr.stdout)

        procs = [decode, rife, sr, enc]
        for p in (decode, rife, sr, enc):
            for pipe in (p.stdin, p.stdout):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
        codes = [enc.wait(), sr.wait(), rife.wait(), decode.wait()]
        chain_s = time.perf_counter() - started
        if any(codes):
            print(f"B2 failed: exit codes {codes}", file=sys.stderr)
            return 1

        mux_start = time.perf_counter()
        mux = subprocess.run([
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", tmp, "-i", args.src,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c", "copy", "-movflags", "+faststart", "-tag:v", "hvc1",
            args.out,
        ], stderr=subprocess.PIPE)
        if mux.returncode != 0:
            print(mux.stderr.decode("utf-8", "replace"), file=sys.stderr)
            return 1
        print(f"B2 {chain_s/60:.1f} min; mux {time.perf_counter()-mux_start:.1f}s; "
              f"total {time.perf_counter()-started:.1f}s")
        print(f"out: {args.out}")
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        if os.path.exists(tmp) and not os.path.exists(args.out):
            try:
                os.remove(tmp)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
