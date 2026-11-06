import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:kazumi/services/sr/sr_engine_service.dart';
import 'package:kazumi/services/sr/sr_presets.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

enum SrStage {
  probing('分析源文件'),
  extracting('提取帧'),
  upscaling('超分辨率'),
  encoding('编码分段'),
  muxing('封装输出'),
  interpolating('插帧'),
  done('完成');

  const SrStage(this.label);
  final String label;
}

/// 导出时把超分结果缩到目标分辨率所用的重采样算法（libswscale）。
enum SrDownscaleFilter {
  lanczos('Lanczos', 'lanczos', '锐利，细节保留好（推荐）'),
  spline('Spline', 'spline', '自然样条，边缘更柔和'),
  bicubic('Bicubic', 'bicubic', '平衡，速度快'),
  area('Area', 'area', '像素平均，整数倍缩小最自然'),
  gauss('Gauss', 'gauss', '高斯，最柔和，观感偏软'),
  neighbor('最近邻', 'neighbor', '硬边，适合像素风/线稿');

  const SrDownscaleFilter(this.label, this.flag, this.description);
  final String label;
  final String flag;
  final String description;
}

/// 目标输出高度。srNative 表示保持超分后的尺寸。
enum SrOutputSize {
  srNative('跟随超分', 0),
  h2160('2160p', 2160),
  h1440('1440p', 1440),
  h1080('1080p', 1080),
  original('原始尺寸', -1);

  const SrOutputSize(this.label, this.height);
  final String label;
  final int height;
}

/// 导出完成后的补帧倍率（RIFE）。
enum SrInterpolation {
  off('关闭', 1),
  x2('2x（24→48）', 2),
  x3('3x（24→72）', 3);

  const SrInterpolation(this.label, this.factor);
  final String label;
  final int factor;
}

enum SrPipelineMode {
  file('文件模式', 'PNG 落盘，稳定，兼容所有引擎'),
  pipe('零落盘管道', 'Ani4K v2 + DirectML，不写临时帧'),
  trt('TensorRT', '最快，需 NVIDIA 专用运行时');

  const SrPipelineMode(this.label, this.description);
  final String label;
  final String description;
}

enum SrEncoder {
  cpu('CPU · x264', '兼容性最好，速度中等'),
  nvenc('NVIDIA NVENC', '显卡编码，快 3-5 倍，画质略低');

  const SrEncoder(this.label, this.description);
  final String label;
  final String description;
}

class SrProbeResult {
  const SrProbeResult({
    required this.duration,
    required this.fps,
    required this.width,
    required this.height,
    required this.hasAudio,
  });

  final Duration duration;
  final double fps;
  final int width;
  final int height;
  final bool hasAudio;

  int get totalFrames => math.max(1, (duration.inMilliseconds / 1000 * fps).round());
}

class SrProgress {
  const SrProgress({
    required this.stage,
    required this.fraction,
    required this.detail,
  });

  final SrStage stage;

  /// 0..1 的总进度。
  final double fraction;
  final String detail;
}

class SrJobConfig {
  const SrJobConfig({
    required this.inputPath,
    required this.outputPath,
    required this.preset,
    required this.scale,
    required this.denoise,
  });

  final String inputPath;
  final String outputPath;
  final SrPreset preset;
  final int scale;
  final int denoise;
}

class SrCancelledException implements Exception {
  const SrCancelledException();
}

/// 把一段视频拆成「提取帧 → 超分 → 编码分段 → 合并 → 封装音轨」的离线管线。
///
/// 逐块处理（默认 240 帧一块），避免一集动画几十 GB 的 PNG 同时落盘。
class SrPipeline {
  SrPipeline({
    this.framesPerChunk = 240,
    this.crf = 16,
    this.encoder = SrEncoder.cpu,
    this.mode = SrPipelineMode.file,
    this.outputSize = SrOutputSize.srNative,
    this.downscale = SrDownscaleFilter.lanczos,
    this.interpolation = SrInterpolation.off,
  });

  final int framesPerChunk;
  final int crf;
  final SrEncoder encoder;
  final SrPipelineMode mode;
  final SrOutputSize outputSize;
  final SrDownscaleFilter downscale;
  final SrInterpolation interpolation;

  /// 目标高度；null 表示不缩放。低于超分输出时返回对应缩放滤镜参数。
  String? scaleFilterFor(int srHeight, int sourceHeight) {
    if (outputSize == SrOutputSize.srNative) return null;
    final target = outputSize == SrOutputSize.original
        ? sourceHeight
        : outputSize.height;
    if (target <= 0 || target >= srHeight) return null;
    return 'scale=-2:$target:flags=${downscale.flag}';
  }

  List<String> get _videoCodecArgs => switch (encoder) {
        SrEncoder.cpu => [
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '$crf',
            '-pix_fmt', 'yuv420p',
          ],
        SrEncoder.nvenc => [
            '-c:v', 'hevc_nvenc',
            '-preset', 'p5',
            '-rc', 'vbr',
            '-cq', '19',
            '-b:v', '0',
            '-pix_fmt', 'yuv420p',
          ],
      };

  final SrEngineService _engines = SrEngineService.instance;

  Process? _activeProcess;
  final List<Process> _pipedProcesses = [];
  bool _cancelRequested = false;

  bool get isRunning => _activeProcess != null;

  void cancel() {
    _cancelRequested = true;
    _activeProcess?.kill();
    for (final process in _pipedProcesses) {
      process.kill();
    }
  }

  Future<SrProbeResult> probe(String inputPath) async {
    // ffmpeg 在没有指定输出时会返回非 0，这是预期行为，信息都在 stderr。
    final result = await Process.run(
      _engines.ffmpegPath,
      ['-hide_banner', '-i', inputPath],
      stdoutEncoding: utf8,
      stderrEncoding: utf8,
    );
    final text = '${result.stdout}\n${result.stderr}';

    final durationMatch =
        RegExp(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)').firstMatch(text);
    if (durationMatch == null) {
      throw Exception('无法解析视频时长，可能不是有效的视频文件');
    }
    final duration = Duration(
      hours: int.parse(durationMatch.group(1)!),
      minutes: int.parse(durationMatch.group(2)!),
      milliseconds:
          (double.parse(durationMatch.group(3)!) * 1000).round(),
    );

    // 只使用第一个视频流，与解码端的 -map 0:v:0 保持一致。
    final lines = const LineSplitter().convert(text);
    final videoBlock = <String>[];
    var videoLine = '';
    for (var i = 0; i < lines.length; i++) {
      if (!lines[i].contains('Stream #') || !lines[i].contains('Video:')) {
        continue;
      }
      videoLine = lines[i];
      for (var j = i; j < lines.length; j++) {
        if (j > i && lines[j].trimLeft().startsWith('Stream #')) break;
        videoBlock.add(lines[j]);
      }
      break;
    }

    final sizeMatch = RegExp(r'(\d{2,5})x(\d{2,5})').firstMatch(videoLine);
    final fpsMatch = RegExp(r'([\d.]+)\s+fps').firstMatch(videoLine);
    final rotation = _rotationDegrees(videoBlock);
    // ffmpeg 默认会按 displaymatrix/rotate 自动旋转解码结果，
    // 所以竖屏素材必须用旋转后的宽高去匹配 host 和静态引擎。
    final normalizedRotation = (rotation?.abs() ?? 0) % 180;
    final swapDimensions = (normalizedRotation - 90).abs() < 1;
    final codedWidth = int.tryParse(sizeMatch?.group(1) ?? '') ?? 0;
    final codedHeight = int.tryParse(sizeMatch?.group(2) ?? '') ?? 0;

    return SrProbeResult(
      duration: duration,
      fps: double.tryParse(fpsMatch?.group(1) ?? '') ?? 24,
      width: swapDimensions ? codedHeight : codedWidth,
      height: swapDimensions ? codedWidth : codedHeight,
      hasAudio: text.contains('Audio:'),
    );
  }

  /// 从第一个视频流的信息块里读取旋转角度（displaymatrix 或 rotate 元数据）。
  double? _rotationDegrees(List<String> videoBlock) {
    for (final line in videoBlock) {
      final display = RegExp(
        r'display\s*matrix\s*:\s*rotation of\s*(-?\d+(?:\.\d+)?)\s*degrees',
        caseSensitive: false,
      ).firstMatch(line);
      if (display != null) return double.tryParse(display.group(1)!);
    }
    for (final line in videoBlock) {
      final rotate =
          RegExp(r'^\s*rotate\s*:\s*(-?\d+(?:\.\d+)?)').firstMatch(line);
      if (rotate != null) return double.tryParse(rotate.group(1)!);
    }
    return null;
  }

  Future<void> run(
    SrJobConfig config, {
    required void Function(SrProgress) onProgress,
    required void Function(String) onLog,
  }) async {
    if (!_engines.isAvailable) {
      throw Exception('超分引擎缺失，请使用完整安装包（sr 目录未找到）');
    }

    _cancelRequested = false;
    final tempRoot = Directory(path.join(
      (await _supportDirectory()).path,
      'sr_temp',
      DateTime.now().millisecondsSinceEpoch.toString(),
    ));
    await tempRoot.create(recursive: true);

    try {
      onProgress(const SrProgress(
        stage: SrStage.probing,
        fraction: 0,
        detail: '读取视频信息…',
      ));
      var probeResult = await probe(config.inputPath);
      onLog('源：${probeResult.width}x${probeResult.height} · '
          '${probeResult.fps.toStringAsFixed(3)} fps · '
          '${_formatDuration(probeResult.duration)} · '
          '约 ${probeResult.totalFrames} 帧');

      if (mode == SrPipelineMode.pipe || mode == SrPipelineMode.trt) {
        await _runPiped(
          config,
          probeResult,
          onProgress: onProgress,
          onLog: onLog,
        );
        onProgress(const SrProgress(
          stage: SrStage.done,
          fraction: 1,
          detail: '完成',
        ));
        return;
      }

      var effectiveInput = config.inputPath;
      if (interpolation != SrInterpolation.off) {
        effectiveInput = await _interpolateSource(
          config.inputPath,
          probeResult,
          tempRoot,
          onProgress: onProgress,
          onLog: onLog,
        );
        probeResult = await probe(effectiveInput);
        onLog('先插帧完成：${probeResult.fps.toStringAsFixed(3)} fps，开始超分');
      }

      final totalFrames = probeResult.totalFrames;
      final chunkCount = (totalFrames / framesPerChunk).ceil();
      final scaleFilter = scaleFilterFor(
        probeResult.height * config.scale,
        probeResult.height,
      );
      if (scaleFilter != null) {
        onLog('输出缩放：$scaleFilter');
      }
      final parts = <File>[];

      for (var chunk = 0; chunk < chunkCount; chunk++) {
        _throwIfCancelled();

        final firstFrame = chunk * framesPerChunk;
        final framesInChunk = math.min(framesPerChunk, totalFrames - firstFrame);
        final startSec = firstFrame / probeResult.fps;
        final chunkSec = framesInChunk / probeResult.fps;

        final chunkDir = Directory(path.join(tempRoot.path, 'chunk_$chunk'));
        final inDir = Directory(path.join(chunkDir.path, 'in'));
        final outDir = Directory(path.join(chunkDir.path, 'out'));
        await inDir.create(recursive: true);
        await outDir.create(recursive: true);

        final baseFraction = chunk / chunkCount;

        onProgress(SrProgress(
          stage: SrStage.extracting,
          fraction: baseFraction,
          detail: '第 ${chunk + 1}/$chunkCount 块 · 提取帧',
        ));
        onLog('[${chunk + 1}/$chunkCount] 提取 '
            '${startSec.toStringAsFixed(2)}s 起 $framesInChunk 帧');

        await _run(
          _engines.ffmpegPath,
          [
            '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-ss', startSec.toStringAsFixed(3),
            '-t', chunkSec.toStringAsFixed(3),
            '-i', effectiveInput,
            '-fps_mode', 'passthrough',
            '-start_number', '0',
            '${inDir.path}${Platform.pathSeparator}%06d.png',
          ],
        );

        final extracted = await _countFiles(inDir);
        if (extracted == 0) {
          throw Exception('第 ${chunk + 1} 块没有提取到任何帧');
        }

        onProgress(SrProgress(
          stage: SrStage.upscaling,
          fraction: baseFraction + 0.05 / chunkCount,
          detail: '第 ${chunk + 1}/$chunkCount 块 · 超分 $extracted 帧',
        ));

        final upscaleFuture = _run(
          _engines.enginePathFor(config.preset),
          _engines.buildArgs(
            preset: config.preset,
            inputDir: inDir.path,
            outputDir: outDir.path,
            scale: config.scale,
            denoise: _engines.clampDenoise(
              config.preset,
              config.scale,
              config.denoise,
            ),
          ),
        );

        // 引擎不输出百分比，用输出目录里的文件数估算进度。
        var polling = true;
        final poller = Timer.periodic(const Duration(milliseconds: 800), (_) async {
          if (!polling) return;
          final done = await _countFiles(outDir);
          onProgress(SrProgress(
            stage: SrStage.upscaling,
            fraction: baseFraction +
                (0.75 * math.min(1, done / extracted)) / chunkCount,
            detail: '第 ${chunk + 1}/$chunkCount 块 · '
                '超分 $done/$extracted 帧',
          ));
        });

        try {
          await upscaleFuture;
        } finally {
          polling = false;
          poller.cancel();
        }

        _throwIfCancelled();

        final partPath = path.join(chunkDir.path, 'part.mp4');
        onProgress(SrProgress(
          stage: SrStage.encoding,
          fraction: baseFraction + 0.8 / chunkCount,
          detail: '第 ${chunk + 1}/$chunkCount 块 · 编码',
        ));

        await _run(
          _engines.ffmpegPath,
          [
            '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-framerate', probeResult.fps.toString(),
            '-start_number', '0',
            '-i', '${outDir.path}${Platform.pathSeparator}%06d.png',
            if (scaleFilter != null) ...['-vf', scaleFilter],
            ..._videoCodecArgs,
            '-an',
            partPath,
          ],
        );

        parts.add(File(partPath));
        await _deleteQuietly(inDir);
        await _deleteQuietly(outDir);

        onProgress(SrProgress(
          stage: SrStage.upscaling,
          fraction: (chunk + 1) / chunkCount,
          detail: '已完成 ${chunk + 1}/$chunkCount 块',
        ));
      }

      _throwIfCancelled();

      onProgress(const SrProgress(
        stage: SrStage.muxing,
        fraction: 1,
        detail: '合并分段并封装原始音轨…',
      ));

      final listFile = File(path.join(tempRoot.path, 'parts.txt'));
      // concat 列表里反斜杠是转义符，Windows 路径必须转成正斜杠。
      await listFile.writeAsString(
        parts
            .map((f) =>
                "file '${f.path.replaceAll('\\', '/').replaceAll("'", r"'\''")}'")
            .join('\n'),
      );

      final output = File(config.outputPath);
      await output.parent.create(recursive: true);
      var muxCompleted = false;
      try {
        await _run(
          _engines.ffmpegPath,
          [
            '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', listFile.path,
            '-i', config.inputPath,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c', 'copy',
            '-movflags', '+faststart',
            if (encoder == SrEncoder.nvenc) ...['-tag:v', 'hvc1'],
            config.outputPath,
          ],
        );
        muxCompleted = true;
      } finally {
        if (!muxCompleted) await _deleteQuietly(output);
      }

      onLog('输出：${config.outputPath}');
      onProgress(const SrProgress(
        stage: SrStage.done,
        fraction: 1,
        detail: '完成',
      ));
    } on SrCancelledException {
      onLog('任务已取消');
      rethrow;
    } finally {
      _activeProcess = null;
      await _deleteQuietly(tempRoot);
    }
  }

  /// Repeat-aware RIFE→SR protocol is opt-out for TRT streaming only.
  ///
  /// Set `KAZUMI_SR_REPEAT_PROTOCOL=0` (or `off`/`false`) to run the legacy
  /// raw path for A/B tests.
  bool get _repeatProtocolEnabled {
    final value =
        Platform.environment['KAZUMI_SR_REPEAT_PROTOCOL']?.toLowerCase();
    return value != '0' && value != 'off' && value != 'false';
  }

  /// 启动 RIFE host：按**输入分辨率**选 TRT 静态引擎，否则回退 ONNX。
  ///
  /// 返回 (进程, 引擎描述, stderr 收集器)；调用方负责接线和退出码。
  Future<(Process, String, _StderrTail)> _startRife(
    int width,
    int height,
    int factor, {
    bool repeatProtocol = false,
  }) async {
    final engine = _engines.rifeTrtEngineFor(width, height);
    final useTrt = engine != null && _engines.hasRifeTrtRuntime;
    if (!useTrt && !_engines.hasRifeRuntime) {
      throw Exception(
        '插帧运行时不可用：${width}x$height 没有匹配的 TensorRT 引擎，'
        '且缺少 rife_host.py / rife47.onnx',
      );
    }
    if (repeatProtocol && !useTrt) {
      throw Exception('重复协议需要 TensorRT 插帧宿主');
    }
    final useRepeat = repeatProtocol && useTrt;
    final args = <String>[
      '-B',
      useTrt ? _engines.rifeTrtHostScript : _engines.rifeHostScript,
      '--width', '$width',
      '--height', '$height',
      if (useTrt)
        ...['--engine', engine]
      else
        ...['--model', _engines.rifeModel],
      '--factor', '$factor',
      if (useRepeat) ...[
        '--repeat-protocol',
        '--repeat-threshold', '0.05',
        '--repeat-scene', '8',
        '--repeat-mad', '0.02',
      ],
      '--quiet',
    ];
    final process = await Process.start(
      useTrt ? _engines.trtPython : _engines.pipePython,
      args,
      workingDirectory: useTrt ? _engines.trtRoot : _engines.pipeRoot,
    );
    _pipedProcesses.add(process);
    final stderrTail = _StderrTail(process);
    final label = useTrt
        ? 'TensorRT（${path.basename(engine)}）'
        : 'DirectML（ONNX）';
    return (process, label, stderrTail);
  }

  /// 零落盘管道：ffmpeg 解码 → （可选 RIFE）→ Ani4K v2 → ffmpeg 编码。
  ///
  /// 插帧固定发生在超分**之前**：先在源分辨率插帧，再逐帧超分。
  Future<void> _runPiped(
    SrJobConfig config,
    SrProbeResult probe, {
    required void Function(SrProgress) onProgress,
    required void Function(String) onLog,
  }) async {
    final isTrt = mode == SrPipelineMode.trt;
    if (isTrt ? !_engines.hasTrtRuntime : !_engines.hasPipeRuntime) {
      throw Exception(
        isTrt ? 'TensorRT 运行时缺失（sr/trt 未找到）' : '管道运行时缺失（sr/pipe 未找到）',
      );
    }
    final width = probe.width;
    final height = probe.height;
    if (width <= 0 || height <= 0) {
      throw Exception('无法确定视频分辨率');
    }
    // Ani4K v2 只做 2x，忽略其它倍数，避免产物尺寸与承诺不一致。
    final outWidth = width * 2;
    final outHeight = height * 2;
    final frameBytes = outWidth * outHeight * 3;
    final sourceFrameBytes = width * height * 3;
    final doInterp = interpolation != SrInterpolation.off;
    final factor = interpolation.factor;
    final rifeEngine = _engines.rifeTrtEngineFor(width, height);
    final rifeUsesTrt =
        rifeEngine != null && _engines.hasRifeTrtRuntime;
    // Only the TRT RIFE host emits the header, only the TRT SR host reads it,
    // and only the streaming pipe can carry it. DirectML pipe SR, DirectML
    // RIFE fallback and file mode stay raw on both sides.
    final repeatProtocol =
        isTrt && doInterp && rifeUsesTrt && _repeatProtocolEnabled;
    // RIFE 处理 N 个输入帧时输出 (N-1)*factor+1 = N*factor-(factor-1) 帧。
    // 解码结束后会用实际解码帧数再校正一次分母。
    var expectedFrames = doInterp
        ? probe.totalFrames * factor - (factor - 1)
        : probe.totalFrames;
    if (doInterp && !(rifeUsesTrt || _engines.hasRifeRuntime)) {
      throw Exception(
        '插帧运行时不可用：${width}x$height 没有匹配的 TensorRT 引擎，'
        '且缺少 rife_host.py / rife47.onnx',
      );
    }
    final scaleFilter = scaleFilterFor(outHeight, height);
    if (scaleFilter != null) {
      onLog('输出缩放：$scaleFilter');
    }

    final tempRoot = Directory(path.join(
      (await _supportDirectory()).path,
      'sr_temp',
      DateTime.now().millisecondsSinceEpoch.toString(),
    ));
    await tempRoot.create(recursive: true);
    final videoOnly = path.join(tempRoot.path, 'video.mp4');

    _StderrTail? decodeErr;
    _StderrTail? rifeErr;
    _StderrTail? hostErr;
    _StderrTail? encodeErr;
    StreamSubscription<List<int>>? encodeStdoutSub;

    try {
      onProgress(const SrProgress(
        stage: SrStage.extracting,
        fraction: 0,
        detail: '启动零落盘管道…',
      ));
      onLog('${isTrt ? 'TensorRT' : '管道'}模式：${width}x$height → '
          '${outWidth}x$outHeight（Ani4K v2 · ${isTrt ? 'TensorRT FP16' : 'DirectML'}）');
      if (doInterp) {
        onLog('顺序：先插帧（${width}x$height ×$factor）→ 再超分');
      }
      if (repeatProtocol) {
        onLog('静态重复协议：0.05% 变化 + 2 对确认（TRT）');
      }
      final rife = doInterp
          ? await _startRife(
              width,
              height,
              factor,
              repeatProtocol: repeatProtocol,
            )
          : null;
      if (rife != null) {
        rifeErr = rife.$3;
        onLog('插帧引擎：${rife.$2}');
      }
      _throwIfCancelled();

      final decode = await Process.start(_engines.ffmpegPath, [
        '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-i', config.inputPath,
        '-map', '0:v:0',
        '-vf', 'format=rgb24',
        // 保留源帧率，不补/不丢帧；旧默认模式会在尾部补重复帧。
        '-fps_mode', 'passthrough',
        '-f', 'rawvideo',
        '-',
      ]);
      _pipedProcesses.add(decode);
      decodeErr = _StderrTail(decode);

      final hostArgs = <String>[
        '-B',
        isTrt ? _engines.trtHostScript : _engines.pipeHostScript,
        '--width', '$width',
        '--height', '$height',
        if (isTrt)
          ...['--engine', _engines.trtEngineFor(width, height)!]
        else
          ...['--model', _engines.pipeModel],
        if (repeatProtocol) '--repeat-protocol',
        '--quiet',
      ];
      final host = await Process.start(
        isTrt ? _engines.trtPython : _engines.pipePython,
        hostArgs,
        workingDirectory: isTrt ? _engines.trtRoot : _engines.pipeRoot,
      );
      _pipedProcesses.add(host);
      hostErr = _StderrTail(host);

      final encode = await Process.start(_engines.ffmpegPath, [
        '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', '${outWidth}x$outHeight',
        '-framerate', (probe.fps * factor).toString(),
        '-i', '-',
        if (scaleFilter != null) ...['-vf', scaleFilter],
        ..._videoCodecArgs,
        '-an',
        videoOnly,
      ]);
      _pipedProcesses.add(encode);
      encodeErr = _StderrTail(encode);
      encodeStdoutSub = encode.stdout.listen((_) {});
      _throwIfCancelled();

      var decodedBytes = 0;
      var processedBytes = 0;
      var lastReport = 0;
      final first = rife == null ? host : rife.$1;
      final decodeDone = _countBytes(decode.stdout, (bytes) {
        decodedBytes += bytes;
      }).pipe(first.stdin).then((_) {
        final decodedFrames = decodedBytes ~/ sourceFrameBytes;
        if (decodedFrames > 0) {
          expectedFrames = doInterp
              ? decodedFrames * factor - (factor - 1)
              : decodedFrames;
        }
      });
      final rifeDone = rife == null
          ? Future<void>.value()
          : rife.$1.stdout.pipe(host.stdin);
      final hostDone = _countBytes(host.stdout, (bytes) {
        processedBytes += bytes;
        if (processedBytes - lastReport >= frameBytes * 12) {
          lastReport = processedBytes;
          final frames = processedBytes ~/ frameBytes;
          onProgress(SrProgress(
            stage: SrStage.upscaling,
            fraction: expectedFrames <= 0
                ? 0
                : (frames / expectedFrames).clamp(0.0, 1.0),
            detail: doInterp
                ? '插帧+超分 $frames/$expectedFrames 帧'
                : '超分 $frames/$expectedFrames 帧',
          ));
        }
      }).pipe(encode.stdin);

      Object? relayError;
      try {
        await Future.wait([decodeDone, rifeDone, hostDone]);
      } catch (error) {
        relayError = error;
        // 下游提前退出时上游可能还卡在写管道，主动收掉避免卡死。
        for (final process in [
          decode,
          host,
          encode,
          if (rife != null) rife.$1,
        ]) {
          process.kill();
        }
      }
      final encodeCode = await encode.exitCode;
      final decodeCode = await decode.exitCode;
      final hostCode = await host.exitCode;
      final rifeCode = rife == null ? null : await rife.$1.exitCode;
      await _flushStderr([decodeErr, rifeErr, hostErr, encodeErr]);
      if (_cancelRequested) throw const SrCancelledException();
      if (relayError != null) {
        final details = <String>['管道转发失败：$relayError'];
        if (decodeCode != 0) {
          details.add(_withTail('解码器退出码 $decodeCode', decodeErr.tail));
        }
        if (rifeCode != null && rifeCode != 0) {
          details.add(_withTail('插帧宿主退出码 $rifeCode', rifeErr?.tail));
        }
        if (hostCode != 0) {
          details.add(_withTail('超分宿主退出码 $hostCode', hostErr.tail));
        }
        if (encodeCode != 0) {
          details.add(_withTail('编码器退出码 $encodeCode', encodeErr.tail));
        }
        throw Exception(details.join('\n'));
      }
      if (decodeCode != 0) {
        throw Exception(
          _withTail('解码器退出码 $decodeCode', decodeErr.tail),
        );
      }
      if (rifeCode != null && rifeCode != 0) {
        throw Exception(
          _withTail('插帧宿主退出码 $rifeCode', rifeErr?.tail),
        );
      }
      if (hostCode != 0) {
        throw Exception(
          _withTail('超分宿主退出码 $hostCode', hostErr.tail),
        );
      }
      if (encodeCode != 0) {
        throw Exception(
          _withTail('编码器退出码 $encodeCode', encodeErr.tail),
        );
      }

      final decodedFrames = decodedBytes ~/ sourceFrameBytes;
      final outputFrames = processedBytes ~/ frameBytes;
      onLog('解码 $decodedFrames 帧 · 输出 $outputFrames 帧');

      onProgress(const SrProgress(
        stage: SrStage.muxing,
        fraction: 1,
        detail: '封装原始音轨…',
      ));

      final output = File(config.outputPath);
      await output.parent.create(recursive: true);
      var muxCompleted = false;
      try {
        await _run(
          _engines.ffmpegPath,
          [
            '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-i', videoOnly,
            '-i', config.inputPath,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c', 'copy',
            '-movflags', '+faststart',
            if (encoder == SrEncoder.nvenc) ...['-tag:v', 'hvc1'],
            config.outputPath,
          ],
        );
        muxCompleted = true;
      } finally {
        if (!muxCompleted) await _deleteQuietly(output);
      }

      onLog('输出：${config.outputPath}');
      onProgress(const SrProgress(
        stage: SrStage.done,
        fraction: 1,
        detail: '完成',
      ));
    } finally {
      // 管道模式的临时目录是函数内新建的，必须自己清理。
      await _stopPipedProcesses();
      await encodeStdoutSub?.cancel();
      await decodeErr?.cancel();
      await rifeErr?.cancel();
      await hostErr?.cancel();
      await encodeErr?.cancel();
      await _deleteQuietly(tempRoot);
    }
  }

  Stream<List<int>> _countBytes(
    Stream<List<int>> input,
    void Function(int bytes) onChunk,
  ) async* {
    await for (final chunk in input) {
      onChunk(chunk.length);
      yield chunk;
    }
  }

  void _throwIfCancelled() {
    if (_cancelRequested) throw const SrCancelledException();
  }

  /// 先把源片插帧到临时视频（分辨率不变，帧率 ×factor）。
  ///
  /// 文件模式（Real-CUGAN / Real-ESRGAN）在抽帧之前调用；管道模式直接在
  /// 解码和超分之间串 RIFE，不经过这个临时文件。
  ///
  /// TODO(phase1): 文件模式仍把 RIFE 输出编码成 1080p48 中间视频，再抽 PNG，
  /// 多一代有损。直接 RIFE→PNG 需要重写按块抽帧/超分/编码的顺序，而且会
  /// 改变输出画面（少一次压缩损失），Phase 0 不做；B2 管道模式不经过这里。
  Future<String> _interpolateSource(
    String inputPath,
    SrProbeResult info,
    Directory tempRoot, {
    required void Function(SrProgress) onProgress,
    required void Function(String) onLog,
  }) async {
    final factor = interpolation.factor;
    final width = info.width;
    final height = info.height;
    if (width <= 0 || height <= 0) {
      throw Exception('无法读取源视频尺寸');
    }
    final outFps = info.fps * factor;
    // RIFE 处理 N 个输入帧时输出 (N-1)*factor+1 = N*factor-(factor-1) 帧。
    var totalFrames = info.totalFrames * factor - (factor - 1);
    final frameBytes = width * height * 3;
    final videoOnly = path.join(tempRoot.path, 'rife_source.mp4');

    onProgress(const SrProgress(
      stage: SrStage.interpolating,
      fraction: 0,
      detail: '先插帧…',
    ));
    final (host, label, hostErr) = await _startRife(width, height, factor);
    onLog('先插帧：${width}x$height ×$factor → ${outFps.toStringAsFixed(3)} fps');
    onLog('插帧引擎：$label');

    _StderrTail? decodeErr;
    _StderrTail? encodeErr;
    StreamSubscription<List<int>>? encodeStdoutSub;
    try {
      _throwIfCancelled();
      final decode = await Process.start(_engines.ffmpegPath, [
        '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-i', inputPath,
        '-map', '0:v:0',
        '-vf', 'format=rgb24',
        // 保留源帧率，不补/不丢帧；旧默认模式会在尾部补重复帧。
        '-fps_mode', 'passthrough',
        '-f', 'rawvideo',
        '-',
      ]);
      _pipedProcesses.add(decode);
      decodeErr = _StderrTail(decode);

      final encode = await Process.start(_engines.ffmpegPath, [
        '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', '${width}x$height',
        '-framerate', outFps.toString(),
        '-i', '-',
        ..._videoCodecArgs,
        '-an',
        videoOnly,
      ]);
      _pipedProcesses.add(encode);
      encodeErr = _StderrTail(encode);
      encodeStdoutSub = encode.stdout.listen((_) {});
      _throwIfCancelled();

      var decodedBytes = 0;
      var processedBytes = 0;
      var lastReport = 0;
      final decodeDone = _countBytes(decode.stdout, (bytes) {
        decodedBytes += bytes;
      }).pipe(host.stdin).then((_) {
        final decodedFrames = decodedBytes ~/ frameBytes;
        if (decodedFrames > 0) {
          totalFrames = decodedFrames * factor - (factor - 1);
        }
      });
      final hostDone = _countBytes(host.stdout, (bytes) {
        processedBytes += bytes;
        if (processedBytes - lastReport >= frameBytes * 12) {
          lastReport = processedBytes;
          final frames = processedBytes ~/ frameBytes;
          onProgress(SrProgress(
            stage: SrStage.interpolating,
            fraction: totalFrames <= 0
                ? 0
                : (frames / totalFrames).clamp(0.0, 1.0),
            detail: '插帧 $frames/$totalFrames 帧',
          ));
        }
      }).pipe(encode.stdin);

      Object? relayError;
      try {
        await Future.wait([decodeDone, hostDone]);
      } catch (error) {
        relayError = error;
        for (final process in [decode, host, encode]) {
          process.kill();
        }
      }
      final encodeCode = await encode.exitCode;
      final decodeCode = await decode.exitCode;
      final hostCode = await host.exitCode;
      await _flushStderr([decodeErr, hostErr, encodeErr]);
      if (_cancelRequested) throw const SrCancelledException();
      if (relayError != null) {
        final details = <String>['管道转发失败：$relayError'];
        if (decodeCode != 0) {
          details.add(_withTail('解码器退出码 $decodeCode', decodeErr.tail));
        }
        if (hostCode != 0) {
          details.add(_withTail('插帧宿主退出码 $hostCode', hostErr.tail));
        }
        if (encodeCode != 0) {
          details.add(_withTail('插帧编码退出码 $encodeCode', encodeErr.tail));
        }
        throw Exception(details.join('\n'));
      }
      if (decodeCode != 0) {
        throw Exception(
          _withTail('解码器退出码 $decodeCode', decodeErr.tail),
        );
      }
      if (hostCode != 0) {
        throw Exception(
          _withTail('插帧宿主退出码 $hostCode', hostErr.tail),
        );
      }
      if (encodeCode != 0) {
        throw Exception(
          _withTail('插帧编码退出码 $encodeCode', encodeErr.tail),
        );
      }
      onLog('插帧完成：$factor 倍，${outFps.toStringAsFixed(3)} fps');
      return videoOnly;
    } finally {
      await _stopPipedProcesses();
      await encodeStdoutSub?.cancel();
      await decodeErr?.cancel();
      await hostErr.cancel();
      await encodeErr?.cancel();
    }
  }

  Future<int> _run(String executable, List<String> args) async {
    _throwIfCancelled();
    final process = await Process.start(executable, args);
    _activeProcess = process;

    final stdoutSub = process.stdout.listen((_) {});
    final stderrTail = _StderrTail(process);
    try {
      final exitCode = await process.exitCode;
      await _flushStderr([stderrTail]);
      if (_cancelRequested) throw const SrCancelledException();
      if (exitCode != 0) {
        throw Exception(
          _withTail(
            '${path.basename(executable)} 退出码 $exitCode',
            stderrTail.tail,
          ),
        );
      }
      return exitCode;
    } finally {
      await stdoutSub.cancel();
      await stderrTail.cancel();
      _activeProcess = null;
    }
  }

  /// 结束后强制收掉管道子进程，避免失败路径留下孤儿进程。
  Future<void> _stopPipedProcesses() async {
    final processes = List<Process>.of(_pipedProcesses);
    _pipedProcesses.clear();
    for (final process in processes) {
      process.kill();
    }
    // 并行回收，最多等 5 秒；已退出的进程会立即返回。
    await Future.wait(
      processes.map(
        (process) => process.exitCode.timeout(
          const Duration(seconds: 5),
          onTimeout: () => -1,
        ),
      ),
    ).catchError((_) => const <int>[]);
  }

  /// 进程退出后给 stderr 流一点时间把尾部行送完，避免报错时丢掉最后一行。
  Future<void> _flushStderr(Iterable<_StderrTail?> tails) async {
    for (final tail in tails.whereType<_StderrTail>()) {
      try {
        await tail.done.timeout(const Duration(milliseconds: 200));
      } catch (_) {
        // 超时则使用已经收到的行。
      }
    }
  }

  String _withTail(String message, String? tail) {
    if (tail == null || tail.isEmpty) return message;
    return '$message\n$tail';
  }

  Future<int> _countFiles(Directory dir) async {
    if (!await dir.exists()) return 0;
    var count = 0;
    await for (final entity in dir.list(followLinks: false)) {
      if (entity is File) count++;
    }
    return count;
  }

  Future<void> _deleteQuietly(FileSystemEntity entity) async {
    try {
      if (await entity.exists()) {
        await entity.delete(recursive: true);
      }
    } catch (_) {
      // 临时目录清理失败不影响结果。
    }
  }

  Future<Directory> _supportDirectory() async {
    // 与 App 其余数据（hive / 日志）保持同一个目录。
    return getApplicationSupportDirectory();
  }

  String _formatDuration(Duration d) {
    final h = d.inHours;
    final m = d.inMinutes.remainder(60);
    final s = d.inSeconds.remainder(60);
    return h > 0 ? '$h:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}'
        : '$m:${s.toString().padLeft(2, '0')}';
  }
}

/// 持续读走子进程 stderr，只保留最后若干行用于报错。
///
/// Process.start 默认使用管道；如果父进程不读，TensorRT/ffmpeg 在写满
/// 64KB 管道后会永久阻塞。所有长驻子进程都必须包一层。
class _StderrTail {
  _StderrTail(Process process) {
    _subscription = process.stderr
        .transform(const Utf8Decoder(allowMalformed: true))
        .transform(const LineSplitter())
        .listen(
          _add,
          onError: (Object _) => _finish(),
          onDone: _finish,
          cancelOnError: false,
        );
  }

  static const int _maxLines = 40;
  final List<String> _lines = [];
  final Completer<void> _done = Completer<void>();
  late final StreamSubscription<String> _subscription;

  void _add(String line) {
    _lines.add(line);
    if (_lines.length > _maxLines) _lines.removeAt(0);
  }

  void _finish() {
    if (!_done.isCompleted) _done.complete();
  }

  String get tail => _lines.reversed.take(20).toList().reversed.join('\n');

  /// stderr 流读完（进程退出）后完成；用于报错前尽量取到最后几行。
  Future<void> get done => _done.future;

  Future<void> cancel() => _subscription.cancel();
}
