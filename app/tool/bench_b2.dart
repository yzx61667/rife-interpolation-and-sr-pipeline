// Standalone B2 benchmark: replicates SrPipeline._runPiped with Process.start
// and Dart stream relays. No Flutter, no path_provider.
//
// Usage (serialized GPU access):
//   dart tool/bench_b2.dart --clip samples/clip10.mp4 --sr-root assets/sr
//   dart tool/bench_b2.dart --seconds 5 --out /tmp/b2.mp4
//
// Use `dart <file>` rather than `dart run`: this project's media_kit build
// hook tries to resolve MSVC and can fail before the script starts.
//
// Every rawvideo frame is forwarded by a Dart stream (_countBytes), so the
// total includes the Dart relay cost and can be compared with the OS-pipe
// sample script (python/sample_b2.py).

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:path/path.dart' as path;

const String _defaultClip = r'samples/clip10.mp4';
const String _defaultSrRoot = r'assets/sr';

Future<void> main(List<String> arguments) async {
  final options = _parseArgs(arguments);
  if (options == null) return;

  final started = <Process>[];
  try {
    final srRoot = options.srRoot;
    final ffmpeg = path.join(srRoot, 'ffmpeg', 'ffmpeg.exe');
    final trtRoot = path.join(srRoot, 'trt');
    final python = path.join(trtRoot, 'runtime', 'python.exe');
    final rifeHost = path.join(trtRoot, 'rife_trt_host.py');
    final srHost = path.join(trtRoot, 'trt_host.py');
    final rifeEngine =
        path.join(trtRoot, 'engines', 'rife_v4.7_1920x1080.engine');
    final srEngine = options.srEngine ??
        _findSrEngine(
          path.join(trtRoot, 'engines'),
          preferStatic: !options.dynamicEngine,
        );

    if (!File(options.clip).existsSync()) {
      throw Exception('找不到源视频: ${options.clip}');
    }
    if (srEngine == null) {
      throw Exception('找不到 ani4k2*.engine');
    }
    final requiredFiles = <String>[
      ffmpeg,
      python,
      rifeHost,
      srHost,
      rifeEngine,
      srEngine,
    ];
    for (final file in requiredFiles) {
      if (!File(file).existsSync()) {
        throw Exception('缺少文件: $file');
      }
    }

    final probe = await _probe(ffmpeg, options.clip);
    if (probe.width != 1920 || probe.height != 1080) {
      throw Exception(
        '当前基准只支持 1080p 源（RIFE TRT 引擎固定 1920x1080），'
        '实际 ${probe.width}x${probe.height}',
      );
    }

    final outPath = await _resolveOutput(options.out);
    final outFile = File(outPath);
    await outFile.parent.create(recursive: true);
    final videoOnly = path.join(
      outFile.parent.path,
      '${path.basenameWithoutExtension(outPath)}_video.mp4',
    );
    final effectiveDuration = options.seconds == null
        ? probe.duration
        : Duration(
            milliseconds: math.min(
              probe.duration.inMilliseconds,
              (options.seconds! * 1000).round(),
            ),
          );

    stdout.writeln('B2 Dart-relay bench');
    stdout.writeln('  clip   : ${options.clip}');
    stdout.writeln('  probe  : ${probe.width}x${probe.height} '
        '${probe.fps.toStringAsFixed(3)} fps '
        '${_formatDuration(probe.duration)}');
    stdout.writeln('  out    : $outPath');
    stdout.writeln('  sr eng : ${path.basename(srEngine)}');
    stdout.writeln('  sr cfg : workers=${options.srWorkers} '
        'batch=${options.srBatch}');
    if (options.seconds != null) {
      stdout.writeln('  limit  : ${options.seconds}s');
    }

    const factor = 2;
    final outWidth = probe.width * factor;
    final outHeight = probe.height * factor;
    final sourceFrameBytes = probe.width * probe.height * 3;
    final frameBytes = outWidth * outHeight * 3;

    final totalWatch = Stopwatch()..start();
    final chainWatch = Stopwatch()..start();

    final rife = await Process.start(
      python,
      [
        '-B', rifeHost,
        '--width', '${probe.width}',
        '--height', '${probe.height}',
        '--engine', rifeEngine,
        '--factor', '$factor',
        if (options.repeat) ...[
          '--repeat-protocol',
          '--repeat-threshold', '0.05',
          '--repeat-scene', '8',
          '--repeat-mad', '0.02',
        ],
        '--quiet',
      ],
      workingDirectory: trtRoot,
    );
    started.add(rife);
    final rifeErr = _StderrTail(rife);

    final decode = await Process.start(ffmpeg, [
      '-hide_banner', '-loglevel', 'error', '-nostdin',
      '-i', options.clip,
      '-map', '0:v:0',
      '-vf', 'format=rgb24',
      // 与 app 一致：保留源帧率，不补/不丢帧。
      '-fps_mode', 'passthrough',
      if (options.seconds != null) ...['-t', '${options.seconds}'],
      '-f', 'rawvideo',
      '-',
    ]);
    started.add(decode);
    final decodeErr = _StderrTail(decode);

    final sr = await Process.start(
      python,
      [
        '-B', srHost,
        '--width', '${probe.width}',
        '--height', '${probe.height}',
        '--engine', srEngine,
        '--workers', '${options.srWorkers}',
        '--batch', '${options.srBatch}',
        if (options.srLegacyIo) '--legacy-io',
        if (options.repeat) '--repeat-protocol',
        '--stats',
        '--quiet',
      ],
      workingDirectory: trtRoot,
    );
    started.add(sr);
    final srErr = _StderrTail(sr);

    final encode = await Process.start(ffmpeg, [
      '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
      '-f', 'rawvideo',
      '-pix_fmt', 'rgb24',
      '-s', '${outWidth}x$outHeight',
      '-framerate', (probe.fps * factor).toString(),
      '-i', '-',
      '-c:v', 'hevc_nvenc',
      '-preset', 'p5',
      '-rc', 'vbr',
      '-cq', '19',
      '-b:v', '0',
      '-pix_fmt', 'yuv420p',
      '-an',
      videoOnly,
    ]);
    started.add(encode);
    final encodeErr = _StderrTail(encode);
    final encodeStdoutSub = encode.stdout.listen((_) {});

    Duration? decodeAt;
    Duration? rifeAt;
    Duration? srAt;
    Duration? encodeAt;
    final decodeExit = decode.exitCode.then((code) {
      decodeAt = chainWatch.elapsed;
      return code;
    });
    final rifeExit = rife.exitCode.then((code) {
      rifeAt = chainWatch.elapsed;
      return code;
    });
    final srExit = sr.exitCode.then((code) {
      srAt = chainWatch.elapsed;
      return code;
    });
    final encodeExit = encode.exitCode.then((code) {
      encodeAt = chainWatch.elapsed;
      return code;
    });

    var decodedBytes = 0;
    var rifeBytes = 0;
    var srBytes = 0;
    final decodeDone = _countBytes(decode.stdout, (bytes) {
      decodedBytes += bytes;
    }).pipe(rife.stdin);
    final rifeDone = _countBytes(rife.stdout, (bytes) {
      rifeBytes += bytes;
    }).pipe(sr.stdin);
    final srDone = _countBytes(sr.stdout, (bytes) {
      srBytes += bytes;
    }).pipe(encode.stdin);

    Object? relayError;
    try {
      await Future.wait([decodeDone, rifeDone, srDone]);
    } catch (error) {
      relayError = error;
      for (final process in started) {
        process.kill();
      }
    }
    final encodeCode = await encodeExit;
    final decodeCode = await decodeExit;
    final rifeCode = await rifeExit;
    final srCode = await srExit;
    await _flushStderr([decodeErr, rifeErr, srErr, encodeErr]);
    final srStats = srErr.tail.trim();
    if (srStats.isNotEmpty) {
      stdout.writeln('sr stats   : $srStats');
    }
    chainWatch.stop();
    await encodeStdoutSub.cancel();

    final decodedFrames = decodedBytes ~/ sourceFrameBytes;
    final rifeOutFrames = rifeBytes ~/ sourceFrameBytes;
    final srOutFrames = srBytes ~/ frameBytes;
    final expectedRifeFrames = decodedFrames * factor - (factor - 1);

    stdout.writeln('per-stage finish (from chain start):');
    stdout.writeln('  decode : ${_seconds(decodeAt)} '
        '(exit $decodeCode, $decodedFrames decoded frames)');
    stdout.writeln('  rife   : ${_seconds(rifeAt)} '
        '(exit $rifeCode, $rifeOutFrames payload frames'
        '${options.repeat ? ' + repeat headers' : ''}, '
        'expanded expected $expectedRifeFrames)');
    stdout.writeln('  sr     : ${_seconds(srAt)} '
        '(exit $srCode, $srOutFrames expanded frames)');
    stdout.writeln('  encode : ${_seconds(encodeAt)} '
        '(exit $encodeCode, $srOutFrames frames in)');

    if (relayError != null) {
      throw Exception('管道转发失败: $relayError');
    }
    if (decodeCode != 0) {
      throw Exception(_withTail('解码器退出码 $decodeCode', decodeErr.tail));
    }
    if (rifeCode != 0) {
      throw Exception(_withTail('插帧宿主退出码 $rifeCode', rifeErr.tail));
    }
    if (srCode != 0) {
      throw Exception(_withTail('超分宿主退出码 $srCode', srErr.tail));
    }
    if (encodeCode != 0) {
      throw Exception(_withTail('编码器退出码 $encodeCode', encodeErr.tail));
    }

    final muxWatch = Stopwatch()..start();
    await _run(
      ffmpeg,
      [
        '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-i', videoOnly,
        '-i', options.clip,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c', 'copy',
        '-movflags', '+faststart',
        '-tag:v', 'hvc1',
        if (effectiveDuration > Duration.zero)
          ...['-t', _formatSeconds(effectiveDuration)],
        outPath,
      ],
    );
    muxWatch.stop();
    totalWatch.stop();

    stdout.writeln('chain wall : ${_seconds(chainWatch.elapsed)}');
    stdout.writeln('mux wall   : ${_seconds(muxWatch.elapsed)}');
    stdout.writeln('total wall : ${_seconds(totalWatch.elapsed)}');
    stdout.writeln('output     : $outPath '
        '(${(outFile.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB)');

    int? outputFrames;
    try {
      outputFrames = await _countFrames(ffmpeg, outPath);
    } catch (_) {
      // 输出校验失败不影响已经测得的计时结果。
    }
    stdout.writeln('mux check  : ${outputFrames ?? '?'} decoded video frames '
        '(verification decode, not in total)');
  } catch (error) {
    for (final process in started) {
      process.kill();
    }
    stderr.writeln('bench_b2 失败: $error');
    exitCode = 1;
  }
}

class _Options {
  String clip = _defaultClip;
  String srRoot = _defaultSrRoot;
  String? out;
  double? seconds;
  bool repeat = true;
  bool dynamicEngine = false;
  String? srEngine;
  int srBatch = 2;
  int srWorkers = 2;
  bool srLegacyIo = false;
}

_Options? _parseArgs(List<String> arguments) {
  final options = _Options();
  try {
    for (var i = 0; i < arguments.length; i++) {
      final argument = arguments[i];
      String nextValue() {
        if (i + 1 >= arguments.length) {
          throw FormatException('$argument 缺少参数值');
        }
        return arguments[++i];
      }

      if (argument == '--clip') {
        options.clip = nextValue();
      } else if (argument == '--sr-root') {
        options.srRoot = nextValue();
      } else if (argument == '--out') {
        options.out = nextValue();
      } else if (argument == '--seconds') {
        final raw = nextValue();
        final seconds = double.tryParse(raw);
        if (seconds == null || seconds <= 0) {
          throw FormatException('--seconds 需要正数，收到 $raw');
        }
        options.seconds = seconds;
      } else if (argument == '--no-repeat') {
        options.repeat = false;
      } else if (argument == '--dynamic-engine') {
        options.dynamicEngine = true;
      } else if (argument == '--sr-engine') {
        options.srEngine = nextValue();
      } else if (argument == '--sr-batch') {
        options.srBatch = _positiveInt(argument, nextValue());
      } else if (argument == '--sr-workers') {
        options.srWorkers = _positiveInt(argument, nextValue());
      } else if (argument == '--sr-legacy-io') {
        options.srLegacyIo = true;
      } else if (argument == '--help' || argument == '-h') {
        _printUsage();
        return null;
      } else {
        throw FormatException('未知参数: $argument');
      }
    }
  } on FormatException catch (error) {
    stderr.writeln(error.message);
    _printUsage();
    exitCode = 64;
    return null;
  }
  return options;
}

int _positiveInt(String flag, String raw) {
  final value = int.tryParse(raw);
  if (value == null || value < 1) {
    throw FormatException('$flag 需要正整数，收到 $raw');
  }
  return value;
}

void _printUsage() {
  stdout.writeln(
    '用法: dart tool/bench_b2.dart '
    '[--clip <视频>] [--sr-root <assets/sr>] [--out <输出.mp4>] '
    '[--seconds <秒>] [--no-repeat] [--dynamic-engine] [--sr-engine <path>] '
    '[--sr-batch <n>] [--sr-workers <n>] [--sr-legacy-io]',
  );
  stdout.writeln('默认 clip: $_defaultClip');
  stdout.writeln('默认启用 TRT repeat protocol；--no-repeat 回退旧路径');
  stdout.writeln('1080p 默认用静态引擎；--dynamic-engine 回退动态引擎做 A/B');
  stdout.writeln('默认 SR: --sr-batch 2 --sr-workers 2（AnimeJaNai 用 1/2）');
}

Future<String> _resolveOutput(String? requested) async {
  if (requested == null || requested.isEmpty) {
    final directory = await Directory.systemTemp.createTemp('bench_b2_');
    return path.join(directory.path, 'bench_b2.mp4');
  }
  if (await FileSystemEntity.isDirectory(requested)) {
    return path.join(requested, 'bench_b2.mp4');
  }
  return requested;
}

class _Probe {
  const _Probe({
    required this.width,
    required this.height,
    required this.fps,
    required this.duration,
  });

  final int width;
  final int height;
  final double fps;
  final Duration duration;
}

Future<_Probe> _probe(String ffmpeg, String input) async {
  final result = await Process.run(
    ffmpeg,
    ['-hide_banner', '-i', input],
    stdoutEncoding: utf8,
    stderrEncoding: utf8,
  );
  final text = '${result.stdout}\n${result.stderr}';
  final durationMatch =
      RegExp(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)').firstMatch(text);
  if (durationMatch == null) {
    throw Exception('无法解析视频时长: $input');
  }
  final duration = Duration(
    hours: int.parse(durationMatch.group(1)!),
    minutes: int.parse(durationMatch.group(2)!),
    milliseconds: (double.parse(durationMatch.group(3)!) * 1000).round(),
  );

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
  final normalizedRotation = (rotation?.abs() ?? 0) % 180;
  final swapDimensions = (normalizedRotation - 90).abs() < 1;
  final codedWidth = int.tryParse(sizeMatch?.group(1) ?? '') ?? 0;
  final codedHeight = int.tryParse(sizeMatch?.group(2) ?? '') ?? 0;
  final width = swapDimensions ? codedHeight : codedWidth;
  final height = swapDimensions ? codedWidth : codedHeight;
  if (width <= 0 || height <= 0) {
    throw Exception('无法解析视频分辨率: $input');
  }
  return _Probe(
    width: width,
    height: height,
    fps: double.tryParse(fpsMatch?.group(1) ?? '') ?? 24,
    duration: duration,
  );
}

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

String? _findSrEngine(String enginesDir, {bool preferStatic = true}) {
  final directory = Directory(enginesDir);
  if (!directory.existsSync()) return null;
  final engines = directory
      .listSync()
      .whereType<File>()
      .where((file) => file.path.endsWith('.engine'))
      .toList();
  if (preferStatic) {
    final staticEngine = File(path.join(
      enginesDir,
      'ani4k2_static1080_fp16.engine',
    ));
    if (staticEngine.existsSync()) return staticEngine.path;
  }
  for (final file in engines) {
    final base = path.basename(file.path);
    if (base.startsWith('ani4k2') && !base.contains('_static')) {
      return file.path;
    }
  }
  for (final file in engines) {
    final base = path.basename(file.path);
    if (!base.startsWith('rife_') &&
        !base.startsWith('sr_pre_') &&
        !base.startsWith('sr_post_') &&
        !base.contains('_static') &&
        base != 'rgb_u8cast.engine') {
      return file.path;
    }
  }
  return null;
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

Future<int> _run(String executable, List<String> args) async {
  final process = await Process.start(executable, args);
  final stdoutSub = process.stdout.listen((_) {});
  final stderrTail = _StderrTail(process);
  try {
    final exitCode = await process.exitCode;
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
  }
}

Future<int?> _countFrames(String ffmpeg, String input) async {
  final result = await Process.run(
    ffmpeg,
    [
      '-hide_banner', '-loglevel', 'info', '-nostdin',
      '-i', input,
      '-map', '0:v:0',
      '-f', 'null', '-',
    ],
    stdoutEncoding: utf8,
    stderrEncoding: utf8,
  );
  final matches = RegExp(r'frame=\s*(\d+)').allMatches('${result.stderr}');
  if (matches.isEmpty) return null;
  return int.tryParse(matches.last.group(1)!);
}

String _withTail(String message, String? tail) {
  if (tail == null || tail.isEmpty) return message;
  return '$message\n$tail';
}

String _formatSeconds(Duration duration) =>
    (duration.inMilliseconds / 1000).toStringAsFixed(3);

String _formatDuration(Duration duration) {
  final h = duration.inHours;
  final m = duration.inMinutes.remainder(60);
  final s = duration.inSeconds.remainder(60);
  return h > 0
      ? '$h:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}'
      : '$m:${s.toString().padLeft(2, '0')}';
}

String _seconds(Duration? duration) {
  if (duration == null) return '?';
  return '${(duration.inMilliseconds / 1000).toStringAsFixed(2)}s';
}

/// 持续读走子进程 stderr，只保留最后若干行用于报错。
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

  Future<void> get done => _done.future;

  Future<void> cancel() => _subscription.cancel();
}

Future<void> _flushStderr(Iterable<_StderrTail?> tails) async {
  for (final tail in tails.whereType<_StderrTail>()) {
    try {
      await tail.done.timeout(const Duration(milliseconds: 200));
    } catch (_) {
      // 超时则使用已经收到的行。
    }
  }
}
