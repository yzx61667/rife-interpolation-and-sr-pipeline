import 'dart:convert';
import 'dart:io';

import 'package:kazumi/services/sr/sr_presets.dart';
import 'package:path/path.dart' as path;

/// 定位随应用分发的超分引擎（ffmpeg / realcugan / realesrgan）。
///
/// 构建时由 windows/CMakeLists.txt 把这些二进制安装到可执行文件旁边的
/// `sr/` 目录，因此运行时不需要解压，也不占用用户的数据目录。
class SrEngineService {
  SrEngineService._();

  static final SrEngineService instance = SrEngineService._();

  /// 可执行文件所在目录（Release 构建即安装包根目录）。
  String get _exeDir => File(Platform.resolvedExecutable).parent.path;

  String get _srRoot => path.join(_exeDir, 'sr');

  String get ffmpegPath => path.join(_srRoot, 'ffmpeg', 'ffmpeg.exe');

  String get realCuganPath =>
      path.join(_srRoot, 'realcugan', 'realcugan-ncnn-vulkan.exe');

  String get realEsrganPath =>
      path.join(_srRoot, 'realesrgan', 'realesrgan-ncnn-vulkan.exe');

  String get realCuganModelsDir => path.join(_srRoot, 'realcugan');

  String get realEsrganModelsDir => path.join(_srRoot, 'realesrgan', 'models');

  // ---- 零落盘管道模式（Ani4K v2 ONNX + DirectML 便携运行时）----
  String get pipeRoot => path.join(_srRoot, 'pipe');
  String get pipePython => path.join(pipeRoot, 'runtime', 'python.exe');
  String get pipeHostScript => path.join(pipeRoot, 'sr_host.py');
  String get pipeModel => path.join(pipeRoot, 'ani4k2.onnx');

  // ---- TensorRT 自用链路（随包分发的精简运行时 + 绑定 GPU 的 engine）----
  String get trtRoot => path.join(_srRoot, 'trt');
  String get trtPython => path.join(trtRoot, 'runtime', 'python.exe');
  String get trtHostScript => path.join(trtRoot, 'trt_host.py');

  /// 超分引擎固定是 Ani4K；同目录还有 RIFE、GPU pre/post 和 uint8 Cast 引擎。
  ///
  /// 兼容旧目录名 engine/ 与 engines/，也兼容不带 ani4k2 前缀的旧 SR 引擎。
  /// 这里只返回**动态**引擎；1920x1080 的静态引擎由 [trtEngineFor] 精确选择，
  /// 不能因目录枚举顺序被当成通用引擎（它的 profile 只接受 1920x1080）。
  String? get trtEnginePath {
    for (final name in const ['engines', 'engine']) {
      final dir = Directory(path.join(trtRoot, name));
      if (!dir.existsSync()) continue;
      final candidates = dir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.engine'))
          .toList();
      for (final file in candidates) {
        final base = path.basename(file.path);
        if (base.startsWith('ani4k2') && !base.contains('_static')) {
          return file.path;
        }
      }
      for (final file in candidates) {
        final base = path.basename(file.path);
        if (!_isAuxiliaryTrtEngine(base) && !base.contains('_static')) {
          return file.path;
        }
      }
    }
    return null;
  }

  bool _isAuxiliaryTrtEngine(String base) =>
      base.startsWith('rife_') ||
      base.startsWith('sr_pre_') ||
      base.startsWith('sr_post_') ||
      base == 'rgb_u8cast.engine';

  /// 按 SR 输入尺寸选择引擎。
  ///
  /// 静态 Ani4K 的 profile 是 min = opt = max = 1920x1080，必须精确匹配；
  /// 其他尺寸（包括更小的 720p/1080p 变体）一律回退动态引擎。
  String? trtEngineFor(int width, int height) {
    if (width == 1920 && height == 1080) {
      for (final dirName in const ['engines', 'engine']) {
        final file = File(
          path.join(trtRoot, dirName, 'ani4k2_static1080_fp16.engine'),
        );
        if (file.existsSync()) return file.path;
      }
    }
    return trtEnginePath;
  }

  bool get hasTrtRuntime =>
      isWindows &&
      File(trtPython).existsSync() &&
      File(trtHostScript).existsSync() &&
      trtEnginePath != null;

  // ---- 插帧（RIFE 4.7 ONNX，复用管道运行时）----
  String get rifeHostScript => path.join(pipeRoot, 'rife_host.py');
  String get rifeModel => path.join(pipeRoot, 'rife47.onnx');

  bool get hasRifeRuntime =>
      hasPipeRuntime &&
      File(rifeHostScript).existsSync() &&
      File(rifeModel).existsSync();

  // ---- 插帧 TensorRT（1080p / 4K 静态引擎，复用 SR 的 TRT 运行时）----
  String get rifeTrtHostScript => path.join(trtRoot, 'rife_trt_host.py');

  /// 成品分辨率对应的 RIFE TRT 引擎；没有精确匹配时返回 null（回退 ONNX）。
  ///
  /// 引擎 profile 是静态的，尺寸不匹配不能硬跑。
  String? rifeTrtEngineFor(int width, int height) {
    final String name;
    if (width == 1920 && height == 1080) {
      name = 'rife_v4.7_1920x1080.engine';
    } else if (width == 3840 && height == 2160) {
      name = 'rife_v4.7_2160x3840.engine';
    } else {
      return null;
    }
    final file = File(path.join(trtRoot, 'engines', name));
    return file.existsSync() ? file.path : null;
  }

  bool get hasRifeTrtRuntime =>
      isWindows &&
      File(trtPython).existsSync() &&
      File(rifeTrtHostScript).existsSync() &&
      File(path.join(trtRoot, 'trt_engine.py')).existsSync();

  bool get hasPipeRuntime =>
      isWindows &&
      File(pipePython).existsSync() &&
      File(pipeHostScript).existsSync() &&
      File(pipeModel).existsSync();

  bool get isWindows => Platform.isWindows;

  /// 引擎是否就绪。缺失时 UI 给出明确提示，而不是等到任务中途才失败。
  bool get isAvailable {
    if (!isWindows) return false;
    return File(ffmpegPath).existsSync() &&
        File(realCuganPath).existsSync() &&
        File(realEsrganPath).existsSync();
  }

  /// 是否带 NVIDIA 硬件编码器（决定分发版默认编码器）。
  Future<bool> hasNvenc() async {
    try {
      final result = await Process.run(
        ffmpegPath,
        ['-hide_banner', '-encoders'],
        stdoutEncoding: utf8,
        stderrEncoding: utf8,
      );
      final text = '${result.stdout}\n${result.stderr}';
      return text.contains('hevc_nvenc') || text.contains('h264_nvenc');
    } catch (_) {
      return false;
    }
  }

  String enginePathFor(SrPreset preset) => switch (preset.backend) {
        SrBackend.realCugan => realCuganPath,
        SrBackend.realEsrgan => realEsrganPath,
      };

  String modelsDirFor(SrPreset preset) => switch (preset.backend) {
        SrBackend.realCugan => realCuganModelsDir,
        SrBackend.realEsrgan => realEsrganModelsDir,
      };

  /// 拼出一次超分调用的参数列表。
  ///
  /// 两个引擎都支持目录级批量处理，输出文件名与输入保持一致。
  ///
  /// 注意：ncnn 版引擎会把 `-m` 当作**相对可执行文件目录**的路径再拼接，
  /// 传绝对路径会被拼成坏路径，所以这里只用相对名。
  List<String> buildArgs({
    required SrPreset preset,
    required String inputDir,
    required String outputDir,
    required int scale,
    required int denoise,
  }) {
    switch (preset.backend) {
      case SrBackend.realCugan:
        return [
          '-i', inputDir,
          '-o', outputDir,
          '-n', '$denoise',
          '-s', '$scale',
          '-m', preset.modelName,
          // 实测（RTX 4060 Laptop，120 帧 360p）：-t 0 -j 2:2:2 约 13s，
          // -t 256 -j 4:4:4 约 7s。自动 tile 会选得很小，白白浪费显存。
          '-t', '256',
          '-j', '4:4:4',
          '-g', '0',
        ];
      case SrBackend.realEsrgan:
        return [
          '-i', inputDir,
          '-o', outputDir,
          '-n', preset.modelName,
          '-s', '$scale',
          '-m', 'models',
          // 实测（RTX 4060 Laptop，120 帧 360p）：-t 0 -j 2:2:2 约 13s，
          // -t 256 -j 4:4:4 约 7s。自动 tile 会选得很小，白白浪费显存。
          '-t', '256',
          '-j', '4:4:4',
          '-g', '0',
        ];
    }
  }

  /// 把 UI 选的降噪等级收敛到该模型/倍数真实存在的档位。
  int clampDenoise(SrPreset preset, int scale, int denoise) {
    final allowed = preset.denoiseFor(scale);
    if (allowed.isEmpty) return 0;
    if (allowed.contains(denoise)) return denoise;
    // 取最接近的可用档位，避免间接崩到不存在的模型文件。
    var best = allowed.first;
    for (final level in allowed) {
      if ((level - denoise).abs() < (best - denoise).abs()) best = level;
    }
    return best;
  }
}
