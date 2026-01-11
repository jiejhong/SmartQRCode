# SmartQRCode

一个以 CameraX + OpenCV + ZXing-C++（NDK）为核心的二维码扫描实验 App。目标是对“低对比度/模糊/缺失 quiet zone/畸变”的码提供更强的恢复能力，并把扫描 pipeline 的内部状态（normal/hard/moduleRecovery 等）以可读文本反馈在界面上，便于调参和定位失败原因。

## 编译方法

### 方式一：Android Studio（推荐）

1. 安装 Android Studio（建议 2024.x+），并确保已安装：
   - Android SDK Platform 34
   - CMake 3.22.1（在 SDK Manager 里安装）
   - NDK 27.0.12077973（在 SDK Manager 里安装）
2. 用 Android Studio 打开仓库根目录 `SmartQRCode/`。
3. 等待 Gradle Sync 完成后，选择 `app` 配置，运行到设备即可。

### 方式二：命令行 Gradle

确保 `JAVA_HOME` 指向 JDK 17（Android Studio 自带的 JBR 也可），并且 Android SDK/NDK/CMake 已安装好。

在仓库根目录执行：

```bash
./gradlew :app:assembleDebug
```

生成的 APK 位于：

`app/build/outputs/apk/debug/app-debug.apk`

可用 adb 安装：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 依赖的程序/组件

- Android Studio（用于开发/运行）
- JDK 17（AGP 8.5.2 + Kotlin 1.9.24）
- Android SDK Platform 34（compileSdk/targetSdk=34，minSdk=24）
- Android NDK 27.0.12077973（编译 native）
- CMake 3.22.1（externalNativeBuild）
- adb（可选，用于命令行安装、抓日志）

### 主要三方库

- CameraX
  - `androidx.camera:camera-camera2`
  - `androidx.camera:camera-lifecycle`
  - `androidx.camera:camera-view`
- OpenCV（Java 包）
  - `org.opencv:opencv:4.11.0`
- ZXing-C++（源码内置）
  - `app/src/main/cpp/third_party/zxing-cpp/`

## 目录结构

```text
SmartQRCode/
  app/
    build.gradle.kts               # Android 模块构建配置（SDK/NDK/CMake/依赖）
    src/main/
      AndroidManifest.xml
      java/com/smartqrcode/
        MainActivity.kt            # UI + CameraX + 调用 ScanPipeline
        ScanPipeline.kt            # 识别主流程（ROI、预处理、native/OpenCV、恢复）
        NativeDecoder.kt           # JNI 封装（decodeGray/parseResult）
        DebugCanvasView.kt         # 下半屏调试画布（显示中间帧/信息）
        RoiTracker 等              # ROI 策略、环形缓存、质量评估
      res/layout/
        activity_main.xml          # 上半屏预览 + 下半屏调试 + 文本 overlay
      cpp/
        CMakeLists.txt             # 编译 smartqrcode_native + 链接 ZXing
        native_decoder.cpp         # ZXing-C++ 解码 + padding/缩放/无效诊断输出
        third_party/zxing-cpp/     # ZXing-C++ 源码
```

## Pipeline 算法说明（当前实现）

代码入口：`ScanPipeline.process(ImageProxy)`（由 `MainActivity` 的 ImageAnalysis 回调驱动）。

### 1）帧输入与 ROI 策略

- 从 CameraX 的 YUV 帧中取 Y 平面作为灰度（luma）。
- ROI 由 `RoiTracker.nextRois()` 产出：包含“跟随上次成功/检测到 box 的 ROI”、周期性全图 ROI、以及若干中心/侧边窗口 ROI，提升在不同构图下的覆盖率。

### 2）native 解码（normal/hard）

- 对每个 ROI 调用 JNI：`NativeDecoder.decodeGray(...)`。
- native 侧使用 ZXing-C++（`ZXing::ReadBarcode`）做解码，并包含一系列针对 quiet zone 缺失/缩放不佳的补偿策略：
  - ROI crop、旋转校正
  - 检测白边（quiet zone）是否不足，按需在四边补白
  - 多个 pad 组合 + 多个 scale 组合尝试
  - 失败时在 `tryHarder=true` 下返回更“内容化”的无效诊断文本（`INVALID(...)`），用于 UI 展示与日志分析

相关实现：`app/src/main/cpp/native_decoder.cpp`、`app/src/main/java/com/smartqrcode/NativeDecoder.kt`。

### 3）OpenCV 辅助（可用时）

- 若 OpenCV 初始化成功（`OpenCVLoader.initDebug()`），pipeline 会在部分阶段使用 `QRCodeDetector.detectAndDecode(...)` 尝试快速解码或做 warmup（具体策略随状态/冷启动/近期是否有 box 而变化）。

### 4）moduleRecovery（强恢复路径）

当长时间无解码、用户点击强制、或处于探索/热区时，会进入更重的恢复分支：

1. 把 ROI resize 到 512×512，并做阈值化得到 `bin`。
2. `estimateModuleCount(bin)` 通过多行统计跳变数估计二维码模块数，并对齐到标准 QR 尺寸（21 + 4k）。
3. 针对候选模块数列表（candidateN±若干步进），调用 `recoverByBeam(...)`：
   - 在相位/缩放/阈值带宽组合上搜索一个“unknown 最少、边界 margin 最大”的采样参数
   - 结构提示（finder/alignment/timing 等）把部分 unknown 固化
   - 对仍不确定的点做 beam search（按概率代价选择翻转组合），并把生成的 512×512 合成图再喂回 native 解码（含 tryHarder）

相关实现：`ScanPipeline.kt` 中的 `moduleRecoveryTry(...)` 与 `recoverByBeam(...)`。

## 用户在 App 内看到的界面与流程

布局：上半屏是相机预览（PreviewView），下半屏是调试画布（DebugCanvasView），并叠加两条文本：

- `flowText`：展示 pipeline 的实时状态（例如 `normal/hard/hard(force)`，当前 stage、旋转角、ROI 数量、无解码时长等）。
- `resultText`：展示最终解码结果；若返回为无效诊断，会显示 `INVALID(...)` 形式的内容片段/错误信息，帮助判断“确实扫到码但校验失败/格式不匹配”等情况。

交互：

- 点击预览区域或下半屏调试区域：触发一次“强制 hard”窗口（短时间内更激进的 tryHarder / moduleRecovery）。

## 日志与调试

建议抓日志只看本应用 tag：

```bash
adb logcat -v time SmartQRCode:I *:S
```

常见关键词：

- `FlowPlan/FlowTryHarder`：当前是否计划走 hard
- `FlowModuleEstimate/FlowModulePlan`：模块数估计与候选列表
- `FlowBeam*`：beam 搜索阶段（Phase/Trim/Start/Decode/Hit/Miss/Stop）
- `NativeDbg`：native 解码路径的参数摘要

## 许可与说明

工程内包含 ZXing-C++ 源码（见 `third_party/zxing-cpp` 目录及其 LICENSE）。其它依赖遵循各自上游许可证。

