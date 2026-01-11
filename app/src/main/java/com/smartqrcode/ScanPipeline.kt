package com.smartqrcode

import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import org.opencv.core.CvType
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.QRCodeDetector
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class ScanPipeline {
    private val roiTracker = RoiTracker()
    private var frameIndex: Long = 0
    private val ringBuffer = RoiRingBuffer(16)
    private var qrDetector: QRCodeDetector? = null
    var debugCallback: ((DebugFrame) -> Unit)? = null
    var statusCallback: ((String) -> Unit)? = null
    private var lastBoxTsNs: Long = 0
    private var lastDecodedTsNs: Long = 0
    private var lastTryHarderTsNs: Long = 0
    private var lastOpenCvTsNs: Long = 0
    private var openCvWarmupUntilNs: Long = 0
    private var openCvWarmupBoxSig: Long = 0
    private var openCvWarmupCollected: Int = 0
    private var firstAnalyzeTsNs: Long = 0
    @Volatile private var pendingUserHardReset: Boolean = false
    @Volatile private var pendingUserWarmup: Boolean = false
    @Volatile private var forceTryHarderFrames: Int = 0
    private var lastModuleRecoveryNs: Long = 0
    private var lastAutoHardNs: Long = 0
    private var lastAutoModuleAttemptNs: Long = 0
    private var lastStatusEmitNs: Long = 0
    private var lastStatusText: String? = null

    private fun isInvalidDiagnosticText(text: String): Boolean {
        return text.startsWith("INVALID(")
    }
    
    private fun emitStatus(text: String) {
        val cb = statusCallback ?: return
        val nowNs = android.os.SystemClock.elapsedRealtimeNanos()
        val changed = text != lastStatusText
        if (!changed && (nowNs - lastStatusEmitNs) < 250_000_000L) return
        lastStatusText = text
        lastStatusEmitNs = nowNs
        cb(text)
    }
    
    fun debugRecoverFromGray512(gray512: Mat, tag: String): String? {
        return moduleRecoveryTry(gray512, debugEnabled = true, tag = tag, bypassInterval = true)
    }

    fun reset() {
        frameIndex = 0
        roiTracker.reset()
        ringBuffer.clear()
        qrDetector = null
        lastBoxTsNs = 0
        lastDecodedTsNs = 0
        lastTryHarderTsNs = 0
        lastOpenCvTsNs = 0
        openCvWarmupUntilNs = 0
        openCvWarmupBoxSig = 0
        openCvWarmupCollected = 0
        firstAnalyzeTsNs = 0
        pendingUserHardReset = false
        pendingUserWarmup = false
        forceTryHarderFrames = 0
        lastModuleRecoveryNs = 0
        lastAutoHardNs = 0
        lastAutoModuleAttemptNs = 0
        lastStatusEmitNs = 0
        lastStatusText = null
    }

    fun requestUserHard(frames: Int = 90) {
        forceTryHarderFrames = max(forceTryHarderFrames, frames)
        pendingUserHardReset = true
        pendingUserWarmup = true
    }

    @ExperimentalGetImage
    fun process(imageProxy: ImageProxy): ScanOutput? {
        imageProxy.use { proxy ->
            val t0 = android.os.SystemClock.elapsedRealtimeNanos()
            val image = proxy.image ?: return null
            val width = image.width
            val height = image.height
            val rotation = proxy.imageInfo.rotationDegrees
            val timestampNs = android.os.SystemClock.elapsedRealtimeNanos()
            val stableCropRect = android.graphics.Rect(proxy.cropRect)
            if (pendingUserHardReset) {
                pendingUserHardReset = false
                frameIndex = 0
                roiTracker.reset()
                ringBuffer.clear()
                qrDetector = null
                lastBoxTsNs = 0
                lastDecodedTsNs = 0
                lastTryHarderTsNs = 0
                lastOpenCvTsNs = 0
                openCvWarmupUntilNs = 0
                openCvWarmupBoxSig = 0
                openCvWarmupCollected = 0
                firstAnalyzeTsNs = 0
                lastAutoHardNs = 0
                lastAutoModuleAttemptNs = 0
            }
            if (firstAnalyzeTsNs == 0L) firstAnalyzeTsNs = timestampNs

            val noDecodeForNs = if (lastDecodedTsNs != 0L) {
                timestampNs - lastDecodedTsNs
            } else {
                timestampNs - firstAnalyzeTsNs
            }
            val autoHardWindowNs = 10_000_000_000L
            val autoModuleWindowNs = 2_500_000_000L
            val autoHardDue = noDecodeForNs >= autoHardWindowNs && (timestampNs - lastAutoHardNs) >= autoHardWindowNs
            if (autoHardDue) {
                lastAutoHardNs = timestampNs
                forceTryHarderFrames = max(forceTryHarderFrames, 180)
                pendingUserWarmup = true
                android.util.Log.i("SmartQRCode", "FlowAutoHard reason=noDecode10s frames=$forceTryHarderFrames")
            }

            val yPlane = image.planes[0]
            val pixelStride = yPlane.pixelStride
            val rowStride = yPlane.rowStride

            val yBuffer = yPlane.buffer
            val yBytes = ByteArray(rowStride * height)
            yBuffer.rewind()
            yBuffer.get(yBytes)

            val lum = if (pixelStride == 1 && rowStride == width) {
                yBytes
            } else {
                val compact = ByteArray(width * height)
                var dst = 0
                var srcRow = 0
                for (r in 0 until height) {
                    var src = srcRow
                    for (c in 0 until width) {
                        compact[dst++] = yBytes[src]
                        src += pixelStride
                    }
                    srcRow += rowStride
                }
                compact
            }

            val rois = roiTracker.nextRois(width, height, frameIndex, timestampNs)
            val hot = lastBoxTsNs != 0L && (timestampNs - lastBoxTsNs) <= 350_000_000L
            val recentlyDecoded = lastDecodedTsNs != 0L && (timestampNs - lastDecodedTsNs) <= 700_000_000L
            val startupBoost = !recentlyDecoded && (timestampNs - firstAnalyzeTsNs) <= 1_200_000_000L
            val forceHarder = forceTryHarderFrames > 0
            if (forceHarder) forceTryHarderFrames = (forceTryHarderFrames - 1).coerceAtLeast(0)
            val explorePeriod = if (startupBoost) 15L else 45L
            val explore = !recentlyDecoded && (frameIndex % explorePeriod == 0L)
            var tryHarderPlanned = forceHarder || hot || explore || startupBoost || (noDecodeForNs >= autoHardWindowNs)
            val debugEnabled = debugCallback != null &&
                ((frameIndex % 3L == 0L) || forceHarder || noDecodeForNs >= 1_500_000_000L)
            val nativeDebugLog = (frameIndex % 15L == 0L)
            val mode = when {
                forceHarder -> "hard(force)"
                tryHarderPlanned -> "hard"
                else -> "normal"
            }
            emitStatus("$mode stage=native0 rot=$rotation rois=${rois.size} noDecodeMs=${noDecodeForNs / 1_000_000L}")
            if (nativeDebugLog) {
                android.util.Log.i(
                    "SmartQRCode",
                    "FlowPlan rot=$rotation planned=$tryHarderPlanned hot=$hot explore=$explore recent=$recentlyDecoded opencv=${OpenCvSupport.ready} rois=${rois.size} noDecodeMs=${noDecodeForNs / 1_000_000L} autoDue=$autoHardDue"
                )
                if (forceHarder || startupBoost) {
                    android.util.Log.i(
                        "SmartQRCode",
                        "FlowPlan2 force=$forceHarder startup=$startupBoost explorePeriod=$explorePeriod"
                    )
                }
            }
            if (debugEnabled) {
                emitLumaPreview("CamGray ${width}x$height rot=$rotation", lum, width, height, rotation)
            }
            frameIndex++

            var bestBox: RoiRect? = null
            var bestQuad: FloatArray? = null
            var bestInvalid: ScanOutput? = null
            var tNative0Ns: Long
            var tNative1Ns: Long
            var tOpenCvNs: Long
            nativeDecodePass(lum, width, height, rotation, rois, timestampNs, stableCropRect, tryHarder = false, debugLog = nativeDebugLog)?.let { out ->
                bestBox = out.box ?: bestBox
                bestQuad = out.quad ?: bestQuad
                if (nativeDebugLog) {
                    if (out.text != null) {
                        android.util.Log.i("SmartQRCode", "FlowHit native th=false text=true box=${out.box != null}")
                    } else {
                        android.util.Log.i("SmartQRCode", "FlowHint native th=false text=false box=${out.box != null}")
                    }
                }
                    if (out.text != null) {
                        if (isInvalidDiagnosticText(out.text)) {
                            bestInvalid = out
                        } else {
                            val t = out.text
                            val head = t.replace('\n', ' ').replace('\r', ' ').take(96)
                            android.util.Log.i("SmartQRCode", "FlowDecoded src=native0 len=${t.length} head=$head")
                            lastDecodedTsNs = timestampNs
                            emitStatus("$mode stage=decoded(native0)")
                            return out
                        }
                    }
            }
            tNative0Ns = android.os.SystemClock.elapsedRealtimeNanos()
            if (bestBox != null) tryHarderPlanned = true
            val boxSig = bestBox?.let { b ->
                (b.left.toLong() shl 48) xor (b.top.toLong() shl 32) xor (b.right.toLong() shl 16) xor b.bottom.toLong()
            } ?: 0L
            if (boxSig != 0L && boxSig != openCvWarmupBoxSig) {
                openCvWarmupBoxSig = boxSig
                openCvWarmupUntilNs = timestampNs + 320_000_000L
                openCvWarmupCollected = 0
                android.util.Log.i("SmartQRCode", "FlowWarmup reset reason=boxSigChange force=$forceHarder")
            }
            val tryHarderNow = tryHarderPlanned && (timestampNs - lastTryHarderTsNs) >= 180_000_000L
            if (nativeDebugLog) {
                val cdMs = ((180_000_000L - (timestampNs - lastTryHarderTsNs)).coerceAtLeast(0L) / 1_000_000L)
                android.util.Log.i("SmartQRCode", "FlowTryHarder planned=$tryHarderPlanned now=$tryHarderNow cdMs=$cdMs box=${bestBox != null}")
            }
            if (tryHarderNow) {
                lastTryHarderTsNs = timestampNs
                emitStatus("$mode stage=native1 rot=$rotation box=${bestBox != null}")
                nativeDecodePass(lum, width, height, rotation, rois, timestampNs, stableCropRect, tryHarder = true, debugLog = nativeDebugLog)?.let { out ->
                    bestBox = out.box ?: bestBox
                    bestQuad = out.quad ?: bestQuad
                    if (nativeDebugLog) {
                        if (out.text != null) {
                            android.util.Log.i("SmartQRCode", "FlowHit native th=true text=true box=${out.box != null}")
                        } else {
                            android.util.Log.i("SmartQRCode", "FlowHint native th=true text=false box=${out.box != null}")
                        }
                    }
                    if (out.text != null) {
                        if (isInvalidDiagnosticText(out.text)) {
                            bestInvalid = out
                        } else {
                            val t = out.text
                            val head = t.replace('\n', ' ').replace('\r', ' ').take(96)
                            android.util.Log.i("SmartQRCode", "FlowDecoded src=native1 len=${t.length} head=$head")
                            lastDecodedTsNs = timestampNs
                            emitStatus("$mode stage=decoded(native1)")
                            return out
                        }
                    }
                }
            }
            tNative1Ns = android.os.SystemClock.elapsedRealtimeNanos()
            if (bestQuad == null && bestBox != null) {
                bestQuad = floatArrayOf(
                    bestBox!!.left.toFloat(), bestBox!!.top.toFloat(),
                    bestBox!!.right.toFloat(), bestBox!!.top.toFloat(),
                    bestBox!!.right.toFloat(), bestBox!!.bottom.toFloat(),
                    bestBox!!.left.toFloat(), bestBox!!.bottom.toFloat()
                )
            }

            if (noDecodeForNs >= autoModuleWindowNs && (timestampNs - lastAutoModuleAttemptNs) >= 600_000_000L) {
                lastAutoModuleAttemptNs = timestampNs
                emitStatus("$mode stage=moduleRecovery(auto) rot=$rotation")
                if (nativeDebugLog) {
                    android.util.Log.i("SmartQRCode", "FlowAutoModule noDecodeMs=${noDecodeForNs / 1_000_000L}")
                }
                try {
                    val full = Mat(height, width, CvType.CV_8UC1)
                    try {
                        full.put(0, 0, lum)
                        val roi = bestBox?.let { b ->
                            val padX = (b.width * 0.30f).toInt().coerceAtLeast(16)
                            val padY = (b.height * 0.30f).toInt().coerceAtLeast(16)
                            RoiRect(
                                (b.left - padX).coerceAtLeast(0),
                                (b.top - padY).coerceAtLeast(0),
                                (b.right + padX).coerceAtMost(width),
                                (b.bottom + padY).coerceAtMost(height)
                            )
                        } ?: run {
                            val side = (min(width, height) * 0.78f).roundToInt().coerceAtLeast(240)
                            val cx = width / 2
                            val cy = height / 2
                            RoiRect(
                                (cx - side / 2).coerceAtLeast(0),
                                (cy - side / 2).coerceAtLeast(0),
                                (cx + side / 2).coerceAtMost(width),
                                (cy + side / 2).coerceAtMost(height)
                            )
                        }
                        val roiRect = Rect(roi.left, roi.top, roi.width, roi.height)
                        val roiMat = Mat(full, roiRect)
                        val patch = Mat()
                        try {
                            val interp = if (roiMat.cols() < 512 || roiMat.rows() < 512) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA
                            Imgproc.resize(roiMat, patch, Size(512.0, 512.0), 0.0, 0.0, interp)
                            if (debugEnabled) emitMat("Auto10s-512", patch)
                            val recovered = moduleRecoveryTry(patch, debugEnabled, "Auto10s", bypassInterval = true)
                            if (recovered != null) {
                                val head = recovered.replace('\n', ' ').replace('\r', ' ').take(96)
                                android.util.Log.i("SmartQRCode", "FlowDecoded src=moduleRecovery:auto len=${recovered.length} head=$head")
                                lastDecodedTsNs = timestampNs
                                roiTracker.onDecoded(roi, timestampNs)
                                emitStatus("$mode stage=decoded(moduleRecovery:auto)")
                                return ScanOutput(recovered, bestBox ?: roi, bestQuad, width, height, rotation, stableCropRect)
                            }
                        } finally {
                            patch.release()
                            roiMat.release()
                        }
                    } finally {
                        full.release()
                    }
                } catch (_: Throwable) {
                }
            }
            val openCvNow = OpenCvSupport.ready &&
                tryHarderPlanned &&
                (timestampNs - lastOpenCvTsNs) >= 260_000_000L
            if (!openCvNow) {
                if (nativeDebugLog) android.util.Log.i(
                    "SmartQRCode",
                    "FlowStop opencvSkip=${!OpenCvSupport.ready} tryHarderSkip=${!tryHarderPlanned} cooldownSkip=${(timestampNs - lastOpenCvTsNs) < 260_000_000L}"
                )
                return bestBox?.let { ScanOutput(null, it, bestQuad, width, height, rotation, stableCropRect) }
            }
            if (!forceHarder && openCvWarmupUntilNs == 0L && (timestampNs - lastOpenCvTsNs) >= 900_000_000L) {
                openCvWarmupUntilNs = timestampNs + 320_000_000L
                openCvWarmupCollected = 0
                openCvWarmupBoxSig = if (boxSig != 0L) boxSig else 1L
            }
            if (pendingUserWarmup && openCvWarmupUntilNs == 0L) {
                pendingUserWarmup = false
                openCvWarmupUntilNs = timestampNs + 320_000_000L
                openCvWarmupCollected = 0
                openCvWarmupBoxSig = if (boxSig != 0L) boxSig else 1L
                android.util.Log.i("SmartQRCode", "FlowWarmup start reason=userTap")
            }
            val warmupActive = (timestampNs < openCvWarmupUntilNs) && (openCvWarmupCollected < 4)
            if (warmupActive) {
                emitStatus("$mode stage=opencvWarmup collected=$openCvWarmupCollected")
                try {
                    val full = Mat(height, width, CvType.CV_8UC1)
                    full.put(0, 0, lum)
                    val candidates = ArrayList<RoiRect>(3)
                    rois.firstOrNull()?.let { candidates.add(it) }
                    bestBox?.let { b ->
                        val padX = (b.width * 0.22f).toInt().coerceAtLeast(12)
                        val padY = (b.height * 0.22f).toInt().coerceAtLeast(12)
                        val r = RoiRect(
                            (b.left - padX).coerceAtLeast(0),
                            (b.top - padY).coerceAtLeast(0),
                            (b.right + padX).coerceAtMost(width),
                            (b.bottom + padY).coerceAtMost(height)
                        )
                        candidates.add(r)
                    }
                    for (roi in candidates.distinct().take(2)) {
                        val roiRect = Rect(roi.left, roi.top, roi.width, roi.height)
                        val roiMat = Mat(full, roiRect)
                        val patch = Mat()
                        try {
                            val interp = if (roiMat.cols() < 512 || roiMat.rows() < 512) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA
                            Imgproc.resize(roiMat, patch, Size(512.0, 512.0), 0.0, 0.0, interp)
                            val quality = RoiQuality.score(patch)
                            ringBuffer.push(RoiFrame(patch.clone(), timestampNs, roi, quality))
                            openCvWarmupCollected++
                            if (forceHarder) {
                                val recovered = moduleRecoveryTry(patch, debugEnabled, "Warmup")
                                if (recovered != null) {
                                    lastDecodedTsNs = timestampNs
                                    roiTracker.onDecoded(roi, timestampNs)
                                    emitStatus("$mode stage=decoded(moduleRecovery:warmup)")
                                    return ScanOutput(recovered, roi, bestQuad, width, height, rotation, stableCropRect)
                                }
                            }
                        } finally {
                            patch.release()
                            roiMat.release()
                        }
                    }
                    full.release()
                } catch (_: Throwable) {
                }
                if (nativeDebugLog) {
                    val best = ringBuffer.topK(1).firstOrNull()?.quality
                    android.util.Log.i(
                        "SmartQRCode",
                        "FlowWarmup opencv collected=$openCvWarmupCollected bestSharp=${best?.sharpness} bestOver=${best?.overexposedRatio} bestContrast=${best?.contrast}"
                    )
                }
                return bestBox?.let { ScanOutput(null, it, bestQuad, width, height, rotation, stableCropRect) }
            }

            tOpenCvNs = android.os.SystemClock.elapsedRealtimeNanos()
            lastOpenCvTsNs = timestampNs
            emitStatus("$mode stage=opencv")
            if (nativeDebugLog || forceHarder) android.util.Log.i(
                "SmartQRCode",
                "FlowEnter opencv force=$forceHarder collected=$openCvWarmupCollected"
            )

            try {
                val full = Mat(height, width, CvType.CV_8UC1)
                try {
                    full.put(0, 0, lum)
                    val bestFrame = ringBuffer.topK(1).firstOrNull()
                    if (bestFrame != null) {
                        val out = decodeFromBranches(bestFrame.gray, bestFrame.roi, tryHarder = true, debugEnabled = debugEnabled)
                        if (out != null) {
                            bestBox = out.box ?: bestBox
                            bestQuad = out.quad ?: bestQuad
                            if (out.text != null) {
                                if (isInvalidDiagnosticText(out.text)) {
                                    bestInvalid = out
                                } else {
                                    lastDecodedTsNs = timestampNs
                                    roiTracker.onDecoded(bestFrame.roi, timestampNs)
                                    return out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                }
                            } else {
                                roiTracker.onBox(bestFrame.roi, timestampNs)
                                return out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                            }
                        }
                    }
                    if (bestQuad != null) {
                        val warped = warpFullByQuad(full, bestQuad!!)
                        if (warped != null) {
                            try {
                                if (debugEnabled) emitMat("WarpFull-512", warped)
                                val bytes = ByteArray(512 * 512)
                                warped.get(0, 0, bytes)
                                val raw = NativeDecoder.decodeGray(bytes, 512, 512, 0, 0, 0, 512, 512, true)
                                val parsed = raw?.let { NativeDecoder.parseResult(it) }
                                if (parsed != null && parsed.text.isNotBlank()) {
                                    if (isInvalidDiagnosticText(parsed.text)) {
                                        bestInvalid = ScanOutput(parsed.text, bestBox, bestQuad, width, height, rotation, stableCropRect)
                                    } else {
                                        lastDecodedTsNs = timestampNs
                                        val box = bestBox
                                        if (box != null) roiTracker.onDecoded(box, timestampNs)
                                        return ScanOutput(parsed.text, box, bestQuad, width, height, rotation, stableCropRect)
                                    }
                                }
                                val recovered = moduleRecoveryTry(warped, debugEnabled, "FullQuad")
                                if (recovered != null) {
                                    lastDecodedTsNs = timestampNs
                                    val box = bestBox
                                    if (box != null) roiTracker.onDecoded(box, timestampNs)
                                    return ScanOutput(recovered, box, bestQuad, width, height, rotation, stableCropRect)
                                }
                            } finally {
                                warped.release()
                            }
                        }
                    }
                    var bestRoi: RoiRect? = null
                    var bestBoxInCv: RoiRect? = null
                    for (roi in rois) {
                        val roiRect = Rect(roi.left, roi.top, roi.width, roi.height)
                        val roiMat = Mat(full, roiRect)
                        val patch = Mat()
                        try {
                            val interp = if (roiMat.cols() < 512 || roiMat.rows() < 512) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA
                            Imgproc.resize(roiMat, patch, Size(512.0, 512.0), 0.0, 0.0, interp)
                            val out = decodeFromBranches(patch, roi, tryHarder = false, debugEnabled = debugEnabled)
                            if (out != null) {
                                bestBoxInCv = out.box ?: bestBoxInCv
                                if (out.box != null) lastBoxTsNs = timestampNs
                                if (out.text != null) {
                                    if (isInvalidDiagnosticText(out.text)) {
                                        bestInvalid = out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                    } else {
                                        roiTracker.onDecoded(roi, timestampNs)
                                        return out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                    }
                                }
                            }
                            if (bestRoi == null) {
                                val quality = RoiQuality.score(patch)
                                ringBuffer.push(RoiFrame(patch.clone(), timestampNs, roi, quality))
                                bestRoi = roi
                            }
                        } finally {
                            patch.release()
                            roiMat.release()
                        }
                    }

                    if (tryHarderPlanned) {
                        for (roi in rois) {
                            val roiRect = Rect(roi.left, roi.top, roi.width, roi.height)
                            val roiMat = Mat(full, roiRect)
                            val patch = Mat()
                            try {
                                val interp = if (roiMat.cols() < 512 || roiMat.rows() < 512) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA
                                Imgproc.resize(roiMat, patch, Size(512.0, 512.0), 0.0, 0.0, interp)
                                val out = decodeFromBranches(patch, roi, tryHarder = true, debugEnabled = debugEnabled)
                                if (out != null) {
                                    bestBoxInCv = out.box ?: bestBoxInCv
                                    if (out.box != null) lastBoxTsNs = timestampNs
                                    if (out.text != null) {
                                        if (isInvalidDiagnosticText(out.text)) {
                                            bestInvalid = out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                        } else {
                                            roiTracker.onDecoded(roi, timestampNs)
                                            return out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                        }
                                    }
                                }
                                if (bestRoi == null) {
                                    val quality = RoiQuality.score(patch)
                                    ringBuffer.push(RoiFrame(patch.clone(), timestampNs, roi, quality))
                                    bestRoi = roi
                                }
                            } finally {
                                patch.release()
                                roiMat.release()
                            }
                        }
                    }

                    val topK = ringBuffer.topK(6)
                    val rectified = topK.mapNotNull { frame ->
                        rectifyTo512(frame.gray)
                    }

                    val primaryRoi = bestRoi ?: rois.firstOrNull()
                    if (rectified.size >= 2 && primaryRoi != null) {
                        val fused = fuseByBlocks(rectified, blockSize = 32)
                        if (debugEnabled) emitMat("Fuse-512", fused)
                        val out = decodeFromBranches(fused, primaryRoi, tryHarder = true, debugEnabled = debugEnabled)
                        fused.release()
                        rectified.forEach { it.release() }
                        if (out != null) {
                            if (out.text != null) {
                                if (isInvalidDiagnosticText(out.text)) {
                                    bestInvalid = out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                } else {
                                    roiTracker.onDecoded(primaryRoi, timestampNs)
                                    return out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                                }
                            } else {
                                roiTracker.onBox(primaryRoi, timestampNs)
                                return out.copy(frameWidth = width, frameHeight = height, rotationDegrees = rotation, cropRect = stableCropRect)
                            }
                        }
                        val recovered = moduleRecoveryTry(fused, debugEnabled, "Fuse")
                        if (recovered != null) {
                            roiTracker.onDecoded(primaryRoi, timestampNs)
                            lastDecodedTsNs = timestampNs
                            return ScanOutput(recovered, bestBoxInCv ?: bestBox, bestQuad, width, height, rotation, stableCropRect)
                        }
                    } else {
                        rectified.forEach { it.release() }
                    }

                    val box = bestBoxInCv ?: bestBox
                    if (box != null) {
                        roiTracker.onBox(box, timestampNs)
                        lastBoxTsNs = timestampNs
                        return ScanOutput(null, box, bestQuad, width, height, rotation, stableCropRect)
                    }
                    return bestInvalid
                } finally {
                    full.release()
                }
            } catch (e: UnsatisfiedLinkError) {
                OpenCvSupport.ready = false
                nativeDecodePass(lum, width, height, rotation, rois, timestampNs, stableCropRect, tryHarder = false, debugLog = nativeDebugLog)?.let { return it }
                if (tryHarderPlanned) {
                    nativeDecodePass(lum, width, height, rotation, rois, timestampNs, stableCropRect, tryHarder = true, debugLog = nativeDebugLog)?.let { return it }
                }
                return bestInvalid
            } finally {
                val tEnd = android.os.SystemClock.elapsedRealtimeNanos()
                val totalMs = (tEnd - t0) / 1_000_000.0
                if (totalMs >= 120.0) {
                    val native0Ms = if (tNative0Ns != 0L) (tNative0Ns - t0) / 1_000_000.0 else -1.0
                    val native1Ms = if (tNative1Ns != 0L) (tNative1Ns - tNative0Ns) / 1_000_000.0 else -1.0
                    val cvMs = if (tOpenCvNs != 0L) (tOpenCvNs - tNative1Ns) / 1_000_000.0 else -1.0
                    android.util.Log.i("SmartQRCode", "FlowTime total=%.1f native0=%.1f native1=%.1f preCv=%.1f".format(totalMs, native0Ms, native1Ms, cvMs))
                }
            }
        }
    }

    private fun emitLumaPreview(step: String, lum: ByteArray, width: Int, height: Int, rotation: Int) {
        val cb = debugCallback ?: return
        val rot = ((rotation % 360) + 360) % 360
        val rotW = if (rot == 90 || rot == 270) height else width
        val rotH = if (rot == 90 || rot == 270) width else height
        val maxW = 720
        val maxH = 720
        val scale = maxOf(rotW.toDouble() / maxW.toDouble(), rotH.toDouble() / maxH.toDouble(), 1.0)
        val outW = (rotW.toDouble() / scale).roundToInt().coerceAtLeast(1)
        val outH = (rotH.toDouble() / scale).roundToInt().coerceAtLeast(1)
        val out = ByteArray(outW * outH)
        for (yOut in 0 until outH) {
            val yRot = (yOut.toDouble() * rotH.toDouble() / outH.toDouble()).toInt().coerceIn(0, rotH - 1)
            val row = yOut * outW
            for (xOut in 0 until outW) {
                val xRot = (xOut.toDouble() * rotW.toDouble() / outW.toDouble()).toInt().coerceIn(0, rotW - 1)
                val (x, y) = when (rot) {
                    90 -> Pair(yRot, (height - 1 - xRot).coerceIn(0, height - 1))
                    180 -> Pair((width - 1 - xRot).coerceIn(0, width - 1), (height - 1 - yRot).coerceIn(0, height - 1))
                    270 -> Pair((width - 1 - yRot).coerceIn(0, width - 1), xRot)
                    else -> Pair(xRot, yRot)
                }
                out[row + xOut] = lum[y * width + x]
            }
        }
        cb(DebugFrame(step, outW, outH, out))
    }

    private fun nativeDecodePass(
        lum: ByteArray,
        width: Int,
        height: Int,
        rotation: Int,
        rois: List<RoiRect>,
        timestampNs: Long,
        cropRect: android.graphics.Rect?,
        tryHarder: Boolean,
        debugLog: Boolean
    ): ScanOutput? {
        val first = rois.firstOrNull()
        for (roi in rois) {
            if (tryHarder) {
                val fullArea = width.toLong() * height.toLong()
                val roiArea = roi.width.toLong() * roi.height.toLong()
                if (roiArea >= (fullArea * 92L) / 100L) continue
            }
            if (debugLog && first === roi) {
                val dbg = NativeDecoder.decodeGrayDebug(
                    lum,
                    width,
                    height,
                    rotation,
                    roi.left,
                    roi.top,
                    roi.right,
                    roi.bottom,
                    tryHarder
                ) ?: "dbg=null"
                android.util.Log.i("SmartQRCode", "NativeDbg rot=$rotation th=$tryHarder roi=${roi.left},${roi.top},${roi.right},${roi.bottom} $dbg")
            }
            val raw = NativeDecoder.decodeGray(
                lum,
                width,
                height,
                rotation,
                roi.left,
                roi.top,
                roi.right,
                roi.bottom,
                tryHarder
            )
            if (raw != null) {
                val parsed = NativeDecoder.parseResult(raw) ?: continue
                val hasText = parsed.text.isNotBlank()
                val isInvalid = hasText && isInvalidDiagnosticText(parsed.text)
                if (hasText && !isInvalid) {
                    roiTracker.onDecoded(parsed.roi, timestampNs)
                } else {
                    roiTracker.onBox(parsed.roi, timestampNs)
                    lastBoxTsNs = timestampNs
                }
                return ScanOutput(parsed.text.takeIf { hasText }, parsed.roi, parsed.quad, width, height, rotation, cropRect)
            }
        }
        return null
    }

    private fun decodeFromBranches(src512: Mat, roi: RoiRect, tryHarder: Boolean, debugEnabled: Boolean): ScanOutput? {
        if (debugEnabled) {
            val bytes = ByteArray(512 * 512)
            src512.get(0, 0, bytes)
            val dbg = NativeDecoder.decodeGrayDebug(bytes, 512, 512, 0, 0, 0, 512, 512, tryHarder) ?: "dbg=null"
            android.util.Log.i("SmartQRCode", "DecodeDbg rot=0 th=$tryHarder roi=${roi.left},${roi.top},${roi.right},${roi.bottom} $dbg")
            emitMat("ROI-512 r=${roi.left},${roi.top},${roi.right},${roi.bottom} $dbg", src512)
        }
        val branches = buildEnhancementBranches(src512)
        try {
            for ((idx, m) in branches.withIndex()) {
                if (debugEnabled) {
                    val name = when (idx) {
                        0 -> "Branch-Raw"
                        1 -> "Branch-CLAHE"
                        2 -> "Branch-Adaptive"
                        3 -> "Branch-AdaptiveClose"
                        4 -> "Branch-Sauvola"
                        5 -> "Branch-SauvolaClose"
                        6 -> "Branch-Unsharp"
                        else -> "Branch-Extra"
                    }
                    emitMat(name, m)
                }
                val bytes = ByteArray(512 * 512)
                m.get(0, 0, bytes)
                val raw = NativeDecoder.decodeGray(
                    bytes,
                    512,
                    512,
                    0,
                    0,
                    0,
                    512,
                    512,
                    tryHarder
                )
                if (raw != null) {
                    val parsed = NativeDecoder.parseResult(raw)
                    if (parsed != null) {
                        val mappedBox = mapPatchBoxToFull(roi, parsed.roi)
                        val mappedQuad = parsed.quad?.let { mapPatchQuadToFull(roi, it) }
                        if (parsed.text.isNotBlank()) {
                            return ScanOutput(parsed.text, mappedBox, mappedQuad)
                        }
                        if (tryHarder && parsed.quad != null) {
                            val rectified = warpByQuad(m, parsed.quad)
                            try {
                                if (debugEnabled) emitMat("WarpByQuad-512", rectified)
                                val recovered = moduleRecoveryTry(rectified, debugEnabled, "Warp")
                                if (recovered != null) return ScanOutput(recovered, mappedBox, mappedQuad)
                            } finally {
                                rectified.release()
                            }
                        }
                        return ScanOutput(null, mappedBox, mappedQuad)
                    }
                }

                if (tryHarder) {
                    val detector = try {
                        qrDetector ?: QRCodeDetector().also { qrDetector = it }
                    } catch (e: UnsatisfiedLinkError) {
                        OpenCvSupport.ready = false
                        null
                    }
                    val allowDetector = idx <= 1
                    if (detector != null && allowDetector) {
                        try {
                            val points = Mat()
                            var patchBox: RoiRect? = null
                            try {
                                val ok = detector.detect(m, points)
                                if (ok && !points.empty() && points.total() >= 4L) {
                                    val pts = FloatArray(8)
                                    points.get(0, 0, pts)
                                    var minX = Int.MAX_VALUE
                                    var minY = Int.MAX_VALUE
                                    var maxX = Int.MIN_VALUE
                                    var maxY = Int.MIN_VALUE
                                    for (i in 0 until 4) {
                                        val x = pts[i * 2].roundToInt()
                                        val y = pts[i * 2 + 1].roundToInt()
                                        minX = min(minX, x)
                                        minY = min(minY, y)
                                        maxX = max(maxX, x)
                                        maxY = max(maxY, y)
                                    }
                                    patchBox = RoiRect(
                                        minX.coerceIn(0, 512),
                                        minY.coerceIn(0, 512),
                                        maxX.coerceIn(0, 512),
                                        maxY.coerceIn(0, 512)
                                    )
                                }
                            } finally {
                                points.release()
                            }

                            val decoded = detector.detectAndDecode(m)
                            if (!decoded.isNullOrBlank()) {
                                val box = patchBox?.let { mapPatchBoxToFull(roi, it) } ?: roi
                                return ScanOutput(decoded, box)
                            }

                            if (patchBox != null) {
                                return ScanOutput(null, mapPatchBoxToFull(roi, patchBox))
                            }
                        } catch (e: UnsatisfiedLinkError) {
                            OpenCvSupport.ready = false
                        }
                    }
                }
            }
            if (tryHarder) {
                moduleRecoveryTry(src512, debugEnabled, "ROI")?.let { recovered ->
                    return ScanOutput(recovered, roi)
                }
            }
            return null
        } finally {
            branches.forEach { it.release() }
        }
    }

    private fun emitMat(step: String, src: Mat) {
        val cb = debugCallback ?: return
        val mat = if (src.type() == CvType.CV_8UC1) {
            src
        } else {
            val tmp = Mat()
            src.convertTo(tmp, CvType.CV_8U)
            tmp
        }
        try {
            val w = mat.cols()
            val h = mat.rows()
            val bytes = ByteArray(w * h)
            mat.get(0, 0, bytes)
            cb(DebugFrame(step, w, h, bytes))
        } finally {
            if (mat !== src) mat.release()
        }
    }

    private fun mapPatchBoxToFull(roi: RoiRect, patchBox: RoiRect): RoiRect {
        val scaleX = roi.width.toDouble() / 512.0
        val scaleY = roi.height.toDouble() / 512.0
        val left = roi.left + (patchBox.left * scaleX).roundToInt()
        val top = roi.top + (patchBox.top * scaleY).roundToInt()
        val right = roi.left + (patchBox.right * scaleX).roundToInt()
        val bottom = roi.top + (patchBox.bottom * scaleY).roundToInt()
        return RoiRect(left, top, right, bottom)
    }

    private fun mapPatchQuadToFull(roi: RoiRect, patchQuad: FloatArray): FloatArray {
        if (patchQuad.size != 8) return patchQuad
        val scaleX = roi.width.toDouble() / 512.0
        val scaleY = roi.height.toDouble() / 512.0
        return FloatArray(8).also { out ->
            for (i in 0 until 4) {
                val x = patchQuad[i * 2].toDouble() * scaleX + roi.left.toDouble()
                val y = patchQuad[i * 2 + 1].toDouble() * scaleY + roi.top.toDouble()
                out[i * 2] = x.toFloat()
                out[i * 2 + 1] = y.toFloat()
            }
        }
    }

    private fun warpByQuad(src512: Mat, quad: FloatArray): Mat {
        val ordered = orderQuad(quad)
        val srcPts = MatOfPoint2f(
            Point(ordered[0].toDouble(), ordered[1].toDouble()),
            Point(ordered[2].toDouble(), ordered[3].toDouble()),
            Point(ordered[4].toDouble(), ordered[5].toDouble()),
            Point(ordered[6].toDouble(), ordered[7].toDouble())
        )
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(511.0, 0.0),
            Point(511.0, 511.0),
            Point(0.0, 511.0)
        )
        val h = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val out = Mat()
        Imgproc.warpPerspective(src512, out, h, Size(512.0, 512.0), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(255.0))
        srcPts.release()
        dstPts.release()
        h.release()
        return out
    }

    private fun warpFullByQuad(fullGray: Mat, quadFull: FloatArray): Mat? {
        if (quadFull.size != 8) return null
        val w = fullGray.cols()
        val h = fullGray.rows()
        if (w <= 0 || h <= 0) return null

        fun cx(v: Float) = v.toInt().coerceIn(0, w - 1).toDouble()
        fun cy(v: Float) = v.toInt().coerceIn(0, h - 1).toDouble()

        val ordered = orderQuad(quadFull)
        val srcPts = MatOfPoint2f(
            Point(cx(ordered[0]), cy(ordered[1])),
            Point(cx(ordered[2]), cy(ordered[3])),
            Point(cx(ordered[4]), cy(ordered[5])),
            Point(cx(ordered[6]), cy(ordered[7]))
        )
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(511.0, 0.0),
            Point(511.0, 511.0),
            Point(0.0, 511.0)
        )
        val m = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val out = Mat()
        Imgproc.warpPerspective(fullGray, out, m, Size(512.0, 512.0), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(255.0))
        srcPts.release()
        dstPts.release()
        m.release()
        return out
    }

    private fun orderQuad(quad: FloatArray): FloatArray {
        if (quad.size != 8) return quad
        data class P(val x: Float, val y: Float)
        val pts = arrayListOf(
            P(quad[0], quad[1]),
            P(quad[2], quad[3]),
            P(quad[4], quad[5]),
            P(quad[6], quad[7])
        )
        var tl = pts[0]
        var tr = pts[0]
        var br = pts[0]
        var bl = pts[0]
        var minSum = Float.POSITIVE_INFINITY
        var maxSum = Float.NEGATIVE_INFINITY
        var minDiff = Float.POSITIVE_INFINITY
        var maxDiff = Float.NEGATIVE_INFINITY
        for (p in pts) {
            val sum = p.x + p.y
            val diff = p.x - p.y
            if (sum < minSum) {
                minSum = sum
                tl = p
            }
            if (sum > maxSum) {
                maxSum = sum
                br = p
            }
            if (diff < minDiff) {
                minDiff = diff
                bl = p
            }
            if (diff > maxDiff) {
                maxDiff = diff
                tr = p
            }
        }
        return floatArrayOf(tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    }

    private fun buildEnhancementBranches(src512: Mat): List<Mat> {
        val out = ArrayList<Mat>(7)
        out.add(src512.clone())

        val claheOut = Mat()
        val clahe: CLAHE = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(src512, claheOut)
        out.add(claheOut)

        val adaptive = Mat()
        Imgproc.adaptiveThreshold(
            src512,
            adaptive,
            255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY,
            35,
            7.0
        )
        out.add(adaptive)

        val sauvola = sauvolaThreshold(src512, window = 25, k = 0.34, r = 128.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val adaptiveClose = Mat()
        Imgproc.morphologyEx(adaptive, adaptiveClose, Imgproc.MORPH_CLOSE, kernel)
        out.add(adaptiveClose)

        out.add(sauvola)
        val sauvolaClose = Mat()
        Imgproc.morphologyEx(sauvola, sauvolaClose, Imgproc.MORPH_CLOSE, kernel)
        out.add(sauvolaClose)
        kernel.release()

        val unsharp = Mat()
        val blur = Mat()
        Imgproc.GaussianBlur(src512, blur, Size(0.0, 0.0), 1.2)
        Core.addWeighted(src512, 1.6, blur, -0.6, 0.0, unsharp)
        blur.release()
        out.add(unsharp)

        return out
    }

    private fun sauvolaThreshold(gray: Mat, window: Int, k: Double, r: Double): Mat {
        val w = if (window % 2 == 1) window else window + 1
        val grayF = Mat()
        gray.convertTo(grayF, CvType.CV_32F)

        val mean = Mat()
        Imgproc.boxFilter(grayF, mean, CvType.CV_32F, Size(w.toDouble(), w.toDouble()), Point(-1.0, -1.0), true, Core.BORDER_REPLICATE)

        val sq = Mat()
        Core.multiply(grayF, grayF, sq)
        val meanSq = Mat()
        Imgproc.boxFilter(sq, meanSq, CvType.CV_32F, Size(w.toDouble(), w.toDouble()), Point(-1.0, -1.0), true, Core.BORDER_REPLICATE)

        val varMat = Mat()
        val meanMean = Mat()
        Core.multiply(mean, mean, meanMean)
        Core.subtract(meanSq, meanMean, varMat)
        Core.max(varMat, Scalar(0.0), varMat)

        val std = Mat()
        Core.sqrt(varMat, std)

        val stdByR = Mat()
        Core.divide(std, Scalar(r), stdByR)
        Core.subtract(stdByR, Scalar(1.0), stdByR)
        Core.multiply(stdByR, Scalar(k), stdByR)
        Core.add(stdByR, Scalar(1.0), stdByR)

        val thresh = Mat()
        Core.multiply(mean, stdByR, thresh)

        val bin = Mat()
        Core.compare(grayF, thresh, bin, Core.CMP_GT)
        bin.convertTo(bin, CvType.CV_8U, 255.0)

        grayF.release()
        mean.release()
        sq.release()
        meanSq.release()
        varMat.release()
        meanMean.release()
        std.release()
        stdByR.release()
        thresh.release()
        return bin
    }

    private fun rectifyTo512(src512: Mat): Mat? {
        val detector = try {
            qrDetector ?: QRCodeDetector().also { qrDetector = it }
        } catch (e: UnsatisfiedLinkError) {
            OpenCvSupport.ready = false
            return null
        }
        val points = Mat()
        val ok = detector.detect(src512, points)
        if (!ok || points.empty() || points.total() < 4L) {
            points.release()
            return null
        }

        val pts = FloatArray(8)
        points.get(0, 0, pts)
        points.release()

        val ordered = orderQuad(pts)
        val srcPts = MatOfPoint2f(
            Point(ordered[0].toDouble(), ordered[1].toDouble()),
            Point(ordered[2].toDouble(), ordered[3].toDouble()),
            Point(ordered[4].toDouble(), ordered[5].toDouble()),
            Point(ordered[6].toDouble(), ordered[7].toDouble())
        )
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(511.0, 0.0),
            Point(511.0, 511.0),
            Point(0.0, 511.0)
        )
        val h = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val out = Mat()
        Imgproc.warpPerspective(src512, out, h, Size(512.0, 512.0), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(255.0))
        srcPts.release()
        dstPts.release()
        h.release()
        return out
    }

    private fun fuseByBlocks(frames: List<Mat>, blockSize: Int): Mat {
        val out = Mat(512, 512, CvType.CV_8UC1)
        val blocks = 512 / blockSize
        val gradEnergy = frames.map { tenengradEnergy(it) }

        for (by in 0 until blocks) {
            for (bx in 0 until blocks) {
                val x = bx * blockSize
                val y = by * blockSize
                var bestIdx = 0
                var bestScore = Double.NEGATIVE_INFINITY
                for (i in frames.indices) {
                    val s = gradEnergy[i].get(y / blockSize, x / blockSize)[0]
                    if (s > bestScore) {
                        bestScore = s
                        bestIdx = i
                    }
                }
                val src = Mat(frames[bestIdx], Rect(x, y, blockSize, blockSize))
                val dst = Mat(out, Rect(x, y, blockSize, blockSize))
                src.copyTo(dst)
                src.release()
                dst.release()
            }
        }

        gradEnergy.forEach { it.release() }
        return out
    }

    private fun tenengradEnergy(gray512: Mat): Mat {
        val gx = Mat()
        val gy = Mat()
        Imgproc.Sobel(gray512, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gray512, gy, CvType.CV_32F, 0, 1, 3)
        val mag = Mat()
        Core.magnitude(gx, gy, mag)

        val blockSize = 32
        val blocks = 512 / blockSize
        val energy = Mat(blocks, blocks, CvType.CV_64F)
        for (by in 0 until blocks) {
            for (bx in 0 until blocks) {
                val x = bx * blockSize
                val y = by * blockSize
                val block = Mat(mag, Rect(x, y, blockSize, blockSize))
                val m = Core.mean(block).`val`[0]
                energy.put(by, bx, m)
                block.release()
            }
        }

        gx.release()
        gy.release()
        mag.release()
        return energy
    }

    private fun moduleRecoveryTry(gray512: Mat, debugEnabled: Boolean, tag: String): String? {
        return moduleRecoveryTry(gray512, debugEnabled, tag, bypassInterval = false)
    }

    private fun moduleRecoveryTry(gray512: Mat, debugEnabled: Boolean, tag: String, bypassInterval: Boolean): String? {
        val nowNs = android.os.SystemClock.elapsedRealtimeNanos()
        val forced = forceTryHarderFrames > 0
        val minIntervalNs = if (forced) 220_000_000L else 900_000_000L
        val intervalNs = nowNs - lastModuleRecoveryNs
        if (!bypassInterval && intervalNs < minIntervalNs) {
            if (forced) {
                android.util.Log.i(
                    "SmartQRCode",
                    "FlowModuleRecSkip tag=$tag forced=true dtMs=%.0f minMs=%.0f".format(
                        intervalNs / 1_000_000.0,
                        minIntervalNs / 1_000_000.0
                    )
                )
            }
            return null
        }
        lastModuleRecoveryNs = nowNs
        emitStatus("${if (forced) "hard(force)" else "hard"} stage=moduleRecovery tag=$tag")
        android.util.Log.i("SmartQRCode", "FlowModuleRec tag=$tag forced=$forced")

        val attempts = arrayListOf<Mat>()
        attempts.add(gray512)
        attempts.add(padAndRescale(gray512, pad = 32))
        attempts.add(padAndRescale(gray512, pad = 64))
        attempts.add(padAndRescale(gray512, pad = 96))
        attempts.add(padAndRescale(gray512, pad = 128))
        attempts.add(padAndRescaleBorder(gray512, pad = 64, borderType = Core.BORDER_CONSTANT))
        attempts.add(padAndRescaleBorder(gray512, pad = 128, borderType = Core.BORDER_CONSTANT))

        try {
            for ((attemptIdx, attempt) in attempts.withIndex()) {
                emitStatus("${if (forced) "hard(force)" else "hard"} stage=moduleRecovery tag=$tag attempt=$attemptIdx")
                if (debugEnabled) emitMat("ModuleIn-$tag", attempt)
                val bin = Mat()
                try {
                    Imgproc.adaptiveThreshold(attempt, bin, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 55, 8.0)
                    val candidateN = estimateModuleCount(bin)
                    val candidates = if (candidateN != null) {
                        val base = listOf(
                            candidateN,
                            candidateN - 4,
                            candidateN + 4,
                            candidateN - 8,
                            candidateN + 8,
                            candidateN - 12,
                            candidateN + 12,
                            candidateN - 16,
                            candidateN + 16
                        )
                        val extra = if (forced) {
                            listOf(
                                candidateN - 20,
                                candidateN + 20,
                                candidateN - 24,
                                candidateN + 24,
                                candidateN - 28,
                                candidateN + 28,
                                candidateN - 32,
                                candidateN + 32,
                                candidateN - 36,
                                candidateN + 36,
                                candidateN - 40,
                                candidateN + 40,
                                candidateN - 44,
                                candidateN + 44,
                                candidateN - 48,
                                candidateN + 48
                            )
                        } else {
                            emptyList()
                        }
                        (base + extra)
                            .distinct()
                            .filter { it >= 21 && (it - 21) % 4 == 0 }
                            .sorted()
                    } else {
                        if (forced) (21..177 step 4).toList() else (21..73 step 4).toList()
                    }
                    android.util.Log.i(
                        "SmartQRCode",
                        "FlowModulePlan tag=$tag attempt=$attemptIdx candN=$candidateN list=${candidates.joinToString(",")}"
                    )
                    for (n in candidates) {
                        emitStatus("${if (forced) "hard(force)" else "hard"} stage=beam tag=$tag n=$n")
                        val recovered0 = recoverByBeam(attempt, n, maxUnknown = 18, beamWidth = 48, blackIsLow = true, debugEnabled = debugEnabled, tag = tag)
                        if (recovered0 != null) return recovered0
                        val recovered1 = recoverByBeam(attempt, n, maxUnknown = 18, beamWidth = 48, blackIsLow = false, debugEnabled = debugEnabled, tag = tag)
                        if (recovered1 != null) return recovered1

                        if (n <= 97) {
                            val recovered2 = recoverByBeam(attempt, n, maxUnknown = 26, beamWidth = 96, blackIsLow = true, debugEnabled = debugEnabled, tag = tag)
                            if (recovered2 != null) return recovered2
                            val recovered3 = recoverByBeam(attempt, n, maxUnknown = 26, beamWidth = 96, blackIsLow = false, debugEnabled = debugEnabled, tag = tag)
                            if (recovered3 != null) return recovered3
                        }
                    }
                } finally {
                    bin.release()
                }
            }
            if (forced || debugEnabled) android.util.Log.i("SmartQRCode", "FlowModuleRecMiss tag=$tag forced=$forced")
            return null
        } finally {
            attempts.drop(1).forEach { it.release() }
        }
    }

    private fun estimateModuleCount(bin: Mat): Int? {
        val row = ByteArray(512)
        val ys = intArrayOf(128, 192, 256, 320, 384)
        var bestTransitions = 0
        var bestSnapped: Int? = null
        var bestNApprox: Int? = null
        for (y in ys) {
            bin.get(y.coerceIn(0, 511), 0, row)
            var transitions = 0
            for (i in 1 until row.size) {
                if ((row[i].toInt() and 0xFF) != (row[i - 1].toInt() and 0xFF)) transitions++
            }
            if (transitions > bestTransitions) {
                val approxModule = (512.0 / (transitions / 2.0)).coerceIn(2.0, 24.0)
                val nApprox = (512.0 / approxModule).roundToInt()
                val snapped = snapToQrSize(nApprox)
                if (snapped != null && snapped in 21..177) {
                    bestTransitions = transitions
                    bestSnapped = snapped
                    bestNApprox = nApprox
                }
            }
        }
        val ok = bestTransitions >= 18
        if (!ok) {
            android.util.Log.i("SmartQRCode", "FlowModuleEstimate fail transitions=$bestTransitions")
        } else {
            android.util.Log.i("SmartQRCode", "FlowModuleEstimate ok transitions=$bestTransitions nApprox=$bestNApprox snapped=$bestSnapped")
        }
        return bestSnapped?.takeIf { ok }
    }

    private fun snapToQrSize(n: Int): Int? {
        val v = ((n - 21) / 4.0).roundToInt() + 1
        if (v < 1) return null
        return 21 + 4 * (v - 1)
    }

    private fun recoverByBeam(
        gray512: Mat,
        modules: Int,
        maxUnknown: Int,
        beamWidth: Int,
        blackIsLow: Boolean,
        debugEnabled: Boolean,
        tag: String
    ): String? {
        val forced = forceTryHarderFrames > 0
        val maxU = when {
            forced -> maxUnknown.coerceAtLeast(40).coerceAtMost(48)
            else -> maxUnknown
        }
        val budgetNs = when {
            forced -> 3_200_000_000L
            debugEnabled -> 7_000_000_000L
            else -> 360_000_000L
        }
        val startNs = android.os.SystemClock.elapsedRealtimeNanos()
        val deadlineNs = startNs + budgetNs
        val moduleSize = 512.0 / modules.toDouble()
        if (moduleSize < 2.0) {
            android.util.Log.i("SmartQRCode", "FlowBeamSkip tag=$tag n=$modules ms=%.2f reason=moduleSizeSmall".format(moduleSize))
            return null
        }

        val levels = estimateBlackWhiteLevels(gray512)
        val blackRef = levels.first
        val whiteRef = levels.second
        val denomFloor = if (forced) 40.0 else 25.0
        val denom = (whiteRef - blackRef).coerceAtLeast(denomFloor)

        val grayBytes = ByteArray(512 * 512)
        gray512.get(0, 0, grayBytes)
        fun mapCenter(x: Int, y: Int, phaseX: Double, phaseY: Double, scale: Double): Pair<Int, Int> {
            val ux = (x + 0.5 + phaseX) / modules.toDouble()
            val uy = (y + 0.5 + phaseY) / modules.toDouble()
            val px = ((ux - 0.5) * scale + 0.5) * 512.0
            val py = ((uy - 0.5) * scale + 0.5) * 512.0
            val cx = px.roundToInt().coerceIn(1, 510)
            val cy = py.roundToInt().coerceIn(1, 510)
            return cx to cy
        }

        fun fill(
            bits: ByteArray,
            margins: DoubleArray,
            pBlacks: DoubleArray,
            phaseX: Double,
            phaseY: Double,
            scale: Double,
            unknownLow: Double,
            unknownHigh: Double
        ): Int {
            var u0 = 0
            for (y in 0 until modules) {
                for (x in 0 until modules) {
                    val (cx, cy) = mapCenter(x, y, phaseX, phaseY, scale)
                    var sum = 0
                    var base = (cy - 1) * 512 + (cx - 1)
                    sum += grayBytes[base].toInt() and 0xFF
                    sum += grayBytes[base + 1].toInt() and 0xFF
                    sum += grayBytes[base + 2].toInt() and 0xFF
                    base += 512
                    sum += grayBytes[base].toInt() and 0xFF
                    sum += grayBytes[base + 1].toInt() and 0xFF
                    sum += grayBytes[base + 2].toInt() and 0xFF
                    base += 512
                    sum += grayBytes[base].toInt() and 0xFF
                    sum += grayBytes[base + 1].toInt() and 0xFF
                    sum += grayBytes[base + 2].toInt() and 0xFF
                    val mean = sum / 9.0

                    val pBlack = if (blackIsLow) {
                        ((whiteRef - mean) / denom).coerceIn(0.0, 1.0)
                    } else {
                        ((mean - blackRef) / denom).coerceIn(0.0, 1.0)
                    }
                    val idx = y * modules + x
                    pBlacks[idx] = pBlack
                    margins[idx] = abs(pBlack - 0.5)
                    when {
                        pBlack > unknownHigh -> bits[idx] = 1
                        pBlack < unknownLow -> bits[idx] = 0
                        else -> {
                            bits[idx] = 2
                            u0++
                        }
                    }
                }
            }
            return u0
        }

        val phaseCandidates = doubleArrayOf(-0.30, 0.0, 0.30)
        val scaleCandidates = doubleArrayOf(0.94, 1.0, 1.06)
        var bestPhaseX = 0.0
        var bestPhaseY = 0.0
        var bestScale = 1.0
        var bestU0 = Int.MAX_VALUE
        var bestU1 = Int.MAX_VALUE
        var bestAvgMargin = -1.0
        var bestMinMargin = -1.0
        var bestLow = 0.38
        var bestHigh = 0.62
        run {
            val bits = ByteArray(modules * modules)
            val margins = DoubleArray(modules * modules)
            val pBlacks = DoubleArray(modules * modules)
            val bands = arrayOf(
                0.35 to 0.65,
                0.40 to 0.60,
                0.44 to 0.56,
                0.47 to 0.53
            )
            for (scale in scaleCandidates) {
                for (phaseY in phaseCandidates) {
                    for (phaseX in phaseCandidates) {
                        for ((low, high) in bands) {
                            val u0 = fill(bits, margins, pBlacks, phaseX, phaseY, scale, low, high)
                            applyStructureHintsIfUnknown(bits, modules)
                            var u1 = 0
                            var sum = 0.0
                            var minM = 1.0
                            for (i in bits.indices) {
                                if (bits[i].toInt() == 2) {
                                    u1++
                                    val m = margins[i]
                                    sum += m
                                    if (m < minM) minM = m
                                }
                            }
                            val avg = if (u1 > 0) sum / u1.toDouble() else -1.0
                            if (u1 == 0) minM = -1.0
                            val better =
                                (u1 < bestU1) ||
                                    (u1 == bestU1 && avg > bestAvgMargin) ||
                                    (u1 == bestU1 && avg == bestAvgMargin && minM > bestMinMargin)
                            if (better) {
                                bestPhaseX = phaseX
                                bestPhaseY = phaseY
                                bestScale = scale
                                bestU0 = u0
                                bestU1 = u1
                                bestAvgMargin = avg
                                bestMinMargin = minM
                                bestLow = low
                                bestHigh = high
                            }
                            if (android.os.SystemClock.elapsedRealtimeNanos() > deadlineNs) return null
                        }
                        if (android.os.SystemClock.elapsedRealtimeNanos() > deadlineNs) return null
                    }
                }
            }
        }

        if (forced || debugEnabled) android.util.Log.i(
            "SmartQRCode",
            "FlowBeamPhase tag=$tag n=$modules blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f u0=$bestU0 u1=$bestU1 marginMin=%.3f marginAvg=%.3f lvl=%.1f-%.1f".format(
                bestPhaseX,
                bestPhaseY,
                bestScale,
                bestMinMargin,
                bestAvgMargin,
                blackRef,
                whiteRef
            )
        )

        val baseBits = ByteArray(modules * modules)
        val baseMargins = DoubleArray(modules * modules)
        val basePBlacks = DoubleArray(modules * modules)
        fill(baseBits, baseMargins, basePBlacks, bestPhaseX, bestPhaseY, bestScale, bestLow, bestHigh)
        applyStructureHintsIfUnknown(baseBits, modules)

        fun tryDecodeBits(bits: ByteArray, level: Int): String? {
            fun tryDecodeRendered(bytes: ByteArray, tryHarder: Boolean): String? {
                val raw = NativeDecoder.decodeGray(bytes, 512, 512, 0, 0, 0, 512, 512, tryHarder)
                val parsed = raw?.let { NativeDecoder.parseResult(it) }
                return parsed?.text?.takeIf { it.isNotBlank() && !isInvalidDiagnosticText(it) }
            }

            val synthesized = renderModulesTo512(bits, modules)
            try {
                val synthesizedBytes = ByteArray(512 * 512)
                synthesized.get(0, 0, synthesizedBytes)
                tryDecodeRendered(synthesizedBytes, tryHarder = false)?.let { return it }
                if (level >= 1) {
                    tryDecodeRendered(synthesizedBytes, tryHarder = true)?.let { return it }
                }

                if (level >= 2) {
                    val padded = padAndRescale(synthesized, pad = 64)
                    try {
                        val paddedBytes = ByteArray(512 * 512)
                        padded.get(0, 0, paddedBytes)
                        tryDecodeRendered(paddedBytes, tryHarder = false)?.let { return it }
                        tryDecodeRendered(paddedBytes, tryHarder = true)?.let { return it }
                    } finally {
                        padded.release()
                    }
                }
                return null
            } finally {
                synthesized.release()
            }
        }

        val unknownsRaw = ArrayList<Pair<Int, Double>>()
        for (i in baseBits.indices) {
            if (baseBits[i].toInt() == 2) unknownsRaw.add(i to baseMargins[i])
        }
        val unknownsBeforeHints = bestU0
        val unknownsAfterHints = unknownsRaw.size

        if (unknownsAfterHints <= 0) {
            val text0 = tryDecodeBits(baseBits, level = 2)
            if (text0 != null) {
                android.util.Log.i("SmartQRCode", "FlowBeamHit tag=$tag n=$modules unk=0 flips=0")
                return text0
            }
            android.util.Log.i(
                "SmartQRCode",
                "FlowBeamSkip tag=$tag n=$modules ms=%.2f blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f u0=$unknownsBeforeHints u1=0 maxU=$maxUnknown lvl=%.1f-%.1f reason=noUnknownDecodeFail".format(
                    moduleSize,
                    bestPhaseX,
                    bestPhaseY,
                    bestScale,
                    blackRef,
                    whiteRef
                )
            )
            return null
        }

        if (android.os.SystemClock.elapsedRealtimeNanos() > deadlineNs) return null

        unknownsRaw.sortBy { it.second }
        val pickRaw = LinkedHashSet<Int>(maxU)
        for ((idx, _) in unknownsRaw.take(min(unknownsRaw.size, maxU))) {
            pickRaw.add(idx)
        }
        if (unknownsRaw.size > maxU && (forced || debugEnabled)) {
            android.util.Log.i(
                "SmartQRCode",
                "FlowBeamTrim tag=$tag n=$modules blackIsLow=$blackIsLow u1=$unknownsAfterHints pick=${pickRaw.size} maxU=$maxU"
            )
        }

        val marginCandidates = ArrayList<Pair<Int, Double>>(modules * modules)
        for (i in baseMargins.indices) {
            val m = baseMargins[i]
            if (m < 0.10) marginCandidates.add(i to m)
        }
        marginCandidates.sortBy { it.second }
        if (pickRaw.size < maxU) {
            for ((idx, _) in marginCandidates) {
                if (pickRaw.size >= maxU) break
                if (pickRaw.contains(idx)) continue
                if (baseBits[idx].toInt() != 2) pickRaw.add(idx)
            }
        }

        val unknowns = ArrayList<Pair<Int, Double>>(pickRaw.size)
        val picked = ArrayList<Pair<Int, Double>>(pickRaw.size)
        var sumMargin = 0.0
        var minMargin = 1.0
        for (idx in pickRaw) {
            val m = baseMargins[idx]
            picked.add(idx to m)
            if (baseBits[idx].toInt() == 2) {
                unknowns.add(idx to m)
                sumMargin += m
                if (m < minMargin) minMargin = m
            }
        }
        picked.sortBy { it.second }
        val unknownsAfterTrim = unknowns.size
        val avgMargin = if (unknowns.isNotEmpty()) sumMargin / unknowns.size.toDouble() else -1.0
        if (unknowns.isEmpty()) minMargin = -1.0

        if (picked.isEmpty()) return null

        if (android.os.SystemClock.elapsedRealtimeNanos() > deadlineNs) return null

        if (unknownsAfterHints > maxU) {
            android.util.Log.i(
                "SmartQRCode",
                "FlowBeamSkip tag=$tag n=$modules ms=%.2f blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f u0=$unknownsBeforeHints u1=$unknownsAfterHints pick=$unknownsAfterTrim maxU=$maxU marginMin=%.3f marginAvg=%.3f lvl=%.1f-%.1f reason=tooManyUnknownTrimmed".format(
                    moduleSize,
                    bestPhaseX,
                    bestPhaseY,
                    bestScale,
                    minMargin,
                    avgMargin,
                    blackRef,
                    whiteRef
                )
            )
        }

        if (forced || debugEnabled) {
            android.util.Log.i(
                "SmartQRCode",
                "FlowBeamStart tag=$tag n=$modules blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f unk=$unknownsAfterTrim pick=${pickRaw.size} u1=$unknownsAfterHints u0=$unknownsBeforeHints maxU=$maxU beam=$beamWidth ms=%.2f marginMin=%.3f marginAvg=%.3f lvl=%.1f-%.1f budgetMs=%.0f".format(
                    bestPhaseX,
                    bestPhaseY,
                    bestScale,
                    moduleSize,
                    minMargin,
                    avgMargin,
                    blackRef,
                    whiteRef,
                    budgetNs / 1_000_000.0
                )
            )
        }
        if (debugEnabled) {
            val base = renderModulesTo512(baseBits, modules)
            try {
                emitMat("Modules-$tag-$modules-${if (blackIsLow) "dark" else "light"}-u${unknowns.size}", base)
            } finally {
                base.release()
            }
        }
        val pick = picked.take(min(picked.size, maxU)).map { it.first }

        fun negLogProb(idx: Int, v: Int): Double {
            val pBlack = basePBlacks[idx].coerceIn(1e-4, 1.0 - 1e-4)
            val p = if (v == 1) pBlack else (1.0 - pBlack)
            return -ln(p.coerceIn(1e-8, 1.0))
        }

        data class BeamState(val bits: ByteArray, val score: Double, val flips: Int)
        var beam = listOf(BeamState(baseBits.copyOf(), 0.0, 0))

        var expandedSteps = 0
        var decodeChecks = 0
        for (idx in pick) {
            val nowNs = android.os.SystemClock.elapsedRealtimeNanos()
            if (nowNs > deadlineNs) {
                if (forced || debugEnabled) android.util.Log.i(
                    "SmartQRCode",
                    "FlowBeamStop tag=$tag n=$modules blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f reason=deadline stage=expand steps=$expandedSteps pick=${pick.size} checks=$decodeChecks elapsedMs=%.0f budgetMs=%.0f".format(
                        bestPhaseX,
                        bestPhaseY,
                        bestScale,
                        (nowNs - startNs) / 1_000_000.0,
                        budgetNs / 1_000_000.0
                    )
                )
                return null
            }
            val next = ArrayList<BeamState>(beam.size * 2)
            for (s in beam) {
                val b0 = s.bits.copyOf()
                b0[idx] = 0
                next.add(BeamState(b0, s.score + negLogProb(idx, 0), s.flips + 1))
                val b1 = s.bits.copyOf()
                b1[idx] = 1
                next.add(BeamState(b1, s.score + negLogProb(idx, 1), s.flips + 1))
            }
            next.sortBy { it.score }
            beam = next.take(beamWidth)
            expandedSteps++
        }

        if (android.os.SystemClock.elapsedRealtimeNanos() > deadlineNs) return null

        var avgDecodeNs = 450_000_000L
        val safetyNs = 100_000_000L
        val fastCheckLimit = when {
            debugEnabled -> 16
            forced -> 6
            else -> Int.MAX_VALUE
        }
        fun chooseLevel(remainNs: Long, decodeChecks: Int): Int {
            if (remainNs < 900_000_000L) return 0
            if (forced) {
                if (decodeChecks <= 2) return 0
                if (decodeChecks <= 4) return 1
                return if (remainNs >= 2_200_000_000L) 2 else 1
            }
            return if (remainNs >= 1_600_000_000L) 1 else 0
        }
        for (s in beam) {
            val nowNs2 = android.os.SystemClock.elapsedRealtimeNanos()
            if (nowNs2 > deadlineNs) {
                if (forced || debugEnabled) android.util.Log.i(
                    "SmartQRCode",
                    "FlowBeamStop tag=$tag n=$modules blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f reason=deadline stage=decode steps=$expandedSteps pick=${pick.size} checks=$decodeChecks elapsedMs=%.0f budgetMs=%.0f".format(
                        bestPhaseX,
                        bestPhaseY,
                        bestScale,
                        (nowNs2 - startNs) / 1_000_000.0,
                        budgetNs / 1_000_000.0
                    )
                )
                return null
            }
            val remainNs = deadlineNs - nowNs2
            if (remainNs < avgDecodeNs + safetyNs) break
            val level = chooseLevel(remainNs, decodeChecks)
            decodeChecks++
            val tDec0 = android.os.SystemClock.elapsedRealtimeNanos()
            val text = tryDecodeBits(s.bits, level = level)
            val tDec1 = android.os.SystemClock.elapsedRealtimeNanos()
            val dtDecNs = tDec1 - tDec0
            avgDecodeNs = ((avgDecodeNs * 3L) + dtDecNs) / 4L
            if (forced || debugEnabled) {
                android.util.Log.i(
                    "SmartQRCode",
                    "FlowBeamDecode tag=$tag n=$modules lv=$level dtMs=%.0f avgMs=%.0f checks=$decodeChecks".format(
                        dtDecNs / 1_000_000.0,
                        avgDecodeNs / 1_000_000.0
                    )
                )
            }
            if (text != null) {
                android.util.Log.i("SmartQRCode", "FlowBeamHit tag=$tag n=$modules unk=${unknowns.size} flips=${s.flips}")
                return text
            }
            if (decodeChecks >= fastCheckLimit) break
        }

        if (forced || debugEnabled) {
            for (s in beam.take(if (debugEnabled) 6 else 3)) {
                val nowNs2 = android.os.SystemClock.elapsedRealtimeNanos()
                if (nowNs2 > deadlineNs) break
                val remainNs = deadlineNs - nowNs2
                if (remainNs < avgDecodeNs + safetyNs) break
                val level = chooseLevel(remainNs, decodeChecks)
                decodeChecks++
                val tDec0 = android.os.SystemClock.elapsedRealtimeNanos()
                val text = tryDecodeBits(s.bits, level = level)
                val tDec1 = android.os.SystemClock.elapsedRealtimeNanos()
                val dtDecNs = tDec1 - tDec0
                avgDecodeNs = ((avgDecodeNs * 3L) + dtDecNs) / 4L
                android.util.Log.i(
                    "SmartQRCode",
                    "FlowBeamDecode tag=$tag n=$modules lv=$level dtMs=%.0f avgMs=%.0f checks=$decodeChecks".format(
                        dtDecNs / 1_000_000.0,
                        avgDecodeNs / 1_000_000.0
                    )
                )
                if (text != null) {
                    android.util.Log.i("SmartQRCode", "FlowBeamHit tag=$tag n=$modules unk=${unknowns.size} flips=${s.flips}")
                    return text
                }
            }
        }

        if (forced || debugEnabled) {
            val endNs = android.os.SystemClock.elapsedRealtimeNanos()
            android.util.Log.i(
                "SmartQRCode",
                "FlowBeamMiss tag=$tag n=$modules blackIsLow=$blackIsLow ph=%.2f,%.2f sc=%.2f unk=$unknownsAfterHints u0=$unknownsBeforeHints pick=${pick.size} beam=$beamWidth checks=$decodeChecks elapsedMs=%.0f budgetMs=%.0f".format(
                    bestPhaseX,
                    bestPhaseY,
                    bestScale,
                    (endNs - startNs) / 1_000_000.0,
                    budgetNs / 1_000_000.0
                )
            )
        }
        return null
    }

    private fun estimateBlackWhiteLevels(gray512: Mat): Pair<Double, Double> {
        val bytes = ByteArray(512 * 512)
        gray512.get(0, 0, bytes)
        val samples = IntArray(128 * 128)
        var k = 0
        for (y in 0 until 512 step 4) {
            val row = y * 512
            for (x in 0 until 512 step 4) {
                samples[k++] = bytes[row + x].toInt() and 0xFF
            }
        }
        java.util.Arrays.sort(samples)
        val lo = samples[(samples.size * 0.05).toInt().coerceIn(0, samples.lastIndex)].toDouble()
        val hi = samples[(samples.size * 0.95).toInt().coerceIn(0, samples.lastIndex)].toDouble()
        val black = min(lo, hi)
        val white = max(lo, hi)
        return black to white
    }

    private fun padAndRescale(gray512: Mat, pad: Int): Mat {
        val padded = Mat()
        Core.copyMakeBorder(gray512, padded, pad, pad, pad, pad, Core.BORDER_CONSTANT, Scalar(255.0))
        val out = Mat()
        Imgproc.resize(padded, out, Size(512.0, 512.0), 0.0, 0.0, Imgproc.INTER_AREA)
        padded.release()
        return out
    }

    private fun padAndRescaleBorder(gray512: Mat, pad: Int, borderType: Int): Mat {
        val padded = Mat()
        if (borderType == Core.BORDER_CONSTANT) {
            Core.copyMakeBorder(gray512, padded, pad, pad, pad, pad, borderType, Scalar(255.0))
        } else {
            Core.copyMakeBorder(gray512, padded, pad, pad, pad, pad, borderType)
        }
        val out = Mat()
        Imgproc.resize(padded, out, Size(512.0, 512.0), 0.0, 0.0, Imgproc.INTER_AREA)
        padded.release()
        return out
    }

    private fun applyStructureHintsIfUnknown(bits: ByteArray, modules: Int) {
        fun idx(x: Int, y: Int) = y * modules + x
        fun setIfUnknown(x: Int, y: Int, v: Int) {
            if (x !in 0 until modules || y !in 0 until modules) return
            val i = idx(x, y)
            if (bits[i].toInt() == 2) bits[i] = v.toByte()
        }

        fun applyFinderAt(sx: Int, sy: Int) {
            for (y in 0..6) {
                for (x in 0..6) {
                    val v = when {
                        x == 0 || x == 6 || y == 0 || y == 6 -> 1
                        x == 1 || x == 5 || y == 1 || y == 5 -> 0
                        else -> 1
                    }
                    setIfUnknown(sx + x, sy + y, v)
                }
            }
            for (i in 0..7) {
                setIfUnknown(sx + i, sy + 7, 0)
                setIfUnknown(sx + 7, sy + i, 0)
            }
        }

        applyFinderAt(0, 0)
        applyFinderAt(modules - 7, 0)
        applyFinderAt(0, modules - 7)

        val start = 8
        val end = modules - 9
        if (end >= start) {
            for (x in start..end) {
                val v = if (((x - start) and 1) == 0) 1 else 0
                setIfUnknown(x, 6, v)
            }
            for (y in start..end) {
                val v = if (((y - start) and 1) == 0) 1 else 0
                setIfUnknown(6, y, v)
            }
        }

        val version = ((modules - 21) / 4) + 1
        if (version >= 2) {
            val num = (version / 7) + 2
            val last = modules - 7
            val centers = IntArray(num)
            centers[0] = 6
            centers[num - 1] = last
            if (num > 2) {
                var step = ((last - 6) + (num - 2)) / (num - 1)
                if ((step and 1) == 1) step++
                for (i in 1 until num - 1) {
                    centers[i] = last - (num - 1 - i) * step
                }
            }

            fun applyAlignmentAt(cx: Int, cy: Int) {
                for (dy in -2..2) {
                    for (dx in -2..2) {
                        val v = when {
                            dx == 0 && dy == 0 -> 1
                            kotlin.math.abs(dx) == 2 || kotlin.math.abs(dy) == 2 -> 1
                            else -> 0
                        }
                        setIfUnknown(cx + dx, cy + dy, v)
                    }
                }
            }

            for (cy in centers) {
                for (cx in centers) {
                    val inTopLeft = cx == 6 && cy == 6
                    val inTopRight = cx == last && cy == 6
                    val inBottomLeft = cx == 6 && cy == last
                    if (inTopLeft || inTopRight || inBottomLeft) continue
                    applyAlignmentAt(cx, cy)
                }
            }
        }

        setIfUnknown(8, modules - 8, 1)
    }

    private fun renderModulesTo512(bits: ByteArray, modules: Int): Mat {
        val qz = 4
        val total = modules + qz * 2
        val small = Mat(total, total, CvType.CV_8UC1, Scalar(255.0))
        val row = ByteArray(modules)
        for (y in 0 until modules) {
            for (x in 0 until modules) {
                val v = bits[y * modules + x].toInt()
                row[x] = when (v) {
                    1 -> 0.toByte()
                    0 -> 255.toByte()
                    else -> 127.toByte()
                }
            }
            small.put(y + qz, qz, row)
        }
        val out = Mat()
        Imgproc.resize(small, out, Size(512.0, 512.0), 0.0, 0.0, Imgproc.INTER_NEAREST)
        small.release()
        return out
    }
}

data class RoiRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width: Int get() = (right - left).coerceAtLeast(0)
    val height: Int get() = (bottom - top).coerceAtLeast(0)
}

data class DecodeParsed(val text: String, val roi: RoiRect, val quad: FloatArray? = null)

data class DebugFrame(val step: String, val width: Int, val height: Int, val gray: ByteArray)

data class ScanOutput(
    val text: String?,
    val box: RoiRect?,
    val quad: FloatArray? = null,
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val rotationDegrees: Int = 0,
    val cropRect: android.graphics.Rect? = null
)

class RoiTracker {
    private var lastRoi: RoiRect? = null
    private var lastBoxRoi: RoiRect? = null
    private var lastSuccessTsNs: Long = 0
    private var lastBoxTsNs: Long = 0

    fun nextRois(width: Int, height: Int, frameIndex: Long, timestampNs: Long): List<RoiRect> {
        val current = lastRoi
        val useSuccess = current != null && (timestampNs - lastSuccessTsNs) <= 1_200_000_000L
        val useBox = !useSuccess && lastBoxRoi != null && (timestampNs - lastBoxTsNs) <= 900_000_000L
        val focus = when {
            useSuccess -> current
            useBox -> lastBoxRoi
            else -> null
        }

        val out = ArrayList<RoiRect>(4)
        if (focus != null) {
            val padX = (focus.width * 0.35f).toInt()
            val padY = (focus.height * 0.35f).toInt()
            out.add(
                RoiRect(
                    (focus.left - padX).coerceAtLeast(0),
                    (focus.top - padY).coerceAtLeast(0),
                    (focus.right + padX).coerceAtMost(width),
                    (focus.bottom + padY).coerceAtMost(height)
                )
            )
        }
        if (frameIndex % 10L == 0L) {
            out.add(RoiRect(0, 0, width, height))
        }

        val side = (minOf(width, height) * 0.72f).toInt()
        val top = (height - side) / 2
        val centerLeft = (width - side) / 2
        out.add(RoiRect(centerLeft, top, centerLeft + side, top + side))
        if (focus == null || frameIndex % 10L == 0L) {
            out.add(RoiRect(0, top, side.coerceAtMost(width), top + side))
            out.add(RoiRect((width - side).coerceAtLeast(0), top, width, top + side))
        }

        val unique = LinkedHashMap<Long, RoiRect>()
        for (r in out) {
            val key = (r.left.toLong() shl 48) xor (r.top.toLong() shl 32) xor (r.right.toLong() shl 16) xor r.bottom.toLong()
            unique[key] = r
        }
        return unique.values.toList()
    }

    fun onDecoded(roi: RoiRect, timestampNs: Long) {
        lastRoi = roi
        lastSuccessTsNs = timestampNs
    }

    fun onBox(roi: RoiRect, timestampNs: Long) {
        lastBoxRoi = roi
        lastBoxTsNs = timestampNs
    }

    fun reset() {
        lastRoi = null
        lastBoxRoi = null
        lastSuccessTsNs = 0
        lastBoxTsNs = 0
    }
}

data class RoiQuality(val sharpness: Double, val overexposedRatio: Double, val contrast: Double) {
    companion object {
        fun score(gray512: Mat): RoiQuality {
            val lap = Mat()
            Imgproc.Laplacian(gray512, lap, CvType.CV_16S, 3, 1.0, 0.0)
            val lapF = Mat()
            lap.convertTo(lapF, CvType.CV_32F)
            val mean = MatOfDouble()
            val std = MatOfDouble()
            Core.meanStdDev(lapF, mean, std)
            val sharpness = std.toArray().firstOrNull()?.let { it * it } ?: 0.0

            val mask = Mat()
            Core.compare(gray512, Scalar(245.0), mask, Core.CMP_GT)
            val over = Core.countNonZero(mask).toDouble() / (512.0 * 512.0)

            val mean2 = MatOfDouble()
            val std2 = MatOfDouble()
            Core.meanStdDev(gray512, mean2, std2)
            val contrast = std2.toArray().firstOrNull() ?: 0.0

            lap.release()
            lapF.release()
            mask.release()
            mean.release()
            std.release()
            mean2.release()
            std2.release()

            return RoiQuality(sharpness, over, contrast)
        }
    }
}

data class RoiFrame(val gray: Mat, val timestampNs: Long, val roi: RoiRect, val quality: RoiQuality)

class RoiRingBuffer(private val capacity: Int) {
    private val frames = ArrayList<RoiFrame>(capacity)

    @Synchronized
    fun push(frame: RoiFrame) {
        if (frames.size == capacity) {
            val old = frames.removeAt(0)
            old.gray.release()
        }
        frames.add(frame)
    }

    @Synchronized
    fun topK(k: Int): List<RoiFrame> {
        return frames
            .sortedWith(
                compareByDescending<RoiFrame> { it.quality.sharpness }
                    .thenBy { it.quality.overexposedRatio }
                    .thenByDescending { it.quality.contrast }
            )
            .take(k)
    }

    @Synchronized
    fun clear() {
        for (f in frames) {
            f.gray.release()
        }
        frames.clear()
    }
}
