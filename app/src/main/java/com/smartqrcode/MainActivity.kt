package com.smartqrcode

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.FrameLayout
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.smartqrcode.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Core
import org.opencv.core.Size as CvSize
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.QRCodeDetector
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

@ExperimentalGetImage
class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private lateinit var pipeline: ScanPipeline
    private lateinit var debugView: DebugCanvasView
    private val toneGenerator = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80)
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null

    private var firstFrameTsNs: Long = 0
    private var lastSeenBoxTsNs: Long = Long.MIN_VALUE
    private var boxNoDecodeStartNs: Long = 0
    private var lastStatusText: String? = null
    private var lastFlowText: String? = null
    private var lastFlowRaw: String? = null
    private var lastFlowStageKey: String? = null
    private var lastFlowStageStartNs: Long = 0
    private var lastDecodedText: String? = null
    private var lastDecodedAtNs: Long = 0
    private var lastBeepTsNs: Long = 0
    private var lastProgressUpdateNs: Long = 0
    private var lastFailureKey: Long = 0
    private var lastMode: UiMode = UiMode.Empty
    private var lastBoxSignature: Long = 0
    @Volatile private var canAnalyze: Boolean = false
    @Volatile private var focusSettled: Boolean = false
    @Volatile private var focusSettledAtNs: Long = 0
    private var cameraStartNs: Long = 0
    private val mainHandler = Handler(Looper.getMainLooper())
    private var lastSavedDebugAtNs: Long = 0
    private val flowTicker = object : Runnable {
        override fun run() {
            renderFlowText()
            mainHandler.postDelayed(this, 200L)
        }
    }

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            binding.resultText.text = "Camera permission denied"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        OpenCvSupport.ready = OpenCVLoader.initDebug()
        debugView = DebugCanvasView(this)
        binding.debugContainer.removeAllViews()
        binding.debugContainer.addView(
            debugView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        pipeline = ScanPipeline().also { p ->
            p.debugCallback = { frame ->
                runOnUiThread {
                    debugView.update(frame)
                }
                maybeSaveDebugFrame(frame)
            }
            p.statusCallback = { status ->
                runOnUiThread { onPipelineStatus(status) }
            }
        }
        binding.previewView.setOnClickListener { onUserForceHard() }
        binding.debugContainer.isClickable = true
        binding.debugContainer.setOnClickListener { onUserForceHard() }
        clearUiAndState()
        mainHandler.removeCallbacks(flowTicker)
        mainHandler.post(flowTicker)

        val debugPath = intent?.getStringExtra("debug_path")
        if (!debugPath.isNullOrBlank()) {
            binding.resultText.text = "Debug decoding…"
            Log.i("SmartQRCode", "DebugDecode start path=$debugPath")
            analysisExecutor.execute { runDebugDecodeFromFile(debugPath) }
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        val debugPath = intent.getStringExtra("debug_path")
        if (!debugPath.isNullOrBlank()) {
            binding.resultText.text = "Debug decoding…"
            Log.i("SmartQRCode", "DebugDecode start path=$debugPath")
            analysisExecutor.execute { runDebugDecodeFromFile(debugPath) }
        }
    }

    private fun runDebugDecodeFromFile(path: String) {
        try {
            val bmp = BitmapFactory.decodeFile(path)
            if (bmp == null) {
                Log.i("SmartQRCode", "DebugDecode fail reason=decodeFileNull path=$path")
                runOnUiThread {
                    binding.resultText.text = "Debug decode failed: decodeFile null"
                }
                return
            }
            val attempts = ArrayList<Pair<String, Bitmap>>(3)
            attempts.add("fit512" to Bitmap.createScaledBitmap(bmp, 512, 512, true))
            if (maxOf(bmp.width, bmp.height) > 1400) {
                val scale = 1400.0 / maxOf(bmp.width, bmp.height).toDouble()
                val w = (bmp.width * scale).roundToInt().coerceAtLeast(1)
                val h = (bmp.height * scale).roundToInt().coerceAtLeast(1)
                attempts.add("down1400" to Bitmap.createScaledBitmap(bmp, w, h, true))
            }
            attempts.add("orig" to bmp)

            var bestText: String? = null
            var bestMeta: String? = null
            fun toGray(b: Bitmap): Pair<ByteArray, Pair<Int, Int>> {
                val w = b.width
                val h = b.height
                val pixels = IntArray(w * h)
                b.getPixels(pixels, 0, w, 0, 0, w, h)
                val gray = ByteArray(w * h)
                for (i in pixels.indices) {
                    val c = pixels[i]
                    val r = (c shr 16) and 0xFF
                    val g = (c shr 8) and 0xFF
                    val bb = c and 0xFF
                    val y = (0.299 * r + 0.587 * g + 0.114 * bb).roundToInt().coerceIn(0, 255)
                    gray[i] = y.toByte()
                }
                return gray to (w to h)
            }

            fun tryNative(gray: ByteArray, w: Int, h: Int, rot: Int, l: Int, t: Int, r: Int, b: Int, tryHarder: Boolean, name: String): String? {
                val raw = NativeDecoder.decodeGray(gray, w, h, rot, l, t, r, b, tryHarder)
                val parsed = raw?.let { NativeDecoder.parseResult(it) }
                val text = parsed?.text?.takeIf { it.isNotBlank() && !it.startsWith("INVALID(") }
                if (text != null) {
                    bestMeta = "name=$name rot=$rot th=$tryHarder roi=$l,$t,$r,$b"
                }
                return text
            }

            fun roiCandidates(w: Int, h: Int): List<RoiRect> {
                val out = ArrayList<RoiRect>(64)
                out.add(RoiRect(0, 0, w, h))
                val side = (minOf(w, h) * 0.78f).roundToInt()
                val cx = w / 2
                val cy = h / 2
                out.add(RoiRect((cx - side / 2).coerceAtLeast(0), (cy - side / 2).coerceAtLeast(0), (cx + side / 2).coerceAtMost(w), (cy + side / 2).coerceAtMost(h)))
                out.add(RoiRect(0, 0, w, (h * 0.60f).roundToInt().coerceIn(1, h)))
                out.add(RoiRect(0, (h * 0.40f).roundToInt().coerceIn(0, h - 1), w, h))

                val wins = intArrayOf(
                    (minOf(w, h) * 0.88f).roundToInt(),
                    (minOf(w, h) * 0.72f).roundToInt(),
                    (minOf(w, h) * 0.56f).roundToInt(),
                    (minOf(w, h) * 0.42f).roundToInt(),
                    (minOf(w, h) * 0.30f).roundToInt()
                ).map { it.coerceAtLeast(220) }.distinct()
                for (win in wins) {
                    val stepX = (win * 0.42f).roundToInt().coerceAtLeast(1)
                    val stepY = (win * 0.32f).roundToInt().coerceAtLeast(1)
                    var y = 0
                    while (y + win <= h) {
                        var x = 0
                        while (x + win <= w) {
                            out.add(RoiRect(x, y, x + win, y + win))
                            x += stepX
                        }
                        y += stepY
                    }
                }

                val unique = LinkedHashMap<Long, RoiRect>()
                for (rr in out) {
                    val key = (rr.left.toLong() shl 48) xor (rr.top.toLong() shl 32) xor (rr.right.toLong() shl 16) xor rr.bottom.toLong()
                    unique[key] = rr
                }
                return unique.values.toList()
            }

            for ((name, b) in attempts) {
                val (gray, dim) = toGray(b)
                val w = dim.first
                val h = dim.second

                val dbg = NativeDecoder.decodeGrayDebug(gray, w, h, 0, 0, 0, w, h, false)
                Log.i("SmartQRCode", "DebugDecode dbg name=$name $dbg")

                if (bestText == null && OpenCvSupport.ready && maxOf(w, h) <= 1600) {
                    val m = Mat(h, w, CvType.CV_8UC1)
                    try {
                        m.put(0, 0, gray)
                        val detector = QRCodeDetector()
                        val opencvText = detector.detectAndDecode(m)
                        if (!opencvText.isNullOrBlank()) {
                            bestText = opencvText
                            bestMeta = "opencvDecode name=$name"
                            Log.i("SmartQRCode", "DebugDecode hit $bestMeta text=$bestText")
                            break
                        }
                        val recovered = pipeline.debugRecoverFromGray512(m, "DebugEarly-$name")
                        if (!recovered.isNullOrBlank()) {
                            bestText = recovered
                            Log.i("SmartQRCode", "DebugDecode recoverEarly name=$name text=$recovered")
                            break
                        }
                    } finally {
                        m.release()
                    }
                }

                fun orderQuad(quad: FloatArray): FloatArray {
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

                fun warpTo512(grayBytes: ByteArray, w: Int, h: Int, quad: FloatArray): Mat? {
                    if (!OpenCvSupport.ready) return null
                    if (quad.size != 8) return null
                    val ordered = orderQuad(quad)
                    val src = Mat(h, w, CvType.CV_8UC1)
                    try {
                        src.put(0, 0, grayBytes)
                        val srcPts = org.opencv.core.MatOfPoint2f(
                            org.opencv.core.Point(ordered[0].toDouble(), ordered[1].toDouble()),
                            org.opencv.core.Point(ordered[2].toDouble(), ordered[3].toDouble()),
                            org.opencv.core.Point(ordered[4].toDouble(), ordered[5].toDouble()),
                            org.opencv.core.Point(ordered[6].toDouble(), ordered[7].toDouble())
                        )
                        val dstPts = org.opencv.core.MatOfPoint2f(
                            org.opencv.core.Point(0.0, 0.0),
                            org.opencv.core.Point(511.0, 0.0),
                            org.opencv.core.Point(511.0, 511.0),
                            org.opencv.core.Point(0.0, 511.0)
                        )
                        val hMat = Imgproc.getPerspectiveTransform(srcPts, dstPts)
                        val out = Mat()
                        Imgproc.warpPerspective(src, out, hMat, CvSize(512.0, 512.0), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, org.opencv.core.Scalar(255.0))
                        srcPts.release()
                        dstPts.release()
                        hMat.release()
                        return out
                    } finally {
                        src.release()
                    }
                }

                for (rot in intArrayOf(0, 90, 180, 270)) {
                    val raw = NativeDecoder.decodeGray(gray, w, h, rot, 0, 0, w, h, true)
                    val parsed = raw?.let { NativeDecoder.parseResult(it) }
                    if (parsed != null) {
                        val text = parsed.text.takeIf { it.isNotBlank() }
                        if (text != null) {
                            bestText = text
                            bestMeta = "name=$name rot=$rot th=true roi=0,0,$w,$h"
                            break
                        }
                        if (parsed.quad != null) {
                            val rectified = warpTo512(gray, w, h, parsed.quad)
                            if (rectified != null) {
                                try {
                                    val rectBytes = ByteArray(512 * 512)
                                    rectified.get(0, 0, rectBytes)
                                    val rectRaw = NativeDecoder.decodeGray(rectBytes, 512, 512, 0, 0, 0, 512, 512, true)
                                    val rectParsed = rectRaw?.let { NativeDecoder.parseResult(it) }
                                    val rectText = rectParsed?.text?.takeIf { it.isNotBlank() }
                                    if (rectText != null) {
                                        bestText = rectText
                                        bestMeta = "nativeRect name=$name rot=$rot th=true"
                                        break
                                    }
                                    val recovered = pipeline.debugRecoverFromGray512(rectified, "DebugHint-$name-r$rot")
                                    if (!recovered.isNullOrBlank()) {
                                        bestText = recovered
                                        bestMeta = "recover name=$name rot=$rot"
                                        break
                                    }
                                } finally {
                                    rectified.release()
                                }
                            }
                        }
                    }
                }
                if (bestText != null) {
                    Log.i("SmartQRCode", "DebugDecode hit $bestMeta text=$bestText")
                    break
                }

                val rois = ArrayList<RoiRect>()
                val primary = ArrayList<RoiRect>(6)
                if (OpenCvSupport.ready) {
                    val m = Mat(h, w, CvType.CV_8UC1)
                    try {
                        m.put(0, 0, gray)
                        val detector = QRCodeDetector()
                        val points = Mat()
                        try {
                            val ok = detector.detect(m, points)
                            if (ok && !points.empty() && points.total() >= 4L) {
                                val pts = FloatArray(8)
                                points.get(0, 0, pts)
                                val rectified = warpTo512(gray, w, h, pts)
                                if (rectified != null) {
                                    try {
                                        val rectBytes = ByteArray(512 * 512)
                                        rectified.get(0, 0, rectBytes)
                                        val rectRaw = NativeDecoder.decodeGray(rectBytes, 512, 512, 0, 0, 0, 512, 512, true)
                                        val rectParsed = rectRaw?.let { NativeDecoder.parseResult(it) }
                                        val rectText = rectParsed?.text?.takeIf { it.isNotBlank() }
                                        if (rectText != null) {
                                            bestText = rectText
                                            bestMeta = "opencvWarp name=$name th=true"
                                        } else {
                                            val recovered = pipeline.debugRecoverFromGray512(rectified, "DebugOpenCvWarp-$name")
                                            if (!recovered.isNullOrBlank()) {
                                                bestText = recovered
                                                bestMeta = "opencvWarpRecover name=$name"
                                            }
                                        }
                                    } finally {
                                        rectified.release()
                                    }
                                }
                                var minX = Float.POSITIVE_INFINITY
                                var minY = Float.POSITIVE_INFINITY
                                var maxX = Float.NEGATIVE_INFINITY
                                var maxY = Float.NEGATIVE_INFINITY
                                for (i in 0 until 4) {
                                    val x = pts[i * 2]
                                    val y = pts[i * 2 + 1]
                                    minX = minOf(minX, x)
                                    minY = minOf(minY, y)
                                    maxX = maxOf(maxX, x)
                                    maxY = maxOf(maxY, y)
                                }
                                val bw = (maxX - minX).coerceAtLeast(1f)
                                val bh = (maxY - minY).coerceAtLeast(1f)
                                val padX = (bw * 0.25f).roundToInt().coerceAtLeast(12)
                                val padY = (bh * 0.25f).roundToInt().coerceAtLeast(12)
                                val l = (minX.roundToInt() - padX).coerceIn(0, w - 1)
                                val t = (minY.roundToInt() - padY).coerceIn(0, h - 1)
                                val r = (maxX.roundToInt() + padX).coerceIn(l + 1, w)
                                val bb = (maxY.roundToInt() + padY).coerceIn(t + 1, h)
                                val rr = RoiRect(l, t, r, bb)
                                rois.add(rr)
                                primary.add(rr)
                                Log.i("SmartQRCode", "DebugDecode opencvDetect name=$name roi=$l,$t,$r,$bb")
                            }
                        } finally {
                            points.release()
                        }
                    } finally {
                        m.release()
                    }
                }
                val baseRois = roiCandidates(w, h)
                rois.addAll(baseRois)
                primary.add(baseRois.first())
                if (baseRois.size >= 2) primary.add(baseRois[1])
                val rots = intArrayOf(0, 90, 180, 270)

                fun scan(tryHarder: Boolean, roiList: List<RoiRect>, budget: Int) {
                    var checked = 0
                    for (roi in roiList) {
                        for (rot in rots) {
                            bestText = tryNative(gray, w, h, rot, roi.left, roi.top, roi.right, roi.bottom, tryHarder, name)
                            checked++
                            if (bestText != null) return
                            if (checked % 40 == 0) {
                                Log.i("SmartQRCode", "DebugDecode prog name=$name th=$tryHarder checked=$checked lastRoi=${roi.left},${roi.top},${roi.right},${roi.bottom}")
                            }
                            if (checked >= budget) return
                        }
                    }
                }

                val uniquePrimary = LinkedHashMap<Long, RoiRect>()
                for (rr in primary) {
                    val key = (rr.left.toLong() shl 48) xor (rr.top.toLong() shl 32) xor (rr.right.toLong() shl 16) xor rr.bottom.toLong()
                    uniquePrimary[key] = rr
                }
                val primaryList = uniquePrimary.values.toList()

                scan(tryHarder = false, roiList = primaryList + rois, budget = 220)
                if (bestText == null) {
                    val topHard = ArrayList<RoiRect>()
                    topHard.addAll(primaryList)
                    topHard.addAll(rois.take(60))
                    scan(tryHarder = true, roiList = topHard, budget = 120)
                }
                if (bestText != null) {
                    Log.i("SmartQRCode", "DebugDecode hit $bestMeta text=$bestText")
                    break
                }

                if (OpenCvSupport.ready && w == 512 && h == 512) {
                    val m = Mat(512, 512, CvType.CV_8UC1)
                    try {
                        m.put(0, 0, gray)
                        val recovered = pipeline.debugRecoverFromGray512(m, "Debug-$name")
                        if (!recovered.isNullOrBlank()) {
                            bestText = recovered
                            Log.i("SmartQRCode", "DebugDecode recover name=$name text=$recovered")
                            break
                        }
                    } finally {
                        m.release()
                    }
                }
            }

            if (!bestText.isNullOrBlank()) {
                runOnUiThread {
                    binding.resultText.text = bestText
                    toneGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 160)
                }
            } else {
                runOnUiThread {
                    binding.resultText.text = "Debug decode: no result"
                }
            }
        } catch (t: Throwable) {
            Log.i("SmartQRCode", "DebugDecode crash ${t::class.java.simpleName}:${t.message}")
            runOnUiThread {
                binding.resultText.text = "Debug decode crashed"
            }
        }
    }

    private fun maybeSaveDebugFrame(frame: DebugFrame) {
        if (frame.width != 512 || frame.height != 512) return
        val step = frame.step
        val allow = step.startsWith("ROI-512") ||
            step.startsWith("Warp") ||
            step.startsWith("Fuse-512")
        if (!allow) return
        val nowNs = SystemClock.elapsedRealtimeNanos()
        if ((nowNs - lastSavedDebugAtNs) < 2_000_000_000L) return
        lastSavedDebugAtNs = nowNs
        val bytes = frame.gray.copyOf()
        analysisExecutor.execute {
            try {
                val dir = getExternalFilesDir(null) ?: filesDir
                val file = File(dir, "last_512.png")
                val w = frame.width
                val h = frame.height
                val pixels = IntArray(w * h)
                for (i in 0 until w * h) {
                    val v = bytes[i].toInt() and 0xFF
                    pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                }
                val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                bmp.setPixels(pixels, 0, w, 0, 0, w, h)
                FileOutputStream(file).use { out ->
                    bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
                bmp.recycle()
                Log.i("SmartQRCode", "DebugFrameSaved path=${file.absolutePath} step=$step")
            } catch (t: Throwable) {
                Log.i("SmartQRCode", "DebugFrameSaved fail ${t::class.java.simpleName}:${t.message}")
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                this.cameraProvider = cameraProvider
                canAnalyze = false
                focusSettled = false
                cameraStartNs = SystemClock.elapsedRealtimeNanos()
                val preview = Preview.Builder().build().also { p ->
                    p.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(Size(1920, 1080))
                    .build()

                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    if (!canAnalyze) {
                        val nowNs = SystemClock.elapsedRealtimeNanos()
                        val elapsedNs = nowNs - cameraStartNs
                        val focusAgeOk = focusSettledAtNs != 0L && (nowNs - focusSettledAtNs) >= 400_000_000L
                        val gateOk = (elapsedNs >= 1_800_000_000L && focusSettled && focusAgeOk) || elapsedNs >= 4_000_000_000L
                        if (gateOk) {
                            canAnalyze = true
                            Log.i("SmartQRCode", "FlowGate ready elapsedMs=${(elapsedNs / 1_000_000L)} focus=$focusSettled")
                        } else {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                    }
                    val ts = imageProxy.imageInfo.timestamp
                    val out = pipeline.process(imageProxy)
                    runOnUiThread {
                        handleFrame(ts, out)
                        updateDebugBox(out)
                    }
                }

                cameraProvider.unbindAll()
                val groupBuilder = UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(analysis)
                binding.previewView.viewPort?.let { vp ->
                    groupBuilder.setViewPort(vp)
                }
                val camera = cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, groupBuilder.build())
                this.camera = camera
                triggerStartupAutoFocus(camera)
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun onUserForceHard() {
        pipeline.requestUserHard(90)
        val cam = camera ?: return
        triggerStartupAutoFocus(cam)
        Log.i("SmartQRCode", "UserForceHard tap")
    }

    private fun triggerStartupAutoFocus(camera: Camera) {
        fun attempt(deadlineNs: Long) {
            val view = binding.previewView
            if (view.width <= 0 || view.height <= 0) {
                view.post { attempt(deadlineNs) }
                return
            }
            val point = view.meteringPointFactory.createPoint(view.width / 2f, view.height / 2f)
            val action = FocusMeteringAction.Builder(
                point,
                FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE or FocusMeteringAction.FLAG_AWB
            ).setAutoCancelDuration(3, TimeUnit.SECONDS).build()
            val future = camera.cameraControl.startFocusAndMetering(action)
            future.addListener(
                {
                    val ok = try {
                        future.get().isFocusSuccessful
                    } catch (_: Throwable) {
                        false
                    }
                    if (ok) {
                        focusSettled = true
                        focusSettledAtNs = SystemClock.elapsedRealtimeNanos()
                        Log.i("SmartQRCode", "FlowGate afOk")
                    } else if (!focusSettled && SystemClock.elapsedRealtimeNanos() < deadlineNs) {
                        mainHandler.postDelayed({ attempt(deadlineNs) }, 350L)
                    }
                },
                ContextCompat.getMainExecutor(this)
            )
        }

        binding.previewView.post {
            focusSettled = false
            focusSettledAtNs = 0
            val deadlineNs = SystemClock.elapsedRealtimeNanos() + 2_500_000_000L
            attempt(deadlineNs)
            mainHandler.postDelayed(
                {
                    if (!focusSettled) {
                        focusSettled = true
                        focusSettledAtNs = SystemClock.elapsedRealtimeNanos()
                        Log.i("SmartQRCode", "FlowGate afTimeout")
                    }
                },
                2500L
            )
        }
    }

    override fun onBackPressed() {
        pipeline.debugCallback = null
        cameraProvider?.unbindAll()
        cameraProvider = null
        camera = null
        clearUiAndState()
        super.onBackPressed()
        finishAndRemoveTask()
    }

    private fun updateDebugBox(out: ScanOutput?) {
        val rect = out?.box ?: run {
            debugView.setBoxRect(null)
            return
        }
        val (bmpW, bmpH) = debugView.bitmapSize() ?: run {
            debugView.setBoxRect(null)
            return
        }
        val rot = ((out.rotationDegrees % 360) + 360) % 360
        val rotW = if (rot == 90 || rot == 270) out.frameHeight else out.frameWidth
        val rotH = if (rot == 90 || rot == 270) out.frameWidth else out.frameHeight
        if (rotW <= 0 || rotH <= 0) {
            debugView.setBoxRect(null)
            return
        }
        val boxF = RectF(rect.left.toFloat(), rect.top.toFloat(), rect.right.toFloat(), rect.bottom.toFloat())
        val rotBox = rotateRect(boxF, out.frameWidth, out.frameHeight, out.rotationDegrees)
        val sx = bmpW.toFloat() / rotW.toFloat()
        val sy = bmpH.toFloat() / rotH.toFloat()
        val mapped = RectF(
            rotBox.left * sx,
            rotBox.top * sy,
            rotBox.right * sx,
            rotBox.bottom * sy
        )
        mapped.intersect(0f, 0f, bmpW.toFloat(), bmpH.toFloat())
        debugView.setBoxRect(mapped)
    }

    private fun handleFrame(timestampNs: Long, out: ScanOutput?) {
        if (firstFrameTsNs == 0L) firstFrameTsNs = timestampNs

        val box = out?.box
        val hasBox = box != null
        val hasText = !out?.text.isNullOrBlank()

        if (hasBox) {
            lastSeenBoxTsNs = timestampNs
            val sig = computeBoxSignature(box!!, out.frameWidth, out.frameHeight)
            if (sig != lastBoxSignature) {
                lastBoxSignature = sig
                boxNoDecodeStartNs = timestampNs
                lastProgressUpdateNs = 0L
                lastFailureKey = 0L
            }
            if (!hasText && boxNoDecodeStartNs == 0L) {
                boxNoDecodeStartNs = timestampNs
            }
        } else {
            lastBoxSignature = 0L
            boxNoDecodeStartNs = 0L
        }

        if (hasText) {
            val text = out?.text.orEmpty()
            val isInvalid = text.startsWith("INVALID(")
            if (!isInvalid) {
                Log.i("SmartQRCode", "Decoded: $text")
                lastDecodedAtNs = timestampNs
                if (text != lastDecodedText || (timestampNs - lastBeepTsNs) > 1_000_000_000L) {
                    toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
                    lastBeepTsNs = timestampNs
                }
                lastDecodedText = text
                setResultText(text)
                lastMode = UiMode.Decoded
                boxNoDecodeStartNs = 0L
                return
            }

            setResultText("已定位到二维码，但解码失败：$text")
            lastMode = UiMode.DecodeFailed
            return
        }

        if (!lastDecodedText.isNullOrBlank() && lastDecodedAtNs != 0L && (timestampNs - lastDecodedAtNs) < 3_000_000_000L) {
            setResultText(lastDecodedText!!)
            lastMode = UiMode.Decoded
            return
        }

        if (!hasBox && lastDecodedText != null && lastSeenBoxTsNs != Long.MIN_VALUE) {
            if ((timestampNs - lastSeenBoxTsNs) > 1_200_000_000L && (timestampNs - lastDecodedAtNs) >= 3_000_000_000L) {
                lastDecodedText = null
                lastDecodedAtNs = 0
                setResultText("")
                lastMode = UiMode.Empty
            }
        }

        if (hasBox && boxNoDecodeStartNs != 0L) {
            val elapsedNs = timestampNs - boxNoDecodeStartNs
            if (elapsedNs < 10_000_000_000L) {
                if ((timestampNs - lastProgressUpdateNs) > 300_000_000L || lastMode != UiMode.Decoding) {
                    val sec = elapsedNs / 1_000_000_000.0
                    setResultText(String.format("已定位二维码，正在解码… %.1fs", sec))
                    lastProgressUpdateNs = timestampNs
                    lastMode = UiMode.Decoding
                }
            } else {
                val reason = buildDecodeFailureReason(out!!)
                val key = computeFailureKey(out)
                if (lastMode != UiMode.DecodeFailed || key != lastFailureKey) {
                    setResultText(reason)
                    lastFailureKey = key
                    lastMode = UiMode.DecodeFailed
                }
            }
            return
        }

        if (!hasBox) {
            if (firstFrameTsNs != 0L && lastSeenBoxTsNs == Long.MIN_VALUE && (timestampNs - firstFrameTsNs) > 10_000_000_000L) {
                if (lastMode != UiMode.NoQr) {
                    setResultText("10秒未找到二维码，请将二维码置于画面中间并保持稳定")
                    lastMode = UiMode.NoQr
                }
            } else {
                if (lastMode != UiMode.Searching && lastDecodedText.isNullOrBlank()) {
                    setResultText("扫描中…")
                    lastMode = UiMode.Searching
                }
            }
        }
    }

    private fun buildDecodeFailureReason(out: ScanOutput): String {
        val box = out.box ?: return "已定位到二维码，但10秒未解码成功：可能被裁切、反光、模糊或对比不足"
        val w = (box.right - box.left).coerceAtLeast(1)
        val h = (box.bottom - box.top).coerceAtLeast(1)
        val areaRatio = (w.toLong() * h.toLong()).toDouble() /
            (out.frameWidth.toLong().coerceAtLeast(1) * out.frameHeight.toLong().coerceAtLeast(1)).toDouble()

        val reasons = ArrayList<String>(5)
        if (areaRatio < 0.03) reasons.add("二维码太小/距离太远")
        if (areaRatio > 0.75) reasons.add("二维码太近/超出取景范围")
        val marginX = (out.frameWidth * 0.02).toInt().coerceAtLeast(8)
        val marginY = (out.frameHeight * 0.02).toInt().coerceAtLeast(8)
        if (box.left <= marginX || box.top <= marginY || box.right >= out.frameWidth - marginX || box.bottom >= out.frameHeight - marginY) {
            reasons.add("二维码贴边被裁切/白边不足")
        }
        reasons.add("可能有反光、黑底反色、模糊或定位符残缺")
        return "已定位到二维码，但10秒未解码成功：${reasons.distinct().joinToString("，")}"
    }

    private fun computeFailureKey(out: ScanOutput): Long {
        val b = out.box ?: return 0
        val w = (b.right - b.left).coerceAtLeast(0)
        val h = (b.bottom - b.top).coerceAtLeast(0)
        val edge = ((b.left <= 8) || (b.top <= 8) || (b.right >= out.frameWidth - 8) || (b.bottom >= out.frameHeight - 8))
        var k = 0L
        k = (k shl 16) xor (w.toLong() and 0xFFFF)
        k = (k shl 16) xor (h.toLong() and 0xFFFF)
        k = (k shl 1) xor (if (edge) 1L else 0L)
        return k
    }

    private fun computeBoxSignature(b: RoiRect, frameW: Int, frameH: Int): Long {
        val fw = frameW.coerceAtLeast(1)
        val fh = frameH.coerceAtLeast(1)
        val cx = (((b.left + b.right) * 0.5 / fw) * 1023.0).roundToInt().coerceIn(0, 1023)
        val cy = (((b.top + b.bottom) * 0.5 / fh) * 1023.0).roundToInt().coerceIn(0, 1023)
        val w = (((b.right - b.left).coerceAtLeast(0).toDouble() / fw) * 1023.0).roundToInt().coerceIn(0, 1023)
        val h = (((b.bottom - b.top).coerceAtLeast(0).toDouble() / fh) * 1023.0).roundToInt().coerceIn(0, 1023)
        var k = 0L
        k = (k shl 10) xor cx.toLong()
        k = (k shl 10) xor cy.toLong()
        k = (k shl 10) xor w.toLong()
        k = (k shl 10) xor h.toLong()
        return k
    }

    private fun setResultText(text: String) {
        if (text == lastStatusText) return
        binding.resultText.text = text
        lastStatusText = text
    }

    private fun setFlowText(text: String) {
        binding.flowText.text = text
        lastFlowText = text
    }

    private fun extractStageKey(status: String): String? {
        val idx = status.indexOf("stage=")
        if (idx < 0) return null
        val start = idx + 6
        val end = status.indexOf(' ', start).let { if (it < 0) status.length else it }
        if (end <= start) return null
        return status.substring(start, end)
    }

    private fun onPipelineStatus(status: String) {
        lastFlowRaw = status
        val stageKey = extractStageKey(status)
        val nowNs = SystemClock.elapsedRealtimeNanos()
        if (stageKey != null && stageKey != lastFlowStageKey) {
            lastFlowStageKey = stageKey
            lastFlowStageStartNs = nowNs
        } else if (lastFlowStageStartNs == 0L) {
            lastFlowStageStartNs = nowNs
        }
        renderFlowText()
    }

    private fun renderFlowText() {
        val raw = lastFlowRaw ?: return
        val startNs = lastFlowStageStartNs
        val elapsedS = if (startNs != 0L) (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000_000.0 else 0.0
        val text = String.format("%s | %.1fs", raw, elapsedS)
        if (text == lastFlowText) return
        setFlowText(text)
    }

    private fun clearUiAndState() {
        firstFrameTsNs = 0
        lastSeenBoxTsNs = Long.MIN_VALUE
        boxNoDecodeStartNs = 0
        lastStatusText = null
        lastFlowText = null
        lastFlowRaw = null
        lastFlowStageKey = null
        lastFlowStageStartNs = 0
        lastDecodedText = null
        lastDecodedAtNs = 0
        lastBeepTsNs = 0
        lastProgressUpdateNs = 0
        lastFailureKey = 0
        lastMode = UiMode.Empty
        setResultText("")
        setFlowText("")
        debugView.setBoxRect(null)
        debugView.clear()
        if (::pipeline.isInitialized) pipeline.reset()
    }

    override fun onStop() {
        super.onStop()
        pipeline.debugCallback = null
        pipeline.statusCallback = null
        mainHandler.removeCallbacks(flowTicker)
        cameraProvider?.unbindAll()
        cameraProvider = null
        camera = null
        clearUiAndState()
        if (!isChangingConfigurations) {
            finishAndRemoveTask()
        }
    }

    private fun rotateRect(r: RectF, w: Int, h: Int, rot: Int): RectF {
        val rr = ((rot % 360) + 360) % 360
        return when (rr) {
            90 -> RectF(r.top, w - r.right, r.bottom, w - r.left)
            180 -> RectF(w - r.right, h - r.bottom, w - r.left, h - r.top)
            270 -> RectF(h - r.bottom, r.left, h - r.top, r.right)
            else -> RectF(r)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        toneGenerator.release()
        analysisExecutor.shutdown()
    }

    private enum class UiMode {
        Empty,
        Searching,
        NoQr,
        Decoding,
        DecodeFailed,
        Decoded
    }

    private class DebugCanvasView(context: android.content.Context) : View(context) {
        private val imagePaint = Paint()
        private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 34f
            style = Paint.Style.FILL
        }
        private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xA0000000.toInt()
            style = Paint.Style.FILL
        }
        private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }

        private var step: String = ""
        private var bitmap: Bitmap? = null
        private var argbBuffer: IntArray? = null
        private var boxRect: RectF? = null

        fun setBoxRect(r: RectF?) {
            boxRect = r
            invalidate()
        }

        fun bitmapSize(): Pair<Int, Int>? {
            val bmp = bitmap ?: return null
            return bmp.width to bmp.height
        }

        fun update(frame: DebugFrame) {
            step = frame.step
            val w = frame.width
            val h = frame.height
            val bmp = bitmap
            if (bmp == null || bmp.width != w || bmp.height != h) {
                bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                argbBuffer = IntArray(w * h)
            }
            val out = argbBuffer ?: return
            val gray = frame.gray
            val n = minOf(out.size, gray.size)
            for (i in 0 until n) {
                val v = gray[i].toInt() and 0xFF
                out[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
            bitmap?.setPixels(out, 0, w, 0, 0, w, h)
            invalidate()
        }

        fun clear() {
            step = ""
            bitmap = null
            argbBuffer = null
            boxRect = null
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            canvas.drawColor(Color.BLACK)
            val bmp = bitmap
            if (bmp != null) {
                val dst = Rect(0, 0, width, height)
                canvas.drawBitmap(bmp, null, dst, imagePaint)
            }
            val r = boxRect
            val bmpW = bmp?.width ?: 0
            val bmpH = bmp?.height ?: 0
            if (r != null && bmpW > 0 && bmpH > 0) {
                val sx = width.toFloat() / bmpW.toFloat()
                val sy = height.toFloat() / bmpH.toFloat()
                val rr = RectF(r.left * sx, r.top * sy, r.right * sx, r.bottom * sy)
                rr.intersect(0f, 0f, width.toFloat(), height.toFloat())
                canvas.drawRect(rr, boxPaint)
            }
            if (step.isNotBlank()) {
                val pad = 16f
                val textW = textPaint.measureText(step)
                val textH = textPaint.textSize
                canvas.drawRect(0f, 0f, textW + pad * 2, textH + pad * 2, shadowPaint)
                canvas.drawText(step, pad, textH + pad, textPaint)
            }
        }
    }
}
