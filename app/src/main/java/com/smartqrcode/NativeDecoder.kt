package com.smartqrcode

object NativeDecoder {
    init {
        System.loadLibrary("smartqrcode_native")
    }

    external fun decodeGray(
        gray: ByteArray,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        roiLeft: Int,
        roiTop: Int,
        roiRight: Int,
        roiBottom: Int,
        tryHarder: Boolean
    ): String?

    external fun decodeGrayDebug(
        gray: ByteArray,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        roiLeft: Int,
        roiTop: Int,
        roiRight: Int,
        roiBottom: Int,
        tryHarder: Boolean
    ): String?

    fun parseResult(raw: String): DecodeParsed? {
        val lastSep = raw.lastIndexOf('|')
        if (lastSep <= 0) return null
        val prevSep = raw.lastIndexOf('|', lastSep - 1)
        if (prevSep <= 0) return null

        val text = raw.substring(0, prevSep)
        val rectStr = raw.substring(prevSep + 1, lastSep)
        val rect = rectStr.split(",")
        if (rect.size != 4) return null
        val roi = RoiRect(
            rect[0].toIntOrNull() ?: return null,
            rect[1].toIntOrNull() ?: return null,
            rect[2].toIntOrNull() ?: return null,
            rect[3].toIntOrNull() ?: return null
        )
        val quadStr = raw.substring(lastSep + 1)
        val quad = run {
            val q = quadStr.split(",")
            if (q.size == 8) {
                FloatArray(8).also { out ->
                    for (i in 0 until 8) {
                        out[i] = (q[i].toFloatOrNull() ?: return null)
                    }
                }
            } else null
        }
        return DecodeParsed(text, roi, quad)
    }
}
