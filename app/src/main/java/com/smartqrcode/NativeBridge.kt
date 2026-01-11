package com.smartqrcode

object NativeBridge {
    init {
        System.loadLibrary("smartqrcode_native")
    }

    external fun hello(): String
}

