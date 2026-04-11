package com.example.musicscore_flutter

import android.content.res.AssetManager

class OmrPipeline {

    external fun nativeInit(assetManager: AssetManager): Boolean

    external fun nativeProcessImageBytes(imageBytes: ByteArray): String?

    companion object {
        init {
            System.loadLibrary("musicscore_flutter")
        }
    }
}
