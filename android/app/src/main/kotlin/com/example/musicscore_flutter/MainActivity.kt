package com.example.musicscore_flutter

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private val pipeline = OmrPipeline()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "com.example.musicscore/omr")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "initialize" -> {
                        val ok = pipeline.nativeInit(assets)
                        result.success(ok)
                    }
                    "processImageBytes" -> {
                        val bytes = call.arguments as? ByteArray
                        if (bytes == null) {
                            result.error("INVALID_ARG", "imageBytes must be ByteArray", null)
                        } else {
                            val xml = pipeline.nativeProcessImageBytes(bytes)
                            result.success(xml)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }
}
