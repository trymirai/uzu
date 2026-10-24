package com.trymirai.uzu

import android.content.Context

object Uzu {
    init {
        // JNI resolves `nativeInitialize` only in libraries the runtime loaded itself; the bindings load libuzu via JNA.
        System.loadLibrary("uzu")
    }

    /**
     * Prepares the engine to run inside this app: HTTPS certificates are checked against Android's trust store, and
     * models are stored in [Context.getNoBackupFilesDir], so gigabytes of weights stay out of cloud backups.
     * Call it before [Engine.create].
     */
    fun initialize(context: Context) {
        nativeInitialize(context.applicationContext, context.noBackupFilesDir.absolutePath)
    }

    @JvmStatic
    private external fun nativeInitialize(context: Context, homePath: String)
}
