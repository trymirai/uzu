package com.trymirai.playground

import android.app.Application
import com.trymirai.playground.audio.AudioPlayer
import com.trymirai.playground.engine.EngineRepository
import com.trymirai.uzu.Uzu

class PlaygroundApplication : Application() {
    val audio = AudioPlayer(this)

    lateinit var engine: EngineRepository
        private set

    override fun onCreate() {
        super.onCreate()
        Uzu.initialize(this)
        engine = EngineRepository()
    }
}
