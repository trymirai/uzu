package com.trymirai.playground.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

sealed interface AudioSource {
    data class Asset(val name: String) : AudioSource

    data class File(val path: String) : AudioSource
}

/** One looping player for the whole app, as on iOS: the home screen's podcast and synthesized speech share it. */
class AudioPlayer(private val context: Context) {
    private var player: MediaPlayer? = null
    private var source: AudioSource? = null
    private val _playing = MutableStateFlow<AudioSource?>(null)

    /** The source that is playing right now, if any. */
    val playing: StateFlow<AudioSource?> = _playing.asStateFlow()

    fun toggle(source: AudioSource) {
        val current = player
        if (current != null && this.source == source) {
            if (current.isPlaying) current.pause() else current.start()
        } else {
            current?.release()
            player = load(source).apply { start() }
            this.source = source
        }
        _playing.value = source.takeIf { player?.isPlaying == true }
    }

    fun pause() {
        player?.takeIf { it.isPlaying }?.pause()
        _playing.value = null
    }

    fun release() {
        player?.release()
        player = null
        source = null
        _playing.value = null
    }

    private fun load(source: AudioSource) = MediaPlayer().apply {
        setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
        )
        when (source) {
            is AudioSource.Asset -> context.assets.openFd(source.name).use { setDataSource(it) }
            is AudioSource.File -> setDataSource(source.path)
        }
        isLooping = true
        prepare()
    }
}
