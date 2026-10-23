package com.trymirai.playground.speech

import com.trymirai.playground.engine.EngineRepository
import com.trymirai.playground.session.SessionViewModel
import com.trymirai.uzu.nagare.TextToSpeechSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

class TextToSpeechViewModel(
    identifier: String,
    engine: EngineRepository,
    private val outputPath: String,
) : SessionViewModel<TextToSpeechSession>({ engine.textToSpeech(identifier) }) {
    private val _output = MutableStateFlow<String?>(null)

    /** The WAV file of the last synthesis, once it is written. */
    val output: StateFlow<String?> = _output.asStateFlow()

    fun generate(text: String) {
        val input = text.trim()
        if (input.isEmpty()) return
        run { session ->
            _output.value = null
            val speech = session.synthesize(input)
            withContext(Dispatchers.IO) { speech.pcmBatch.saveAsWav(outputPath) }
            _output.value = outputPath
        }
    }
}
