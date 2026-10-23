package com.trymirai.playground.speech

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.trymirai.playground.engine.EngineRepository
import com.trymirai.playground.session.SessionState
import com.trymirai.playground.session.sessionStateOf
import com.trymirai.uzu.nagare.TextToSpeechSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TextToSpeechViewModel(
    identifier: String,
    engine: EngineRepository,
    private val outputPath: String,
) : ViewModel() {
    private val _state = MutableStateFlow<SessionState>(SessionState.Loading)
    val state: StateFlow<SessionState> = _state.asStateFlow()

    private val _output = MutableStateFlow<String?>(null)

    /** The WAV file of the last synthesis, once it is written. */
    val output: StateFlow<String?> = _output.asStateFlow()

    private var session: TextToSpeechSession? = null

    init {
        viewModelScope.launch {
            _state.value = sessionStateOf {
                session = engine.textToSpeech(identifier)
                SessionState.Idle
            }
        }
    }

    fun generate(text: String) {
        val session = session ?: return
        val input = text.trim()
        if (input.isEmpty() || _state.value != SessionState.Idle) return
        _state.value = SessionState.Generating
        _output.value = null
        viewModelScope.launch {
            _state.value = sessionStateOf {
                val speech = session.synthesize(input)
                withContext(Dispatchers.IO) { speech.pcmBatch.saveAsWav(outputPath) }
                _output.value = outputPath
                SessionState.Idle
            }
        }
    }

    override fun onCleared() {
        session?.close()
    }
}
