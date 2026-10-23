package com.trymirai.playground.session

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** Loads one model's session, runs one generation on it at a time, and frees it with the view model. */
abstract class SessionViewModel<S : AutoCloseable>(load: suspend () -> S) : ViewModel() {
    private val _state = MutableStateFlow<SessionState>(SessionState.Loading)
    val state: StateFlow<SessionState> = _state.asStateFlow()

    private var session: S? = null
    private var generation: Job? = null

    init {
        viewModelScope.launch {
            _state.value = sessionStateOf {
                session = load()
                SessionState.Idle
            }
        }
    }

    /** Runs [block] on the loaded session, unless it is still loading or already generating. */
    protected fun run(block: suspend (S) -> Unit) {
        val session = session ?: return
        if (_state.value != SessionState.Idle) return
        _state.value = SessionState.Generating
        generation = viewModelScope.launch {
            _state.value = sessionStateOf {
                block(session)
                SessionState.Idle
            }
        }
    }

    fun stop() {
        generation?.cancel()
        _state.value = SessionState.Idle
    }

    override fun onCleared() {
        // The session holds the model's weights; free them now instead of when the collector finds the object.
        session?.close()
    }
}

/** Runs [block], turning an engine failure into [SessionState.Failed]; cancellation still propagates. */
private suspend fun sessionStateOf(block: suspend () -> SessionState): SessionState =
    try {
        block()
    } catch (error: CancellationException) {
        throw error
    } catch (error: Exception) {
        SessionState.Failed(error.message ?: error.toString())
    }
