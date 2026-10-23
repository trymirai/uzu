package com.trymirai.playground.session

import kotlinx.coroutines.CancellationException

sealed interface SessionState {
    data object Loading : SessionState

    data object Idle : SessionState

    data object Generating : SessionState

    data class Failed(val message: String) : SessionState
}

/** Runs [block], turning an engine failure into [SessionState.Failed]; cancellation still propagates. */
suspend fun sessionStateOf(block: suspend () -> SessionState): SessionState =
    try {
        block()
    } catch (error: CancellationException) {
        throw error
    } catch (error: Exception) {
        SessionState.Failed(error.message ?: error.toString())
    }
