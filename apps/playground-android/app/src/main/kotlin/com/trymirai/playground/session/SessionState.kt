package com.trymirai.playground.session

sealed interface SessionState {
    /** Whether a generation can start: the session loaded, and its last generation, if any, has ended. */
    val canStart: Boolean get() = this == Idle || (this is Failed && loaded)

    data object Loading : SessionState

    data object Idle : SessionState

    data object Generating : SessionState

    data class Failed(val message: String, val loaded: Boolean) : SessionState
}

/** What a screen's prompt or action shows for this state, given its own words for [idle] and [generating]. */
fun SessionState.label(idle: String, generating: String): String = when (this) {
    SessionState.Loading -> "Loading…"
    SessionState.Generating -> generating
    SessionState.Idle, is SessionState.Failed -> idle
}
