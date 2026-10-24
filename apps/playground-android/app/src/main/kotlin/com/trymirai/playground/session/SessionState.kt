package com.trymirai.playground.session

sealed interface SessionState {
    data object Loading : SessionState

    data object Idle : SessionState

    data object Generating : SessionState

    data class Failed(val message: String) : SessionState
}

/** What a screen shows for this state, given its own words for [idle] and [generating]. */
fun SessionState.label(idle: String, generating: String): String = when (this) {
    SessionState.Loading -> "Loading…"
    SessionState.Generating -> generating
    is SessionState.Failed -> "Error: $message"
    SessionState.Idle -> idle
}
