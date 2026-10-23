package com.trymirai.playground

import androidx.navigation3.runtime.NavKey
import kotlinx.serialization.Serializable

@Serializable
sealed interface Route : NavKey {
    @Serializable
    data object Home : Route

    @Serializable
    data class ModelSelection(val next: Feature) : Route

    @Serializable
    data class Chat(val identifier: String) : Route

    @Serializable
    data class Classification(val identifier: String) : Route

    @Serializable
    data class Summarization(val identifier: String) : Route

    @Serializable
    data class TextToSpeech(val identifier: String) : Route

    @Serializable
    data object About : Route

    @Serializable
    data object ModelManagement : Route
}

/** A use case the home screen offers; each one starts by picking a model. */
@Serializable
enum class Feature {
    Chat,
    Classification,
    Summarization,
    TextToSpeech;

    fun route(identifier: String): Route = when (this) {
        Chat -> Route.Chat(identifier)
        Classification -> Route.Classification(identifier)
        Summarization -> Route.Summarization(identifier)
        TextToSpeech -> Route.TextToSpeech(identifier)
    }
}
