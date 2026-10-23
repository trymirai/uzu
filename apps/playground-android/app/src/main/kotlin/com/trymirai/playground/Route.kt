package com.trymirai.playground

import androidx.navigation3.runtime.NavKey
import com.trymirai.playground.task.TextTask
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
    data class TextTask(val identifier: String, val task: com.trymirai.playground.task.TextTask) : Route

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
        Classification -> Route.TextTask(identifier, TextTask.Classification)
        Summarization -> Route.TextTask(identifier, TextTask.Summarization)
        TextToSpeech -> Route.TextToSpeech(identifier)
    }
}
