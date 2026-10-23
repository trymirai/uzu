package com.trymirai.playground.engine

import com.trymirai.uzu.Engine
import com.trymirai.uzu.EngineCallback
import com.trymirai.uzu.EngineConfig
import com.trymirai.uzu.create
import com.trymirai.uzu.downloadmanager.DownloadState
import com.trymirai.uzu.nagare.ChatSession
import com.trymirai.uzu.nagare.TextToSpeechSession
import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.Model
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** The app's one engine and the catalog it publishes, refreshed whenever the engine reports a change. */
class EngineRepository {
    private val _failure = MutableStateFlow<String?>(null)

    /** The last engine call that failed, if any: creating the engine, or starting, pausing or deleting a download. */
    val failure: StateFlow<String?> = _failure.asStateFlow()

    private val scope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default + CoroutineExceptionHandler { _, error ->
            _failure.value = error.message ?: error.toString()
        },
    )
    private val engine = CompletableDeferred<Engine>()
    private val _models = MutableStateFlow<List<Model>>(emptyList())
    private val _downloadStates = MutableStateFlow<Map<String, DownloadState>>(emptyMap())

    val models: StateFlow<List<Model>> = _models.asStateFlow()
    val downloadStates: StateFlow<Map<String, DownloadState>> = _downloadStates.asStateFlow()

    init {
        scope.launch {
            val engine = try {
                Engine.create(EngineConfig.create())
            } catch (error: Exception) {
                engine.completeExceptionally(error)
                throw error
            }
            engine.registerCallback(EngineCallback.create { scope.launch { refresh(engine) } })
            this@EngineRepository.engine.complete(engine)
            refresh(engine)
        }
    }

    suspend fun chat(identifier: String): ChatSession {
        val engine = engine.await()
        return engine.chat(engine.requireModel(identifier), ChatConfig.create())
    }

    suspend fun textToSpeech(identifier: String): TextToSpeechSession {
        val engine = engine.await()
        return engine.textToSpeech(engine.requireModel(identifier))
    }

    fun download(model: Model) {
        scope.launch { engine.await().downloader(model).resume() }
    }

    fun pause(model: Model) {
        scope.launch { engine.await().downloader(model).pause() }
    }

    fun delete(model: Model) {
        scope.launch { engine.await().downloader(model).delete() }
    }

    private suspend fun refresh(engine: Engine) {
        _models.value = engine.models()
        _downloadStates.value = engine.downloadStates()
    }

    private suspend fun Engine.requireModel(identifier: String): Model =
        model(identifier) ?: throw IllegalArgumentException("$identifier is not in the catalog")
}
