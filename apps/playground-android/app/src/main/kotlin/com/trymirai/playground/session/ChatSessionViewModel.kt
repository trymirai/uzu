package com.trymirai.playground.session

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.trymirai.playground.engine.EngineRepository
import com.trymirai.uzu.asFlow
import com.trymirai.uzu.nagare.ChatSession
import com.trymirai.uzu.nagare.ChatSessionStreamChunk
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReply
import com.trymirai.uzu.shoji.ChatReplyConfig
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** Loads a chat session for one model and streams replies from it, as chat, classification and summarization do. */
abstract class ChatSessionViewModel(identifier: String, engine: EngineRepository) : ViewModel() {
    private val _state = MutableStateFlow<SessionState>(SessionState.Loading)
    val state: StateFlow<SessionState> = _state.asStateFlow()

    private var session: ChatSession? = null
    private var generation: Job? = null

    init {
        viewModelScope.launch {
            _state.value = sessionStateOf {
                session = engine.chat(identifier)
                SessionState.Idle
            }
        }
    }

    /** Answers [input] from a fresh conversation, passing every partial reply to [onReply]. */
    protected fun reply(input: List<ChatMessage>, config: ChatReplyConfig, onReply: (ChatReply) -> Unit) {
        val session = session ?: return
        if (_state.value != SessionState.Idle) return
        _state.value = SessionState.Generating
        generation = viewModelScope.launch {
            _state.value = sessionStateOf {
                session.reset()
                session.replyWithStream(input, config).asFlow().collect { chunk ->
                    when (chunk) {
                        is ChatSessionStreamChunk.Replies -> chunk.replies.lastOrNull()?.let(onReply)
                        is ChatSessionStreamChunk.Error -> throw chunk.error
                        is ChatSessionStreamChunk.ToolResults -> Unit
                    }
                }
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
