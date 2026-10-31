package com.trymirai.playground.chat

import com.trymirai.playground.engine.EngineRepository
import com.trymirai.playground.session.ChatSessionViewModel
import com.trymirai.playground.session.SessionState
import com.trymirai.uzu.assistant
import com.trymirai.uzu.create
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.shoji.ChatReplyStats
import com.trymirai.uzu.user
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class Message(
    val isUser: Boolean,
    val content: String,
    val reasoning: String? = null,
    val stats: ChatReplyStats? = null,
)

class ChatViewModel(identifier: String, engine: EngineRepository) : ChatSessionViewModel(identifier, engine) {
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    fun send(text: String) {
        val content = text.trim()
        if (content.isEmpty() || !state.value.canStart) return
        // A failed turn ends with its user message and unfinished reply; the new message replaces both.
        val completed = if (state.value is SessionState.Failed) _messages.value.dropLast(2) else _messages.value
        val history = completed + Message(isUser = true, content = content)
        _messages.value = history + Message(isUser = false, content = "")
        val input = history.map {
            (if (it.isUser) ChatMessage.user() else ChatMessage.assistant()).withText(it.content)
        }
        reply(input, ChatReplyConfig.create().withTokenLimit(1024u)) { reply ->
            _messages.update { messages ->
                messages.dropLast(1) + Message(
                    isUser = false,
                    content = reply.message.text().orEmpty().trim(),
                    reasoning = reply.message.reasoning()?.trim(),
                    stats = reply.stats,
                )
            }
        }
    }
}
