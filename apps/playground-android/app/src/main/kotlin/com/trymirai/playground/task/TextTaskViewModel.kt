package com.trymirai.playground.task

import com.trymirai.playground.engine.EngineRepository
import com.trymirai.playground.session.ChatSessionViewModel
import com.trymirai.uzu.create
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReply
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.shoji.ReasoningEffort
import com.trymirai.uzu.shoji.SamplingMethod
import com.trymirai.uzu.system
import com.trymirai.uzu.user
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class TextTaskViewModel(
    identifier: String,
    engine: EngineRepository,
    val task: TextTask,
) : ChatSessionViewModel(identifier, engine) {
    private val _result = MutableStateFlow<ChatReply?>(null)
    val result: StateFlow<ChatReply?> = _result.asStateFlow()

    fun run(text: String) {
        val content = text.trim()
        if (content.isEmpty()) return
        _result.value = null
        reply(
            listOf(
                ChatMessage.system().withReasoningEffort(ReasoningEffort.DISABLED),
                ChatMessage.user().withText(task.prompt(content)),
            ),
            ChatReplyConfig.create().withSamplingMethod(SamplingMethod.Greedy).withTokenLimit(task.tokenLimit),
        ) { _result.value = it }
    }
}
