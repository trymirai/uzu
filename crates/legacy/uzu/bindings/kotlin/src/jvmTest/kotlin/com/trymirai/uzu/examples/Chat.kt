package com.trymirai.uzu.examples

import com.trymirai.uzu.Engine
import com.trymirai.uzu.EngineConfig
import com.trymirai.uzu.asFlow
import com.trymirai.uzu.create
import com.trymirai.uzu.nagare.ChatSessionStreamChunk
import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.system
import com.trymirai.uzu.user

suspend fun runChat() {
    val engine = Engine.create(EngineConfig.create())

    val model = checkNotNull(engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4")) { "Model not found" }
    engine.download(model).asFlow().collect { update ->
        println("Download progress: %.2f%%".format(update.progress() * 100))
    }

    val messages = listOf(
        ChatMessage.system().withText("You are a helpful assistant"),
        ChatMessage.user().withText("Tell me a short, funny story about a robot"),
    )
    val session = engine.chat(model, ChatConfig.create())
    var message: ChatMessage? = null
    session.replyWithStream(messages, ChatReplyConfig.create()).asFlow().collect { update ->
        when (update) {
            is ChatSessionStreamChunk.Replies -> {
                val reply = update.replies.lastOrNull()
                message = reply?.message
                println("Generated tokens: ${reply?.stats?.tokensCountOutput ?: 0u}")
            }
            is ChatSessionStreamChunk.Error -> println("Error: ${update.error}")
            is ChatSessionStreamChunk.ToolResults -> Unit
        }
    }
    println("Reasoning: ${message?.reasoning() ?: "empty"}")
    println("Text: ${message?.text() ?: "empty"}")
}
