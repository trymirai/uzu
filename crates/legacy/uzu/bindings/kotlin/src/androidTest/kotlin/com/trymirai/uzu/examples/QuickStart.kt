package com.trymirai.uzu.examples

import android.content.Context
import com.trymirai.uzu.Engine
import com.trymirai.uzu.EngineConfig
import com.trymirai.uzu.Uzu
import com.trymirai.uzu.asFlow
import com.trymirai.uzu.create
import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.system
import com.trymirai.uzu.user

suspend fun runQuickStart(context: Context) {
    Uzu.initialize(context)
    val engine = Engine.create(EngineConfig.create())

    val model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4") ?: return
    engine.download(model).asFlow().collect { update ->
        println("Download progress: %.2f%%".format(update.progress() * 100))
    }

    val session = engine.chat(model, ChatConfig.create())

    val messages = listOf(
        ChatMessage.system().withText("You are a helpful assistant"),
        ChatMessage.user().withText("Tell me a short, funny story about a robot"),
    )

    val message = session.reply(messages, ChatReplyConfig.create()).lastOrNull()?.message ?: return

    println("Reasoning: ${message.reasoning() ?: "empty"}")
    println("Text: ${message.text() ?: "empty"}")
}
