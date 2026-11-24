package com.trymirai.uzu

import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertNotNull

class EngineTest {
    @Test
    fun chatReplyProducesText(): Unit = runBlocking {
        val engine = Engine.create(EngineConfig.create())

        val model = checkNotNull(engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4")) { "Model not found" }
        engine.download(model).asFlow().collect()

        val session = engine.chat(model, ChatConfig.create())
        val messages = listOf(
            ChatMessage.system().withText("You are a helpful assistant"),
            ChatMessage.user().withText("Hi"),
        )
        val message = checkNotNull(session.reply(messages, ChatReplyConfig.create()).lastOrNull()?.message) {
            "Reply has no messages"
        }
        assertNotNull(message.reasoning())
        assertNotNull(message.text())
    }
}
