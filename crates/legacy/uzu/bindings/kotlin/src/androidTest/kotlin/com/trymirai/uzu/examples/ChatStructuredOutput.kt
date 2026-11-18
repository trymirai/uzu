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
import com.trymirai.uzu.shoji.Grammar
import com.trymirai.uzu.shoji.ReasoningEffort
import com.trymirai.uzu.system
import com.trymirai.uzu.user
import org.json.JSONArray

private const val COUNTRIES_SCHEMA = """
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": { "name": { "type": "string" }, "capital": { "type": "string" } },
    "required": ["name", "capital"]
  }
}
"""

suspend fun runChatStructuredOutput(context: Context) {
    Uzu.initialize(context)
    val engine = Engine.create(EngineConfig.create())

    val model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4") ?: return
    engine.download(model).asFlow().collect { update ->
        println("Download progress: %.2f%%".format(update.progress() * 100))
    }

    val messages = listOf(
        ChatMessage.system().withReasoningEffort(ReasoningEffort.DISABLED),
        ChatMessage.user().withText(
            "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields",
        ),
    )

    val session = engine.chat(model, ChatConfig.create())
    val config = ChatReplyConfig.create().withGrammar(Grammar.JsonSchema(COUNTRIES_SCHEMA))
    val text = session.reply(messages, config).lastOrNull()?.message?.text() ?: return

    val countries = JSONArray(text)
    for (index in 0 until countries.length()) {
        val country = countries.getJSONObject(index)
        println("${country.getString("name")}: ${country.getString("capital")}")
    }
}
