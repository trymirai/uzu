<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord&color=brightgreen" alt="Discord"></a> <a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-brightgreen" alt="Contact us"></a> <a href="https://docs.trymirai.com"><img src="https://img.shields.io/badge/Read-Docs-brightgreen" alt="Read docs"></a> [![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE) [![Build](https://github.com/trymirai/uzu/actions/workflows/tests.yml/badge.svg)](https://github.com/trymirai/uzu/actions) [![Kotlin](https://img.shields.io/badge/Kotlin-purple)](crates/legacy/uzu/bindings/kotlin) 

# uzu

A high-performance inference engine for AI models. It allows you to deploy AI directly in your app with **zero latency**, **full data privacy**, and **no inference costs**. Key features:

- Simple, high-level API
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices
- [Broad model support](https://trymirai.com/local-models)

## Quick Start



Add the dependency:

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        // The JVM half of rustls-platform-verifier, which checks the engine's HTTPS certificates.
        maven("https://github.com/rustls/rustls-platform-verifier/raw/maven-archive/android-release-support/maven/")
    }
}
include(":uzu")
project(":uzu").projectDir = file("<uzu checkout>/crates/legacy/uzu/bindings/kotlin")

// app/build.gradle.kts
dependencies {
    implementation(project(":uzu"))
}
```

Building the module compiles the engine for arm64 with `cargo tools build kotlin`, which needs the Android NDK and
cargo-ndk (`cargo tools install kotlin`). Set `minSdk = 31` and run on an arm64 device with Vulkan 1.4.
Call `Uzu.initialize(context)` once before creating the engine.

Run the code below:

```kotlin
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
```


<br>

Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

You can run any example via `cargo tools example` \<**kotlin**\> \<**chat** | **chat-cloud** | **chat-shared-instance** | **chat-structured-output** | **quick-start** | **tool-calls**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

```kotlin
package com.trymirai.uzu.examples

import android.content.Context
import com.trymirai.uzu.Engine
import com.trymirai.uzu.EngineConfig
import com.trymirai.uzu.Uzu
import com.trymirai.uzu.asFlow
import com.trymirai.uzu.create
import com.trymirai.uzu.nagare.ChatSessionStreamChunk
import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.system
import com.trymirai.uzu.user

suspend fun runChat(context: Context) {
    Uzu.initialize(context)
    val engine = Engine.create(EngineConfig.create())

    val model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4") ?: return
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
```

<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

```kotlin
package com.trymirai.uzu.examples

import android.content.Context
import com.trymirai.uzu.Engine
import com.trymirai.uzu.EngineConfig
import com.trymirai.uzu.Uzu
import com.trymirai.uzu.create
import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.shoji.ReasoningEffort
import com.trymirai.uzu.system
import com.trymirai.uzu.user

suspend fun runChatCloud(context: Context) {
    Uzu.initialize(context)
    val engineConfig = EngineConfig.create().withOpenaiApiKey("OPENAI_API_KEY")
    val engine = Engine.create(engineConfig)

    val model = engine.model("gpt-5") ?: return

    val messages = listOf(
        ChatMessage.system().withReasoningEffort(ReasoningEffort.LOW),
        ChatMessage.user().withText("How LLMs work"),
    )

    val session = engine.chat(model, ChatConfig.create())
    val message = session.reply(messages, ChatReplyConfig.create()).lastOrNull()?.message ?: return

    println("Reasoning: ${message.reasoning() ?: "empty"}")
    println("Text: ${message.text() ?: "empty"}")
}
```

### Chat with shared instance

This example shows how to reuse chat instance without reloading model into memory:

```kotlin
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
import com.trymirai.uzu.user

suspend fun runChatSharedInstance(context: Context) {
    Uzu.initialize(context)
    val engine = Engine.create(EngineConfig.create())

    val model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4") ?: return
    engine.download(model).asFlow().collect { update ->
        println("Download progress: %.2f%%".format(update.progress() * 100))
    }

    // The chatInstance owns the loaded model and can be shared between sessions.
    val chatInstance = engine.chatInstance(model, ChatConfig.create())

    val firstSession = engine.chatWithInstance(chatInstance)
    val replies = firstSession.reply(
        listOf(ChatMessage.user().withText("Tell me a short, funny story about a robot")),
        ChatReplyConfig.create(),
    )
    replies.lastOrNull()?.message?.let { message ->
        println("First session reasoning: ${message.reasoning().orEmpty()}")
        println("First session text: ${message.text().orEmpty()}")
    }

    // The second session reuses the already-loaded weights instead of loading the model again.
    val secondSession = engine.chatWithInstance(chatInstance)
    val secondReplies = secondSession.reply(
        listOf(ChatMessage.user().withText("What is the capital of France?")),
        ChatReplyConfig.create(),
    )
    secondReplies.lastOrNull()?.message?.let { message ->
        println("\nSecond session reasoning: ${message.reasoning().orEmpty()}")
        println("Second session text: ${message.text().orEmpty()}")
    }
}
```

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

```kotlin
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
```

### Tool calls

This example shows how to use external tools:

```kotlin
package com.trymirai.uzu.examples

import android.content.Context
import com.trymirai.uzu.Engine
import com.trymirai.uzu.EngineConfig
import com.trymirai.uzu.Uzu
import com.trymirai.uzu.asFlow
import com.trymirai.uzu.create
import com.trymirai.uzu.nagare.ForeignTool
import com.trymirai.uzu.nagare.ForeignToolHandler
import com.trymirai.uzu.shoji.ChatConfig
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReplyConfig
import com.trymirai.uzu.shoji.SamplingMethod
import com.trymirai.uzu.shoji.ToolFunction
import com.trymirai.uzu.shoji.Value
import com.trymirai.uzu.system
import com.trymirai.uzu.user

private const val COORDINATE_SCHEMA = """
{
  "type": "object",
  "properties": {
    "latitude": { "type": "number", "description": "Latitude in decimal degrees." },
    "longitude": { "type": "number", "description": "Longitude in decimal degrees." }
  },
  "required": ["latitude", "longitude"]
}
"""

private fun tool(name: String, description: String, parameters: String, call: suspend (String) -> String) =
    ForeignTool(
        ToolFunction(name, description, Value(parameters), returnDefinition = null),
        object : ForeignToolHandler {
            override suspend fun invokeJson(argumentsJson: String) = call(argumentsJson)
        },
    )

private val getCurrentLocation = tool(
    "get_current_location",
    "Returns current location in coordinates",
    """{ "type": "object", "properties": {} }""",
) { """{ "latitude": 51.5074, "longitude": -0.1278 }""" }

private val getCurrentTemperature = tool(
    "get_current_temperature",
    "Returns temperature in provided location",
    COORDINATE_SCHEMA,
) { "25.0" }

suspend fun runToolCalls(context: Context) {
    Uzu.initialize(context)
    val engine = Engine.create(EngineConfig.create())
    val model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4") ?: error("Model not found")
    engine.download(model).asFlow().collect { update ->
        println("Download progress: %.2f%%".format(update.progress() * 100))
    }

    val session = engine.chat(model, ChatConfig.create())
    session.addForeignTool(getCurrentLocation)
    session.addForeignTool(getCurrentTemperature)

    val messages = listOf(
        ChatMessage.system().withText("You are a helpful assistant"),
        ChatMessage.user().withText("What temperature is it now at my location?"),
    )
    val replyConfig = ChatReplyConfig.create().withSamplingMethod(SamplingMethod.Greedy)
    val message = session.reply(messages, replyConfig).lastOrNull()?.message ?: return

    println("Reasoning: ${message.reasoning().orEmpty()}")
    println("Text: ${message.text().orEmpty()}")
}
```



## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
