<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-python.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord&color=brightgreen" alt="Discord"></a> <a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-brightgreen" alt="Contact us"></a> <a href="https://docs.trymirai.com"><img src="https://img.shields.io/badge/Read-Docs-brightgreen" alt="Read docs"></a> [![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE) [![Build](https://github.com/trymirai/uzu/actions/workflows/tests.yml/badge.svg)](https://github.com/trymirai/uzu/actions) [![Python](https://img.shields.io/badge/Python-orange)](bindings/python) [![Package](https://img.shields.io/pypi/v/uzu?color=orange&label=Package&v=0.4.4)](https://pypi.org/project/uzu/) [![Python](https://img.shields.io/pypi/pyversions/uzu?color=orange&label=Python&v=0.4.4)](https://pypi.org/project/uzu/) 

# uzu

A high-performance inference engine for AI models. It allows you to deploy AI directly in your app with **zero latency**, **full data privacy**, and **no inference costs**. Key features:

- Simple, high-level API
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices
- [Broad model support](https://trymirai.com/models)

## Quick Start



Add the dependency:

```bash
uv add uzu==0.4.4
```

Run the code below:

```python
import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        return

    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    session = await engine.chat(model, ChatConfig.create())

    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]

    replies = await session.reply(messages, ChatReplyConfig.create())
    if not replies:
        return

    message = replies[-1].message
    print(f"Reasoning: {message.reasoning}")
    print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
```


<br>

Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

You can run any example via `cargo tools example` \<**python**\> \<**chat** | **chat-cloud** | **chat-speculation-classification** | **chat-speculation-summarization** | **chat-structured-output** | **classification** | **quick-start** | **text-to-speech**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

```python
import asyncio

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSessionStreamChunk,
    Engine,
    EngineConfig,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]
    session = await engine.chat(model, ChatConfig.create())
    stream = await session.reply_with_stream(messages, ChatReplyConfig.create())
    message: ChatMessage | None = None
    async for chunk in stream.iterator():
        if isinstance(chunk, ChatSessionStreamChunk.Replies):
            replies = chunk.replies
            if replies:
                reply = replies[0]
                message = reply.message
                print(f"Generated tokens: {reply.stats.tokens_count_output}")
        elif isinstance(chunk, ChatSessionStreamChunk.Error):
            print(f"Error: {chunk.error}")
    if message is not None:
        print(f"Reasoning: {message.reasoning}")
        print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

```python
import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort


async def main() -> None:
    engine_config = EngineConfig.create().with_openai_api_key("OPENAI_API_KEY")
    engine = await Engine.create(engine_config)

    model = await engine.model("gpt-5")
    if model is None:
        raise RuntimeError("Model not found")

    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Low),
        ChatMessage.user().with_text("How LLMs work"),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(messages, ChatReplyConfig.create())
    if replies:
        message = replies[0].message
        print(f"Reasoning: {message.reasoning}")
        print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Chat using speculation preset for classification

In this example, we will use the `classification` speculation preset to determine the sentiment of the user's input:

```python
import asyncio

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSpeculationPreset,
    Engine,
    EngineConfig,
    Feature,
    ReasoningEffort,
    SamplingMethod,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    feature = Feature(
        "sentiment",
        ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Disgusted"],
    )
    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Classification(feature))
    session = await engine.chat(model, chat_config)

    text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling."
    prompt = (
        f'Text is: "{text_to_detect_feature}". '
        f"Choose {feature.name} from the list: {', '.join(feature.values)}. "
        "Answer with one word. Don't add a dot at the end."
    )
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(prompt),
    ]

    chat_reply_config = ChatReplyConfig.create().with_token_limit(32).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    if replies:
        reply = replies[0]
        print(f"Prediction: {reply.message.text}")
        print(f"Generated tokens: {reply.stats.tokens_count_output}")


if __name__ == "__main__":
    asyncio.run(main())
```

<br>You can view the stats to see that the answer will be ready immediately after the prefill step, and actual generation won’t even start due to speculative decoding, which significantly improves generation speed.

### Chat using speculation preset for summarization

In this example, we will use the `summarization` speculation preset to generate a summary of the input text:

```python
import asyncio

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSpeculationPreset,
    Engine,
    EngineConfig,
    ReasoningEffort,
    SamplingMethod,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    text_to_summarize = (
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. "
        "It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. "
        "LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. "
        "These models have a wide range of applications, including chatbots, content creation, translation, and code generation. "
        "One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. "
        "They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. "
        "Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. "
        "As these models grow in size and sophistication, they continue to enhance human-computer interactions, "
        "making AI-powered communication more natural and effective."
    )
    prompt = f'Text is: "{text_to_summarize}". Write only summary itself.'
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(prompt),
    ]

    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Summarization())
    session = await engine.chat(model, chat_config)

    chat_reply_config = ChatReplyConfig.create().with_token_limit(256).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    if replies:
        reply = replies[0]
        print(f"Summary: {reply.message.text}")
        print(f"Generation t/s: {reply.stats.generate_tokens_per_second}")


if __name__ == "__main__":
    asyncio.run(main())
```

<br>You will notice that the model’s run count is lower than the actual number of generated tokens due to speculative decoding, which significantly improves generation speed.

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

```python
import asyncio
import json

from pydantic import BaseModel

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    Grammar,
    ReasoningEffort,
)


class Country(BaseModel):
    name: str
    capital: str


class CountryList(BaseModel):
    countries: list[Country]


def structured_response(response: str | None, model_type: type[BaseModel]) -> BaseModel | None:
    if not response:
        return None
    return model_type.model_validate_json(response)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    schema_string = json.dumps(CountryList.model_json_schema())
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(
            "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields"
        ),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(
        messages,
        ChatReplyConfig.create().with_grammar(Grammar.JsonSchema(schema_string)),
    )
    if replies:
        countries = structured_response(replies[0].message.text, CountryList)
        print(countries)


if __name__ == "__main__":
    asyncio.run(main())
```

### Classification

In this example, we will use a classification model to determine whether the user's input is safe from a moderation perspective:

```python
import asyncio

from uzu import ClassificationMessage, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("trymirai/chat-moderation-router")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    messages = [ClassificationMessage.user("Hi")]

    session = await engine.classification(model)
    output = await session.classify(messages)
    print(f"Output: {output.probabilities.values}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Text to Speech

In this example, we will generate audio from text:

```python
import asyncio
from pathlib import Path

from uzu import Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("fishaudio/s1-mini")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    text = (
        "London is the capital of United Kingdom and one of the world's most influential cities, "
        "known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. "
        "Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace "
        "with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum "
        "and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year."
    )
    output_path = Path.home() / "Desktop" / "output.wav"
    session = await engine.text_to_speech(model)
    pcm_batch = await session.synthesize(text)
    pcm_batch.save_as_wav(str(output_path))
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
```



## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
