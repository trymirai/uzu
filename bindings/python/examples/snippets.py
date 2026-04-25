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


async def quick_start_example() -> None:
    # snippet:quick-start
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")

    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(messages, ChatReplyConfig.create())
    message = replies[0].message if replies else None
    # endsnippet:quick-start

    if message is not None:
        print(f"Reasoning: {message.reasoning}")
        print(f"Text: {message.text}")


async def chat_example() -> None:
    # snippet:engine-create
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)
    # endsnippet:engine-create

    # snippet:model-choose
    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    # endsnippet:model-choose

    # snippet:model-download
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")
    # endsnippet:model-download

    # snippet:session-create-general
    session = await engine.chat(model, ChatConfig.create())
    # endsnippet:session-create-general

    # snippet:session-input-general
    messages = [
        ChatMessage.system().with_text("You are a helpful assistant"),
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]
    # endsnippet:session-input-general

    # snippet:session-run-general
    replies = await session.reply(messages, ChatReplyConfig.create())
    message = replies[0].message if replies else None
    # endsnippet:session-run-general

    if message is not None:
        print(f"Reasoning: {message.reasoning}")
        print(f"Text: {message.text}")


async def summarization_example() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")

    # snippet:session-input-summarization
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
    # endsnippet:session-input-summarization

    # snippet:session-create-summarization
    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Summarization())
    session = await engine.chat(model, chat_config)
    # endsnippet:session-create-summarization

    # snippet:session-run-summarization
    chat_reply_config = ChatReplyConfig.create().with_token_limit(256).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    reply = replies[0] if replies else None
    # endsnippet:session-run-summarization

    if reply is not None:
        print(f"Summary: {reply.message.text}")
        print(f"Generation t/s: {reply.stats.generate_tokens_per_second}")


async def classification_example() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")

    # snippet:session-create-classification
    feature = Feature(
        "sentiment",
        ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Disgusted"],
    )
    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Classification(feature))
    session = await engine.chat(model, chat_config)
    # endsnippet:session-create-classification

    # snippet:session-input-classification
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
    # endsnippet:session-input-classification

    # snippet:session-run-classification
    chat_reply_config = ChatReplyConfig.create().with_token_limit(32).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    reply = replies[0] if replies else None
    # endsnippet:session-run-classification

    if reply is not None:
        print(f"Prediction: {reply.message.text}")
        print(f"Generated tokens: {reply.stats.tokens_count_output}")


async def main() -> None:
    await quick_start_example()
    await chat_example()
    await summarization_example()
    await classification_example()


if __name__ == "__main__":
    asyncio.run(main())
