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
        print(f"Download progress: {update.progress()}")

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
