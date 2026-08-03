import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort


async def main() -> None:
    engine_config = EngineConfig.create().with_openai_api_key("OPENAI_API_KEY")
    engine = await Engine.create(engine_config)

    model = await engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"\rDownload progress: {update.progress:.2%}", end="", flush=True)
    print()

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
