import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        return

    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")

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
