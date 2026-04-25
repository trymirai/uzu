import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("cartesia-ai/Llamba-1B-4bit-mlx")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")

    messages = [
        ChatMessage.user().with_text("Tell me a short, funny story about a robot"),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(messages, ChatReplyConfig.create())
    if replies:
        message = replies[0].message
        print(f"Text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
