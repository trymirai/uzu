import asyncio

from uzu import (
    ChatConfig,
    ChatContentBlock,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    ReasoningEffort,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("trymirai/Qwen3.5-9B-M-Vision")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"\rDownload progress: {update.progress:.2%}", end="", flush=True)
    print()

    session = await engine.chat(model, ChatConfig.create())
    messages = [
        ChatMessage.system().with_text("You are a helpful assistant").with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user()
        .with_block(ChatContentBlock.Image(url="https://assets.trymirai.com/images/artifacts/photo-cat.jpeg"))
        .with_block(ChatContentBlock.Image(url="https://assets.trymirai.com/images/artifacts/photo-london.jpeg"))
        .with_text("What's on the images?"),
    ]
    replies = await session.reply(messages, ChatReplyConfig.create().with_token_limit(256))
    if replies:
        message = replies[-1].message
        print(f"Reasoning: {message.reasoning or ''}")
        print(f"Text: {message.text or ''}")


if __name__ == "__main__":
    asyncio.run(main())
