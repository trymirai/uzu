import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4")
    if model is None:
        raise RuntimeError("Model not found")

    async for update in (await engine.download(model)).iterator():
        print(f"\rDownload progress: {update.progress:.2%}", end="", flush=True)
    print()

    # The chat_instance owns the loaded model and can be shared between sessions.
    chat_instance = await engine.chat_instance(model, ChatConfig.create())

    first_session = await engine.chat_with_instance(chat_instance)
    replies = await first_session.reply(
        [ChatMessage.user().with_text("Tell me a short, funny story about a robot")],
        ChatReplyConfig.create(),
    )
    if replies:
        message = replies[-1].message
        print(f"First session reasoning: {message.reasoning}")
        print(f"First session text: {message.text}")

    # The second session reuses the already-loaded weights instead of loading the model again.
    second_session = await engine.chat_with_instance(chat_instance)
    replies = await second_session.reply(
        [ChatMessage.user().with_text("What is the capital of France?")],
        ChatReplyConfig.create(),
    )
    if replies:
        message = replies[-1].message
        print(f"\nSecond session reasoning: {message.reasoning}")
        print(f"Second session text: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
