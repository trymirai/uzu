import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig


def test_chat_reply_produces_text() -> None:
    async def run() -> None:
        engine = await Engine.create(EngineConfig.create())

        model = await engine.model("Qwen/Qwen3-0.6B")
        assert model is not None, "Model not found"

        async for update in (await engine.download(model)).iterator():
            print(f"Download progress: {update.progress}")

        session = await engine.chat(model, ChatConfig.create())

        messages = [
            ChatMessage.system().with_text("You are a helpful assistant"),
            ChatMessage.user().with_text("Hi"),
        ]

        replies = await session.reply(messages, ChatReplyConfig.create())
        assert replies, "Reply has no messages"

        message = replies[-1].message
        assert message.reasoning is not None
        assert message.text is not None

    asyncio.run(run())
