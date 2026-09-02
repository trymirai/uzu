import asyncio

from uzu import ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort, SamplingMethod


def test_chat_reply_produces_text() -> None:
    async def run() -> None:
        engine = await Engine.create(EngineConfig.create())

        model = await engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4")
        assert model is not None, "Model not found"

        async for update in (await engine.download(model)).iterator():
            print(f"Download progress: {update.progress}")

        session = await engine.chat(model, ChatConfig.create())

        messages = [
            ChatMessage.system()
            .with_text("You are a helpful assistant")
            .with_reasoning_effort(ReasoningEffort.Disabled),
            ChatMessage.user().with_text("Hi"),
        ]

        config = ChatReplyConfig.create().with_token_limit(64).with_sampling_method(SamplingMethod.Greedy())
        replies = await session.reply(messages, config)
        assert replies, "Reply has no messages"

        message = replies[-1].message
        assert message.text is not None

    asyncio.run(run())
