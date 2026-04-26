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
