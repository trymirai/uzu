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


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    feature = Feature(
        "sentiment",
        ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Disgusted"],
    )
    chat_config = ChatConfig.create().with_speculation_preset(ChatSpeculationPreset.Classification(feature))
    session = await engine.chat(model, chat_config)

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

    chat_reply_config = ChatReplyConfig.create().with_token_limit(32).with_sampling_method(SamplingMethod.Greedy())
    replies = await session.reply(messages, chat_reply_config)
    if replies:
        reply = replies[0]
        print(f"Prediction: {reply.message.text}")
        print(f"Generated tokens: {reply.stats.tokens_count_output}")


if __name__ == "__main__":
    asyncio.run(main())
