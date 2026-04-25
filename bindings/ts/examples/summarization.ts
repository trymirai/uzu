import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPresetSummarization, Engine, EngineConfig, ReasoningEffort, SamplingMethodGreedy } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const textToSummarize =
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    const prompt = `Text is: "${textToSummarize}". Write only summary itself.`;
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];

    let chatConfig = ChatConfig.create().withSpeculationPreset(new ChatSpeculationPresetSummarization);
    let session = await engine.chat(model, chatConfig);

    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(256).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];

    if (reply) {
        console.log('Summary: ', reply.message.text);
        console.log('Generation t\\s: ', reply.stats.generateTokensPerSecond);
    }
}

main().catch((error) => {
    console.error(error);
});
