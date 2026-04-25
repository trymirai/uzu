import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPresetClassification, ChatSpeculationPresetSummarization, Engine, EngineConfig, Feature, ReasoningEffort, SamplingMethodGreedy } from '@trymirai/uzu';

async function quickStartExample() {
    // snippet:quick-start
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];

    let session = await engine.chat(model, ChatConfig.create());
    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;
    // endsnippet:quick-start

    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

async function generalExample() {
    // snippet:engine-create
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);
    // endsnippet:engine-create

    // snippet:model-choose
    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    // endsnippet:model-choose

    // snippet:model-download
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }
    // endsnippet:model-download

    // snippet:session-create-general
    let session = await engine.chat(model, ChatConfig.create());
    // endsnippet:session-create-general

    // snippet:session-input-general
    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];
    // endsnippet:session-input-general

    // snippet:session-run-general
    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;
    // endsnippet:session-run-general

    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

async function summarizationExample() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    // snippet:session-input-summarization
    const textToSummarize =
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    const prompt = `Text is: "${textToSummarize}". Write only summary itself.`;
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];
    // endsnippet:session-input-summarization

    // snippet:session-create-summarization
    let chatConfig = ChatConfig.create().withSpeculationPreset(new ChatSpeculationPresetSummarization);
    let session = await engine.chat(model, chatConfig);
    // endsnippet:session-create-summarization

    // snippet:session-run-summarization
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(256).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];
    // endsnippet:session-run-summarization

    if (reply) {
        console.log('Summary: ', reply.message.text);
        console.log('Generation t\\s: ', reply.stats.generateTokensPerSecond);
    }
}

async function classificationExample() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    // snippet:session-create-classification
    const feature = new Feature('sentiment', [
        'Happy',
        'Sad',
        'Angry',
        'Fearful',
        'Surprised',
        'Disgusted',
    ]);
    let chatConfig = ChatConfig.create().withSpeculationPreset(new ChatSpeculationPresetClassification(feature));
    let session = await engine.chat(model, chatConfig);
    // endsnippet:session-create-classification

    // snippet:session-input-classification
    const textToDetectFeature =
        "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    const prompt =
        `Text is: "${textToDetectFeature}". Choose ${feature.name} from the list: ${feature.values.join(', ')}. ` +
        "Answer with one word. Don't add a dot at the end.";
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];
    // endsnippet:session-input-classification

    // snippet:session-run-classification
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(32).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];
    // endsnippet:session-run-classification

    if (reply) {
        console.log('Prediction: ', reply.message.text);
        console.log('Generated tokens: ', reply.stats.tokensCountOutput);
    }
}

async function main() {
    await quickStartExample();
    await generalExample();
    await summarizationExample();
    await classificationExample();
}

main().catch((error) => {
    console.error(error);
});
