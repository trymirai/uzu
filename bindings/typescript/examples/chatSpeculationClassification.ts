import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPresetClassification, Engine, EngineConfig, Feature, ReasoningEffort, SamplingMethodGreedy } from '@trymirai/uzu';

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

    const textToDetectFeature =
        "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    const prompt =
        `Text is: "${textToDetectFeature}". Choose ${feature.name} from the list: ${feature.values.join(', ')}. ` +
        "Answer with one word. Don't add a dot at the end.";
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];

    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(32).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];

    if (reply) {
        console.log('Prediction: ', reply.message.text);
        console.log('Generated tokens: ', reply.stats.tokensCountOutput);
    }
}

main().catch((error) => {
    console.error(error);
});
