import { ClassificationMessage, Engine, EngineConfig } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('trymirai/chat-moderation-router');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    let messages = [
        ClassificationMessage.user('Hi')
    ];

    let session = await engine.classification(model);
    let output = await session.classify(messages);
    console.log('Output: ', output.probabilities.values);
}

main().catch((error) => {
    console.error(error);
});
