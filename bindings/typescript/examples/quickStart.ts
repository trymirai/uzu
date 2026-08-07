import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }

    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    console.log();

    let session = await engine.chat(model, ChatConfig.create());

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];

    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;

    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
