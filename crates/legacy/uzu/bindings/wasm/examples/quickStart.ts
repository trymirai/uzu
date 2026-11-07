import {ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig} from '@trymirai/uzu-wasm';

async function main() {
    let output = document.getElementById('output')!;

    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }

    for await (const update of await engine.download(model)) {
        output.textContent = `Download progress: ${(update.progress * 100).toFixed(2)}%`;
    }
    output.textContent += 'Model loaded\n'

    let session = await engine.chat(model, ChatConfig.create());
    output.textContent += 'Session created\n'

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];

    output.textContent += 'Inference started\n'
    let replyConfig = ChatReplyConfig.create().withTokenLimit(256)
    let reply = await session.reply(messages, replyConfig);

    let message = reply[0]?.message;

    if (message) {
        output.textContent += `\nReasoning: ${message.reasoning}\nText: ${message.text}`;
    }
}

main().catch((error) => {
    console.error(error);
    document.getElementById('output')!.textContent += `\nError: ${error}`;
});
