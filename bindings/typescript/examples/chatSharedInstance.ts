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

    // The chat instance owns the loaded model and can be shared between sessions.
    let chatInstance = await engine.chatInstance(model, ChatConfig.create());

    let firstSession = await engine.chatWithInstance(chatInstance);
    let replies = await firstSession.reply(
        [ChatMessage.user().withText('Tell me a short, funny story about a robot')],
        ChatReplyConfig.create(),
    );
    let reply = replies[replies.length - 1];
    if (reply) {
        console.log('First session reasoning: ', reply.message.reasoning);
        console.log('First session text: ', reply.message.text);
    }

    // The second session reuses the already-loaded weights instead of loading the model again.
    let secondSession = await engine.chatWithInstance(chatInstance);
    replies = await secondSession.reply(
        [ChatMessage.user().withText('What is the capital of France?')],
        ChatReplyConfig.create(),
    );
    reply = replies[replies.length - 1];
    if (reply) {
        console.log('\nSecond session reasoning: ', reply.message.reasoning);
        console.log('Second session text: ', reply.message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
