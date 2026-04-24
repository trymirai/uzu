import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSessionStreamChunkError, ChatSessionStreamChunkReplies, Engine, EngineConfig } from '@trymirai/uzu';

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

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];
    let session = await engine.chat(model, ChatConfig.create());
    let stream = await session.replyWithStream(messages, ChatReplyConfig.create());
    let message: ChatMessage | undefined;
    for await (const chunk of stream) {
        if (chunk instanceof ChatSessionStreamChunkReplies) {
            message = chunk.replies[0]?.message;
            console.log('Generated tokens: ', chunk.replies[0]?.stats.tokensCountOutput);
        } else if (chunk instanceof ChatSessionStreamChunkError) {
            console.error('Error: ', chunk.error);
        }
    }
    console.log('Reasoning: ', message?.reasoning);
    console.log('Text: ', message?.text);
}

main().catch((error) => {
    console.error(error);
});
