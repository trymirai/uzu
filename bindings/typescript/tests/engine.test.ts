import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@trymirai/uzu';

jest.setTimeout(600_000);

test('chat reply produces text', async () => {
    const engine = await Engine.create(EngineConfig.create());

    const model = await engine.model('Qwen/Qwen3-0.6B');
    expect(model).toBeTruthy();

    for await (const update of await engine.download(model!)) {
        console.log('Download progress:', update.progress);
    }

    const session = await engine.chat(model!, ChatConfig.create());

    const messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Hi'),
    ];

    const reply = await session.reply(messages, ChatReplyConfig.create());
    const message = reply[reply.length - 1]?.message;

    expect(message).toBeDefined();
    expect(message!.reasoning).not.toBeNull();
    expect(message!.text).not.toBeNull();
});
