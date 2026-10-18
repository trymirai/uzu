import init, { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@trymirai/uzu/browser';

const repository = 'trymirai/Qwen3.5-0.8B-M';
export async function main() {
    await init();
    const engine = await Engine.create(EngineConfig.create());
    const model = await engine.model(repository);
    if (!model) throw new Error('Model not found');

    const download = await engine.download(model);
    try {
        for (;;) {
            const state = await download.next();
            if (!state) break;
            try {
                console.log(`${state.name}: ${(state.progress * 100).toFixed(1)}%`);
                if ('Error' in state.phase) throw new Error(state.phase.Error.message);
            } finally {
                state.free();
            }
        }
    } finally {
        download.free();
    }
    if (!(await engine.modelPath(model))) throw new Error('Download did not produce a usable model');

    console.log('Loading model');
    const session = await engine.chat(model, ChatConfig.create().withSamplingSeed({ Custom: { seed: 0 } }));
    console.log('Generating reply');
    try {
        const replies = await session.reply(
            [ChatMessage.user().withText('What is 2 + 2 and why?')],
            ChatReplyConfig.create(),
        );
        const reply = replies[0];
        if (!reply) throw new Error('No reply generated');
        const message = reply.message;
        const result = { reasoning: message.reasoning, text: message.text };
        console.log('Reasoning:', result.reasoning);
        console.log('Text:', result.text);
        message.free();
        for (const reply of replies) reply.free();
        return result;
    } finally {
        session.free();
        engine.free();
    }
}
