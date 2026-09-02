import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort } from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create().withOpenaiApiKey('OPENAI_API_KEY');
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('gpt-5');
    if (!model) {
        throw new Error('Model not found');
    }

    let messages = [
        ChatMessage.system().withReasoningEffort("Low" as ReasoningEffort),
        ChatMessage.user().withText('How LLMs work')
    ];

    let session = await engine.chat(model, ChatConfig.create());
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
