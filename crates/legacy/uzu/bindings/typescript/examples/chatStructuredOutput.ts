import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, GrammarJsonSchema, ReasoningEffort } from '@trymirai/uzu';
import * as z from "zod";

const CountryType = z.object({
    name: z.string(),
    capital: z.string(),
});
const CountryListType = z.array(CountryType);

function structuredResponse<T extends z.ZodType>(response: string | null | undefined, type: T): z.infer<T> | undefined {
    if (!response) {
        return undefined;
    }
    const data = JSON.parse(response);
    const result = type.parse(data);
    return result;
}

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

    let schema = z.toJSONSchema(CountryListType);
    let schemaString = JSON.stringify(schema);
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText('Give me a JSON object containing a list of 3 countries, where each country has name and capital fields')
    ];

    let session = await engine.chat(model, ChatConfig.create());
    let reply = await session.reply(messages, ChatReplyConfig.create().withGrammar(new GrammarJsonSchema(schemaString)));
    let message = reply[0]?.message;
    let countries = structuredResponse(message?.text, CountryListType);
    console.log(countries);
}

main().catch((error) => {
    console.error(error);
});
