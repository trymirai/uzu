<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-typescript.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord&color=brightgreen" alt="Discord"></a> <a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-brightgreen" alt="Contact us"></a> <a href="https://docs.trymirai.com"><img src="https://img.shields.io/badge/Read-Docs-brightgreen" alt="Read docs"></a> [![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE) [![Build](https://github.com/trymirai/uzu/actions/workflows/tests.yml/badge.svg)](https://github.com/trymirai/uzu/actions) [![TypeScript](https://img.shields.io/badge/TypeScript-yellow)](bindings/typescript) [![Package](https://img.shields.io/npm/v/@trymirai/uzu?color=yellow&label=Package)](https://www.npmjs.com/package/@trymirai/uzu) [![Downloads](https://img.shields.io/npm/dm/@trymirai/uzu?color=yellow&label=Downloads)]((https://www.npmjs.com/package/@trymirai/uzu)) 

# uzu

A high-performance inference engine for AI models. It allows you to deploy AI directly in your app with **zero latency**, **full data privacy**, and **no inference costs**. Key features:

- Simple, high-level API
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices
- [Broad model support](https://trymirai.com/models)

## Quick Start



Add the dependency:

```bash
pnpm add @trymirai/uzu@0.1.9
```

Run the code below:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@trymirai/uzu';

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
```


<br>
Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

You can run any example via `cargo tools example` \<**typescript**\> \<**chat** | **chat-cloud** | **chat-speculation-classification** | **chat-speculation-summarization** | **chat-structured-output** | **classification** | **quick-start** | **text-to-speech**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

```ts
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
```

<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

```ts
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
```

### Chat using speculation preset for classification

In this example, we will use the `classification` speculation preset to determine the sentiment of the user's input:

```ts
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
```

<br>You can view the stats to see that the answer will be ready immediately after the prefill step, and actual generation won’t even start due to speculative decoding, which significantly improves generation speed.

### Chat using speculation preset for summarization

In this example, we will use the `summarization` speculation preset to generate a summary of the input text:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPresetSummarization, Engine, EngineConfig, ReasoningEffort, SamplingMethodGreedy } from '@trymirai/uzu';

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

    const textToSummarize =
        "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    const prompt = `Text is: "${textToSummarize}". Write only summary itself.`;
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText(prompt)
    ];

    let chatConfig = ChatConfig.create().withSpeculationPreset(new ChatSpeculationPresetSummarization);
    let session = await engine.chat(model, chatConfig);

    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(256).withSamplingMethod(new SamplingMethodGreedy());
    let reply = (await session.reply(messages, chatReplyConfig))[0];

    if (reply) {
        console.log('Summary: ', reply.message.text);
        console.log('Generation t\\s: ', reply.stats.generateTokensPerSecond);
    }
}

main().catch((error) => {
    console.error(error);
});
```

<br>You will notice that the model’s run count is lower than the actual number of generated tokens due to speculative decoding, which significantly improves generation speed.

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

```ts
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

    let model = await engine.model('Qwen/Qwen3-0.6B');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

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
```

### Classification

In this example, we will use a classification model to determine whether the user's input is safe from a moderation perspective:

```ts
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
```

### Text to Speech

In this example, we will generate audio from text:

```ts
import { Engine, EngineConfig } from '@trymirai/uzu';
import { homedir } from "os";
import { join } from "path";

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('fishaudio/s1-mini');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const text = "London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.";
    const outputPath = join(homedir(), "Desktop", "output.wav");
    let session = await engine.textToSpeech(model);
    let pcmBatch = await session.synthesize(text);
    pcmBatch.saveAsWav(outputPath);
    console.log('Output saved to: ', outputPath);
}

main().catch((error) => {
    console.error(error);
});
```



## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
