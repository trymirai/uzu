<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-typescript.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord" alt="Discord"></a>
<a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
<a href="https://docs.trymirai.com/app-integration/overview"><img src="https://img.shields.io/badge/Read-Docs-blue" alt="Read docs"></a>
[![npm (scoped)](https://img.shields.io/npm/v/%40trymirai%2Fuzu)](https://www.npmjs.com/package/@trymirai/uzu)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# uzu-ts

Node package for [uzu](https://github.com/trymirai/uzu), a **high-performance** inference engine for AI models on Apple Silicon. It allows you to deploy AI directly in your app with **zero latency**, **full data privacy**, and **no inference costs**. You don’t need an ML team or weeks of setup - one developer can handle everything in minutes. Key features:

- Simple, high-level API
- Specialized configurations with significant performance boosts for common use cases like classification and summarization
- [Broad model support](https://trymirai.com/models)

## Quick Start

Add the `uzu` dependency to your project's `package.json`:

```json
"dependencies": {
    "@trymirai/uzu": "{{VERSION}}"
}
```

Run the snippet below:

```ts
// include:examples/snippets.ts#quick-start
```

Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

Run it using one of the following commands:

```bash
pnpm run tsn examples/chat.ts
pnpm run tsn examples/summarization.ts
pnpm run tsn examples/classification.ts
pnpm run tsn examples/cloud.ts
pnpm run tsn examples/structuredOutput.ts
pnpm run tsn examples/classifier.ts
pnpm run tsn examples/textToSpeech.ts
```

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

```ts
// include:examples/chat.ts
```

### Summarization

In this example, we will use the `summarization` preset to generate a summary of the input text:

```ts
// include:examples/summarization.ts
```

You will notice that the model’s run count is lower than the actual number of generated tokens due to speculative decoding, which significantly improves generation speed.

### Classification

In this example, we will use the `classification` preset to determine the sentiment of the user's input:

```ts
// include:examples/classification.ts
```

You can view the stats to see that the answer will be ready immediately after the prefill step, and actual generation won’t even start due to speculative decoding, which significantly improves generation speed.

### Cloud

Sometimes you want to create a complex pipeline where some requests are processed on-device and the more complex ones are handled in the cloud using a larger model. With `uzu`, you can do this easily: just choose the cloud model you want to use and perform all requests through the same API:

```ts
// include:examples/cloud.ts
```

### Structured Output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema, or define the schema using [Zod](https://github.com/colinhacks/zod).

```ts
// include:examples/structuredOutput.ts
```

## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
