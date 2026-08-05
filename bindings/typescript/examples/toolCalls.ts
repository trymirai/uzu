import {
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    SamplingMethodGreedy,
    SamplingPolicyCustom,
    uzuToolFunction,
} from '@trymirai/uzu';
import * as z from 'zod';


const Coordinate = z.object({
    latitude: z.number().describe('Latitude in decimal degrees.'),
    longitude: z.number().describe('Longitude in decimal degrees.'),
});

type Coordinate = z.infer<typeof Coordinate>;


const getCurrentLocation = uzuToolFunction({
    name: 'get_location',
    description: 'Return the current location in coordinates',
    parameters: z.object({}),
    returns: Coordinate,
    handler: (): Coordinate => ({
        latitude: 51.5074,
        longitude: -0.1278,
    }),
});


async function calculateCurrentTemperature({latitude, longitude}: Coordinate): Promise<number> {
    if (!Number.isFinite(Math.hypot(latitude, longitude))) {
        throw new RangeError('Coordinates must be finite');
    }
    return 25;
}

const getCurrentTemperature = uzuToolFunction({
    name: 'get_current_temperature',
    description: 'Return the temperature at the provided coordinates',
    parameters: Coordinate,
    returns: z.number(),
    handler: calculateCurrentTemperature,
});


async function main() {
    const engine = await Engine.create(EngineConfig.create());
    const model = await engine.model('mlx-community/Qwen3.5-9B-MLX-8bit');
    if (!model) {
        throw new Error('Model not found');
    }

    for await (const update of await engine.download(model)) {
        console.log('Download progress:', update.progress);
    }

    const session = await engine.chat(model, ChatConfig.create());
    await session.addTool(getCurrentLocation);
    await session.addTool(getCurrentTemperature);

    const messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('What temperature is it now at my location?'),
    ];
    const config = ChatReplyConfig.create().withSamplingPolicy(
        new SamplingPolicyCustom(new SamplingMethodGreedy()),
    );
    const replies = await session.reply(messages, config);
    const message = replies[replies.length - 1]?.message;
    if (message) {
        console.log('Reasoning:', message.reasoning ?? '');
        console.log('Text:', message.text ?? '');
    }
}

main().catch((error: unknown) => {
    console.error(error);
});
