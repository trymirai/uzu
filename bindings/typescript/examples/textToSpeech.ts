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
    let output = await session.synthesize(text);
    output.pcmBatch.saveAsWav(outputPath);
    console.log('Output saved to: ', outputPath);
}

main().catch((error) => {
    console.error(error);
});
