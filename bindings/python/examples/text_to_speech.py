import asyncio
from pathlib import Path

from uzu import Engine, EngineConfig


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("fishaudio/s1-mini")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress}")

    text = (
        "London is the capital of United Kingdom and one of the world's most influential cities, "
        "known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. "
        "Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace "
        "with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum "
        "and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year."
    )
    output_path = Path.home() / "Desktop" / "output.wav"
    session = await engine.text_to_speech(model)
    pcm_batch = await session.synthesize(text)
    pcm_batch.save_as_wav(str(output_path))
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
