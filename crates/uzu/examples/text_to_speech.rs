use uzu::{
    engine::{Engine, EngineConfig},
    session::text_to_speech::TextToSpeechSessionStreamChunk,
    types::basic::PcmBatch,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("fishaudio/s1-mini".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let text = "London is the capital of United Kingdom and one of the world's most influential cities, \
        known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. \
        Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace \
        with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum \
        and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.";
    let output_path = dirs::home_dir().ok_or("Home not found")?.join("Desktop").join("output.wav");

    let session = engine.text_to_speech(model).await?;
    let stream = session.synthesize_stream(text.to_string()).await;
    let mut pcm_batches: Vec<PcmBatch> = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            TextToSpeechSessionStreamChunk::Output {
                output,
            } => {
                pcm_batches.push(output.pcm_batch);
            },
            TextToSpeechSessionStreamChunk::Error {
                error,
            } => {
                println!("Error: {error}");
            },
        }
    }

    let pcm_batch_first = pcm_batches.first().ok_or("No batches")?;
    let pcm_batch_full = PcmBatch {
        samples: pcm_batches.iter().flat_map(|batch| batch.samples.iter().copied()).collect(),
        sample_rate: pcm_batch_first.sample_rate,
        channels: pcm_batch_first.channels,
        lengths: vec![pcm_batches.iter().flat_map(|batch| batch.lengths.iter().copied()).sum()],
    };
    pcm_batch_full.save_as_wav(output_path.to_string_lossy().to_string())?;
    println!("Output saved to: {}", output_path.display());

    Ok(())
}
