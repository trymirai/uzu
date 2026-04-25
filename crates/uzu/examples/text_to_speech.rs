use uzu::engine::{Engine, EngineConfig};

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
    let pcm_batch = session.synthesize(text.to_string()).await?;
    pcm_batch.save_as_wav(output_path.to_string_lossy().to_string())?;
    println!("Output saved to: {}", output_path.display());

    Ok(())
}
