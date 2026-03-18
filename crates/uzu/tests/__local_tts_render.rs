#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use uzu::{
    audio::AudioPcmBatch,
    backends::metal::Metal,
    prelude::{Input, TtsSession},
    session::config::TtsRunConfig,
};

fn env_var(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("missing env var: {name}"))
}

fn write_pcm_batch_as_wav(
    pcm: &AudioPcmBatch,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let channels = pcm.channels() as u16;
    let sample_rate = pcm.sample_rate();
    let bits_per_sample = 16u16;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let mut pcm_bytes = Vec::with_capacity(pcm.samples().len() * 2);
    for &sample in pcm.samples() {
        let clamped = sample.clamp(-1.0, 1.0);
        let quantized = (clamped * f32::from(i16::MAX)).round() as i16;
        pcm_bytes.extend_from_slice(&quantized.to_le_bytes());
    }

    let data_len = pcm_bytes.len() as u32;
    let riff_len = 36u32 + data_len;
    let output_path = PathBuf::from(path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_len.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    writer.write_all(&pcm_bytes)?;
    writer.flush()?;
    Ok(())
}

#[test]
fn local_tts_render() {
    let model_path = PathBuf::from(env_var("LOCAL_TTS_MODEL_PATH"));
    let output_wav = env_var("LOCAL_TTS_OUTPUT_WAV");
    let prompt = env_var("LOCAL_TTS_PROMPT");
    let runs: usize = env_var("LOCAL_TTS_RUNS").parse().expect("LOCAL_TTS_RUNS");
    let seed: u64 = env_var("LOCAL_TTS_SEED").parse().expect("LOCAL_TTS_SEED");

    let mut session = TtsSession::<Metal>::new(model_path).expect("tts session");
    let input = Input::Text(prompt);
    let mut last_pcm = None;

    for run_index in 0..runs {
        let pcm = session
            .synthesize_with_seed_and_config(input.clone(), seed, &TtsRunConfig::default())
            .expect("synthesize");
        let stats = session.last_execution_stats().expect("execution stats");
        let audio_seconds = pcm.total_frames() as f64 / pcm.sample_rate() as f64;
        let wall_seconds = stats.semantic_decode_seconds + stats.audio_decode_seconds;
        let rtf = wall_seconds / audio_seconds;
        println!(
            "run={} wall_s={:.6} audio_s={:.6} rtf={:.6} semantic_s={:.6} audio_decode_s={:.6} command_buffers={} host_waits={} semantic_frames={} audio_decode_calls={}",
            run_index + 1,
            wall_seconds,
            audio_seconds,
            rtf,
            stats.semantic_decode_seconds,
            stats.audio_decode_seconds,
            stats.command_buffers_submitted,
            stats.host_waits,
            stats.semantic_frames,
            stats.audio_decode_calls,
        );
        last_pcm = Some(pcm);
    }

    let last_pcm = last_pcm.expect("last pcm");
    write_pcm_batch_as_wav(&last_pcm, &output_wav).expect("write wav");
    println!("wrote_wav={output_wav}");
}
