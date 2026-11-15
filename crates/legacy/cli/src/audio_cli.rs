use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::ChatConfig,
};
use uzu_engine::audio::lfm3_audio::pipeline::GenerationMode;

use crate::common::audio::{AudioRuntime, DEFAULT_ASR_SYSTEM_PROMPT};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum AudioModeArg {
    Text,
    Audio,
    Interleaved,
}

impl From<AudioModeArg> for GenerationMode {
    fn from(arg: AudioModeArg) -> Self {
        match arg {
            AudioModeArg::Text => GenerationMode::Text,
            AudioModeArg::Audio => GenerationMode::Audio,
            AudioModeArg::Interleaved => GenerationMode::Interleaved,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run_audio_cli(
    model_name: String,
    input_path: Option<String>,
    text: Option<String>,
    system: Option<String>,
    output_path: Option<String>,
    mode: Option<AudioModeArg>,
    max_tokens: usize,
    temperature: Option<f32>,
    seed: Option<i64>,
) -> Result<()> {
    if input_path.is_none() && text.as_deref().map(str::trim).is_none_or(str::is_empty) {
        bail!("provide --input <WAV_PATH> (ASR/speech-to-speech) or --text <TEXT> (TTS)");
    }

    let engine = Engine::new(EngineConfig::default()).await?;
    let resolved = crate::common::hf::resolve_model_or_prompt_hf(&engine, &model_name, true).await?;
    let model_dir = match &resolved.accessibility {
        uzu::types::model::ModelAccessibility::OnDevice {
            source: uzu::types::model::ModelSource::Filesystem { path },
        } => PathBuf::from(path),
        _ => bail!("Audio models require a local filesystem model directory"),
    };

    let session = engine
        .chat(resolved, ChatConfig::default())
        .await
        .context("Failed to create chat session")?;
    let backend_inst = session
        .chat_token_backend_instance()
        .await
        .context("No backend instance available")?;
    let uzu_backend = backend_inst
        .as_any()
        .downcast_ref::<uzu::engine::bridge::UzuChatTokenBackendInstance<uzu_engine::backends::metal::Metal>>()
        .context("Backend is not Metal")?;
    let lm = uzu_backend.language_model().clone();

    let mut runtime = AudioRuntime::load(&model_dir, &lm).await?;
    let wav_in = input_path
        .as_deref()
        .map(|p| std::fs::read(p).with_context(|| format!("failed to read input WAV '{p}'")))
        .transpose()?;

    let resolved_mode = mode.map(GenerationMode::from).or_else(|| {
        if output_path.is_some() {
            Some(GenerationMode::Audio)
        } else if wav_in.is_some() {
            Some(GenerationMode::Text)
        } else {
            None
        }
    });
    let decode_audio = output_path.is_some();
    let prompt = text.unwrap_or_default();
    let default_sys = if wav_in.is_some() && matches!(resolved_mode, Some(GenerationMode::Text)) {
        Some(DEFAULT_ASR_SYSTEM_PROMPT)
    } else {
        None
    };
    let sys_ref = system.as_deref().or(default_sys);

    let out = runtime.generate(
        &lm,
        wav_in.as_deref(),
        sys_ref,
        &prompt,
        resolved_mode,
        decode_audio,
        max_tokens,
        temperature,
        seed,
    )?;

    if let (Some(path), Some(wav)) = (&output_path, &out.wav_bytes) {
        std::fs::write(path, wav).with_context(|| format!("failed to write output WAV '{path}'"))?;
        eprintln!(
            "🔊 Saved {} frames ({} Hz) to {path} [prep={:.1}ms prefill={:.1}ms depth={:.1}ms backbone={:.1}ms codec={:.1}ms]",
            out.frames.len(),
            out.sample_rate,
            out.prep_ms,
            out.prefill_ms,
            out.depth_ms,
            out.backbone_ms,
            out.codec_ms
        );
    } else {
        eprintln!(
            "📝 [prep={:.1}ms prefill={:.1}ms depth={:.1}ms backbone={:.1}ms tokens={}]",
            out.prep_ms,
            out.prefill_ms,
            out.depth_ms,
            out.backbone_ms,
            out.tokens.len()
        );
    }
    if !out.text.is_empty() {
        println!("{}", out.text);
    }
    Ok(())
}
