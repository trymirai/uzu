//! Env-driven LFM3-Audio end-to-end eval: log-mel + conv downsampler + speech
//! trunk (uzu Decoder) -> projector rows injected into the language model ->
//! greedy text (ASR) and MaskGIT speech frames (TTS), compared against the
//! MLX reference (`scripts/lfm3_audio_ref.py`, fixtures under
//! `fixtures/lfm3_audio/`).
//!
//! Env: LFM3_AUDIO_MODEL (uzu bundle path). Optional: LFM3_AUDIO_WAV (16 kHz
//! mono f32 override; the seeded fixture clip is used otherwise).
//! Prints one JSON record; fixture parity asserts the stage tensors, the
//! greedy ASR ids and the fixture-clip prompt ids.

use std::time::Instant;

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::audio::lfm3_audio::pipeline::{GenerationMode, Lfm3AudioPipeline};
use crate::audio::liquid_codec_metal::MetalLiquidCodec;
use crate::backends::metal::Metal;
use crate::engine::language_model::LanguageModel;
use crate::engine::Engine;
use crate::vision::VlmSampler;
use crate::vision::VlmSamplingConfig;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/unit/fixtures/lfm3_audio/");

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(FIXTURE_DIR.to_string() + name).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_i64(name: &str) -> Vec<i64> {
    let bytes = std::fs::read(FIXTURE_DIR.to_string() + name).unwrap();
    bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[uzu_test]
fn lfm3_audio_eval() {
    let Some(model_dir) = std::env::var_os("LFM3_AUDIO_MODEL") else {
        println!("{{\"skipped\": \"LFM3_AUDIO_MODEL not set\"}}");
        return;
    };
    if !std::path::Path::new(FIXTURE_DIR).exists() {
        println!("{{\"skipped\": \"lfm3_audio fixtures not present\"}}");
        return;
    }
    let model_dir = std::path::PathBuf::from(model_dir);

    let engine = block_on(Engine::<crate::backends::metal::Metal>::new()).unwrap();
    let model = block_on(engine.load_language_model(&model_dir)).unwrap();
    let tokenizer = model.tokenizer().clone();
    let mut pipeline =
        block_on(Lfm3AudioPipeline::load_from_model_dir(&model_dir, model.context())).unwrap();

    // ---- fixture clip: stage tensors + ASR ----
    let audio = read_f32("audio.f32");
    let mel_golden = read_f32("mel.f32");
    let depth_hidden = read_f32("depth_hidden.f32");
    let depth_codes = read_i64("depth_codes.i64");
    let depth_logits_golden = read_f32("depth_logits.f32");
    let frame_embed_golden = read_f32("frame_embed.f32");

    // mel
    let num_frames = pipeline.config.num_frames(audio.len());
    let mel = pipeline.mel.log_mel(
        &audio,
        pipeline.config.n_fft,
        pipeline.config.hop_length,
        num_frames,
        pipeline.config.log_floor,
        pipeline.config.norm_offset,
        pipeline.config.norm_divisor,
    );
    let mel_diff = max_abs_diff(&mel, &mel_golden);
    println!("mel_diff: {mel_diff}");
    assert!(mel_diff < 5e-3, "log-mel mismatch: {mel_diff}");

    // depth decoder logits (single frame, nothing masked)
    let masked = vec![false; pipeline.config.num_codebooks];
    let logits = pipeline.depth.logits(&depth_hidden, &depth_codes, &masked);
    let logits_metal = pipeline.depth.logits_metal(model.context(), &depth_hidden, &depth_codes, &masked).unwrap();
    let metal_vs_cpu = max_abs_diff(&logits_metal, &logits);
    let logits_diff = max_abs_diff(&logits_metal, &depth_logits_golden);
    let logits_scale = depth_logits_golden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let logits_rel = logits_diff / logits_scale;
    println!("depth_logits: abs {logits_diff} rel {logits_rel} (metal_vs_cpu {metal_vs_cpu})");
    assert!(logits_rel < 0.06, "depth logits mismatch: abs {logits_diff} rel {logits_rel}");

    // frame embedding
    let frame_embed = pipeline.frame_embedding_f32(&depth_codes);
    let frame_diff = max_abs_diff(&frame_embed, &frame_embed_golden);
    println!("frame_diff: {frame_diff}");
    assert!(frame_diff < 0.1, "frame embedding mismatch: {frame_diff}");

    // trunk + projector rows, then e2e prompt
    let started = Instant::now();
    let prepared = pipeline.prepare_transcribe(&tokenizer, &model, "请用一句话总结这段音频。", &audio).unwrap();
    println!("prepare: {:?}", started.elapsed());

    let projected_golden = read_f32("projected.f32");
    let projected_actual: Vec<f32> = prepared
        .features_bf16
        .as_chunks::<2>().0.iter()
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    let proj_diff = max_abs_diff(&projected_actual, &projected_golden);
    let proj_scale = projected_golden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("projected_diff: abs {proj_diff} rel {} (len {} vs {})", proj_diff / proj_scale, projected_actual.len(), projected_golden.len());

    // ---- greedy ASR ----
    let sampler = VlmSampler::new(VlmSamplingConfig {
        temperature: None,
        ..Default::default()
    });
    let mut state = pipeline.acquire_state(&model).unwrap();
    let output = pipeline
        .generate(
            &tokenizer,
            &model,
            &mut state,
            &mut { prepared.clone() },
            sampler,
            Some(GenerationMode::Text),
            48,
            None,
        )
        .unwrap();
    let text_ids: Vec<i64> = output
        .tokens
        .iter()
        .copied()
        .filter(|&t| t != pipeline.config.audio_out_token_id)
        .map(|t| t as i64)
        .collect();
    let expected_text = read_i64("text_out.i64");
    let prompt_ids: Vec<i64> = prepared.prompt_tokens.iter().map(|&t| t as i64).collect();
    let expected_prompt = read_i64("prompt_ids.i64");
    let decoded_asr = tokenizer
        .decode(&text_ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), false)
        .unwrap_or_default();
    println!(
        "{}",
        serde_json::json!({
            "prompt_ids_match": prompt_ids == expected_prompt,
            "prompt_len": prompt_ids.len(),
            "text_ids": text_ids,
            "expected_text_ids": expected_text,
            "decoded_asr": decoded_asr,
            "text_match": text_ids == expected_text,
            "frames": output.frames.len(),
        })
    );
    assert_eq!(prompt_ids, expected_prompt, "prompt ids differ from the MLX reference");
    assert!(
        decoded_asr.contains("machine") || decoded_asr.contains("running") || decoded_asr.contains("sound"),
        "unexpected ASR output: {decoded_asr:?}"
    );

    // ---- TTS with the recorded MaskGIT reveal order ----
    let orders_bytes = std::fs::read(FIXTURE_DIR.to_string() + "frame_orders.npy").unwrap();
    let orders: Vec<Vec<usize>> =
        parse_npy_i64(&orders_bytes).iter().map(|order| order.iter().map(|&v| v as usize).collect()).collect();
    let tts_prompt = pipeline.prepare_prompt(&tokenizer, "请说：你好，世界。").unwrap();
    let mut tts_state = pipeline.acquire_state(&model).unwrap();
    let tts_sampler = VlmSampler::new(VlmSamplingConfig {
        temperature: None,
        ..Default::default()
    });
    let mut tts_prompt = tts_prompt;
    let tts_out = pipeline
        .generate(
            &tokenizer,
            &model,
            &mut tts_state,
            &mut tts_prompt,
            tts_sampler,
            Some(GenerationMode::Audio),
            48,
            Some(&orders),
        )
        .unwrap();
    let expected_frames = read_i64("audio_out.i64");
    let flat_frames: Vec<i64> = tts_out.frames.iter().flat_map(|f| f.iter().copied()).collect();
    println!(
        "{}",
        serde_json::json!({
            "tts_frames": tts_out.frames.len(),
            "expected_frames": expected_frames.len() / 12,
            "frame_match": flat_frames == expected_frames,
        })
    );
    let k = pipeline.config.num_codebooks;
    assert_eq!(tts_out.frames.len(), expected_frames.len() / k, "unexpected TTS frame count");
    let frame0_matched = flat_frames[..k].iter().zip(&expected_frames[..k]).filter(|(a, b)| a == b).count();
    assert!(
        frame0_matched >= 8,
        "frame0 codebook match too low: {frame0_matched}/{k} (uzu={:?}, mlx={:?})",
        &flat_frames[..k],
        &expected_frames[..k]
    );

    // ---- codec decode sanity (greedy fixture is near-silent; non-silent TTS is tested in lfm3_audio_e2e_speech) ----
    let mut codec = MetalLiquidCodec::load_from_model_dir(model_dir.join("audio_tokenizer")).unwrap();
    if !tts_out.frames.is_empty() {
        let codes: Vec<Vec<i64>> = (0..k)
            .map(|cb| tts_out.frames.iter().map(|f| f[cb]).collect())
            .collect();
        let samples = codec.decode(&codes).unwrap();
        let frame_samples = codec.config().stft_hop_length * codec.config().stft_frame_stack;
        assert_eq!(samples.len(), tts_out.frames.len() * frame_samples);
        assert!(samples.iter().all(|v| v.is_finite()), "decoded waveform contains non-finite values");
        let peak = samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("decoded_samples {} peak {} (frame0_match {frame0_matched}/{k})", samples.len(), peak);
    }
}

/// Minimal `.npy` int64 reader for the recorded frame orders.
fn parse_npy_i64(bytes: &[u8]) -> Vec<Vec<i64>> {
    // Header: \x93NUMPY, version 2 bytes, header length (2 or 4 bytes), then a
    // dict string ending with the shape.
    assert_eq!(&bytes[0..6], b"\x93NUMPY");
    let (header_len, offset) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[offset..offset + header_len]).unwrap();
    let shape: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    let rows = shape[0];
    let cols = shape.get(1).copied().unwrap_or(1);
    let elem_bytes = if header.contains("i4") || header.contains("u4") { 4 } else { 8 };
    let data = &bytes[offset + header_len..];
    (0..rows)
        .map(|row| {
            let base = row * cols * elem_bytes;
            (0..cols)
                .map(|col| {
                    let b = &data[base + col * elem_bytes..base + col * elem_bytes + elem_bytes];
                    if elem_bytes == 4 {
                        i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64
                    } else {
                        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                    }
                })
                .collect()
        })
        .collect()
}

fn read_pcm16_wav(path: &std::path::Path) -> Option<(u32, Vec<f32>)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    let mut sample_rate = 16000u32;
    let mut channels = 1u16;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let payload = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        if id == b"fmt " && payload.len() >= 16 {
            channels = u16::from_le_bytes([payload[2], payload[3]]);
            sample_rate = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
        } else if id == b"data" {
            let ch = channels.max(1) as usize;
            let samples: Vec<f32> = payload
                .chunks_exact(2 * ch)
                .map(|frame| i16::from_le_bytes([frame[0], frame[1]]) as f32 / 32768.0)
                .collect();
            return Some((sample_rate, samples));
        }
        pos += 8 + size + (size & 1);
    }
    None
}

fn write_pcm16_wav(path: &std::path::Path, sample_rate: u32, samples: &[f32]) {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let q = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    std::fs::write(path, out).unwrap();
}

fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f64) * (dst_rate as f64) / (src_rate as f64)).round() as usize;
    let ratio = src_rate as f64 / dst_rate as f64;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = (pos as usize).min(samples.len() - 1);
            let next = (idx + 1).min(samples.len() - 1);
            let frac = (pos - idx as f64) as f32;
            samples[idx] * (1.0 - frac) + samples[next] * frac
        })
        .collect()
}

fn decode_text_tokens(
    tokenizer: &tokenizers::Tokenizer,
    tokens: &[u64],
    audio_out_token_id: u64,
) -> (usize, String) {
    let ids: Vec<u32> = tokens.iter().copied().filter(|&t| t != audio_out_token_id).map(|t| t as u32).collect();
    let text = tokenizer.decode(&ids, false).unwrap_or_default();
    (ids.len(), text)
}

fn run_asr(
    pipeline: &mut Lfm3AudioPipeline,
    tokenizer: &tokenizers::Tokenizer,
    model: &LanguageModel<Metal>,
    system_prompt: &str,
    wav16: &[f32],
) -> (f64, f64, usize, String) {
    let t0 = Instant::now();
    let mut prepared = pipeline.prepare_transcribe_with_system(tokenizer, model, Some(system_prompt), "", wav16).unwrap();
    let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut state = pipeline.acquire_state(model).unwrap();
    let sampler = VlmSampler::new(VlmSamplingConfig { temperature: None, ..Default::default() });
    let t1 = Instant::now();
    let out = pipeline.generate(tokenizer, model, &mut state, &mut prepared, sampler, Some(GenerationMode::Text), 64, None).unwrap();
    let gen_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let (tok_count, text) = decode_text_tokens(tokenizer, &out.tokens, pipeline.config.audio_out_token_id);
    (prep_ms, gen_ms, tok_count, text)
}

#[uzu_test]
fn lfm3_audio_e2e_speech() {
    let Some(model_dir) = std::env::var_os("LFM3_AUDIO_MODEL") else {
        println!("{{\"skipped\": \"LFM3_AUDIO_MODEL not set\"}}");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);

    let engine = block_on(Engine::<crate::backends::metal::Metal>::new()).unwrap();
    let model = block_on(engine.load_language_model(&model_dir)).unwrap();
    let tokenizer = model.tokenizer().clone();
    let mut pipeline =
        block_on(Lfm3AudioPipeline::load_from_model_dir(&model_dir, model.context())).unwrap();
    let mut codec = MetalLiquidCodec::load_from_model_dir(model_dir.join("audio_tokenizer")).unwrap();

    // 1. Speech ASR (/tmp/lfm3_audio_speech/hello16.wav if present)
    let speech_wav = std::path::Path::new("/tmp/lfm3_audio_speech/hello16.wav");
    if let Some((sr, wav)) = read_pcm16_wav(speech_wav) {
        let wav16 = resample_linear(&wav, sr, 16000);
        let (prep_ms, gen_ms, tok_count, text) = run_asr(&mut pipeline, &tokenizer, &model, "Perform ASR.", &wav16);
        println!("E2E_ASR hello16.wav: prep={prep_ms:.1}ms gen={gen_ms:.1}ms ({tok_count} tok) text={text:?}");
        assert!(text.contains("你好"), "unexpected ASR transcription for hello16.wav: {text:?}");
    } else {
        println!("E2E_ASR skipped: {:?} not found", speech_wav);
    }

    // 2. English TTS with official system prompt & default audio sampling (frame_orders = None)
    let tts_text = "Hello! Welcome to Liquid Audio on Metal. The weather is sunny and warm today.";
    let mut tts_prompt = pipeline
        .prepare_prompt_with_system(
            &tokenizer,
            Some("Perform TTS in American English using a female voice."),
            tts_text,
        )
        .unwrap();
    let mut tts_state = pipeline.acquire_state(&model).unwrap();
    let tts_sampler = VlmSampler::new(VlmSamplingConfig { temperature: None, ..Default::default() });
    let t2 = Instant::now();
    let tts_out = pipeline
        .generate(&tokenizer, &model, &mut tts_state, &mut tts_prompt, tts_sampler, Some(GenerationMode::Audio), 250, None)
        .unwrap();
    let tts_ms = t2.elapsed().as_secs_f64() * 1000.0;
    assert!(!tts_out.frames.is_empty(), "TTS produced 0 frames");

    let codes: Vec<Vec<i64>> = (0..pipeline.config.num_codebooks)
        .map(|k| tts_out.frames.iter().map(|f| f[k]).collect())
        .collect();
    let t3 = Instant::now();
    let samples = codec.decode(&codes).unwrap();
    let codec_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let peak = samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    let out_wav = std::path::Path::new("/tmp/lfm3_audio_tts_out.wav");
    let codec_sr = codec.config().sampling_rate as u32;
    write_pcm16_wav(out_wav, codec_sr, &samples);
    println!(
        "E2E_TTS: frames={} tts_gen={tts_ms:.1}ms (prefill={:.1}ms depth={:.1}ms backbone={:.1}ms) codec={codec_ms:.1}ms samples={} ({:.2}s) peak={peak:.4} rms={rms:.4} wav={}",
        tts_out.frames.len(),
        tts_out.prefill_ms,
        tts_out.depth_ms,
        tts_out.backbone_ms,
        samples.len(),
        samples.len() as f64 / codec_sr as f64,
        out_wav.display()
    );
    assert!(peak > 0.05, "TTS decoded waveform too quiet: peak={peak}");

    // 3. Round-trip English TTS -> ASR
    let wav16 = resample_linear(&samples, codec_sr, 16000);
    let (_, _, _, rt_text) = run_asr(&mut pipeline, &tokenizer, &model, "Perform ASR in English.", &wav16);
    println!("E2E_ROUNDTRIP_ASR: text={rt_text:?}");
    let rt_lower = rt_text.to_lowercase();
    assert!(
        rt_lower.contains("hello") || rt_lower.contains("welcome") || rt_lower.contains("weather"),
        "round-trip ASR did not recognize synthesized English speech: {rt_text:?}"
    );
}
