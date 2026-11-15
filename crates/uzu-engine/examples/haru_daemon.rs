use std::{
    env,
    fs::File,
    io::{BufReader, Read, Write},
    net::Shutdown,
    os::unix::net::{UnixListener, UnixStream},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::json;
use uzu_engine::haru::{HaruFrame, HaruPipeline, HaruSessionConfig, session::HaruPrompt};

// Wire protocol framing matching Haru native.py (`<BI` kind:u8 + len:u32 LE + payload)
const MSG_AUDIO: u8 = 0x01;
const MSG_START: u8 = 0x02;
const MSG_PREPARE: u8 = 0x03;
const MSG_REFERENCE: u8 = 0x04;

const MSG_HELLO: u8 = 0x10;
const MSG_FRAME: u8 = 0x11;
const MSG_TEXT: u8 = 0x12;
const MSG_ASSETS: u8 = 0x13;
const MSG_ERROR: u8 = 0x14;
const MSG_READY: u8 = 0x15;

struct CliArgs {
    model_dir: PathBuf,
    verify_goldens: Option<PathBuf>,
    bench_frames: Option<usize>,
    socket_path: Option<PathBuf>,
    sink_window: Option<u32>,
}

fn parse_args() -> Result<CliArgs> {
    let mut args = env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut verify_goldens: Option<PathBuf> = None;
    let mut bench_frames: Option<usize> = None;
    let mut socket_path: Option<PathBuf> = None;
    let mut sink_window: Option<u32> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                let p = args.next().ok_or_else(|| anyhow!("--model requires a path"))?;
                model_dir = Some(PathBuf::from(p));
            },
            "--verify-goldens" => {
                let p = args.next().ok_or_else(|| anyhow!("--verify-goldens requires a dir"))?;
                verify_goldens = Some(PathBuf::from(p));
            },
            "--bench" => {
                let n = args.next().ok_or_else(|| anyhow!("--bench requires frame count"))?;
                bench_frames = Some(n.parse::<usize>().context("invalid --bench integer")?);
            },
            "--socket" => {
                let p = args.next().ok_or_else(|| anyhow!("--socket requires a path"))?;
                socket_path = Some(PathBuf::from(p));
            },
            "--sink-window" => {
                let n = args.next().ok_or_else(|| anyhow!("--sink-window requires frame count (0=off)"))?;
                let w = n.parse::<u32>().context("invalid --sink-window integer")?;
                sink_window = (w > 0).then_some(w);
            },
            "-h" | "--help" => {
                println!(
                    "Usage: haru_daemon --model <dir> [--verify-goldens <dir>] [--bench <N>] [--socket <path>] [--sink-window <W>]"
                );
                std::process::exit(0);
            },
            other => bail!("Unknown argument: {other}"),
        }
    }

    let model_dir = model_dir.ok_or_else(|| anyhow!("--model <dir> is required"))?;
    Ok(CliArgs {
        model_dir,
        verify_goldens,
        bench_frames,
        socket_path,
        sink_window,
    })
}

#[derive(Deserialize)]
struct TensorSlice {
    offset: usize,
    len: usize,
}

#[derive(Deserialize)]
struct GoldensManifest {
    silence_latent: TensorSlice,
    proj_in_raw: TensorSlice,
    model_proj_out: TensorSlice,
    user_proj_out: TensorSlice,
    codec_pcm_in: TensorSlice,
    codec_encode_out: TensorSlice,
    codec_decode_out: TensorSlice,
    head_cond: TensorSlice,
    head_noise: TensorSlice,
    guidance_scale: f32,
    head_norm_out: TensorSlice,
    head_raw_out: TensorSlice,
    prompt_tokens: TensorSlice,
    prompt_model_latents: TensorSlice,
    prompt_user_latents: TensorSlice,
    prompt_model_mask: TensorSlice,
    prompt_user_mask: TensorSlice,
    prompt_hidden_out: TensorSlice,
    prompt_pad_logit: f32,
    prompt_eop_logit: f32,
    prompt_argmax_token: u32,
    prompt_top5_ids: Vec<u32>,
}

fn read_f32_slice(bin: &[u8], s: &TensorSlice) -> Result<Vec<f32>> {
    let end = s.offset + s.len * 4;
    if end > bin.len() {
        bail!("Slice out of bounds: offset={} len={} bin={}", s.offset, s.len, bin.len());
    }
    let mut out = Vec::with_capacity(s.len);
    for &chunk in bin[s.offset..end].as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(chunk));
    }
    Ok(out)
}

fn read_u32_slice(bin: &[u8], s: &TensorSlice) -> Result<Vec<u32>> {
    let end = s.offset + s.len * 4;
    if end > bin.len() {
        bail!("Slice out of bounds: offset={} len={} bin={}", s.offset, s.len, bin.len());
    }
    let mut out = Vec::with_capacity(s.len);
    for &chunk in bin[s.offset..end].as_chunks::<4>().0 {
        out.push(u32::from_le_bytes(chunk));
    }
    Ok(out)
}

struct Metrics {
    cos: f64,
    mae: f64,
    rmse: f64,
    max_abs: f64,
    snr_db: f64,
    all_finite: bool,
}

fn compare_vecs(got: &[f32], want: &[f32]) -> Result<Metrics> {
    if got.len() != want.len() || got.is_empty() {
        bail!("length mismatch: got {} vs want {}", got.len(), want.len());
    }
    let mut dot = 0.0f64;
    let mut norm_g = 0.0f64;
    let mut norm_w = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut all_finite = true;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if !g.is_finite() || !w.is_finite() {
            all_finite = false;
        }
        let gf = g as f64;
        let wf = w as f64;
        dot += gf * wf;
        norm_g += gf * gf;
        norm_w += wf * wf;
        let diff = (gf - wf).abs();
        sum_abs += diff;
        sum_sq += diff * diff;
        if diff > max_abs {
            max_abs = diff;
        }
    }
    let n = got.len() as f64;
    let cos = if norm_g > 0.0 && norm_w > 0.0 {
        dot / (norm_g.sqrt() * norm_w.sqrt())
    } else {
        0.0
    };
    let mae = sum_abs / n;
    let rmse = (sum_sq / n).sqrt();
    let snr_db = if sum_sq > 0.0 && norm_w > 0.0 {
        10.0 * (norm_w / sum_sq).log10()
    } else if sum_sq == 0.0 {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    Ok(Metrics {
        cos,
        mae,
        rmse,
        max_abs,
        snr_db,
        all_finite,
    })
}

fn record_metric(failures: &mut Vec<String>, name: &str, m: &Metrics, min_cos: f64) {
    println!(
        "[golden] {name:<26} finite={:<5} cos={:.6} mae={:.6} rmse={:.6} max_abs={:.6} snr={:.2}dB",
        m.all_finite, m.cos, m.mae, m.rmse, m.max_abs, m.snr_db
    );
    if !m.all_finite {
        failures.push(format!("{name}: non-finite values detected"));
    } else if m.cos < min_cos {
        failures.push(format!("{name}: cosine {:.6} < threshold {:.6}", m.cos, min_cos));
    }
}

fn verify_goldens(pipeline: &mut HaruPipeline, dir: &Path) -> Result<()> {
    println!("=== Verifying Goldens from {} ===", dir.display());
    let manifest_bytes = std::fs::read(dir.join("goldens.json"))?;
    let m: GoldensManifest = serde_json::from_slice(&manifest_bytes)?;
    let bin = std::fs::read(dir.join("goldens.bin"))?;
    let mut failures = Vec::new();

    // 1. silence_latent (dynamically encoded via U8 codec vs BF16 reference)
    let g_silence = read_f32_slice(&bin, &m.silence_latent)?;
    let got_silence = pipeline.codec.silence_latent().to_vec();
    record_metric(&mut failures, "codec.silence_latent", &compare_vecs(&got_silence, &g_silence)?, 0.995);

    // 2. model_audio_proj & user_audio_proj
    let proj_in = read_f32_slice(&bin, &m.proj_in_raw)?;
    let g_model_proj = read_f32_slice(&bin, &m.model_proj_out)?;
    let g_user_proj = read_f32_slice(&bin, &m.user_proj_out)?;
    let got_model_proj = pipeline.run_audio_proj_f32(false, &proj_in)?;
    let got_user_proj = pipeline.run_audio_proj_f32(true, &proj_in)?;
    record_metric(&mut failures, "model_audio_proj", &compare_vecs(&got_model_proj, &g_model_proj)?, 0.999);
    record_metric(&mut failures, "user_audio_proj", &compare_vecs(&got_user_proj, &g_user_proj)?, 0.999);

    // 3. codec.encode_f32 (frame-by-frame streaming over 3 frames)
    let pcm_in = read_f32_slice(&bin, &m.codec_pcm_in)?;
    let g_enc = read_f32_slice(&bin, &m.codec_encode_out)?;
    let fs = pipeline.config.frame_samples as usize;
    let num_frames = pcm_in.len() / fs;
    let mut enc_state = pipeline.codec.create_encoder_state(pipeline.context())?;
    let mut got_enc = Vec::with_capacity(g_enc.len());
    for i in 0..num_frames {
        let lat = pipeline.codec_encode_f32(&pcm_in[i * fs..(i + 1) * fs], &mut enc_state)?;
        got_enc.extend_from_slice(&lat);
    }
    record_metric(&mut failures, "codec.encode_f32", &compare_vecs(&got_enc, &g_enc)?, 0.98);

    // 4. codec.decode_f32 (frame-by-frame streaming over 3 frames from golden latents)
    let g_dec = read_f32_slice(&bin, &m.codec_decode_out)?;
    let lat_dim = pipeline.config.latent_dim as usize;
    let mut dec_state = pipeline.codec.create_decoder_state(pipeline.context())?;
    let mut got_dec = Vec::with_capacity(g_dec.len());
    for i in 0..num_frames {
        let wav = pipeline.codec_decode_f32(&g_enc[i * lat_dim..(i + 1) * lat_dim], &mut dec_state)?;
        got_dec.extend_from_slice(&wav);
    }
    record_metric(&mut failures, "codec.decode_f32", &compare_vecs(&got_dec, &g_dec)?, 0.98);

    // 5. MeanFlowHead (JANGTQ 3-bit FFN + U8 ada)
    let head_cond = read_f32_slice(&bin, &m.head_cond)?;
    let head_noise = read_f32_slice(&bin, &m.head_noise)?;
    let g_head_norm = read_f32_slice(&bin, &m.head_norm_out)?;
    let g_head_raw = read_f32_slice(&bin, &m.head_raw_out)?;
    let (got_head_norm, got_head_raw) = pipeline.run_head_f32(&head_cond, &head_noise, m.guidance_scale)?;
    record_metric(&mut failures, "head.norm_latent", &compare_vecs(&got_head_norm, &g_head_norm)?, 0.94);
    record_metric(&mut failures, "head.raw_latent", &compare_vecs(&got_head_raw, &g_head_raw)?, 0.94);

    // 6. Full-model prompt forward pass
    let prompt = HaruPrompt {
        tokens: read_u32_slice(&bin, &m.prompt_tokens)?,
        model_latents: read_f32_slice(&bin, &m.prompt_model_latents)?,
        user_latents: read_f32_slice(&bin, &m.prompt_user_latents)?,
        model_mask: read_u32_slice(&bin, &m.prompt_model_mask)?,
        user_mask: read_u32_slice(&bin, &m.prompt_user_mask)?,
    };
    let g_hidden = read_f32_slice(&bin, &m.prompt_hidden_out)?;
    let (got_hidden, pad_logit, eop_logit, argmax, top5) = pipeline.forward_prompt_goldens(&prompt)?;
    record_metric(&mut failures, "lm.prompt_final_hidden", &compare_vecs(&got_hidden, &g_hidden)?, 0.90);
    println!(
        "[golden] lm.logits pad: got={:.3} bf16={:.3} (diff={:.3}), eop: got={:.3} bf16={:.3} (diff={:.3})",
        pad_logit,
        m.prompt_pad_logit,
        (pad_logit - m.prompt_pad_logit).abs(),
        eop_logit,
        m.prompt_eop_logit,
        (eop_logit - m.prompt_eop_logit).abs(),
    );
    println!(
        "[golden] lm.argmax: got={} bf16={}, got_top5={:?}, bf16_top5={:?}",
        argmax, m.prompt_argmax_token, top5, m.prompt_top5_ids
    );
    if !pad_logit.is_finite() || !eop_logit.is_finite() {
        failures.push("lm.logits: non-finite pad/eop logit".to_string());
    }
    if argmax != m.prompt_argmax_token && !m.prompt_top5_ids.contains(&argmax) {
        failures.push(format!(
            "lm.argmax mismatch: got {} which is not in bf16 top-5 {:?}",
            argmax, m.prompt_top5_ids
        ));
    }

    if !failures.is_empty() {
        bail!("Golden verification failures:\n  - {}", failures.join("\n  - "));
    }

    println!("=== Goldens verification PASSED ===");
    Ok(())
}

struct LatencyStats {
    min_ms: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

fn compute_stats(samples_ms: &[f64]) -> LatencyStats {
    let mut s = samples_ms.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let sum: f64 = s.iter().sum();
    let idx = |q: f64| -> f64 { s[((n - 1) as f64 * q).round() as usize] };
    LatencyStats {
        min_ms: s[0],
        mean_ms: sum / (n as f64),
        p50_ms: idx(0.50),
        p95_ms: idx(0.95),
        max_ms: s[n - 1],
    }
}

fn run_bench(pipeline: &mut HaruPipeline, frames: usize, sink_window: Option<u32>) -> Result<()> {
    println!("=== Benchmarking {frames} frames on Metal (sink_window={sink_window:?}) ===");
    let mut cfg = HaruSessionConfig::from_config(&pipeline.config);
    cfg.seed = Some(42);
    cfg.max_frames = frames + 16;
    cfg.sink_window = sink_window;

    let t_start = Instant::now();
    let (mut session, frame0) =
        pipeline.start_session(cfg, "You are Haru, a helpful voice assistant.", None)?;
    let start_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[bench] start_session (prefill len={}, emit0): {:.2} ms, token0={}, rms0={:.5}",
        session.prompt_length,
        start_ms,
        frame0.token,
        rms(&frame0.audio)
    );

    let fs = pipeline.config.frame_samples as usize;
    let user_silence = vec![0.0f32; fs];

    // Warmup 5 frames
    for _ in 0..5 {
        session.feed(pipeline, &user_silence)?;
        let f = session.step(pipeline, None)?;
        if f.audio.iter().any(|x| !x.is_finite()) {
            bail!("Non-finite audio sample in warmup");
        }
    }

    let mut feed_ms = Vec::with_capacity(frames);
    let mut step_ms = Vec::with_capacity(frames);
    let mut total_ms = Vec::with_capacity(frames);
    let mut non_pad_count = 0usize;
    let mut decoded_text = String::new();

    for i in 0..frames {
        let t0 = Instant::now();
        let f = session.step(pipeline, Some(&user_silence))?;
        let t2 = Instant::now();

        if f.audio.iter().any(|x| !x.is_finite()) {
            bail!("Non-finite audio sample at bench frame {i}");
        }
        if f.token != pipeline.config.pad_token_id {
            non_pad_count += 1;
        }
        if !f.text.is_empty() {
            decoded_text.push_str(&f.text);
        }

        let sm = (t2 - t0).as_secs_f64() * 1000.0;
        feed_ms.push(0.0);
        step_ms.push(sm);
        total_ms.push(sm);
    }

    let sf = compute_stats(&feed_ms);
    let ss = compute_stats(&step_ms);
    let st = compute_stats(&total_ms);
    println!(
        "[bench] feed(codec.enc): min={:.2} mean={:.2} p50={:.2} p95={:.2} max={:.2} ms",
        sf.min_ms, sf.mean_ms, sf.p50_ms, sf.p95_ms, sf.max_ms
    );
    println!(
        "[bench] step(lm+head+dec): min={:.2} mean={:.2} p50={:.2} p95={:.2} max={:.2} ms",
        ss.min_ms, ss.mean_ms, ss.p50_ms, ss.p95_ms, ss.max_ms
    );
    println!(
        "[bench] total(feed+step):  min={:.2} mean={:.2} p50={:.2} p95={:.2} max={:.2} ms",
        st.min_ms, st.mean_ms, st.p50_ms, st.p95_ms, st.max_ms
    );
    println!("[bench] non_pad_tokens={non_pad_count}/{frames} text_preview={:?}", decoded_text);
    pipeline.bench_breakdown(12)?;
    Ok(())
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f32 = x.iter().map(|&v| v * v).sum();
    (s / x.len() as f32).sqrt()
}

fn send_frame(stream: &mut UnixStream, kind: u8, payload: &[u8]) -> Result<()> {
    let mut header = [0u8; 5];
    header[0] = kind;
    header[1..5].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    stream.write_all(&header)?;
    if !payload.is_empty() {
        stream.write_all(payload)?;
    }
    Ok(())
}

fn send_json(stream: &mut UnixStream, kind: u8, val: &serde_json::Value) -> Result<()> {
    let body = serde_json::to_vec(val)?;
    send_frame(stream, kind, &body)
}

fn read_frame(stream: &mut UnixStream) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut header = [0u8; 5];
    match stream.read_exact(&mut header) {
        Ok(()) => {},
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let kind = header[0];
    let len = u32::from_le_bytes(header[1..5].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; len];
    if len > 0 {
        stream.read_exact(&mut payload)?;
    }
    Ok(Some((kind, payload)))
}

fn downsample_2x(pcm_48k: &[f32], carry: &mut Option<f32>) -> Vec<f32> {
    let mut combined = Vec::with_capacity(pcm_48k.len() + 1);
    if let Some(prev) = carry.take() {
        combined.push(prev);
    }
    combined.extend_from_slice(pcm_48k);
    if combined.len() % 2 == 1 {
        *carry = combined.pop();
    }
    combined.as_chunks::<2>().0.iter().map(|c| 0.5 * (c[0] + c[1])).collect()
}

fn upsample_2x(pcm_24k: &[f32], prev_sample: &mut f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(pcm_24k.len() * 2);
    for &s in pcm_24k {
        out.push(0.5 * (*prev_sample + s));
        out.push(s);
        *prev_sample = s;
    }
    out
}

fn decode_le_f32(payload: &[u8]) -> Vec<f32> {
    if let Ok(aligned) = bytemuck::try_cast_slice::<u8, f32>(payload) {
        return aligned.to_vec();
    }
    payload
        .as_chunks::<4>().0.iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn emit_wire_frame(stream: &mut UnixStream, frame: &HaruFrame, prev_out_sample: &mut f32) -> Result<()> {
    let up = upsample_2x(&frame.audio, prev_out_sample);
    let mut payload = Vec::with_capacity(8 + up.len() * 4);
    payload.extend_from_slice(&(frame.index as u32).to_le_bytes());
    payload.extend_from_slice(&(1000.0 * frame.seconds as f32).to_le_bytes());
    payload.extend_from_slice(bytemuck::cast_slice(&up));
    send_frame(stream, MSG_FRAME, &payload)?;
    if !frame.text.is_empty() {
        send_frame(stream, MSG_TEXT, frame.text.as_bytes())?;
    }
    Ok(())
}

fn emit_raw_24k_frame(stream: &mut UnixStream, frame: &HaruFrame) -> Result<()> {
    let text_bytes = frame.text.as_bytes();
    let text_len = text_bytes.len().min(u16::MAX as usize);
    let mut payload = Vec::with_capacity(16 + frame.audio.len() * 4 + text_len);
    payload.extend_from_slice(&(frame.index as u32).to_le_bytes());
    payload.extend_from_slice(&(1000.0 * frame.seconds as f32).to_le_bytes());
    payload.extend_from_slice(&frame.token.to_le_bytes());
    payload.push(if frame.done { 1 } else { 0 });
    payload.extend_from_slice(&(text_len as u16).to_le_bytes());
    payload.push(0);
    payload.extend_from_slice(bytemuck::cast_slice(&frame.audio));
    payload.extend_from_slice(&text_bytes[..text_len]);
    send_frame(stream, MSG_FRAME, &payload)?;
    Ok(())
}

#[derive(Deserialize, Default)]
struct StartPayload {
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    eop_bias: Option<f32>,
    #[serde(default)]
    speak_first: Option<bool>,
    #[serde(default)]
    max_frames: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    reference_pcm: Option<Vec<f32>>,
    #[serde(default)]
    reference_wav: Option<String>,
    #[serde(default)]
    raw_24k: bool,
    #[serde(default)]
    auto_tick: bool,
    #[serde(default)]
    sink_window: Option<u32>,
}

#[derive(Deserialize, Default)]
struct PreparePayload {
    #[serde(default)]
    seq: u64,
}

fn load_wav_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut raw = Vec::new();
    file.read_to_end(&mut raw)?;
    if raw.len() < 44 || &raw[0..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        bail!("Unsupported WAV file: {}", path.display());
    }
    let mut pos = 12usize;
    let mut audio_format = 1u16;
    let mut channels = 1u16;
    let mut bits_per_sample = 16u16;
    let mut data_slice: Option<&[u8]> = None;
    while pos + 8 <= raw.len() {
        let id = &raw[pos..pos + 4];
        let sz = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + sz > raw.len() {
            break;
        }
        if id == b"fmt " && sz >= 16 {
            audio_format = u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap());
            channels = u16::from_le_bytes(raw[pos + 2..pos + 4].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(raw[pos + 14..pos + 16].try_into().unwrap());
        } else if id == b"data" {
            data_slice = Some(&raw[pos..pos + sz]);
            break;
        }
        pos += sz + (sz % 2);
    }
    let data = data_slice.ok_or_else(|| anyhow!("Missing data chunk in {}", path.display()))?;
    let ch = channels.max(1) as usize;
    let mono: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => data
            .chunks_exact(2 * ch)
            .map(|fr| {
                let mut sum = 0.0f32;
                for c in 0..ch {
                    sum += i16::from_le_bytes(fr[c * 2..c * 2 + 2].try_into().unwrap()) as f32 / 32768.0;
                }
                sum / ch as f32
            })
            .collect(),
        (3, 32) => data
            .chunks_exact(4 * ch)
            .map(|fr| {
                let mut sum = 0.0f32;
                for c in 0..ch {
                    sum += f32::from_le_bytes(fr[c * 4..c * 4 + 4].try_into().unwrap());
                }
                sum / ch as f32
            })
            .collect(),
        _ => bail!("Unsupported WAV format={} bits={}", audio_format, bits_per_sample),
    };
    Ok(mono)
}

fn scan_assets(model_dir: &Path) -> Vec<String> {
    let assets_dir = model_dir.join("assets");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(assets_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                out.push(name.to_string());
            }
        }
    }
    out.sort();
    out
}

fn start_daemon_session(
    start: StartPayload,
    pending_reference_pcm: &mut Option<Vec<f32>>,
    pipeline: &mut HaruPipeline,
    model_dir: &Path,
    default_sink_window: Option<u32>,
) -> Result<(uzu_engine::haru::HaruDuplexSession, HaruFrame)> {
    let mut cfg = HaruSessionConfig::from_config(&pipeline.config);
    if let Some(v) = start.guidance_scale {
        cfg.guidance_scale = v;
    }
    if let Some(v) = start.temperature {
        cfg.temperature = v;
    }
    if let Some(v) = start.top_k {
        cfg.top_k = v;
    }
    if let Some(v) = start.top_p {
        cfg.top_p = v;
    }
    if let Some(v) = start.eop_bias {
        cfg.eop_bias = v;
    }
    if let Some(v) = start.speak_first {
        cfg.speak_first = v;
    }
    if let Some(v) = start.max_frames {
        cfg.max_frames = v;
    }
    cfg.sink_window = match start.sink_window {
        Some(0) => None,
        Some(w) => Some(w),
        None => default_sink_window,
    };
    cfg.seed = start.seed;

    let ref_pcm = if let Some(pcm) = start.reference_pcm {
        *pending_reference_pcm = None;
        Some(pcm)
    } else if let Some(pcm) = pending_reference_pcm.take() {
        Some(pcm)
    } else if let Some(wav_rel) = start.reference_wav {
        let p = model_dir.join("assets").join(&wav_rel);
        let p = if p.exists() { p } else { PathBuf::from(&wav_rel) };
        Some(load_wav_mono_f32(&p)?)
    } else {
        None
    };
    let sys = start
        .system_prompt
        .unwrap_or_else(|| "You are Haru, a helpful voice assistant.".to_string());

    pipeline.start_session(cfg, &sys, ref_pcm.as_deref())
}

fn emit_any_frame(
    stream: &mut UnixStream,
    frame: &HaruFrame,
    raw_24k: bool,
    prev_out_24k: &mut f32,
) -> Result<()> {
    if raw_24k {
        emit_raw_24k_frame(stream, frame)
    } else {
        emit_wire_frame(stream, frame, prev_out_24k)
    }
}

fn serve_client(
    mut stream: UnixStream,
    pipeline: &mut HaruPipeline,
    model_dir: &Path,
    default_sink_window: Option<u32>,
) -> Result<()> {
    let assets = scan_assets(model_dir);
    send_json(&mut stream, MSG_ASSETS, &json!({ "assets": assets }))?;

    let mut session = None;
    let mut pending_reference_pcm: Option<Vec<f32>> = None;
    let mut raw_24k = false;
    let mut auto_tick = false;
    let mut carry_48k: Option<f32> = None;
    let mut prev_out_24k = 0.0f32;
    let fs = pipeline.config.frame_samples as usize;
    let frame_sec = pipeline.config.frame_samples as f64 / pipeline.config.sample_rate as f64;
    let frame_dur = Duration::from_secs_f64(frame_sec);
    let mut next_tick = Instant::now();

    loop {
        if session.is_some() && auto_tick {
            let now = Instant::now();
            let timeout = next_tick.saturating_duration_since(now).max(Duration::from_millis(1));
            stream.set_read_timeout(Some(timeout))?;
        } else {
            stream.set_read_timeout(None)?;
        }

        match read_frame(&mut stream) {
            Ok(Some((MSG_REFERENCE, payload))) => {
                if payload.is_empty() {
                    pending_reference_pcm = None;
                } else if payload.len() % 4 == 0 {
                    pending_reference_pcm = Some(decode_le_f32(&payload));
                } else {
                    send_json(
                        &mut stream,
                        MSG_ERROR,
                        &json!({ "message": format!("Invalid MSG_REFERENCE byte length {}", payload.len()) }),
                    )?;
                }
                continue;
            },
            Ok(Some((MSG_PREPARE, payload))) => {
                let prep: PreparePayload = serde_json::from_slice(&payload).unwrap_or_default();
                send_json(&mut stream, MSG_READY, &json!({ "seq": prep.seq, "ok": true }))?;
                continue;
            },
            Ok(Some((MSG_START, payload))) => {
                session = None;
                let start: StartPayload = serde_json::from_slice(&payload).unwrap_or_default();
                raw_24k = start.raw_24k;
                auto_tick = start.auto_tick;
                carry_48k = None;
                prev_out_24k = 0.0f32;
                match start_daemon_session(start, &mut pending_reference_pcm, pipeline, model_dir, default_sink_window) {
                    Ok((sess, frame0)) => {
                        let prompt_length = sess.prompt_length;
                        let (sr, f_len, step_ms) = if raw_24k {
                            (pipeline.config.sample_rate, fs, frame0.seconds * 1000.0)
                        } else {
                            (pipeline.config.sample_rate * 2, fs * 2, frame_sec * 1000.0)
                        };
                        send_json(
                            &mut stream,
                            MSG_HELLO,
                            &json!({
                                "sample_rate": sr,
                                "frame_samples": f_len,
                                "step_ms": step_ms,
                                "prompt_length": prompt_length,
                            }),
                        )?;
                        emit_any_frame(&mut stream, &frame0, raw_24k, &mut prev_out_24k)?;
                        next_tick = Instant::now() + frame_dur;
                        session = Some(sess);
                    },
                    Err(err) => {
                        send_json(&mut stream, MSG_ERROR, &json!({ "message": format!("{err:#}") }))?;
                    },
                }
                continue;
            },
            Ok(Some((MSG_AUDIO, payload))) => {
                if let Some(sess) = session.as_mut()
                    && payload.len() % 4 == 0
                {
                    let samples = decode_le_f32(&payload);
                    if raw_24k {
                        if !samples.is_empty() {
                            if auto_tick {
                                sess.feed(pipeline, &samples)?;
                            } else {
                                sess.push_pcm(&samples);
                            }
                        }
                    } else {
                        let pcm_24k = downsample_2x(&samples, &mut carry_48k);
                        if !pcm_24k.is_empty() {
                            if auto_tick {
                                sess.feed(pipeline, &pcm_24k)?;
                            } else {
                                sess.push_pcm(&pcm_24k);
                            }
                        }
                    }
                    if !auto_tick {
                        while sess.pending_frames_ready(fs) > 0 {
                            let frame = sess.step_pending(pipeline)?;
                            emit_any_frame(&mut stream, &frame, raw_24k, &mut prev_out_24k)?;
                            if frame.done {
                                session = None;
                                break;
                            }
                        }
                    }
                }
            },
            Ok(Some((kind, _))) => {
                if session.is_none() {
                    send_json(
                        &mut stream,
                        MSG_ERROR,
                        &json!({ "message": format!("Unexpected pre-start message kind 0x{kind:02x}") }),
                    )?;
                }
            },
            Ok(None) => break,
            Err(e)
                if session.is_some()
                    && auto_tick
                    && matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) => {},
            Err(e) => return Err(e.into()),
        }

        if auto_tick
            && let Some(sess) = session.as_mut()
            && (sess.frames_ready() > 0 || Instant::now() >= next_tick)
        {
            let user_fallback = if sess.frames_ready() == 0 {
                Some(vec![0.0f32; fs])
            } else {
                None
            };
            let frame = sess.step(pipeline, user_fallback.as_deref())?;
            emit_any_frame(&mut stream, &frame, raw_24k, &mut prev_out_24k)?;
            next_tick = Instant::now().max(next_tick + frame_dur);
            if frame.done {
                session = None;
            }
        }
    }

    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
}

fn serve_socket(
    socket_path: &Path,
    pipeline: &mut HaruPipeline,
    model_dir: &Path,
    default_sink_window: Option<u32>,
) -> Result<()> {
    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }
    let listener = UnixListener::bind(socket_path)?;
    println!("[haru_daemon] listening on {} (sink_window={default_sink_window:?})", socket_path.display());
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                if let Err(err) = serve_client(stream, pipeline, model_dir, default_sink_window) {
                    eprintln!("[haru_daemon] client session error: {err:#}");
                }
            },
            Err(err) => {
                eprintln!("[haru_daemon] accept error: {err}");
            },
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let t0 = Instant::now();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let mut pipeline = rt.block_on(HaruPipeline::load(&args.model_dir))?;
    println!(
        "Loaded Haru pipeline from {} in {:.2} s",
        args.model_dir.display(),
        t0.elapsed().as_secs_f64()
    );

    if let Some(dir) = &args.verify_goldens {
        verify_goldens(&mut pipeline, dir)?;
    }
    if let Some(n) = args.bench_frames {
        run_bench(&mut pipeline, n, args.sink_window)?;
    }
    if let Some(sock) = &args.socket_path {
        serve_socket(sock, &mut pipeline, &args.model_dir, args.sink_window)?;
    }
    Ok(())
}
