//! LFM3-Omni numerical alignment & end-to-end tests against `mlx-vlm` reference.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_tests {
    use std::path::PathBuf;

    use futures::executor::block_on;
    use half::bf16;
    use serde::Deserialize;
    use uzu_engine_macros::uzu_test;

    use crate::audio::lfm3_audio::pipeline::{GenerationMode, Lfm3AudioGenerateOptions};
    use crate::audio::liquid_codec_metal::MetalLiquidCodec;
    use crate::backends::metal::Metal;
    use crate::engine::Engine;
    use crate::omni::{Lfm3OmniPipeline, OmniMessage, OmniVideoInput, group_seconds, wrap_media};
    use crate::vision::{VlmSampler, VlmSamplingConfig};

    fn model_dir() -> Option<PathBuf> {
        // Weight-dependent tests are opt-in: point LFM3_OMNI_MODEL_DIR at the
        // local uzu bundle (e.g. ~/.mirai/models/LiquidAI-LFM3-Omni-2B-MLX-4bit-uzu).
        std::env::var("LFM3_OMNI_MODEL_DIR").ok().map(PathBuf::from)
    }

    fn read_f32_bin(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read f32 bin");
        bytes
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<2>().0.iter()
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (xf, yf) = (x as f64, y as f64);
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[derive(Deserialize)]
    struct PromptRef {
        input_ids: Vec<u64>,
    }

    #[uzu_test]
    fn lfm3_omni_eval() {
        let Some(dir) = model_dir() else {
            println!("{{\"skipped\": \"set LFM3_OMNI_MODEL_DIR to run\"}}");
            return;
        };
        if !dir.is_dir() {
            println!("{{\"skipped\": \"LFM3-Omni uzu model not present\"}}");
            return;
        }
        let ref_dir = std::path::Path::new("/tmp/lfm3_omni_ref");
        if !ref_dir.join("joint_prompt_ids.json").is_file() {
            println!("{{\"skipped\": \"/tmp/lfm3_omni_ref fixtures not present\"}}");
            return;
        }

        // 1. Unit checks for wrap_media and group_seconds
        assert_eq!(
            wrap_media("<|image|>hi<|vision_start|><|audio|><|vision_end|><|video|>"),
            "<|vision_start|><|image|><|vision_end|>hi<|vision_start|><|audio|><|vision_end|><|vision_start|><|video|><|vision_end|>"
        );
        assert!((group_seconds(0, 1, 2.0) - 0.25).abs() < 1e-9);
        assert!((group_seconds(2, 3, 2.0) - 1.25).abs() < 1e-9);

        assert!(Lfm3OmniPipeline::is_lfm3_omni(&dir));
        let engine = block_on(Engine::<Metal>::new()).expect("engine");
        let model = block_on(engine.load_language_model(&dir)).expect("load lm");
        let tokenizer = model.tokenizer().clone();
        let omni = block_on(Lfm3OmniPipeline::load_from_model_dir(&dir, &tokenizer, model.context()))
            .expect("load omni");

        let img = image::open(ref_dir.join("test_img.png")).expect("open test_img");
        let frames: Vec<_> = (0..4)
            .map(|i| image::open(ref_dir.join(format!("frame_{i}.png"))).expect("open frame"))
            .collect();
        let wav = read_f32_bin(&ref_dir.join("test_wav.f32"));

        let msgs = vec![
            OmniMessage {
                role: "system".to_string(),
                content: Some("Respond with text.".to_string()),
                ..Default::default()
            },
            OmniMessage {
                role: "user".to_string(),
                content: Some("<|image|><|video|><|audio|>Describe what you see and hear.".to_string()),
                images: vec![img],
                videos: vec![OmniVideoInput {
                    frames,
                    fps: 2.0,
                    frame_indices: None,
                }],
                audios: vec![wav],
                ..Default::default()
            },
        ];

        let prepared = omni
            .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
            .expect("prepare_conversation");

        let prompt_ref: PromptRef =
            serde_json::from_str(&std::fs::read_to_string(ref_dir.join("joint_prompt_ids.json")).unwrap()).unwrap();
        assert_eq!(
            prepared.prompt_tokens, prompt_ref.input_ids,
            "prompt_tokens must match mlx-vlm input_ids exactly"
        );

        let model_dim = model.decoder().model_dim() as usize;
        let all_feats = bf16_bytes_to_f32(&prepared.features_bf16);
        let vis_ref = read_f32_bin(&ref_dir.join("vision_projected.f32"));
        let vid_ref = read_f32_bin(&ref_dir.join("video_projected.f32"));
        let aud_ref = read_f32_bin(&ref_dir.join("audio_projected.f32"));

        let vis_len = vis_ref.len();
        let vid_len = vid_ref.len();
        let aud_len = aud_ref.len();
        assert_eq!(all_feats.len(), vis_len + vid_len + aud_len);

        let vis_got = &all_feats[0..vis_len];
        let vid_got = &all_feats[vis_len..vis_len + vid_len];
        let aud_got = &all_feats[vis_len + vid_len..];

        let vis_cos = cosine_sim(vis_got, &vis_ref);
        let vid_cos = cosine_sim(vid_got, &vid_ref);
        let aud_cos = cosine_sim(aud_got, &aud_ref);
        let vis_max = max_abs_diff(vis_got, &vis_ref);
        let vid_max = max_abs_diff(vid_got, &vid_ref);
        let aud_max = max_abs_diff(aud_got, &aud_ref);

        println!(
            "lfm3_omni_eval: model_dim={model_dim} vis_cos={vis_cos:.6} (max={vis_max:.4}) vid_cos={vid_cos:.6} (max={vid_max:.4}) aud_cos={aud_cos:.6} (max={aud_max:.4})"
        );
        assert!(vis_cos > 0.995, "vision cosine {vis_cos}");
        assert!(vid_cos > 0.999, "video cosine {vid_cos}");
        assert!(aud_cos > 0.975, "audio cosine {aud_cos}");
    }

    #[uzu_test]
    fn lfm3_omni_e2e() {
        let Some(dir) = model_dir() else {
            println!("{{\"skipped\": \"set LFM3_OMNI_MODEL_DIR to run\"}}");
            return;
        };
        let engine = block_on(Engine::<Metal>::new()).expect("engine");
        let model = block_on(engine.load_language_model(&dir)).expect("load lm");
        let tokenizer = model.tokenizer().clone();
        let mut omni = block_on(Lfm3OmniPipeline::load_from_model_dir(&dir, &tokenizer, model.context()))
            .expect("load omni");
        let mut codec = MetalLiquidCodec::load_from_model_dir(dir.join("audio_tokenizer")).expect("load codec");

        // 1. Pure TTS -> waveform
        {
            let mut state = omni.acquire_state(&model).expect("state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Perform TTS in American English using a female voice.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("Hello from LFM3 Omni.".to_string()),
                    ..Default::default()
                },
            ];
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, Some(""))
                .expect("prepare tts");
            let sampler = VlmSampler::new(VlmSamplingConfig::default());
            let out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Audio),
                    128,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |_| {},
                    |_| {},
                )
                .expect("tts generate");
            assert!(!out.frames.is_empty(), "TTS produced no audio frames");
            let mut by_cb = (0..12).map(|_| Vec::with_capacity(out.frames.len())).collect::<Vec<_>>();
            for frame in &out.frames {
                for (cb, &c) in frame.iter().enumerate() {
                    by_cb[cb].push(c);
                }
            }
            let pcm_24k = codec.decode(&by_cb).expect("codec decode");
            assert!(!pcm_24k.is_empty());

            // 2. Pure ASR roundtrip on the synthesized speech (resampled 24k -> 16k)
            let pcm_16k: Vec<f32> = (0..(pcm_24k.len() * 2 / 3))
                .map(|i| pcm_24k[(i * 3) / 2])
                .collect();
            let mut asr_state = omni.acquire_state(&model).expect("asr state");
            let asr_msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Perform ASR in English.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    audios: vec![pcm_16k],
                    ..Default::default()
                },
            ];
            let mut asr_prep = omni
                .prepare_conversation(&tokenizer, &model, &asr_msgs, false, None, true, None)
                .expect("prepare asr");
            let asr_sampler = VlmSampler::new(VlmSamplingConfig::default());
            let mut asr_ids = Vec::new();
            let _asr_out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut asr_state,
                    &mut asr_prep,
                    asr_sampler,
                    Some(GenerationMode::Text),
                    64,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| asr_ids.push(id as u32),
                    |_| {},
                )
                .expect("asr generate");
            let asr_text = tokenizer.decode(&asr_ids, true).unwrap();
            println!("ASR roundtrip: {asr_text:?}");
            assert!(!asr_text.trim().is_empty(), "ASR returned empty text");
        }

        // 3. Video + Text -> Text
        {
            let frames: Vec<_> = (0..4)
                .map(|t| {
                    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([20 + t as u8 * 40, 100, 200]));
                    image::DynamicImage::ImageRgb8(img)
                })
                .collect();
            let mut state = omni.acquire_state(&model).expect("video state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Respond with text.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("What color is in this video?".to_string()),
                    videos: vec![OmniVideoInput {
                        frames,
                        fps: 2.0,
                        frame_indices: None,
                    }],
                    ..Default::default()
                },
            ];
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
                .expect("prepare video");
            let sampler = VlmSampler::new(VlmSamplingConfig::default());
            let mut vid_ids = Vec::new();
            let _out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Text),
                    48,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| vid_ids.push(id as u32),
                    |_| {},
                )
                .expect("video generate");
            let vid_text = tokenizer.decode(&vid_ids, true).unwrap();
            println!("Video VQA: {vid_text:?}");
            assert!(!vid_text.trim().is_empty(), "Video VQA returned empty text");
        }

        // 4. Image + Audio -> Interleaved Speech
        {
            let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 64, image::Rgb([220, 30, 30])));
            let wav = vec![0.0f32; 8000];
            let mut state = omni.acquire_state(&model).expect("interleaved state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Respond with interleaved text and audio.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("What color is this image?".to_string()),
                    images: vec![img],
                    audios: vec![wav],
                    ..Default::default()
                },
            ];
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
                .expect("prepare interleaved");
            let sampler = VlmSampler::new(VlmSamplingConfig::default());
            let mut text_ids = Vec::new();
            let out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Interleaved),
                    96,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| text_ids.push(id as u32),
                    |_| {},
                )
                .expect("interleaved generate");
            let text = tokenizer.decode(&text_ids, true).unwrap();
            println!("Image+Audio Interleaved: text={text:?} frames={}", out.frames.len());
            assert!(!text.trim().is_empty(), "Interleaved produced no text");
            assert!(!out.frames.is_empty(), "Interleaved produced no audio frames");
        }

        // 5. Multilingual text chat (Japanese)
        {
            let mut state = omni.acquire_state(&model).expect("ja state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Respond with text.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("日本の首都はどこですか？一言で答えてください。".to_string()),
                    ..Default::default()
                },
            ];
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
                .expect("prepare ja");
            let sampler = VlmSampler::new(VlmSamplingConfig::default());
            let mut ja_ids = Vec::new();
            let _out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Text),
                    32,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| ja_ids.push(id as u32),
                    |_| {},
                )
                .expect("ja generate");
            let ja_text = tokenizer.decode(&ja_ids, true).unwrap();
            println!("Multilingual (JA): {ja_text:?}");
            assert!(!ja_text.trim().is_empty(), "JA returned empty text");
        }
    }

    #[uzu_test]
    fn lfm3_omni_bench() {
        use std::cell::Cell;
        use std::time::Instant;

        if std::env::var("UZU_BENCH").ok().as_deref() != Some("1") {
            return;
        }
        let Some(dir) = model_dir() else {
            eprintln!("skipping lfm3_omni_bench: set LFM3_OMNI_MODEL_DIR to the uzu bundle dir");
            return;
        };
        if !dir.exists() {
            eprintln!("skipping lfm3_omni_bench: model dir missing at {}", dir.display());
            return;
        }
        let ref_dir = PathBuf::from("/tmp/lfm3_omni_ref");
        if !ref_dir.join("test_img.png").exists() {
            eprintln!("skipping lfm3_omni_bench: /tmp/lfm3_omni_ref missing");
            return;
        }

        let t_load = Instant::now();
        let engine = block_on(Engine::<Metal>::new()).expect("engine");
        let model = block_on(engine.load_language_model(&dir)).expect("load lm");
        let tokenizer = model.tokenizer().clone();
        let mut omni = block_on(Lfm3OmniPipeline::load_from_model_dir(&dir, &tokenizer, model.context()))
            .expect("load omni");
        let mut codec = MetalLiquidCodec::load_from_model_dir(dir.join("audio_tokenizer")).expect("load codec");
        println!("BENCH_LOAD: {:.1} ms", t_load.elapsed().as_secs_f64() * 1000.0);

        let img = image::open(ref_dir.join("test_img.png")).expect("open test_img.png");
        let vid_frames: Vec<_> = (0..4)
            .map(|i| image::open(ref_dir.join(format!("frame_{i}.png"))).expect("open frame"))
            .collect();
        let wav_bytes = std::fs::read(ref_dir.join("test_wav.f32")).expect("read test_wav.f32");
        let wav_f32: Vec<f32> = wav_bytes
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // 1. Image + Text VQA (warmup + 2 timed runs)
        for run in 0..3 {
            let mut state = omni.acquire_state(&model).expect("state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Respond with text.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("Describe this image in detail.".to_string()),
                    images: vec![img.clone()],
                    ..Default::default()
                },
            ];
            let t_prep = Instant::now();
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
                .expect("prepare");
            let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
            let prompt_tokens = prep.prompt_tokens.len();

            let sampler = VlmSampler::new(VlmSamplingConfig {
                temperature: Some(0.0),
                ..Default::default()
            });
            let t_gen = Instant::now();
            let ttft_ms = Cell::new(None::<f64>);
            let mut gen_ids = Vec::new();
            let _out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Text),
                    64,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| {
                        if ttft_ms.get().is_none() {
                            ttft_ms.set(Some(t_gen.elapsed().as_secs_f64() * 1000.0));
                        }
                        gen_ids.push(id as u32);
                    },
                    |_| {},
                )
                .expect("generate");
            let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
            let ttft = ttft_ms.get().unwrap_or(gen_ms);
            let dec_ms = (gen_ms - ttft).max(0.001);
            let n_tok = gen_ids.len();
            let dec_tps = if n_tok > 1 { (n_tok - 1) as f64 / (dec_ms / 1000.0) } else { 0.0 };
            let prefill_tps = prompt_tokens as f64 / (ttft / 1000.0);
            if run > 0 {
                println!(
                    "BENCH_IMAGE_VQA run={run}: prompt_tok={prompt_tokens} gen_tok={n_tok} prep={prep_ms:.1}ms ttft={ttft:.1}ms (prefill={prefill_tps:.1} tok/s) dec={dec_ms:.1}ms (decode={dec_tps:.1} tok/s) total={:.1}ms",
                    prep_ms + gen_ms
                );
            }
        }

        // 2. Video + Text VQA (warmup + 2 timed runs)
        for run in 0..3 {
            let mut state = omni.acquire_state(&model).expect("state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Respond with text.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("Describe what changes in this video.".to_string()),
                    videos: vec![OmniVideoInput {
                        frames: vid_frames.clone(),
                        fps: 2.0,
                        frame_indices: None,
                    }],
                    ..Default::default()
                },
            ];
            let t_prep = Instant::now();
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
                .expect("prepare");
            let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
            let prompt_tokens = prep.prompt_tokens.len();

            let sampler = VlmSampler::new(VlmSamplingConfig {
                temperature: Some(0.0),
                ..Default::default()
            });
            let t_gen = Instant::now();
            let ttft_ms = Cell::new(None::<f64>);
            let mut gen_ids = Vec::new();
            let _out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Text),
                    64,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| {
                        if ttft_ms.get().is_none() {
                            ttft_ms.set(Some(t_gen.elapsed().as_secs_f64() * 1000.0));
                        }
                        gen_ids.push(id as u32);
                    },
                    |_| {},
                )
                .expect("generate");
            let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
            let ttft = ttft_ms.get().unwrap_or(gen_ms);
            let dec_ms = (gen_ms - ttft).max(0.001);
            let n_tok = gen_ids.len();
            let dec_tps = if n_tok > 1 { (n_tok - 1) as f64 / (dec_ms / 1000.0) } else { 0.0 };
            let prefill_tps = prompt_tokens as f64 / (ttft / 1000.0);
            if run > 0 {
                println!(
                    "BENCH_VIDEO_VQA run={run}: prompt_tok={prompt_tokens} gen_tok={n_tok} prep={prep_ms:.1}ms ttft={ttft:.1}ms (prefill={prefill_tps:.1} tok/s) dec={dec_ms:.1}ms (decode={dec_tps:.1} tok/s) total={:.1}ms",
                    prep_ms + gen_ms
                );
            }
        }

        // 3. Image + Audio -> Interleaved Speech (warmup + 2 timed runs)
        for run in 0..3 {
            let mut state = omni.acquire_state(&model).expect("state");
            let msgs = vec![
                OmniMessage {
                    role: "system".to_string(),
                    content: Some("Respond with interleaved text and audio.".to_string()),
                    ..Default::default()
                },
                OmniMessage {
                    role: "user".to_string(),
                    content: Some("Describe this image briefly.".to_string()),
                    images: vec![img.clone()],
                    audios: vec![wav_f32.clone()],
                    ..Default::default()
                },
            ];
            let t_prep = Instant::now();
            let mut prep = omni
                .prepare_conversation(&tokenizer, &model, &msgs, false, None, true, None)
                .expect("prepare");
            let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
            let prompt_tokens = prep.prompt_tokens.len();

            let sampler = VlmSampler::new(VlmSamplingConfig {
                temperature: Some(0.0),
                ..Default::default()
            });
            let t_gen = Instant::now();
            let ttft_ms = Cell::new(None::<f64>);
            let mut gen_ids = Vec::new();
            let out = omni
                .generate_with_options_and_callbacks(
                    &tokenizer,
                    &model,
                    &mut state,
                    &mut prep,
                    sampler,
                    Some(GenerationMode::Interleaved),
                    128,
                    None,
                    &Lfm3AudioGenerateOptions::default(),
                    |id| {
                        if ttft_ms.get().is_none() {
                            ttft_ms.set(Some(t_gen.elapsed().as_secs_f64() * 1000.0));
                        }
                        gen_ids.push(id as u32);
                    },
                    |_| {
                        if ttft_ms.get().is_none() {
                            ttft_ms.set(Some(t_gen.elapsed().as_secs_f64() * 1000.0));
                        }
                    },
                )
                .expect("generate");
            let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
            let t_codec = Instant::now();
            let mut by_cb = (0..12).map(|_| Vec::with_capacity(out.frames.len())).collect::<Vec<_>>();
            for frame in &out.frames {
                for (cb, &c) in frame.iter().enumerate() {
                    by_cb[cb].push(c);
                }
            }
            let pcm = if out.frames.is_empty() {
                Vec::new()
            } else {
                codec.decode(&by_cb).expect("codec decode")
            };
            let codec_ms = t_codec.elapsed().as_secs_f64() * 1000.0;
            let audio_dur_s = pcm.len() as f64 / 24000.0;
            let total_ms = prep_ms + gen_ms + codec_ms;
            let rtf = if audio_dur_s > 0.0 { (total_ms / 1000.0) / audio_dur_s } else { 0.0 };
            if run > 0 {
                println!(
                    "BENCH_INTERLEAVED run={run}: prompt_tok={prompt_tokens} text_tok={} frames={} audio_s={audio_dur_s:.2}s prep={prep_ms:.1}ms ttft={:.1}ms gen={gen_ms:.1}ms codec={codec_ms:.1}ms total={total_ms:.1}ms rtf={rtf:.3}",
                    gen_ids.len(),
                    out.frames.len(),
                    ttft_ms.get().unwrap_or(gen_ms),
                );
            }
        }
    }
}
