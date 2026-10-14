//! Match the browser's non-speculative full-model test on native backends.
//! cargo run -p uzu-web --release --features native-metal --example native-benchmark -- metal MODEL_DIR PROMPT 64
//! UZU_SAMPLING_POLICY accepts the browser policy JSON; null uses checkpoint defaults.
//! UZU_BENCH_TRIALS defaults to 3; 1 is useful for a slow CPU token reference.
#[allow(dead_code)]
#[path = "../src/chat.rs"]
mod chat;

use std::{path::PathBuf, time::Instant};

use serde_json::{Value, json};
use uzu_engine::{
    backends::{common::Backend, cpu::Cpu},
    engine::{
        Engine,
        language_model::{LanguageModel, stream::SamplingMethod},
    },
};

type Error = Box<dyn std::error::Error + Send + Sync>;

struct FileWeights {
    file: std::fs::File,
    size: u64,
}

impl uzu_engine::ParameterReadSource for FileWeights {
    fn len(&self) -> u64 {
        self.size
    }

    fn read_exact_at(
        &self,
        destination: &mut [u8],
        offset: u64,
    ) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(destination, offset)
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = &self.file;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(destination)
        }
    }
}

#[cfg(unix)]
fn process_cpu() -> Option<(f64, f64)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // getrusage initializes the entire output on success. CPU time includes the
    // process's driver/background threads, but excludes GPU execution time.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let seconds = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 / 1_000_000.0;
    Some((seconds(usage.ru_utime), seconds(usage.ru_stime)))
}

#[cfg(not(unix))]
fn process_cpu() -> Option<(f64, f64)> {
    None
}

fn trial<B: Backend + std::fmt::Debug>(
    model: &LanguageModel<B>,
    input: &[u64],
    limit: usize,
    sampling: &SamplingMethod,
) -> Result<(Vec<u32>, Value), Error>
where
    B::Error: Send + Sync + 'static,
{
    let cpu_start = process_cpu();
    let start = Instant::now();
    let mut state = model.create_empty_state(Some(256), 42)?;
    let mut options = model.default_stream_options();
    options.sampling_method = sampling.clone();
    let mut stream = model.stream(input, &mut state, options)?;
    let stream_ready = start.elapsed().as_secs_f64();
    let mut first = None;
    let mut first_cpu = None;
    let mut ids = Vec::new();
    for token in (&mut stream).take(limit) {
        let token = token?;
        if first.is_none() {
            first = Some(Instant::now());
            first_cpu = process_cpu();
        }
        if model.generation_config().stop_token_ids.contains(&token) {
            break;
        }
        ids.push(token as u32);
    }
    // Both the browser and this reference wait for the in-flight stream before
    // finishing, including work already submitted beyond the final output token.
    drop(stream);
    let finished = Instant::now();
    let cpu_end = process_cpu();
    let first = first.unwrap_or(finished);
    let prefill = first.duration_since(start).as_secs_f64();
    let decode = finished.duration_since(first).as_secs_f64();
    let cpu_delta = |begin: Option<(f64, f64)>, end: Option<(f64, f64)>| {
        begin
            .zip(end)
            .map(|(a, b)| json!({"userSeconds":b.0-a.0,"systemSeconds":b.1-a.1,"totalSeconds":b.0+b.1-a.0-a.1}))
    };
    Ok((
        ids.clone(),
        json!({
            "prefillSeconds":prefill, "streamCreationSeconds":stream_ready,
            "generationSeconds":decode, "totalSeconds":finished.duration_since(start).as_secs_f64(),
            "inputTokens":input.len(), "outputTokens":ids.len(),
            "tokensPerSecond":if decode>0.0 {ids.len() as f64/decode} else {0.0},
            "prefillHostCpu":cpu_delta(cpu_start,first_cpu),
            "generationHostCpu":cpu_delta(first_cpu,cpu_end),
            "totalHostCpu":cpu_delta(cpu_start,cpu_end),
        }),
    ))
}

fn benchmark<B: Backend + std::fmt::Debug>(
    path: PathBuf,
    prompt: String,
    limit: usize,
) -> Result<(), Error>
where
    B::Error: Send + Sync + 'static,
{
    let config_bytes = std::fs::read(path.join("config.json"))?;
    let config: chat::ModelConfig = serde_json::from_slice(&config_bytes)?;
    let policy: Value =
        serde_json::from_str(&std::env::var("UZU_SAMPLING_POLICY").unwrap_or_else(|_| "{\"type\":\"Greedy\"}".into()))?;
    let params: chat::RunParams = serde_json::from_value(json!({
        "messages":[{"role":"user","content":prompt}],"maxOutputTokens":limit,"samplingPolicy":policy,
    }))?;
    let rendered = config.token_codec_config.render(&params, chrono::Utc::now())?;
    let engine = Engine::<B>::new()?;
    let started = Instant::now();
    // The source entrypoint excludes sibling speculators, and reads each tensor
    // directly into its backend buffer without a whole-checkpoint allocation.
    let file = std::fs::File::open(path.join("model.safetensors"))?;
    let weights = FileWeights {
        size: file.metadata()?.len(),
        file,
    };
    let model = engine.load_language_model_from_source(
        &config_bytes,
        &std::fs::read(path.join("tokenizer.json"))?,
        &weights,
        None,
    )?;
    let load = started.elapsed().as_secs_f64();
    let input: Vec<u64> = model.tokenizer().encode(rendered, false)?.get_ids().iter().copied().map(u64::from).collect();
    let sampling = chat::sampling_method(params.sampling_policy.as_ref(), model.default_sampling_method())?;
    let trial_count = std::env::var("UZU_BENCH_TRIALS").unwrap_or_else(|_| "3".into()).parse::<usize>()?;
    if trial_count == 0 {
        return Err("UZU_BENCH_TRIALS must be positive".into());
    }
    let (_, warmup) = trial(&model, &input, 1, &sampling)?;
    let mut trials = Vec::new();
    let mut expected = None;
    for _ in 0..trial_count {
        let (ids, metrics) = trial(&model, &input, limit, &sampling)?;
        if let Some(previous) = &expected {
            if previous != &ids {
                return Err("Seeded token sequence changed between trials".into());
            }
        } else {
            expected = Some(ids);
        }
        trials.push(metrics);
    }
    let ids = expected.unwrap();
    println!(
        "{}",
        json!({"backend":B::NAME,"prompt":prompt,"contextLength":256,"speculator":false,"warmupOutputTokens":1,"samplingPolicy":policy,
            "inputTokens":input,"tokenIds":ids,"text":model.tokenizer().decode(&ids,false)?,
            "loadSeconds":load,"warmup":warmup,"trials":trials})
    );
    Ok(())
}

fn main() -> Result<(), Error> {
    let mut arguments = std::env::args().skip(1);
    let backend = arguments.next().ok_or("expected backend: cpu or metal")?;
    let path = PathBuf::from(arguments.next().ok_or("expected model directory")?);
    let prompt = arguments.next().unwrap_or_else(|| "Explain why the sky is blue in detail.".into());
    let limit = arguments.next().as_deref().unwrap_or("64").parse()?;
    match backend.as_str() {
        "cpu" => benchmark::<Cpu>(path, prompt, limit),
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        "metal" => benchmark::<uzu_engine::backends::metal::Metal>(path, prompt, limit),
        _ => Err("Use cpu, or metal on macOS with --features native-metal".into()),
    }
}
