use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow, bail};
use benchmarks::runner::{Runner, types::Task};
use clap::Parser;
use serde::Serialize;
use tokenizers::Tokenizer;
#[cfg(feature = "metal-backend")]
use uzu::backends::metal::Metal;
use uzu::{
    backends::{common::Backend, cpu::Cpu},
    config::ModelMetadata,
    language_model::LanguageModelGenerator,
    session::{
        config::DecodingConfig,
        helpers::{InputProcessor, InputProcessorDefault},
        parameter::{ContextLength, PrefillStepSize, SamplingMethod},
        types::Input,
    },
};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    task: PathBuf,
    output: PathBuf,
    #[arg(long)]
    prefill_step_size: Option<usize>,
}

#[derive(Serialize)]
struct ProbeResult {
    identifier: String,
    memory_used: Option<u64>,
    time_to_first_token: f64,
    prompt_tokens_per_second: f64,
    generate_tokens_per_second: Option<f64>,
    text: String,
    input_tokens: usize,
    prefill_top_token: u64,
    prefill_logits: Vec<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let task = load_task(&args.task)?;
    let result = match std::env::var("UZU_BACKEND").map(|value| value.to_ascii_lowercase()) {
        Ok(value) if value == "cpu" => run::<Cpu>(&args.model, &task, args.prefill_step_size)?,
        #[cfg(feature = "metal-backend")]
        Ok(value) if value == "metal" => run::<Metal>(&args.model, &task, args.prefill_step_size)?,
        #[cfg(not(feature = "metal-backend"))]
        Ok(value) if value == "metal" => bail!("metal backend was not built; rebuild with `--features metal-backend`"),
        Ok(value) => bail!("Unsupported backend: {value}"),
        #[cfg(feature = "metal-backend")]
        Err(_) => run::<Metal>(&args.model, &task, args.prefill_step_size)?,
        #[cfg(not(feature = "metal-backend"))]
        Err(_) => run::<Cpu>(&args.model, &task, args.prefill_step_size)?,
    };
    serde_json::to_writer_pretty(File::create(&args.output)?, &result)?;
    Ok(())
}

fn load_task(path: &Path) -> Result<Task> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn run<B: Backend>(
    model_path: &Path,
    task: &Task,
    prefill_step_size: Option<usize>,
) -> Result<ProbeResult>
where
    B::Error: std::error::Error + Send + Sync + 'static,
{
    let runner = Runner::new(task.clone(), model_path.display().to_string(), prefill_step_size);
    let perf = runner.run(None::<fn(f64)>).map_err(|err| anyhow!("runner failed: {err}"))?;
    let perf = perf.into_iter().collect::<Vec<_>>();
    assert!(!perf.is_empty(), "prefill probe expects at least one result");
    let first = perf.first().expect("missing first result");
    assert!(
        perf.iter().all(|result| result.text == first.text),
        "prefill probe expects deterministic text across runs"
    );
    let time_to_first_token = perf.iter().map(|result| result.time_to_first_token).sum::<f64>() / perf.len() as f64;
    let prompt_tokens_per_second =
        perf.iter().map(|result| result.prompt_tokens_per_second).sum::<f64>() / perf.len() as f64;
    let generate_tokens_per_second = if perf.iter().all(|result| result.generate_tokens_per_second.is_some()) {
        Some(
            perf.iter().map(|result| result.generate_tokens_per_second.expect("checked above")).sum::<f64>()
                / perf.len() as f64,
        )
    } else {
        None
    };
    let prefill_logits = extract_sampled_prefill_logits::<B>(model_path, task, prefill_step_size)?;
    let prefill_top_token = prefill_logits
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.partial_cmp(rhs.1).expect("logits must be finite"))
        .map(|(index, _)| index as u64)
        .expect("logits must be non-empty");

    Ok(ProbeResult {
        identifier: task.identifier.clone(),
        memory_used: first.memory_used,
        time_to_first_token,
        prompt_tokens_per_second,
        generate_tokens_per_second,
        text: first.text.clone(),
        input_tokens: first.tokens_count_input as usize,
        prefill_top_token,
        prefill_logits,
    })
}

fn extract_sampled_prefill_logits<B: Backend>(
    model_path: &Path,
    task: &Task,
    prefill_step_size: Option<usize>,
) -> Result<Vec<f32>>
where
    B::Error: std::error::Error + Send + Sync + 'static,
{
    let metadata: ModelMetadata = serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
    let lm_config = metadata.model_config.as_language_model().ok_or_else(|| anyhow!("language model config"))?;
    let input_processor = InputProcessorDefault::new(lm_config.message_processor_config.clone());
    let prompt = input_processor
        .process(&Input::Messages(task.messages.clone()), false, task.tokens_limit > 0)
        .map_err(|err| anyhow!("input processor: {err}"))?;
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|_| anyhow!("tokenizer"))?;
    let tokens = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|_| anyhow!("encode prompt"))?
        .get_ids()
        .iter()
        .map(|&token| token as u64)
        .collect::<Vec<_>>();
    assert!(!tokens.is_empty(), "prompt must produce at least one token");

    let mut decoding_config = DecodingConfig::default()
        .with_context_length(ContextLength::Custom(tokens.len() + task.tokens_limit as usize + 1));
    if let Some(prefill_step_size) = prefill_step_size {
        decoding_config = decoding_config.with_prefill_step_size(PrefillStepSize::Custom(prefill_step_size));
    }

    let mut generator =
        LanguageModelGenerator::<B>::new(model_path, decoding_config).map_err(|err| anyhow!("generator: {err}"))?;
    generator.probe_prefill_logits(tokens, SamplingMethod::Greedy).map_err(|err| anyhow!("prefill probe: {err}"))
}
