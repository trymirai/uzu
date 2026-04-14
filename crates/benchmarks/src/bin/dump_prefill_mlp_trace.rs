#[cfg(not(feature = "tracing"))]
compile_error!("dump_prefill_mlp_trace requires `--features tracing`");

use std::{
    fs::{self, File},
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow, bail};
use benchmarks::runner::types::Task;
use clap::Parser;
use serde::Serialize;
use tokenizers::Tokenizer;
#[cfg(feature = "metal-backend")]
use uzu::backends::metal::Metal;
use uzu::{
    backends::{common::Backend, cpu::Cpu},
    config::ModelMetadata,
    language_model::{LanguageModelGenerator, language_model_generator::PrefillTraceDump},
    session::{
        config::DecodingConfig,
        helpers::{InputProcessor, InputProcessorDefault},
        parameter::{ContextLength, PrefillStepSize},
        types::Input,
    },
};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    task: PathBuf,
    output_dir: PathBuf,
    #[arg(long)]
    prefill_step_size: Option<usize>,
}

#[derive(Serialize)]
struct LayerManifest {
    layer_index: usize,
    rows: usize,
    cols: usize,
    pre_mlp_norm_file: String,
    mlp_file: String,
}

#[derive(Serialize)]
struct TraceManifest {
    identifier: String,
    input_tokens: usize,
    logits_rows: usize,
    logits_cols: usize,
    logits_file: String,
    layers: Vec<LayerManifest>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let task = load_task(&args.task)?;
    match std::env::var("UZU_BACKEND").map(|value| value.to_ascii_lowercase()) {
        Ok(value) if value == "cpu" => run::<Cpu>(&args.model, &task, args.prefill_step_size, &args.output_dir),
        #[cfg(feature = "metal-backend")]
        Ok(value) if value == "metal" => run::<Metal>(&args.model, &task, args.prefill_step_size, &args.output_dir),
        #[cfg(not(feature = "metal-backend"))]
        Ok(value) if value == "metal" => bail!("metal backend was not built; rebuild with `--features metal-backend`"),
        Ok(value) => bail!("Unsupported backend: {value}"),
        #[cfg(feature = "metal-backend")]
        Err(_) => run::<Metal>(&args.model, &task, args.prefill_step_size, &args.output_dir),
        #[cfg(not(feature = "metal-backend"))]
        Err(_) => run::<Cpu>(&args.model, &task, args.prefill_step_size, &args.output_dir),
    }
}

fn load_task(path: &Path) -> Result<Task> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn run<B: Backend>(
    model_path: &Path,
    task: &Task,
    prefill_step_size: Option<usize>,
    output_dir: &Path,
) -> Result<()>
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

    let mut decoding_config = DecodingConfig::default().with_context_length(ContextLength::Custom(tokens.len() + 1));
    if let Some(prefill_step_size) = prefill_step_size {
        decoding_config = decoding_config.with_prefill_step_size(PrefillStepSize::Custom(prefill_step_size));
    }

    let mut generator =
        LanguageModelGenerator::<B>::new(model_path, decoding_config).map_err(|err| anyhow!("generator: {err}"))?;
    let traces = generator.probe_prefill_trace_dump(tokens.clone()).map_err(|err| anyhow!("prefill trace: {err}"))?;
    write_trace_dump(output_dir, &task.identifier, tokens.len(), &traces)
}

fn write_trace_dump(
    output_dir: &Path,
    identifier: &str,
    input_tokens: usize,
    traces: &PrefillTraceDump,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let layers = traces
        .layers
        .iter()
        .map(|trace| {
            let pre_mlp_norm_file = format!("layer_{:02}_pre_mlp_norm.f32", trace.layer_index);
            let mlp_file = format!("layer_{:02}_mlp.f32", trace.layer_index);
            write_f32_file(&output_dir.join(&pre_mlp_norm_file), &trace.pre_mlp_norm)?;
            write_f32_file(&output_dir.join(&mlp_file), &trace.mlp)?;
            Ok(LayerManifest {
                layer_index: trace.layer_index,
                rows: trace.rows,
                cols: trace.cols,
                pre_mlp_norm_file,
                mlp_file,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let logits_file = "logits.f32".to_string();
    write_f32_file(&output_dir.join(&logits_file), &traces.logits.logits)?;
    let manifest = TraceManifest {
        identifier: identifier.to_string(),
        input_tokens,
        logits_rows: traces.logits.rows,
        logits_cols: traces.logits.cols,
        logits_file,
        layers,
    };
    serde_json::to_writer_pretty(File::create(output_dir.join("manifest.json"))?, &manifest)?;
    Ok(())
}

fn write_f32_file(
    path: &Path,
    values: &[f32],
) -> Result<()> {
    let mut file = File::create(path)?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
