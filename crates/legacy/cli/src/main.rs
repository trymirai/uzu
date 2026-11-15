use anyhow::Result;
use clap::{Parser, Subcommand};
use shoji::types::basic::ReasoningEffort;

mod audio_cli;
mod bench;
mod common;
mod complete;
mod interactive;
mod server;
mod storage;
mod vlm_cli;

#[derive(Parser)]
#[command(name = "cli", bin_name = "cli", version, args_conflicts_with_subcommands = true)]
struct Cli {
    /// Model identifier, repository ID, or local model directory.
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
    /// Send one chat message, stream the reply, and exit.
    #[arg(short, long, value_name = "TEXT", requires = "model")]
    message: Option<String>,
    /// Reasoning effort: disabled, default, low, medium or high.
    /// Overrides the saved preference for this run only; never persisted.
    #[arg(long, value_name = "EFFORT")]
    reasoning_effort: Option<ReasoningEffort>,
    /// Sampling seed for chat sessions. In server mode, the seed for requests that don't set `seed`.
    #[arg(long, value_name = "SEED", allow_negative_numbers = true, global = true)]
    seed: Option<i64>,
    /// Disable built-in tools in interactive chat sessions.
    #[arg(long)]
    no_tools: bool,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Bench {
        model_path: String,
        task_path: String,
        output_path: String,
    },
    ListCheckpoints {
        /// Model ID shown by `list-models`.
        #[arg(value_name = "MODEL_ID")]
        model_id: String,
    },
    ListModels,
    Server {
        /// Model identifier, repository ID, or local model directory. Omit to
        /// boot idle and load models dynamically via POST /v1/models/load.
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,
        /// Directory scanned by GET /v1/models; its subdirectories (with a
        /// config.json) are the loadable model names.
        #[arg(long, value_name = "DIR")]
        models_dir: Option<String>,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Reuse the previous request's state when the new prompt extends it (enabled by default).
        #[arg(long = "prefix-cache", action = clap::ArgAction::SetTrue, conflicts_with = "no_prefix_cache")]
        prefix_cache: bool,
        /// Reset the session on every request, disabling prefix cache reuse.
        #[arg(long = "no-prefix-cache", action = clap::ArgAction::SetTrue)]
        no_prefix_cache: bool,
        /// Print verbose information into the file log
        #[arg(long)]
        verbose_file_log: bool,
        /// Verbose console output: engine diagnostics (prefill/vision breakdowns, GPU operator spans) and debug logs. Also enabled by UZU_VERBOSE=1.
        #[arg(long)]
        verbose: bool,
        /// Session context length in tokens (default 131072, hard-capped at 131072 for now).
        #[arg(long, value_name = "TOKENS")]
        max_context_tokens: Option<u32>,
        /// Disable the HTTP request journal (JSONL access log under the cache dir).
        #[arg(long)]
        no_request_log: bool,
    },
    Storage {
        #[arg(long, value_enum, default_value_t = storage::DownloadManagerCliType::default())]
        download_manager: storage::DownloadManagerCliType,
    },
    /// Raw completion: prefill PROMPT verbatim (no chat template) and show the model's raw continuation.
    Complete {
        /// Prompt text (verbatim, no template). Omit to use --tokens.
        #[arg(value_name = "PROMPT")]
        prompt: Option<String>,
        /// Literal token ids to prefill instead of text (comma-separated).
        #[arg(long, value_name = "IDS", value_delimiter = ',')]
        tokens: Option<Vec<u64>>,
        /// Ensure this model is resident before completing (models-dir directory name).
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,
        #[arg(long, default_value_t = 8124)]
        port: u16,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long)]
        top_k: Option<i64>,
        /// Sampling seed.
        #[arg(long, allow_negative_numbers = true)]
        seed: Option<i64>,
        /// Stop sequence; generation ends before it (repeatable).
        #[arg(long, value_name = "SEQ")]
        stop: Vec<String>,
        /// Stream raw text deltas as they decode.
        #[arg(long, short = 's')]
        stream: bool,
        /// Models dir used when starting a server (optional).
        #[arg(long, value_name = "DIR")]
        models_dir: Option<String>,
    },
    /// Run VLM inference on image(s) and text prompt, supporting multi-image input and multi-turn interactive chat.
    Vlm {
        /// Model identifier, repository ID, or local model directory.
        #[arg(long, value_name = "MODEL")]
        model: String,
        /// Path to one or more local image files.
        #[arg(long, value_name = "IMAGE_PATH", num_args = 1..)]
        image: Vec<String>,
        /// Optional video file (sampled at 2 fps and encoded as a video prompt).
        #[arg(long, value_name = "VIDEO_PATH")]
        video: Option<String>,
        /// User prompt text. If omitted or if --interactive is passed, enters interactive multi-turn chat mode.
        #[arg(long)]
        prompt: Option<String>,
        /// Start interactive multi-turn chat session with KV cache reuse across turns.
        #[arg(long, short = 'i')]
        interactive: bool,
        /// Maximum number of tokens to generate per turn.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Maximum context tokens to retain in KV Cache before capping (prevents unbounded memory growth).
        #[arg(long, default_value_t = 8192)]
        max_context_tokens: usize,
        /// Sampling temperature (0.0 for greedy decoding).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
    },
    /// Run LFM3-Audio inference (ASR, TTS, or speech-to-speech).
    Audio {
        /// Model identifier, repository ID, or local model directory.
        #[arg(long, value_name = "MODEL")]
        model: String,
        /// Optional input WAV file for ASR or speech-to-speech.
        #[arg(long, value_name = "WAV_PATH")]
        input: Option<String>,
        /// Input text for TTS or user prompt alongside --input.
        #[arg(long, value_name = "TEXT")]
        text: Option<String>,
        /// Optional system prompt (defaults to "Perform ASR in English." for ASR).
        #[arg(long, value_name = "TEXT")]
        system: Option<String>,
        /// Optional output WAV file path for TTS / speech output.
        #[arg(long, value_name = "WAV_PATH")]
        output: Option<String>,
        /// Generation mode: text, audio, or interleaved.
        #[arg(long, value_enum)]
        mode: Option<audio_cli::AudioModeArg>,
        /// Maximum number of tokens/frames to generate.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// Optional text sampling temperature.
        #[arg(long)]
        temperature: Option<f32>,
        /// Optional sampling seed.
        #[arg(long, allow_negative_numbers = true)]
        seed: Option<i64>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Bench {
            model_path,
            task_path,
            output_path,
        }) => bench::run_bench(model_path, task_path, output_path).await?,
        Some(Commands::ListCheckpoints {
            model_id,
        }) => interactive::run_list_checkpoints(model_id).await?,
        Some(Commands::ListModels) => interactive::run_list_models().await?,
        Some(Commands::Server {
            model,
            models_dir,
            port,
            host,
            prefix_cache,
            no_prefix_cache,
            verbose_file_log,
            verbose,
            max_context_tokens,
            no_request_log,
        }) => {
            server::run_server(model, models_dir, host, port, prefix_cache || !no_prefix_cache, !no_request_log, verbose_file_log, verbose, cli.seed, max_context_tokens)
                .await?
        },
        Some(Commands::Complete {
            prompt,
            tokens,
            model,
            port,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            stop,
            stream,
            models_dir,
        }) => {
            complete::run_complete(complete::CompleteArgs {
                prompt,
                tokens,
                model,
                port,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                stop,
                stream,
                models_dir,
            })
            .await?
        },
        Some(Commands::Storage {
            download_manager,
        }) => storage::run(download_manager).await?,
        Some(Commands::Vlm {
            model,
            image,
            video,
            prompt,
            interactive,
            max_tokens,
            max_context_tokens,
            temperature,
        }) => vlm_cli::run_vlm_cli(model, image, video, prompt, interactive, max_tokens, max_context_tokens, temperature).await?,
        Some(Commands::Audio {
            model,
            input,
            text,
            system,
            output,
            mode,
            max_tokens,
            temperature,
            seed,
        }) => audio_cli::run_audio_cli(model, input, text, system, output, mode, max_tokens, temperature, seed).await?,
        None => match cli.message {
            Some(message) => {
                interactive::run_non_interactive(
                    cli.model.expect("--message requires --model"),
                    message,
                    cli.reasoning_effort,
                    cli.seed,
                )
                .await?;
            },
            None => interactive::run_interactive(cli.model, cli.reasoning_effort, cli.seed, cli.no_tools).await?,
        },
    }

    Ok(())
}
