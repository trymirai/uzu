use clap::{CommandFactory, Parser, Subcommand};
use cli::handlers::{
    handle_bench, handle_kv_fidelity, handle_kv_spectral_calibrate, handle_run, handle_serve, handle_trace_fidelity,
};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a model with the specified path
    Run {
        /// Folder with model's files
        model_path: String,
        /// Prefill step size
        prefill_step_size: Option<usize>,
        // Seed
        #[arg(long)]
        seed: Option<u64>,
        // Speculator
        #[arg(long)]
        speculator: Option<String>,
        /// Non-interactive mode: run a single message and exit
        #[arg(long, short)]
        message: Option<String>,
        #[arg(long, short)]
        no_thinking: bool,
    },
    /// Start a server with the specified model path
    Serve {
        /// Folder with model's files
        model_path: String,
        /// Prefill step size
        prefill_step_size: Option<usize>,
    },
    /// Run benchmarks for the specified model
    Bench {
        /// Folder with model's files
        model_path: String,
        /// Path to the task file
        task_path: String,
        /// Path to the output file
        output_path: String,
    },
    /// Compare baseline KV tensors against a compressed KV method after prefill
    KvFidelity {
        /// Folder with model's files
        model_path: String,
        /// Path to the task file
        task_path: String,
        /// Path to the output file
        output_path: String,
        /// Compression method, for example shearkv or turboquant
        #[arg(long)]
        method: String,
        /// Codec bit width for shearkv or turboquant
        #[arg(long)]
        turboquant_bits: Option<usize>,
        /// TurboQuant storage target, only used by turboquant
        #[arg(long)]
        turboquant_target: Option<String>,
        /// Stop after generation prefill and compare only prompt-token cache rows
        #[arg(long)]
        after_prefill: bool,
        /// Stop after the first generated token and compare prompt plus first generated token rows
        #[arg(long)]
        after_first_generate: bool,
        /// Stop after this many generated tokens and compare prompt plus generated-token cache rows
        #[arg(long)]
        after_generate_tokens: Option<usize>,
        /// After generation, compare only prompt-token cache rows
        #[arg(long)]
        compare_prompt_only: bool,
    },
    /// Build an offline spectral calibration file from a real prompt prefill
    KvSpectralCalibrate {
        /// Folder with model's files
        model_path: String,
        /// Path to the task file
        task_path: String,
        /// Path to the output file
        output_path: String,
    },
    /// Compare per-layer activation traces for a generated step
    TraceFidelity {
        /// Folder with model's files
        model_path: String,
        /// Path to the task file
        task_path: String,
        /// Path to the output file
        output_path: String,
        /// Compression method, for example shearkv or sparsevalue_turboquant
        #[arg(long)]
        method: String,
        /// Codec bit width for shearkv or turboquant
        #[arg(long)]
        turboquant_bits: Option<usize>,
        /// TurboQuant storage target, only used by turboquant
        #[arg(long)]
        turboquant_target: Option<String>,
        /// Number of generated tokens to run before capturing traces
        #[arg(long)]
        generated_tokens: usize,
        /// Active-row index to compare
        #[arg(long, default_value_t = 0)]
        row_index: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run {
            model_path,
            prefill_step_size,
            seed,
            speculator,
            message,
            no_thinking,
        }) => {
            handle_run(model_path, 2048, prefill_step_size, seed, speculator, message, no_thinking);
        },
        Some(Commands::Serve {
            model_path,
            prefill_step_size,
        }) => {
            handle_serve(model_path, prefill_step_size);
        },
        Some(Commands::Bench {
            model_path,
            task_path,
            output_path,
        }) => {
            if let Err(err) = handle_bench(model_path, task_path, output_path) {
                eprintln!("{err}");
                std::process::exit(1);
            }
        },
        Some(Commands::KvFidelity {
            model_path,
            task_path,
            output_path,
            method,
            turboquant_bits,
            turboquant_target,
            after_prefill,
            after_first_generate,
            after_generate_tokens,
            compare_prompt_only,
        }) => {
            if let Err(err) = handle_kv_fidelity(
                model_path,
                task_path,
                output_path,
                method,
                turboquant_bits,
                turboquant_target,
                after_prefill,
                after_first_generate,
                after_generate_tokens,
                compare_prompt_only,
            ) {
                eprintln!("{err}");
                std::process::exit(1);
            }
        },
        Some(Commands::KvSpectralCalibrate {
            model_path,
            task_path,
            output_path,
        }) => {
            if let Err(err) = handle_kv_spectral_calibrate(model_path, task_path, output_path) {
                eprintln!("{err}");
                std::process::exit(1);
            }
        },
        Some(Commands::TraceFidelity {
            model_path,
            task_path,
            output_path,
            method,
            turboquant_bits,
            turboquant_target,
            generated_tokens,
            row_index,
        }) => {
            if let Err(err) = handle_trace_fidelity(
                model_path,
                task_path,
                output_path,
                method,
                turboquant_bits,
                turboquant_target,
                generated_tokens,
                row_index,
            ) {
                eprintln!("{err}");
                std::process::exit(1);
            }
        },
        None => {
            let mut cmd = Cli::command();
            cmd.print_help().unwrap();
        },
    }
}
