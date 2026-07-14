use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::{CommandFactory, Parser, Subcommand};
use cli_tools::{
    configs::{HOST_TARGET, PlatformsConfig},
    languages::{
        LanguageBackend, PythonLanguageBackend, RustLanguageBackend, SwiftLanguageBackend, TypeScriptLanguageBackend,
    },
    release::run_release,
    sync::run_sync,
    types::{Capability, Command, Configuration, Language},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
enum CollectMetricsSourceMode {
    Registry,
    Local,
}

#[derive(Parser)]
#[command(name = "uzu-tools", bin_name = "uzu-tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Install rustup / uv / pnpm and required targets
    Setup {
        #[arg(long)]
        include_platform_specific: bool,
    },
    /// Install tools for a specific language
    Install {
        #[arg(value_enum)]
        language: Language,
    },
    /// Build for a specific language
    Build {
        #[arg(value_enum, default_value = "rust")]
        language: Language,
        #[arg(value_enum, default_value = "release")]
        configuration: Configuration,
        #[arg(long, value_delimiter = ',', default_value = HOST_TARGET)]
        targets: Vec<String>,
        #[arg(long, value_enum, value_delimiter = ',')]
        capabilities: Vec<Capability>,
    },
    /// Run tests for a specific language
    Test {
        #[arg(value_enum)]
        language: Language,
    },
    /// Run an example for a specific language
    Example {
        #[arg(value_enum)]
        language: Language,
        name: String,
    },
    /// Synchronize project files with platform.toml
    Sync {
        #[arg(long)]
        check: bool,
    },
    /// Verify that the working tree has no uncommitted changes after building each language
    Verify,
    /// Bump version, sync, build all languages, and stage release artifacts
    Release {
        version: String,
    },
    /// Measure per-model power, energy, and DRAM traffic across a prefill/generate sweep (macOS only)
    CollectMetrics {
        #[arg(long, default_value = "metrics.csv")]
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = CollectMetricsSourceMode::Registry)]
        source: CollectMetricsSourceMode,
        #[arg(long)]
        storage: Option<PathBuf>,
        #[arg(long = "model-id")]
        model_ids: Vec<String>,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,6,8,10,12,16,32,64,128,256")]
        prefill: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "32,128")]
        generate: Vec<usize>,
        #[arg(long, default_value_t = 6)]
        iterations: usize,
    },
}

fn run_setup(include_platform_specific: bool) -> Result<()> {
    if include_platform_specific && cfg!(target_vendor = "apple") {
        Command::xcodebuild_first_launch().run()?;
        Command::xcodebuild_download_metal_toolchain().run()?;
        Command::cmake_setup().run()?;
        Command::clang_format_setup().run()?;
    }

    Command::rustup_setup().run()?;
    Command::rustup_update().run()?;
    Command::rustup_show().run()?;
    Command::uv_setup().run()?;
    Command::pnpm_setup().run()?;

    Ok(())
}

fn run_verify(config: &PlatformsConfig) -> Result<()> {
    for language in config.languages.keys() {
        let backend = language_backend(*language, config.clone())?;
        backend.build(Configuration::Release, vec![config.host_target()?], vec![])?;
    }
    let (output, _) = Command::git_status_porcelain().output()?;
    if !output.is_empty() {
        eprintln!("{output}");
        return Err(anyhow!("The repository has uncommitted changes after building all languages"));
    }
    Ok(())
}

#[cfg(feature = "collect-metrics")]
async fn run_collect_metrics(
    output: PathBuf,
    source: CollectMetricsSourceMode,
    storage: Option<PathBuf>,
    model_ids: Vec<String>,
    prefill: Vec<usize>,
    generate: Vec<usize>,
    iterations: usize,
) -> Result<()> {
    let source = match source {
        CollectMetricsSourceMode::Registry => cli_tools::collect_metrics::SourceMode::Registry,
        CollectMetricsSourceMode::Local => cli_tools::collect_metrics::SourceMode::Local,
    };
    cli_tools::collect_metrics::run(cli_tools::collect_metrics::Options {
        source,
        storage,
        output,
        model_ids,
        prefill,
        generate,
        iterations,
    })
    .await
}

#[cfg(not(feature = "collect-metrics"))]
async fn run_collect_metrics(
    _output: PathBuf,
    _source: CollectMetricsSourceMode,
    _storage: Option<PathBuf>,
    _model_ids: Vec<String>,
    _prefill: Vec<usize>,
    _generate: Vec<usize>,
    _iterations: usize,
) -> Result<()> {
    Err(anyhow!(
        "this binary was built without the `collect-metrics` feature; rebuild with `--features collect-metrics` on macOS"
    ))
}

fn language_backend(
    language: Language,
    config: PlatformsConfig,
) -> Result<Box<dyn LanguageBackend>> {
    match language {
        Language::Rust => Ok(Box::new(RustLanguageBackend::new(config))),
        Language::Python => Ok(Box::new(PythonLanguageBackend::new(config))),
        Language::Swift => Ok(Box::new(SwiftLanguageBackend::new(config))),
        Language::TypeScript => Ok(Box::new(TypeScriptLanguageBackend::new(config))),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(Commands::CollectMetrics {
        output,
        source,
        storage,
        model_ids,
        prefill,
        generate,
        iterations,
    }) = cli.command
    {
        return run_collect_metrics(output, source, storage, model_ids, prefill, generate, iterations).await;
    }

    let config = PlatformsConfig::load()?;
    let host_target = config.host_target()?;

    match cli.command {
        Some(Commands::Setup {
            include_platform_specific,
        }) => run_setup(include_platform_specific)?,
        Some(Commands::Install {
            language,
        }) => language_backend(language, config)?.install()?,
        Some(Commands::Build {
            language,
            configuration,
            targets,
            capabilities,
        }) => language_backend(language, config)?.build(configuration, targets, capabilities)?,
        Some(Commands::Test {
            language,
        }) => {
            let configuration = Configuration::Release;
            let capabilities = vec![];
            let backend = language_backend(language, config.clone())?;
            if backend.expects_prebuild_for_run() {
                backend.build(configuration, vec![host_target.clone()], capabilities.clone())?;
            }
            backend.test(configuration, host_target.clone(), capabilities.clone())?
        },
        Some(Commands::Example {
            language,
            name,
        }) => {
            let configuration = Configuration::Release;
            let capabilities = vec![];
            let backend = language_backend(language, config.clone())?;
            if backend.expects_prebuild_for_run() {
                backend.build(configuration, vec![host_target.clone()], capabilities.clone())?;
            }
            backend.example(&name, configuration, host_target.clone(), capabilities.clone())?
        },
        Some(Commands::Sync {
            check,
        }) => run_sync(check)?,
        Some(Commands::Verify) => run_verify(&config)?,
        Some(Commands::Release {
            version,
        }) => run_release(&version)?,
        Some(Commands::CollectMetrics {
            ..
        }) => unreachable!("handled above"),
        None => {
            let mut cmd = Cli::command();
            cmd.print_help()?;
        },
    }

    Ok(())
}
