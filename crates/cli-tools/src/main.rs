use anyhow::{Result, anyhow};
use clap::{CommandFactory, Parser, Subcommand};
use cli_tools::{
    configs::{HOST_TARGET, PlatformsConfig},
    languages::{
        LanguageBackend, PythonLanguageBackend, RustLanguageBackend, SwiftLanguageBackend, TypeScriptLanguageBackend,
    },
    sync::run_sync,
    types::{Capability, Command, Configuration, Language},
};

#[derive(Parser)]
#[command(name = "uzu-tools", bin_name = "uzu-tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Install rustup / uv and required toolchains
    Setup,
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
}

fn run_setup() -> Result<()> {
    if cfg!(target_vendor = "apple") {
        Command::xcodebuild_first_launch().run()?;
        Command::xcodebuild_download_metal_toolchain().run()?;
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
        let backend = language_backend(language.clone(), config.clone())?;
        backend.build(Configuration::Release, vec![config.host_target()?], vec![])?;
    }
    let output = Command::git_status_porcelain().output()?;
    if !output.is_empty() {
        eprintln!("{output}");
        return Err(anyhow!("The repository has uncommitted changes after building all languages"));
    }
    Ok(())
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = PlatformsConfig::load()?;
    let host_target = config.host_target()?;

    match cli.command {
        Some(Commands::Setup) => run_setup()?,
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
        None => {
            let mut cmd = Cli::command();
            cmd.print_help()?;
        },
    }

    Ok(())
}
