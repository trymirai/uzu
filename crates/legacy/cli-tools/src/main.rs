use anyhow::{Result, anyhow};
use clap::{CommandFactory, Parser, Subcommand};
use cli_tools::{
    configs::{HOST_TARGET, PlatformsConfig},
    hanashi::export_tokenizers,
    release::run_release,
    sync::run_sync,
    types::{Capability, Command, Configuration, Language},
};
use kiban::rt::RuntimeHandle;

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
        /// Directory to stage release artifacts into
        path: std::path::PathBuf,
    },
    HanashiTokenizers,
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
    let host_target = config.host_target()?;
    for language in config.languages.keys() {
        // Device-only bindings, like Kotlin for Android, have no host build to verify.
        if config.targets_for_language(*language, vec![host_target.clone()])?.is_empty() {
            println!("Skipping {}: it has no {host_target} build", language.name());
            continue;
        }
        language.backend(config.clone()).build(Configuration::Release, vec![host_target.clone()], vec![])?;
    }
    let (output, _) = Command::git_status_porcelain().output()?;
    if !output.is_empty() {
        eprintln!("{output}");
        return Err(anyhow!("The repository has uncommitted changes after building all languages"));
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = PlatformsConfig::load()?;
    let host_target = config.host_target()?;

    match cli.command {
        Some(Commands::Setup {
            include_platform_specific,
        }) => run_setup(include_platform_specific)?,
        Some(Commands::Install {
            language,
        }) => language.backend(config).install()?,
        Some(Commands::Build {
            language,
            configuration,
            targets,
            capabilities,
        }) => language.backend(config).build(configuration, targets, capabilities)?,
        Some(Commands::Test {
            language,
        }) => {
            let configuration = Configuration::Release;
            let capabilities = vec![];
            let backend = language.backend(config.clone());
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
            let backend = language.backend(config.clone());
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
            path,
        }) => run_release(&version, &path)?,
        Some(Commands::HanashiTokenizers) => export_tokenizers(RuntimeHandle::current()).await?,
        None => {
            let mut cmd = Cli::command();
            cmd.print_help()?;
        },
    }

    Ok(())
}
