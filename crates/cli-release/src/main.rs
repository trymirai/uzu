use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use cli_release::{
    handlers::{
        SyncSource, prepare_bindings_python, prepare_bindings_swift, prepare_bindings_ts, prepare_workspace_swift,
        prepare_workspace_swift_spm, prepare_workspace_ts, prepare_workspace_ts_napi, sync_into_repo,
    },
    types::{Environment, Error},
};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare bindings
    PrepareBindings {
        /// Optional language filter (prepare only that language)
        #[arg(value_enum)]
        language: Option<BindingLanguage>,
    },
    /// Prepare workspace
    PrepareWorkspace {
        /// Package version
        version: String,
    },
    /// Sync into repo
    SyncIntoRepo {
        /// Source
        source: SyncSource,
        /// Repository path
        repo_path: PathBuf,
    },
}

#[derive(ValueEnum, Clone, Copy)]
enum BindingLanguage {
    Swift,
    Ts,
    Python,
}

fn prepare_bindings(
    environment: &Environment,
    language: Option<BindingLanguage>,
) -> Result<(), Error> {
    match language {
        Some(BindingLanguage::Swift) => prepare_bindings_swift(environment)?,
        Some(BindingLanguage::Ts) => {
            prepare_workspace_ts_napi(environment)?;
            prepare_bindings_ts(environment)?;
        },
        Some(BindingLanguage::Python) => prepare_bindings_python(environment)?,
        None => {
            prepare_workspace_ts_napi(environment)?;
            prepare_bindings_ts(environment)?;
            prepare_bindings_swift(environment)?;
            prepare_bindings_python(environment)?;
        },
    }
    Ok(())
}

fn prepare_workspace(
    environment: &Environment,
    version: String,
) -> Result<(), Error> {
    prepare_bindings(&environment, None)?;
    prepare_workspace_ts(environment, version.clone())?;
    let swift_spm_checksum = prepare_workspace_swift_spm(environment, version.clone())?;
    prepare_workspace_swift(environment, version.clone(), swift_spm_checksum)?;
    Ok(())
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();
    let environment = Environment::new()?;

    match cli.command {
        Some(Commands::PrepareBindings {
            language,
        }) => {
            prepare_bindings(&environment, language)?;
        },
        Some(Commands::PrepareWorkspace {
            version,
        }) => {
            prepare_workspace(&environment, version)?;
        },
        Some(Commands::SyncIntoRepo {
            source,
            repo_path,
        }) => {
            sync_into_repo(&environment, source, repo_path)?;
        },
        None => {
            let mut cmd = Cli::command();
            cmd.print_help().unwrap();
        },
    }

    Ok(())
}
