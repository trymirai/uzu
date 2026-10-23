use std::{collections::BTreeMap, env, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use colored::Colorize;
use download_manager::{DownloadManager, DownloadManagerType, DownloadPhase, DownloadTaskRequest};
use futures_util::StreamExt;
use kiban::rt::RuntimeHandle;
use serde_json::Value;
use tokenizers::{AddedToken, Tokenizer};

use crate::configs::Paths;

pub async fn export_tokenizers(runtime_handle: RuntimeHandle) -> Result<()> {
    let data_path = Paths::new()?.crate_path("hanashi").join("tests").join("data");
    let registry: Vec<Value> = serde_json::from_str(&fs::read_to_string(data_path.join("registry.json"))?)?;
    let manager = DownloadManager::new(DownloadManagerType::Universal, runtime_handle);
    let mut failed_repo_ids = Vec::new();
    for model in
        registry.iter().filter(|model| model["encodings"].as_array().is_some_and(|encodings| !encodings.is_empty()))
    {
        let repo_id = model["repo_id"].as_str().context("Registry model without repo_id")?;
        let name = repo_id.replace('/', "_");
        match export_tokenizer(
            &manager,
            repo_id,
            model["harmony"]["tokenizer_url"].as_str(),
            &data_path.join("configs").join(&name),
            &data_path.join("tokenizers").join(&name),
        )
        .await
        {
            Ok(()) => println!("{} {repo_id}", "Exported".green()),
            Err(error) => {
                eprintln!("{} {repo_id}: {error:#}", "Failed".red());
                failed_repo_ids.push(repo_id);
            },
        }
    }
    if !failed_repo_ids.is_empty() {
        return Err(anyhow!("Failed to export tokenizers for {}", failed_repo_ids.join(", ")));
    }
    Ok(())
}

async fn export_tokenizer(
    manager: &DownloadManager,
    repo_id: &str,
    harmony_tokenizer_url: Option<&str>,
    configs_directory: &Path,
    tokenizer_directory: &Path,
) -> Result<()> {
    let mut subrequests: Vec<_> = ["tokenizer_config.json", "tokenizer.json"]
        .into_iter()
        .map(|file_name| {
            DownloadTaskRequest::file()
                .destination(file_name)
                .source_url(format!("https://huggingface.co/{repo_id}/resolve/main/{file_name}"))
                .maybe_bearer_token(env::var("HF_TOKEN").ok())
                .build()
        })
        .collect();
    let harmony_tokenizer_name = match harmony_tokenizer_url {
        Some(url) => {
            let file_name = Path::new(url).file_name().context("Harmony tokenizer URL without file name")?;
            subrequests.push(DownloadTaskRequest::file().destination(file_name).source_url(url).build());
            Some(file_name)
        },
        None => None,
    };
    let task = manager
        .download_task(DownloadTaskRequest::group().destination(configs_directory).subrequests(subrequests).build())
        .await?;
    let mut progress = task.progress();
    task.download().await?;
    loop {
        match task.state().phase {
            DownloadPhase::Downloaded {} => break,
            phase if phase.is_in_progress() => {
                progress.next().await.context("Download progress ended")?;
            },
            DownloadPhase::Error {
                message,
            } => return Err(anyhow!(message)),
            phase => return Err(anyhow!("Download stopped: {phase:?}")),
        }
    }

    let tokenizer_config: Value = serde_json::from_slice(&fs::read(configs_directory.join("tokenizer_config.json"))?)?;
    let tokenizer_bytes = fs::read(configs_directory.join("tokenizer.json"))?;
    let has_prepend_normalizer = serde_json::from_slice::<Value>(&tokenizer_bytes)?["normalizer"]["normalizers"]
        .as_array()
        .is_some_and(|normalizers| normalizers.iter().any(|normalizer| normalizer["type"] == "Prepend"));
    let (special_tokens, tokens): (Vec<_>, Vec<_>) =
        serde_json::from_value::<Option<BTreeMap<u32, AddedToken>>>(tokenizer_config["added_tokens_decoder"].clone())?
            .unwrap_or_default()
            .into_values()
            .map(|token| AddedToken {
                normalized: token.normalized && !has_prepend_normalizer,
                ..token
            })
            .partition(|token| token.special);
    let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(|error| anyhow!(error))?;
    tokenizer.add_special_tokens(&special_tokens);
    tokenizer.add_tokens(&tokens);

    fs::create_dir_all(tokenizer_directory)?;
    tokenizer.save(tokenizer_directory.join("tokenizer.json"), true).map_err(|error| anyhow!(error))?;
    if let Some(file_name) = harmony_tokenizer_name {
        fs::copy(configs_directory.join(file_name), tokenizer_directory.join(file_name))?;
    }
    Ok(())
}
