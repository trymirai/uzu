use std::io::{IsTerminal, Write};

use anyhow::{Context, Result, bail};
use uzu::{
    engine::{Engine, config::KEY_HF_TOKEN},
    registry::hf_cache::{download_hf_repo, hf_repo_exists, looks_like_hf_repo_id},
    types::model::Model,
};

/// Resolve `target` via `Engine::model`, and if not found locally but `target`
/// looks like a Hugging Face `<org>/<repo>`, check HF Hub and either prompt to
/// download (when `interactive_prompt` is true on a TTY) or return an actionable
/// download hint.
pub async fn resolve_model_or_prompt_hf(
    engine: &Engine,
    target: &str,
    interactive_prompt: bool,
) -> Result<Model> {
    if let Some(model) = engine.model(target.to_string()).await? {
        return Ok(model);
    }

    if !looks_like_hf_repo_id(target) {
        bail!("Model not found: {target}");
    }

    let token = std::env::var(KEY_HF_TOKEN).ok();
    let exists = hf_repo_exists(target, token.as_deref()).await?;
    if !exists {
        bail!("Model '{target}' not found locally or on Hugging Face Hub");
    }

    if interactive_prompt && std::io::stdin().is_terminal() {
        eprint!("Model '{target}' is not in your local Hugging Face cache. Download from Hugging Face? [y/N]: ");
        std::io::stderr().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).context("Failed to read confirmation")?;
        let confirmed = matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes");
        if !confirmed {
            bail!("Model '{target}' not downloaded. Run `hf download {target}` to fetch it.");
        }

        eprintln!("⬇️  Downloading {target} into Hugging Face cache...");
        let snapshot_dir = download_hf_repo(target, token.as_deref()).await?;
        eprintln!("✅ Downloaded {target} to {}", snapshot_dir.display());

        let _ = engine.refresh_models().await?;
        if let Some(model) = engine.model(target.to_string()).await? {
            return Ok(model);
        }
        let mut model = engine
            .model_by_path(snapshot_dir.to_string_lossy().into_owned())
            .await?
            .with_context(|| format!("Downloaded '{target}', but failed to load from {}", snapshot_dir.display()))?;
        model.identifier = target.to_string();
        return Ok(model);
    }

    bail!(
        "Model '{target}' was found on Hugging Face Hub, but is not in your local HF cache. \
         Run `hf download {target}` (or run interactively to download)."
    );
}
