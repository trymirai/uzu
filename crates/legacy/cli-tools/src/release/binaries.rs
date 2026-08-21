use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

use crate::{
    configs::{BinaryConfig, Paths, PlatformsConfig},
    types::{Command, Configuration},
};

pub fn stage_binaries(
    paths: &Paths,
    platforms: &PlatformsConfig,
) -> Result<()> {
    for (binary_name, binary_config) in &platforms.binaries {
        let root = paths.release_binary_path(binary_name);
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir_all(&root)?;

        stage_binary(paths, platforms, &root, binary_name, binary_config)?;
    }

    Ok(())
}

fn stage_binary(
    paths: &Paths,
    platforms: &PlatformsConfig,
    root: &Path,
    binary_name: &str,
    binary_config: &BinaryConfig,
) -> Result<()> {
    for target in &binary_config.targets {
        let backend = platforms.backend_for_target(target.clone())?;
        let capabilities = platforms.capabilities_for_target(target.clone(), Vec::new())?;
        let features =
            [vec![backend.feature()], capabilities.iter().map(|capability| capability.feature()).collect()].concat();

        Command::cargo_build(binary_config.crate_name.clone(), target.clone(), features, Configuration::Release)
            .with_envs(platforms.required_envs_for_target(target.clone())?)
            .run()?;

        let source_path = binary_path(paths, target, &binary_config.crate_name);
        let destination_path = root.join(artifact_name(binary_name, target));
        fs::copy(&source_path, &destination_path).with_context(|| {
            format!("Failed to copy binary from {} to {}", source_path.display(), destination_path.display())
        })?;
    }

    Ok(())
}

fn binary_path(
    paths: &Paths,
    target: &str,
    binary_name: &str,
) -> PathBuf {
    paths.target_path().join(target).join(Configuration::Release.name()).join(executable_name(binary_name, target))
}

fn artifact_name(
    binary_name: &str,
    target: &str,
) -> String {
    format!("{binary_name}-{target}")
}

fn executable_name(
    binary_name: &str,
    target: &str,
) -> String {
    if target.contains("windows") {
        format!("{binary_name}.exe")
    } else {
        binary_name.to_string()
    }
}
