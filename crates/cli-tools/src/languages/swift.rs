use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};

use crate::{
    configs::{Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget, generate_swift_extensions},
    types::{Command, Configuration, Language},
};

pub struct SwiftLanguageBackend {
    config: PlatformsConfig,
}

impl SwiftLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }
}

impl LanguageBackend for SwiftLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Swift
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let crate_path = paths.crate_path(&paths.main_crate);
        let xcframework_path = paths.swift_xcframework_path();
        let generated_sources_path = paths.swift_generated_sources_path();
        let extensions_path = generated_sources_path.join("Extensions.swift");

        let slices_path = paths.swift_slices_path();
        let output_path = crate_path.join(&paths.main_crate);

        if slices_path.exists() {
            fs::remove_dir_all(&slices_path)?;
        }
        fs::create_dir_all(&slices_path)?;

        for target in targets.iter() {
            if output_path.exists() {
                fs::remove_dir_all(&output_path)?;
            }

            Command::cargo_swift_package(
                paths.main_crate.clone(),
                target.name.clone(),
                target.features.clone(),
                configuration,
            )
            .with_current_path(&crate_path)
            .with_envs(self.config.required_envs_for_target(target.name.clone())?)
            .run()?;

            let slice_dir = slices_path.join(&target.name);
            fs::rename(&output_path, &slice_dir).context("Moving cargo-swift output to slice dir")?;
        }

        if xcframework_path.exists() {
            fs::remove_dir_all(&xcframework_path)?;
        }
        let slice_libs_with_headers = collect_slice_libs_with_headers(&paths.main_crate, &slices_path)?;
        Command::xcodebuild_create_xcframework(slice_libs_with_headers, xcframework_path.clone()).run()?;
        Command::codesign_adhoc(xcframework_path).run()?;

        if generated_sources_path.exists() {
            fs::remove_dir_all(&generated_sources_path)?;
        }
        let any_slice_path = fs::read_dir(&slices_path)?.next().context("No slices produced")??;
        let sources_path = any_slice_path.path().join("Sources");
        fs::create_dir_all(&generated_sources_path)?;
        copy_directory_recursive(&sources_path, &generated_sources_path)?;
        generate_swift_extensions(paths.crates_path(), extensions_path.clone())?;

        Ok(())
    }

    fn test(&self) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        Command::swift_test().with_current_path(&bindings_path).run()
    }
}

fn collect_slice_libs_with_headers(
    name: &str,
    slices_path: &Path,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut results = Vec::new();
    for entry in fs::read_dir(slices_path)? {
        let entry = entry?;
        let xcframework_path = entry.path().join(format!("{}.xcframework", name));
        if !xcframework_path.exists() {
            continue;
        }
        for entry in fs::read_dir(&xcframework_path)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let static_lib_path = find_static_lib(&path)?;
            let headers_path = find_modulemap_directory(&path.join("Headers"))?;
            results.push((static_lib_path, headers_path));
        }
    }
    if results.is_empty() {
        return Err(anyhow!("No slices produced"));
    }
    Ok(results)
}

fn find_modulemap_directory(headers_path: &Path) -> Result<PathBuf> {
    if headers_path.join("module.modulemap").exists() {
        return Ok(headers_path.to_path_buf());
    }
    for entry in fs::read_dir(headers_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("module.modulemap").exists() {
            return Ok(path);
        }
    }
    Err(anyhow!("module.modulemap not found in {}", headers_path.display()))
}

fn find_static_lib(slice_path: &Path) -> Result<PathBuf> {
    fn walk(path: &Path) -> Option<PathBuf> {
        let entries = fs::read_dir(path).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|extension| extension.to_str()) == Some("a") {
                return Some(path);
            }
            if path.is_dir() {
                if let Some(found) = walk(&path) {
                    return Some(found);
                }
            }
        }
        None
    }
    walk(slice_path).context("Static lib (.a) not found in the slice")
}

fn copy_directory_recursive(
    source_path: &Path,
    destination_path: &Path,
) -> Result<()> {
    fs::create_dir_all(destination_path)?;
    for entry in fs::read_dir(source_path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from_path = entry.path();
        let to_path = destination_path.join(entry.file_name());
        if file_type.is_dir() {
            copy_directory_recursive(&from_path, &to_path)?;
        } else if file_type.is_file() {
            fs::copy(&from_path, &to_path)?;
        }
    }
    Ok(())
}
