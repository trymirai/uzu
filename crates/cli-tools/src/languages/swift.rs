use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};

use crate::{
    configs::{ALL_TARGET, Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget, generate_swift_extensions},
    types::{Capability, Command, Configuration, Language},
    utilities::fs::copy_directory,
};

const FRAMEWORK_URL_TEMPLATE: &str =
    "https://artifacts.trymirai.com/uzu-swift/releases/{version}.zip";

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
        copy_directory(&sources_path, &generated_sources_path)?;
        generate_swift_extensions(paths.crates_path(), extensions_path.clone())?;

        Ok(())
    }

    fn test_target(
        &self,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        Command::swift_test().with_current_path(&bindings_path).run()
    }

    fn example_target(
        &self,
        name: &str,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        let name = self.language().convert_command_name(name);
        Command::swift_run_example(name).with_current_path(&bindings_path).run()
    }

    fn release(
        &self,
        version: &str,
    ) -> Result<()> {
        self.build(Configuration::Release, vec![ALL_TARGET.to_string()], Vec::<Capability>::new())?;

        let paths = Paths::new()?;
        let xcframework_path = paths.swift_xcframework_path();
        if !xcframework_path.exists() {
            anyhow::bail!("Missing xcframework at {}", xcframework_path.display());
        }

        let spm_root = paths.release_swift_spm_path();
        if spm_root.exists() {
            fs::remove_dir_all(&spm_root)?;
        }
        fs::create_dir_all(&spm_root)?;

        let zip_path = spm_root.join(format!("{version}.zip"));
        let staging_dir = spm_root.join(version);
        fs::create_dir_all(&staging_dir)?;
        let staged_xcframework = staging_dir.join(format!("{}.xcframework", paths.main_crate));
        copy_directory(&xcframework_path, &staged_xcframework)?;
        Command::zip_directory(staging_dir.clone(), zip_path.clone()).run()?;
        fs::remove_dir_all(&staging_dir)?;

        let checksum = Command::swift_compute_checksum(zip_path.clone()).output()?;
        let checksum = checksum
            .lines()
            .filter(|line| !line.trim().is_empty())
            .last()
            .context("Empty checksum output")?
            .to_string();

        let source_package_swift =
            paths.bindings_for_language_path(Language::Swift).join("Package.swift");
        let body = fs::read_to_string(&source_package_swift)
            .with_context(|| format!("Failed to read {}", source_package_swift.display()))?;
        let url = FRAMEWORK_URL_TEMPLATE.replace("{version}", version);
        let replacement = format!("url: \"{url}\",\n            checksum: \"{checksum}\"");
        let body = body.replace("path: \"uzu.xcframework\"", &replacement);

        let target_paths = [
            (".target(", "\"Uzu\"", "bindings/swift/Sources/Uzu"),
            (".executableTarget(", "\"Examples\"", "bindings/swift/Sources/Examples"),
            (".testTarget(", "\"UzuTests\"", "bindings/swift/Tests/UzuTests"),
        ];
        let mut body = body;
        for (target_kind, name_literal, source_path) in target_paths {
            let needle = format!("{target_kind}\n            name: {name_literal},");
            let replacement =
                format!("{needle}\n            path: \"{source_path}\",");
            let updated = body.replace(&needle, &replacement);
            if updated == body {
                anyhow::bail!("Failed to inject path for {target_kind} {name_literal}");
            }
            body = updated;
        }

        fs::write(paths.root_package_swift_path(), body)?;

        Ok(())
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

