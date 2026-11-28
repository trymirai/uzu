use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};

use crate::{
    configs::{ALL_TARGET, Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget, generate_swift_extensions},
    types::{Command, Configuration, Language},
    utilities::fs::copy_directory,
};

const FRAMEWORK_URL_TEMPLATE: &str = "https://artifacts.trymirai.com/uzu-swift/releases/{version}.zip";

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
        let slice_libs_with_headers = targets
            .iter()
            .map(|target| {
                slice_libs_with_headers(
                    &slices_path.join(&target.name).join(format!("{}.xcframework", paths.main_crate)),
                )
            })
            .collect::<Result<Vec<_>>>()?
            .concat();
        Command::xcodebuild_create_xcframework(slice_libs_with_headers, xcframework_path.clone()).run()?;
        Command::codesign_adhoc(xcframework_path).run()?;

        if generated_sources_path.exists() {
            fs::remove_dir_all(&generated_sources_path)?;
        }
        // Every slice generates the same sources.
        let first_target = targets.first().context("No targets to build")?;
        let sources_path = slices_path.join(&first_target.name).join("Sources");
        fs::create_dir_all(&generated_sources_path)?;
        copy_directory(&sources_path, &generated_sources_path)?;
        generate_swift_extensions(&paths.crates_path(), &extensions_path)?;

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
        release_path: &Path,
    ) -> Result<()> {
        self.build(Configuration::Release, vec![ALL_TARGET.to_string()])?;

        let paths = Paths::new()?;
        let xcframework_path = paths.swift_xcframework_path();
        if !xcframework_path.exists() {
            anyhow::bail!("Missing xcframework at {}", xcframework_path.display());
        }

        let spm_root = release_path.join("swift-spm");
        if spm_root.exists() {
            fs::remove_dir_all(&spm_root)?;
        }
        fs::create_dir_all(&spm_root)?;

        let zip_path = spm_root.join(format!("{version}.zip"));
        let xcframework_parent = xcframework_path.parent().context("xcframework path has no parent")?.to_path_buf();
        let xcframework_name =
            xcframework_path.file_name().context("xcframework path has no file name")?.to_string_lossy().to_string();
        Command::zip_directory(xcframework_name.into(), zip_path.clone())
            .with_current_path(&xcframework_parent)
            .run()?;

        let (checksum, _) = Command::swift_compute_checksum(zip_path.clone()).output()?;
        let checksum =
            checksum.lines().rfind(|line| !line.trim().is_empty()).context("Empty checksum output")?.to_string();

        let source_package_swift = paths.bindings_for_language_path(Language::Swift).join("Package.swift");
        let body = fs::read_to_string(&source_package_swift)
            .with_context(|| format!("Failed to read {}", source_package_swift.display()))?;
        let url = FRAMEWORK_URL_TEMPLATE.replace("{version}", version);
        let replacement = format!("url: \"{url}\",\n            checksum: \"{checksum}\"");
        let body = body.replace("path: \"uzu.xcframework\"", &replacement);
        let body = body.replace("path: \"", "path: \"crates/legacy/uzu/bindings/swift/");

        fs::write(paths.root_package_swift_path(), body)?;

        Ok(())
    }
}

// The static library and headers of each platform in one target's xcframework.
fn slice_libs_with_headers(xcframework_path: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut results = Vec::new();
    for entry in
        fs::read_dir(xcframework_path).with_context(|| format!("Failed to read {}", xcframework_path.display()))?
    {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let static_lib_path =
            find_static_lib(&path)?.with_context(|| format!("No static library in {}", path.display()))?;
        let headers_path = find_modulemap_directory(&path.join("Headers"))?;
        results.push((static_lib_path, headers_path));
    }
    if results.is_empty() {
        return Err(anyhow!("No platforms in {}", xcframework_path.display()));
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

fn find_static_lib(path: &Path) -> Result<Option<PathBuf>> {
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|extension| extension.to_str()) == Some("a") {
            return Ok(Some(path));
        }
        if path.is_dir()
            && let Some(found) = find_static_lib(&path)?
        {
            return Ok(Some(found));
        }
    }
    Ok(None)
}
