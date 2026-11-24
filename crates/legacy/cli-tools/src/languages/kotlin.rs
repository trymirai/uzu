use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use uniffi::{GenerateOptions, TargetLanguage};
use walkdir::WalkDir;

use crate::{
    configs::{Paths, PlatformsConfig},
    languages::{
        LanguageBackend, LanguageBackendTarget, kotlin_error_renames::apply_kotlin_error_renames,
        kotlin_extensions::generate_kotlin_extensions,
    },
    types::{Command, Configuration, Language},
};

const TESTS_CLASS: &str = "com.trymirai.uzu.EngineTest";
const EXAMPLES_CLASS: &str = "com.trymirai.uzu.examples.ExamplesTest";

pub struct KotlinLanguageBackend {
    config: PlatformsConfig,
}

impl KotlinLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }
}

// The classpath directory JNA loads a library from on the target's platform, and the library's file name there.
fn jna_library(
    target: &str,
    library: &str,
) -> Result<(&'static str, String)> {
    match target {
        "aarch64-apple-darwin" => Ok(("darwin-aarch64", format!("lib{library}.dylib"))),
        "x86_64-unknown-linux-gnu" => Ok(("linux-x86-64", format!("lib{library}.so"))),
        "x86_64-pc-windows-msvc" => Ok(("win32-x86-64", format!("{library}.dll"))),
        _ => Err(anyhow!("No JNA platform for target {target}")),
    }
}

// The Kotlin files UniFFI generated, which the post-processing passes rewrite.
pub fn kotlin_sources(generated_sources_path: &Path) -> Result<Vec<PathBuf>> {
    let mut sources = Vec::new();
    for entry in WalkDir::new(generated_sources_path) {
        let path = entry?.into_path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("kt") {
            sources.push(path);
        }
    }
    Ok(sources)
}

// UniFFI reads the u16 API checksums through JNA as an Int, but x86-64 Windows leaves the upper bits of a u16 return
// undefined, so every check fails there. Compare the low 16 bits, which are the checksum.
fn compare_checksums_as_u16(generated_sources_path: &Path) -> Result<()> {
    for path in kotlin_sources(generated_sources_path)? {
        let source = fs::read_to_string(&path)?;
        let mut masked_count = 0;
        let masked = source
            .lines()
            .map(|line| match line.split_once("if (lib.") {
                Some((indent, check)) if check.contains("_checksum_") => match check.split_once("() != ") {
                    Some((function, rest)) => {
                        masked_count += 1;
                        format!("{indent}if ((lib.{function}() and 0xffff) != {rest}")
                    },
                    None => line.to_string(),
                },
                _ => line.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if source.contains("_checksum_") && masked_count == 0 {
            bail!("No UniFFI checksum check to mask in {}; the generated code changed shape", path.display());
        }
        fs::write(&path, masked + "\n")?;
    }
    Ok(())
}

// The Windows library must link the C runtime statically (platforms.toml); RUSTFLAGS and CARGO_ENCODED_RUSTFLAGS
// replace the target's flags, so either one set without crt-static would build a library that crashes in some JVMs.
fn check_static_runtime(target: &str) -> Result<()> {
    if !target.contains("windows") {
        return Ok(());
    }
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"] {
        if let Ok(flags) = std::env::var(name)
            && !flags.is_empty()
            && !flags.contains("+crt-static")
        {
            bail!("{name} is set without -C target-feature=+crt-static, which the Windows library needs");
        }
    }
    Ok(())
}

impl LanguageBackend for KotlinLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Kotlin
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let natives_path = paths.kotlin_natives_path();
        if natives_path.exists() {
            fs::remove_dir_all(&natives_path)?;
        }
        let mut library_path = None;
        for target in targets.iter() {
            check_static_runtime(&target.name)?;
            Command::cargo_build(paths.main_crate.clone(), target.name.clone(), target.features.clone(), configuration)
                .with_envs(self.config.required_envs_for_target(target.name.clone())?)
                .run()?;
            let (platform, file_name) = jna_library(&target.name, &paths.main_crate)?;
            let built_path = paths.target_profile_path(&target.name, configuration).join(&file_name);
            let destination_path = natives_path.join(platform);
            fs::create_dir_all(&destination_path)?;
            fs::copy(&built_path, destination_path.join(&file_name))
                .with_context(|| format!("Cannot copy {}", built_path.display()))?;
            library_path = Some(built_path);
        }
        let library_path = library_path.context("No libraries built")?;

        let generated_sources_path = paths.kotlin_generated_sources_path();
        if generated_sources_path.exists() {
            fs::remove_dir_all(&generated_sources_path)?;
        }
        uniffi::generate(GenerateOptions {
            languages: vec![TargetLanguage::Kotlin],
            source: library_path.to_str().context("Library path is not UTF-8")?.into(),
            out_dir: generated_sources_path.to_str().context("Generated sources path is not UTF-8")?.into(),
            config_override: None,
            format: false,
            crate_filter: None,
            metadata_no_deps: false,
        })?;
        apply_kotlin_error_renames(&paths.crates_path(), &generated_sources_path)?;
        compare_checksums_as_u16(&generated_sources_path)?;
        generate_kotlin_extensions(&paths.crates_path(), &generated_sources_path, &paths.main_crate)
    }

    fn test_target(
        &self,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        Command::gradle_test(&paths.bindings_for_language_path(self.language()), "jvmTest", TESTS_CLASS).run()
    }

    fn example_target(
        &self,
        name: &str,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let method = self.language().convert_command_name(name);
        Command::gradle_test(
            &paths.bindings_for_language_path(self.language()),
            "jvmExamples",
            &format!("{EXAMPLES_CLASS}.{method}"),
        )
        .run()
    }
}
