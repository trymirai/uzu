use std::fs;

use anyhow::{Context, Result};
use uniffi::{GenerateOptions, TargetLanguage};

use crate::{
    configs::{ALL_TARGET, Paths, PlatformsConfig},
    languages::{
        LanguageBackend, LanguageBackendTarget, kotlin_error_renames::apply_kotlin_error_renames,
        kotlin_extensions::generate_kotlin_extensions,
    },
    types::{Command, Configuration, Language},
};

const SDK_PROJECT: &str = ":uzu";
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

impl LanguageBackend for KotlinLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Kotlin
    }

    fn expects_prebuild_for_run(&self) -> bool {
        false
    }

    fn default_targets(&self) -> Vec<String> {
        vec![ALL_TARGET.to_string()]
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let staging_path = paths.kotlin_staging_jni_libs_path();
        let jni_libs_path = paths.kotlin_jni_libs_path();
        for path in [&staging_path, &jni_libs_path] {
            if path.exists() {
                fs::remove_dir_all(path)?;
            }
        }
        for target in targets.iter() {
            Command::cargo_ndk_build(
                paths.main_crate.clone(),
                target.name.clone(),
                target.features.clone(),
                configuration,
                staging_path.clone(),
            )
            .with_envs(self.config.required_envs_for_target(target.name.clone())?)
            .run()?;
        }

        let library_name = format!("lib{}.so", paths.main_crate);
        for abi in fs::read_dir(&staging_path)? {
            let abi_path = abi?.path();
            let destination_path = jni_libs_path.join(abi_path.file_name().context("ABI directory has no name")?);
            fs::create_dir_all(&destination_path)?;
            for name in [library_name.as_str(), "libc++_shared.so"] {
                fs::copy(abi_path.join(name), destination_path.join(name))?;
            }
        }

        let abi_path = fs::read_dir(&jni_libs_path)?.next().context("No libraries produced")??.path();
        let library_path = abi_path.join(library_name);
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
        generate_kotlin_extensions(&paths.crates_path(), &generated_sources_path, &paths.main_crate)
    }

    fn test(
        &self,
        _configuration: Configuration,
        _target: String,
    ) -> Result<()> {
        let paths = Paths::new()?;
        Command::gradle_connected_android_test(&paths.playground_android_path(), SDK_PROJECT, TESTS_CLASS).run()
    }

    fn example(
        &self,
        name: &str,
        _configuration: Configuration,
        _targets: Vec<String>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let method = self.language().convert_command_name(name);
        Command::gradle_connected_android_test(
            &paths.playground_android_path(),
            SDK_PROJECT,
            &format!("{EXAMPLES_CLASS}#{method}"),
        )
        .run()
    }
}
