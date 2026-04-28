use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, context};
use serde::Serialize;

use crate::{
    configs::{Paths, PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
    types::Language,
};

pub struct ReadmeSyncTask {
    languages: Vec<Language>,
}

impl ReadmeSyncTask {
    pub fn new(languages: Vec<Language>) -> Self {
        Self {
            languages,
        }
    }
}

#[derive(Serialize)]
struct LanguageContext {
    name: String,
    display: String,
    code_fence: String,
    dependency: String,
    quick_start: String,
    examples: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ExampleEntry {
    name: String,
    title: String,
    description: String,
    explanation: Option<String>,
}

#[derive(Serialize)]
struct TargetEntry {
    name: String,
    in_progress: bool,
}

impl SyncTask for ReadmeSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        _input: &str,
    ) -> Result<String> {
        validate_examples(platforms)?;

        let paths = Paths::new()?;

        let primary_language = self.languages.first().copied().context("Languages list is empty")?;
        let metadata = platforms
            .languages
            .get(&primary_language)
            .with_context(|| format!("Missing [languages.{}]", primary_language.name()))?
            .metadata
            .clone();
        let badges = resolve_badges(platforms, &metadata.badges)?;

        let template_body = fs::read_to_string(paths.readme_template_path())?;
        let mut environment = Environment::new();
        environment.add_template("readme", &template_body)?;
        let template = environment.get_template("readme")?;

        let mut language_contexts = Vec::new();
        for language in &self.languages {
            language_contexts.push(language_context(*language, platforms, workspace, &paths)?);
        }

        let intro = fs::read_to_string(paths.readme_fragments_path().join("intro.md"))?.trim().to_string();

        let example_entries: Vec<ExampleEntry> = platforms
            .examples
            .iter()
            .map(|(name, config)| ExampleEntry {
                name: name.clone(),
                title: config.title.clone(),
                description: config.description.clone(),
                explanation: config.explanation.clone(),
            })
            .collect();

        let mut binding_targets: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for language_config in platforms.languages.values() {
            if !language_config.bindings.is_empty() {
                binding_targets.extend(language_config.targets.iter().cloned());
            }
        }
        let targets: Vec<TargetEntry> = platforms
            .targets
            .keys()
            .map(|name| TargetEntry {
                name: name.clone(),
                in_progress: !binding_targets.contains(name),
            })
            .collect();

        let mut backends: Vec<String> = Vec::new();
        for target_config in platforms.targets.values() {
            let name = target_config.backend.name();
            if !backends.contains(&name) {
                backends.push(name);
            }
        }

        let rendered = template.render(context! {
            title => "uzu",
            version => workspace.workspace.package.version,
            image_url => metadata.image_url,
            badges => badges,
            has_multiple_languages => self.languages.len() > 1,
            intro => intro,
            languages => language_contexts,
            examples => example_entries,
            targets => targets,
            backends => backends,
        })?;

        Ok(format!("{}\n", rendered.trim_end()))
    }
}

fn resolve_badges(
    platforms: &PlatformsConfig,
    badge_ids: &[String],
) -> Result<Vec<String>> {
    badge_ids
        .iter()
        .map(|id| {
            platforms
                .badges
                .get(id)
                .cloned()
                .with_context(|| format!("Unknown badge id '{id}' in platforms.toml [badges]"))
        })
        .collect()
}

fn language_context(
    language: Language,
    platforms: &PlatformsConfig,
    workspace: &WorkspaceManifest,
    paths: &Paths,
) -> Result<LanguageContext> {
    let language_config =
        platforms.languages.get(&language).with_context(|| format!("Missing [languages.{}]", language.name()))?;
    let fragments_root = paths.readme_fragments_path().join(language.name());

    let dependency = render_fragment(&fragments_root.join("dependency.md"), &workspace.workspace.package.version)?;

    let examples_root = paths.root_path.join(&language_config.examples_path);
    let mut examples = BTreeMap::new();
    for name in platforms.examples.keys() {
        examples.insert(name.clone(), read_example(&examples_root, language, name)?);
    }
    let quick_start = examples
        .get("quick-start")
        .cloned()
        .with_context(|| format!("Missing 'quick-start' example for {}", language.name()))?;

    Ok(LanguageContext {
        name: language.name(),
        display: language.display(),
        code_fence: language.code_fence().to_string(),
        dependency,
        quick_start,
        examples,
    })
}

fn render_fragment(
    path: &Path,
    version: &str,
) -> Result<String> {
    let body = fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let mut environment = Environment::new();
    environment.add_template("fragment", &body)?;
    let rendered = environment.get_template("fragment")?.render(context! {
        version => version,
    })?;
    Ok(rendered.trim_end().to_string())
}

fn read_example(
    examples_root: &Path,
    language: Language,
    canonical_name: &str,
) -> Result<String> {
    let converted_name = language.convert_file_name(canonical_name);
    let path = examples_root.join(format!("{converted_name}.{}", language.file_extension()));
    fs::read_to_string(&path)
        .with_context(|| format!("Missing example '{canonical_name}' for {} at {}", language.name(), path.display()))
        .map(|body| body.trim_end().to_string())
}

fn validate_examples(platforms: &PlatformsConfig) -> Result<()> {
    let paths = Paths::new()?;
    for (language, language_config) in &platforms.languages {
        let examples_root = paths.root_path.join(&language_config.examples_path);
        for name in platforms.examples.keys() {
            let converted_name = language.convert_file_name(name);
            let path = examples_root.join(format!("{converted_name}.{}", language.file_extension()));
            if !path.exists() {
                return Err(anyhow!(
                    "Example '{}' missing for {} (expected at {})",
                    name,
                    language.name(),
                    path.display()
                ));
            }
        }
    }
    Ok(())
}
