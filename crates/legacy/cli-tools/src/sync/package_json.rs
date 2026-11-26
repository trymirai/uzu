use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::ser::PrettyFormatter;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
    types::Language,
};

pub enum PackageJsonSyncTask {
    /// A bindings package, whose version, description, author, repository and license follow the workspace.
    Bindings(Language),
    /// An app package, which only follows the workspace version.
    App,
}

// Every key the repository's package.json files use, in the order sync writes them. An unknown key is an error, so
// a new one is added here first.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PackageJson {
    name: String,
    version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    types: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    main: Option<String>,
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    repository: Option<Repository>,
    license: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    keywords: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engines: Option<IndexMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    os: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu: Option<Vec<String>>,
    package_manager: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    napi: Option<Napi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pi: Option<Pi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    private: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    publish_config: Option<PublishConfig>,
    scripts: IndexMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prettier: Option<Prettier>,
    dependencies: IndexMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peer_dependencies: Option<IndexMap<String, String>>,
    dev_dependencies: IndexMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exports: Option<IndexMap<String, Export>>,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum Repository {
    Url(String),
    Directory {
        r#type: String,
        url: String,
        directory: String,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Napi {
    binary_name: String,
    targets: Vec<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pi {
    extensions: Vec<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PublishConfig {
    access: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    directory: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Prettier {
    single_quote: bool,
    trailing_comma: String,
    print_width: u32,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum Export {
    Path(String),
    Conditions(IndexMap<String, Export>),
}

impl SyncTask for PackageJsonSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let package = &workspace.workspace.package;
        let manifest: PackageJson = serde_json::from_str(input)?;
        let mut manifest = PackageJson {
            version: package.version.clone(),
            ..manifest
        };
        let indent = match self {
            Self::Bindings(language) => {
                manifest.description = Some(package.description.clone());
                manifest.author = Some(package.authors.join(", "));
                manifest.repository = Some(Repository::Url(package.repository.clone()));
                manifest.license.clone_from(&package.license);
                // The umbrella package installs anywhere: each target's binary is its own package, which napi
                // generates with that target's `os` and `cpu` (see the TypeScript release).
                if *language == Language::TypeScript {
                    let targets = &platforms
                        .languages
                        .get(language)
                        .with_context(|| format!("Missing [languages.{}] in platforms.toml", language.name()))?
                        .targets;
                    manifest.napi.as_mut().context("Missing napi in package.json")?.targets.clone_from(targets);
                }
                "    "
            },
            Self::App => "  ",
        };

        // The layout `JSON.stringify` gives, which is what the formatters keep package.json in.
        let mut output = Vec::new();
        manifest.serialize(&mut serde_json::Serializer::with_formatter(
            &mut output,
            PrettyFormatter::with_indent(indent.as_bytes()),
        ))?;
        output.push(b'\n');
        Ok(String::from_utf8(output)?)
    }
}
