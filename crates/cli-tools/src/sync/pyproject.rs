use anyhow::{Context, Result};
use toml_edit::{Array, DocumentMut, InlineTable, Item, Value};

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct PyprojectSyncTask;

impl SyncTask for PyprojectSyncTask {
    fn process(
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let package = &workspace.workspace.package;

        let mut document: DocumentMut = input.parse()?;

        let project = document
            .get_mut("project")
            .and_then(Item::as_table_mut)
            .context("Missing [project] table in pyproject.toml")?;
        project.insert("description", Item::Value(Value::from(package.description.as_str())));
        project.insert("readme", Item::Value(Value::from(package.readme.as_str())));
        project.insert("license", Item::Value(Value::from(package.license.as_str())));

        let mut authors = Array::new();
        for author in &package.authors {
            authors.push(Value::InlineTable(parse_author(author)?));
        }
        project.insert("authors", Item::Value(Value::Array(authors)));

        let mut keywords = Array::new();
        for keyword in &package.keywords {
            let mut value = Value::from(keyword.as_str());
            value.decor_mut().set_prefix("\n    ");
            keywords.push_formatted(value);
        }
        keywords.set_trailing("\n");
        keywords.set_trailing_comma(true);
        project.insert("keywords", Item::Value(Value::Array(keywords)));

        let urls = document
            .get_mut("project")
            .and_then(Item::as_table_mut)
            .and_then(|table| table.get_mut("urls"))
            .and_then(Item::as_table_like_mut)
            .context("Missing [project.urls] table in pyproject.toml")?;
        urls.insert("Homepage", Item::Value(Value::from(package.homepage.as_str())));
        urls.insert("Repository", Item::Value(Value::from(package.repository.as_str())));
        urls.insert("Issues", Item::Value(Value::from(format!("{}/issues", package.repository))));

        Ok(document.to_string())
    }
}

fn parse_author(value: &str) -> Result<InlineTable> {
    let (name, rest) =
        value.rsplit_once('<').with_context(|| format!("Author '{value}' must be in 'Name <email>' format"))?;
    let email = rest.trim_end_matches('>').trim();
    let mut table = InlineTable::new();
    table.insert("name", Value::from(name.trim()));
    table.insert("email", Value::from(email));
    Ok(table)
}
