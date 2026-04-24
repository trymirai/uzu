use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use regex::Regex;

use crate::types::{Error, Example, Snippet};

pub fn update_readme(
    readme_path: PathBuf,
    snippets: Vec<Snippet>,
    examples: Vec<Example>,
    version: String,
) -> Result<(), Error> {
    let original_content = fs::read_to_string(&readme_path).map_err(|_| Error::UnableToUpdateREADME)?;

    let include_pattern = Regex::new(r#"(?m)^\s*//\s*include:(?P<path>[^\s#]+)(?:#(?P<snippet>[^\s]+))?\s*$"#)
        .map_err(|_| Error::UnableToUpdateREADME)?;

    // Index examples by normalized relative path
    let mut examples_by_relative_path: HashMap<String, &Example> = HashMap::new();
    for example in &examples {
        examples_by_relative_path.insert(normalize_slashes(&example.relative_path), example);
    }

    // Index snippets for quick lookup
    let mut snippets_by_name_and_source_path: HashMap<(String, String), &Snippet> = HashMap::new();
    let mut snippets_by_name: HashMap<String, Vec<&Snippet>> = HashMap::new();

    for snippet in &snippets {
        snippets_by_name_and_source_path
            .insert((snippet.name.clone(), normalize_slashes(&snippet.source_path)), snippet);
        snippets_by_name.entry(snippet.name.clone()).or_default().push(snippet);
    }

    let mut updated_content = String::with_capacity(original_content.len());
    let mut last_position = 0;

    for capture in include_pattern.captures_iter(&original_content) {
        let full_match = capture.get(0).ok_or(Error::UnableToUpdateREADME)?;
        updated_content.push_str(&original_content[last_position..full_match.start()]);

        let include_path = capture.name("path").ok_or(Error::UnableToUpdateREADME)?.as_str();
        let snippet_name = capture.name("snippet").map(|m| m.as_str());
        let normalized_include_path = normalize_slashes_str(include_path);

        // Find the matching example (by relative path)
        let example = *examples_by_relative_path.get(&normalized_include_path).ok_or(Error::UnableToUpdateREADME)?;

        // Choose replacement content (no indentation modification)
        let replacement_content = if let Some(snippet_name) = snippet_name {
            let exact_key = (snippet_name.to_string(), normalize_slashes(&example.full_path));

            if let Some(snippet) = snippets_by_name_and_source_path.get(&exact_key) {
                snippet.content.clone()
            } else {
                snippets_by_name
                    .get(snippet_name)
                    .and_then(|list| list.first().map(|snippet| snippet.content.clone()))
                    .ok_or(Error::UnableToUpdateREADME)?
            }
        } else {
            example.content.clone()
        };

        updated_content.push_str(&replacement_content);
        last_position = full_match.end();
    }

    updated_content.push_str(&original_content[last_position..]);

    // Replace all {{VERSION}} placeholders
    let final_content = updated_content.replace("{{VERSION}}", version.as_str());

    fs::write(&readme_path, final_content).map_err(|_| Error::UnableToUpdateREADME)?;
    Ok(())
}

fn normalize_slashes(path: &Path) -> String {
    normalize_slashes_str(&path.to_string_lossy())
}

fn normalize_slashes_str<S: AsRef<str>>(text: S) -> String {
    text.as_ref().replace('\\', "/")
}
