use std::{fs, path::PathBuf};

use crate::{
    types::{Error, Example},
    utilities::{kebab_filename, relative_path},
};

const EXPORTED_EXAMPLE_NAMES: [&str; 12] = [
    "quick-start",
    "chat",
    "summarization",
    "classification",
    "cloud",
    "chat-static-context",
    "chat-dynamic-context",
    "ssm",
    "structured-output",
    "chat-with-speculator",
    "classifier",
    "text-to-speech",
];

pub fn extract_examples(
    examples_path: PathBuf,
    root_path: PathBuf,
) -> Result<Vec<Example>, Error> {
    let mut examples = Vec::new();
    for entry in fs::read_dir(examples_path).map_err(|_| Error::UnableToExtractExamples)? {
        let entry = entry.map_err(|_| Error::UnableToExtractExamples)?;
        let entry_path = entry.path();
        let content = fs::read_to_string(entry_path.clone()).map_err(|_| Error::UnableToExtractExamples)?;
        let relative_path =
            relative_path(entry_path.clone(), root_path.clone()).ok_or(Error::UnableToExtractExamples)?;
        let name = kebab_filename(&entry_path).ok_or(Error::UnableToExtractExamples)?;
        examples.push(Example::new(
            name.clone(),
            relative_path,
            entry_path.clone(),
            content,
            EXPORTED_EXAMPLE_NAMES.contains(&name.clone().as_str()),
        ));
    }
    Ok(examples)
}
