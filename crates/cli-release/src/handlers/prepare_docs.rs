use std::{fs, path::PathBuf};

use crate::types::{Environment, Error, Example, Language, Snippet};

pub fn prepare_docs(
    environment: &Environment,
    version: String,
    language: Language,
    snippets: Vec<Snippet>,
    examples: Vec<Example>,
) -> Result<(), Error> {
    let snippets = [snippets, vec![dependency_snippet(language.clone(), version.clone())]].concat();

    let docs_snippets_path = environment.workspace().docs_snippets_path().join(language.clone().to_string());
    if docs_snippets_path.exists() {
        fs::remove_dir_all(docs_snippets_path.clone()).map_err(|_| Error::UnableToPrepareDocs)?;
    }
    fs::create_dir_all(docs_snippets_path.clone()).map_err(|_| Error::UnableToPrepareDocs)?;

    for snippet in snippets {
        let snippet_path = docs_snippets_path.join(format!("{}.mdx", snippet.name));
        fs::write(snippet_path, md_content(language.clone(), snippet.content))
            .map_err(|_| Error::UnableToPrepareDocs)?;
    }

    let docs_examples_path = environment.workspace().docs_examples_path().join(language.clone().to_string());
    if docs_examples_path.exists() {
        fs::remove_dir_all(docs_examples_path.clone()).map_err(|_| Error::UnableToPrepareDocs)?;
    }
    fs::create_dir_all(docs_examples_path.clone()).map_err(|_| Error::UnableToPrepareDocs)?;

    for example in examples {
        if !example.exportable {
            continue;
        }
        let example_path = docs_examples_path.join(format!("{}.mdx", example.name));
        fs::write(example_path, md_content(language.clone(), example.content))
            .map_err(|_| Error::UnableToPrepareDocs)?;
    }

    Ok(())
}

fn md_content(
    language: Language,
    content: String,
) -> String {
    format!("```{}\n{}\n```", language.to_string(), content)
}

fn dependency_snippet(
    language: Language,
    version: String,
) -> Snippet {
    let content = match language {
        Language::Swift => format!(
            "dependencies: [\n    .package(url: \"https://github.com/trymirai/uzu-swift.git\", from: \"{}\")\n]",
            version
        )
        .to_string(),
        Language::TS => format!("\"dependencies\": {{\n    \"@trymirai/uzu\": \"{}\"\n}}", version).to_string(),
    };
    Snippet {
        name: "dependency".to_string(),
        content,
        source_path: PathBuf::new(),
    }
}
