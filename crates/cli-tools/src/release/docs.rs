use std::fs;

use anyhow::Result;

use crate::configs::{Paths, PlatformsConfig};

pub fn stage_docs(
    paths: &Paths,
    platforms: &PlatformsConfig,
) -> Result<()> {
    let examples_root = paths.release_docs_path().join("examples");
    if examples_root.exists() {
        fs::remove_dir_all(&examples_root)?;
    }

    for (language, language_config) in &platforms.languages {
        let language_dir = examples_root.join(language.name());
        fs::create_dir_all(&language_dir)?;

        let source_root = paths.root_path.join(&language_config.examples_path);
        for example_name in platforms.examples.keys() {
            let converted_name = language.convert_file_name(example_name);
            let source_path = source_root.join(format!("{converted_name}.{}", language.file_extension()));
            let body = fs::read_to_string(&source_path)?;
            let mdx = format!("```{}\n{}\n```\n", language.code_fence(), body.trim_end());
            fs::write(language_dir.join(format!("{example_name}.mdx")), mdx)?;
        }
    }

    Ok(())
}
