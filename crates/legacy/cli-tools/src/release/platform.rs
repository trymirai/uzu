use std::{fs, path::Path};

use anyhow::Result;

use crate::configs::{Paths, PlatformsConfig};

pub fn stage_platform(
    paths: &Paths,
    platforms: &PlatformsConfig,
    release_path: &Path,
) -> Result<()> {
    let root = release_path.join("platform");
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }

    for (language, language_config) in &platforms.languages {
        let language_dir = root.join(language.name());
        fs::create_dir_all(&language_dir)?;

        let examples_root = paths.root_path.join(&language_config.examples_path);
        for example_name in platforms.examples.keys() {
            let converted_name = language.convert_file_name(example_name);
            let source_path = examples_root.join(format!("{converted_name}.{}", language.file_extension()));
            let body = fs::read_to_string(&source_path)?;
            fs::write(language_dir.join(format!("{example_name}.txt")), body)?;
        }
    }

    Ok(())
}
