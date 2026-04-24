use std::fs;

use crate::types::{Environment, Error, Example, Language};

pub fn prepare_platform(
    environment: &Environment,
    language: Language,
    examples: Vec<Example>,
) -> Result<(), Error> {
    let platform_examples_path = environment.workspace().platform_path().join(language.clone().to_string());
    if platform_examples_path.exists() {
        fs::remove_dir_all(platform_examples_path.clone()).map_err(|_| Error::UnableToPreparePlatform)?;
    }
    fs::create_dir_all(platform_examples_path.clone()).map_err(|_| Error::UnableToPreparePlatform)?;

    for example in examples {
        if !example.exportable {
            continue;
        }
        let example_path = platform_examples_path.join(format!("{}.txt", example.name));
        fs::write(example_path, example.content).map_err(|_| Error::UnableToPreparePlatform)?;
    }

    Ok(())
}
