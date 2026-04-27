use std::path::Path;

mod toolchains;

use anyhow::{Ok, Result, anyhow};
pub use toolchains::ToolchainsSyncTask;

use crate::configs::{Paths, PlatformsConfig};

pub trait SyncTask {
    fn process(
        config: &PlatformsConfig,
        input: &str,
    ) -> Result<String>;

    fn run(
        config: &PlatformsConfig,
        input_path: &Path,
        check: bool,
    ) -> Result<()> {
        let input = std::fs::read_to_string(input_path)?;
        let output = Self::process(config, &input)?;
        if check {
            if input != output {
                return Err(anyhow!("The file is out of sync: {}", input_path.display()));
            }
        } else {
            std::fs::write(input_path, output)?;
        }
        Ok(())
    }
}

pub fn run_sync(check: bool) -> Result<()> {
    let paths = Paths::new()?;
    let config = PlatformsConfig::load()?;
    ToolchainsSyncTask::run(&config, &paths.root_path.join("rust-toolchain.toml"), check)?;
    Ok(())
}
