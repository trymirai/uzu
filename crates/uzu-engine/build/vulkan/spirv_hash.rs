use std::{fs, path::Path};

use anyhow::Context;
use walkdir::WalkDir;

pub fn export_spirv_hash(output_directory: &Path) -> anyhow::Result<()> {
    let mut spirv = blake3::Hasher::new();
    for entry in WalkDir::new(output_directory).sort_by_file_name() {
        let path = entry.with_context(|| format!("cannot walk {}", output_directory.display()))?.into_path();
        if path.extension().is_some_and(|extension| extension == "spv") {
            spirv.update(&fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?);
        }
    }
    println!("cargo::rustc-env=UZU_VULKAN_SPIRV_HASH={}", &spirv.finalize().to_hex()[..16]);
    Ok(())
}
