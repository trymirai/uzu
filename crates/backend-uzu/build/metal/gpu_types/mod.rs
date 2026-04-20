use std::path::Path;

use anyhow::Context;
use itertools::Itertools;

use crate::common::gpu_types::{GpuType, GpuTypeFile, GpuTypes};

mod gpu_type_enum;
mod gpu_type_struct;

use gpu_type_enum::gpu_type_gen_enum;
use gpu_type_struct::gpu_type_gen_struct;

pub async fn gpu_type_gen(
    gpu_types_dir: &Path,
    gpu_types: &GpuTypes,
) -> anyhow::Result<()> {
    for gpu_type_file in &gpu_types.files {
        gpu_type_gen_file(&gpu_types_dir.join(gpu_type_file.name.as_ref()).with_extension("h"), gpu_type_file)
            .await
            .with_context(|| format!("Cannot generate gpu types for {}", gpu_type_file.name.as_ref()))?;
    }

    Ok(())
}

async fn gpu_type_gen_file(
    file_path: &Path,
    gpu_types_file: &GpuTypeFile,
) -> anyhow::Result<()> {
    let module_name = gpu_types_file.name.as_ref();

    let generated = gpu_types_file
        .types
        .iter()
        .map(|gpu_type| match gpu_type {
            GpuType::Enum(gpu_type_enum) => gpu_type_gen_enum(gpu_type_enum)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_enum:?}")),
            GpuType::Struct(gpu_type_struct) => gpu_type_gen_struct(gpu_type_struct)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_struct:?}")),
        })
        .process_results(|mut it| it.join("\n\n"))?;

    let new_contents = format!(include_str!("template.ht"), module_name = module_name, generated = generated);

    // Avoid advancing mtime if the contents are the same
    if !tokio::fs::read(&file_path).await.is_ok_and(|old_contents| &old_contents == new_contents.as_bytes()) {
        tokio::fs::write(&file_path, new_contents).await.context("cannot write output")?;
    }

    Ok(())
}
