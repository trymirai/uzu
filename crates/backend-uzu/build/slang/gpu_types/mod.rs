use std::{
    collections::{HashMap, hash_map::Entry},
    path::Path,
};

use anyhow::{Context, bail};
use itertools::Itertools;

use crate::common::gpu_types::{GpuType, GpuTypeEnum, GpuTypeFile, GpuTypeStruct, GpuTypes};

mod gpu_type_enum;
mod gpu_type_struct;

use gpu_type_enum::gpu_type_gen_enum;
use gpu_type_struct::gpu_type_gen_struct;

pub async fn gpu_type_gen(
    gpu_types_dir: &Path,
    gpu_types: &GpuTypes,
) -> anyhow::Result<HashMap<String, String>> {
    let mut gpu_types_map = HashMap::new();

    for gpu_type_file in &gpu_types.files {
        gpu_type_gen_file(&gpu_types_dir.join(gpu_type_file.name.as_ref()).with_extension("slang"), gpu_type_file)
            .await
            .with_context(|| format!("Cannot generate gpu types for {}", gpu_type_file.name.as_ref()))?;
        for gpu_type in &gpu_type_file.types {
            let (GpuType::Enum(GpuTypeEnum {
                name,
                ..
            })
            | GpuType::Struct(GpuTypeStruct {
                name,
                ..
            })) = gpu_type;

            match gpu_types_map.entry(name.to_string()) {
                Entry::Occupied(_) => bail!("{name} is duplicated"),
                Entry::Vacant(vacant) => {
                    vacant.insert(format!("crate::backends::common::gpu_types::{}::{}", gpu_type_file.name, name))
                },
            };
        }
    }

    Ok(gpu_types_map)
}

async fn gpu_type_gen_file(
    file_path: &Path,
    gpu_types_file: &GpuTypeFile,
) -> anyhow::Result<()> {
    let new_contents = gpu_types_file
        .types
        .iter()
        .map(|gpu_type| match gpu_type {
            GpuType::Enum(gpu_type_enum) => gpu_type_gen_enum(gpu_type_enum)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_enum:?}")),
            GpuType::Struct(gpu_type_struct) => gpu_type_gen_struct(gpu_type_struct)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_struct:?}")),
        })
        .process_results(|mut it| it.join("\n\n"))?;

    // Avoid advancing mtime if the contents are the same
    if !tokio::fs::read(&file_path).await.is_ok_and(|old_contents| &old_contents == new_contents.as_bytes()) {
        tokio::fs::create_dir_all(&file_path.parent().unwrap()).await.context("cannot create output directory")?;
        tokio::fs::write(&file_path, new_contents).await.context("cannot write output")?;
    }

    Ok(())
}
