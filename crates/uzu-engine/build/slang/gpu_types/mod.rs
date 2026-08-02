use std::{
    collections::{HashMap, hash_map::Entry},
    path::Path,
};

use anyhow::{Context, bail};
use itertools::Itertools;

use crate::common::gpu_types::{GpuType, GpuTypeFile, GpuTypes};

mod gpu_type_constant;
mod gpu_type_enum;
mod gpu_type_option_set;
mod gpu_type_struct;

use gpu_type_constant::gpu_type_gen_constant;
use gpu_type_enum::gpu_type_gen_enum;
use gpu_type_option_set::gpu_type_gen_option_set;
use gpu_type_struct::gpu_type_gen_struct;

pub fn gpu_type_gen(
    gpu_types_dir: &Path,
    gpu_types: &GpuTypes,
) -> anyhow::Result<HashMap<String, String>> {
    let mut gpu_types_map = HashMap::new();

    for gpu_type_file in &gpu_types.files {
        gpu_type_gen_file(&gpu_types_dir.join(gpu_type_file.name.as_ref()).with_extension("slang"), gpu_type_file)
            .with_context(|| format!("Cannot generate gpu types for {}", gpu_type_file.name.as_ref()))?;
        for gpu_type in &gpu_type_file.types {
            let name: &str = match gpu_type {
                GpuType::Enum(gpu_type_enum) => gpu_type_enum.name.as_ref(),
                GpuType::Struct(gpu_type_struct) => gpu_type_struct.name.as_ref(),
                GpuType::OptionSet(gpu_type_option_set) => gpu_type_option_set.name.as_ref(),
                GpuType::Constant(_) => continue,
            };

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

fn gpu_type_gen_file(
    file_path: &Path,
    gpu_types_file: &GpuTypeFile,
) -> anyhow::Result<()> {
    let new_contents = gpu_types_file
        .types
        .iter()
        .map(|gpu_type| match gpu_type {
            GpuType::Constant(gpu_type_constant) => gpu_type_gen_constant(gpu_type_constant)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_constant:?}")),
            GpuType::Enum(gpu_type_enum) => gpu_type_gen_enum(gpu_type_enum)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_enum:?}")),
            GpuType::Struct(gpu_type_struct) => gpu_type_gen_struct(gpu_type_struct)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_struct:?}")),
            GpuType::OptionSet(gpu_type_option_set) => gpu_type_gen_option_set(gpu_type_option_set)
                .with_context(|| format!("Failed to generate bindings for {gpu_type_option_set:?}")),
        })
        .process_results(|mut it| it.join("\n\n"))?;

    // Avoid advancing mtime if the contents are the same
    if !std::fs::read(file_path).is_ok_and(|old_contents| old_contents == new_contents.as_bytes()) {
        std::fs::create_dir_all(file_path.parent().unwrap()).context("cannot create output directory")?;
        std::fs::write(file_path, new_contents).context("cannot write output")?;
    }

    Ok(())
}
