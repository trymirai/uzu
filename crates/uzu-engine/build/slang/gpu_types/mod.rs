use std::{
    collections::{BTreeMap, HashMap, HashSet, hash_map::Entry},
    fs,
    path::Path,
};

use anyhow::{Context, bail};
use itertools::Itertools;

use crate::common::gpu_types::{GpuType, GpuTypes};

mod gpu_type_constant;
mod gpu_type_enum;
mod gpu_type_option_set;
mod gpu_type_struct;

use gpu_type_constant::gpu_type_gen_constant;
use gpu_type_enum::gpu_type_gen_enum;
use gpu_type_option_set::gpu_type_gen_option_set;
use gpu_type_struct::gpu_type_gen_struct;

/// Writes one Slang file per GPU types file name into `gpu_types_dir`, merging the files of that name in `gpu_types`,
/// and removes any other file there; returns the canonical path of each named type by its short name.
pub fn gpu_type_gen(
    gpu_types_dir: &Path,
    gpu_types: &[&GpuTypes],
) -> anyhow::Result<HashMap<String, String>> {
    let mut gpu_types_map = HashMap::new();
    let mut files: BTreeMap<&str, Vec<&GpuType>> = BTreeMap::new();

    for gpu_types in gpu_types {
        for gpu_type_file in &gpu_types.files {
            files.entry(&gpu_type_file.name).or_default().extend(&gpu_type_file.types);
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
                        vacant.insert(format!("{}::{}::{}", gpu_types.module_path, gpu_type_file.name, name))
                    },
                };
            }
        }
    }

    fs::create_dir_all(gpu_types_dir).context("cannot create output directory")?;
    let mut written = HashSet::new();
    for (name, types) in files {
        let file_path = gpu_types_dir.join(name).with_extension("slang");
        gpu_type_gen_file(&file_path, &types).with_context(|| format!("Cannot generate gpu types for {name}"))?;
        written.insert(file_path);
    }

    // A types file removed must not leave its Slang file behind.
    for entry in fs::read_dir(gpu_types_dir).context("cannot list generated gpu types")? {
        let path = entry?.path();
        if !written.contains(&path) {
            fs::remove_file(&path).with_context(|| format!("cannot remove {}", path.display()))?;
        }
    }

    Ok(gpu_types_map)
}

fn gpu_type_gen_file(
    file_path: &Path,
    types: &[&GpuType],
) -> anyhow::Result<()> {
    let new_contents = types
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
    if !fs::read(file_path).is_ok_and(|old_contents| old_contents == new_contents.as_bytes()) {
        fs::write(file_path, new_contents).context("cannot write output")?;
    }

    Ok(())
}
