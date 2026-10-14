use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Context;
use itertools::Itertools;

use crate::common::{
    gpu_type_paths::GpuTypePaths,
    gpu_types::{GpuType, GpuTypes},
};

mod gpu_type_constant;
mod gpu_type_enum;
mod gpu_type_option_set;
mod gpu_type_struct;

use gpu_type_constant::gpu_type_gen_constant;
use gpu_type_enum::gpu_type_gen_enum;
use gpu_type_option_set::gpu_type_gen_option_set;
use gpu_type_struct::gpu_type_gen_struct;

pub fn rust_path<'a>(
    gpu_type_paths: &'a GpuTypePaths,
    c_type: &str,
) -> Option<&'a str> {
    let path = c_type.strip_prefix("uzu::")?;
    let short_name = path.rsplit("::").next()?;
    let rust_path = gpu_type_paths.full_path_for(short_name)?;
    let (_, type_path) = rust_path.split_once("::gpu_types::")?;
    (type_path == path).then_some(rust_path)
}

fn rust_to_metal(ty: &str) -> anyhow::Result<&'static str> {
    match ty {
        "i8" => Ok("int8_t"),
        "i16" => Ok("int16_t"),
        "i32" => Ok("int32_t"),
        "i64" => Ok("int64_t"),
        "u8" => Ok("uint8_t"),
        "u16" => Ok("uint16_t"),
        "u32" => Ok("uint32_t"),
        "u64" => Ok("uint64_t"),
        "f32" => Ok("float"),
        "bool" => Ok("bool"),
        unknown => anyhow::bail!("Unsupported GPU type: {unknown}"),
    }
}

pub fn gpu_type_gen(
    gpu_types_dir: &Path,
    gpu_types: &[&GpuTypes],
) -> anyhow::Result<()> {
    fs::create_dir_all(gpu_types_dir).context("cannot create gpu types directory")?;
    let mut files: BTreeMap<&str, Vec<&GpuType>> = BTreeMap::new();
    for gpu_types in gpu_types {
        for file in &gpu_types.files {
            files.entry(&file.name).or_default().extend(file.types.iter());
        }
    }
    for (module_name, types) in files {
        gpu_type_gen_file(&gpu_types_dir.join(module_name).with_extension("h"), module_name, &types)
            .with_context(|| format!("Cannot generate gpu types for {module_name}"))?;
    }

    Ok(())
}

fn gpu_type_gen_file(
    file_path: &Path,
    module_name: &str,
    types: &[&GpuType],
) -> anyhow::Result<()> {
    let generated = types
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

    let new_contents = format!(include_str!("template.ht"), module_name = module_name, generated = generated);

    // Avoid advancing mtime if the contents are the same
    if !fs::read(file_path).is_ok_and(|old_contents| old_contents == new_contents.as_bytes()) {
        fs::write(file_path, new_contents).context("cannot write output")?;
    }

    Ok(())
}
