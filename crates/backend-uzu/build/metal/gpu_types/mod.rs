use std::path::Path;

use anyhow::Context;
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
        "usize" => Ok("size_t"),
        "isize" => Ok("ptrdiff_t"),
        unknown => anyhow::bail!("Unsupported GPU type: {unknown}"),
    }
}

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
    if !tokio::fs::read(&file_path).await.is_ok_and(|old_contents| old_contents == new_contents.as_bytes()) {
        tokio::fs::write(&file_path, new_contents).await.context("cannot write output")?;
    }

    Ok(())
}
