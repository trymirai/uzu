use std::iter::once;

use anyhow::anyhow;
use itertools::Itertools;

use crate::common::gpu_types::{GpuTypeStruct, GpuTypeStructFieldType};

fn r2c(ty: &str) -> anyhow::Result<&'static str> {
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
        _ => Err(anyhow!("Only primitive types are supported, found {ty}")),
    }
}

pub fn gpu_type_gen_struct(gpu_type_struct: &GpuTypeStruct) -> anyhow::Result<String> {
    once(Ok("typedef struct {".into()))
        .chain(gpu_type_struct.fields.iter().map(|field| match &field.ty {
            GpuTypeStructFieldType::Scalar(ty) => Ok(format!("  {} {};", r2c(ty.as_ref())?, field.name)),
            GpuTypeStructFieldType::Array {
                element,
                length,
            } => Ok(format!("  {} {}[{}];", r2c(element.as_ref())?, field.name, *length)),
        }))
        .chain(once(Ok(format!("}} {};", gpu_type_struct.name.as_ref()))))
        .process_results(|mut it| it.join("\n"))
}
