use std::iter::once;

use itertools::Itertools;

use super::rust_to_metal;
use crate::common::gpu_types::{GpuTypeStruct, GpuTypeStructFieldType};

pub fn gpu_type_gen_struct(gpu_type_struct: &GpuTypeStruct) -> anyhow::Result<String> {
    once(Ok("typedef struct {".into()))
        .chain(gpu_type_struct.fields.iter().map(|field| match &field.ty {
            GpuTypeStructFieldType::Scalar(ty) => Ok(format!("  {} {};", rust_to_metal(ty.as_ref())?, field.name)),
            GpuTypeStructFieldType::Array {
                element,
                length,
            } => Ok(format!("  {} {}[{}];", rust_to_metal(element.as_ref())?, field.name, *length)),
        }))
        .chain(once(Ok(format!("}} {};", gpu_type_struct.name.as_ref()))))
        .process_results(|mut it| it.join("\n"))
}
