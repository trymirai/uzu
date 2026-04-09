use std::iter::once;

use anyhow::Context;
use itertools::Itertools;

use crate::{
    common::gpu_types::{GpuTypeStruct, GpuTypeStructFieldType},
    slang::types::rust2slang,
};

pub fn gpu_type_gen_struct(gpu_type_struct: &GpuTypeStruct) -> anyhow::Result<String> {
    once(Ok(format!("struct {} {{", gpu_type_struct.name.as_ref())))
        .chain(gpu_type_struct.fields.iter().map(|field| match &field.ty {
            GpuTypeStructFieldType::Scalar(ty) => Ok(format!(
                "  {} {};",
                rust2slang(ty.as_ref()).with_context(|| format!("cannot convert {} to slang type", ty.as_ref()))?,
                field.name
            )),
            GpuTypeStructFieldType::Array {
                element,
                length,
            } => Ok(format!(
                    "  {} {}[{}];",
                    rust2slang(element.as_ref())
                        .with_context(|| format!("cannot convert {} to slang type", element.as_ref()))?,
                    field.name,
                    *length
                )),
        }))
        .chain(once(Ok("};".to_string())))
        .process_results(|mut it| it.join("\n"))
}
