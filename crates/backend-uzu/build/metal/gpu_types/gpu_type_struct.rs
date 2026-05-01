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

fn is_byte_scalar(ty: &str) -> bool {
    matches!(ty, "bool" | "i8" | "u8")
}

pub fn gpu_type_gen_struct(gpu_type_struct: &GpuTypeStruct) -> anyhow::Result<String> {
    let all_bytes = gpu_type_struct.fields.iter().all(|f| match &f.ty {
        GpuTypeStructFieldType::Scalar(ty) => is_byte_scalar(ty.as_ref()),
        _ => false,
    });
    let uint_compatible =
        all_bytes && gpu_type_struct.fields.len() <= 4 && gpu_type_struct.alignment == Some(4);

    let name = gpu_type_struct.name.as_ref();

    let field_lines: Vec<String> = gpu_type_struct
        .fields
        .iter()
        .map(|field| match &field.ty {
            GpuTypeStructFieldType::Scalar(ty) => Ok(format!("  {} {};", r2c(ty.as_ref())?, field.name)),
            GpuTypeStructFieldType::Array {
                element,
                length,
            } => Ok(format!("  {} {}[{}];", r2c(element.as_ref())?, field.name, *length)),
        })
        .collect::<anyhow::Result<_>>()?;

    let alignas = match gpu_type_struct.alignment {
        Some(n) => format!(" alignas({n})"),
        None => String::new(),
    };

    if uint_compatible {
        let init_lines = gpu_type_struct
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| {
                let shift = i * 8;
                let expr = if shift == 0 {
                    "uint8_t(__dsl_v)".to_string()
                } else {
                    format!("uint8_t(__dsl_v >> {shift})")
                };
                if i == 0 {
                    format!("    : {}({expr})", field.name)
                } else {
                    format!("    , {}({expr})", field.name)
                }
            })
            .collect::<Vec<_>>();

        Ok(once(format!("struct{alignas} {name} {{"))
            .chain(field_lines.into_iter())
            .chain(once(format!("  inline {name}() = default;")))
            .chain(once(format!("  inline {name}(uint __dsl_v)")))
            .chain(init_lines.into_iter())
            .chain(once("  {}".into()))
            .chain(once("};".into()))
            .join("\n"))
    } else {
        Ok(once(format!("typedef struct{alignas} {{"))
            .chain(field_lines.into_iter())
            .chain(once(format!("}} {name};")))
            .join("\n"))
    }
}
