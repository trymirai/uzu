use anyhow::bail;

use crate::common::gpu_types::GpuTypeConstant;

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
        unknown => bail!("Unsupported GPU constant type: {unknown}"),
    }
}

pub fn gpu_type_gen_constant(constant: &GpuTypeConstant) -> anyhow::Result<String> {
    let ty = rust_to_metal(&constant.ty)?;
    Ok(format!("static constant constexpr {ty} {} = {};", constant.name, constant.value_expression))
}
