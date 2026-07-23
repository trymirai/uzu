use super::rust_to_metal;
use crate::common::gpu_types::GpuTypeConstant;

pub fn gpu_type_gen_constant(constant: &GpuTypeConstant) -> anyhow::Result<String> {
    let ty = rust_to_metal(&constant.ty)?;
    Ok(format!("static constant constexpr {ty} {} = {};", constant.name, constant.value_expression))
}
