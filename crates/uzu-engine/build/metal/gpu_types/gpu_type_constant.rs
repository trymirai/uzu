use super::rust_to_metal;
use crate::common::gpu_types::GpuTypeConstant;

pub fn gpu_type_gen_constant(constant: &GpuTypeConstant) -> anyhow::Result<String> {
    let ty = rust_to_metal(&constant.ty)?;
    let array_suffix = constant.array_length.as_ref().map(|length| format!("[{length}]")).unwrap_or_default();
    let separator = if constant.array_length.is_some() {
        "\n    "
    } else {
        " "
    };
    Ok(format!(
        "static constant constexpr {ty} {}{array_suffix} ={separator}{};",
        constant.name, constant.value_expression
    ))
}
