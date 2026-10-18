use anyhow::Context;

use crate::{common::gpu_types::GpuTypeConstant, slang::types::rust2slang};

pub fn gpu_type_gen_constant(constant: &GpuTypeConstant) -> anyhow::Result<String> {
    let ty = rust2slang(&constant.ty).with_context(|| format!("cannot convert {} to slang type", constant.ty))?;
    let array_suffix = constant.array_length.as_ref().map(|length| format!("[{length}]")).unwrap_or_default();
    Ok(format!("static const {ty} {}{array_suffix} = {};", constant.name, constant.value_expression))
}
