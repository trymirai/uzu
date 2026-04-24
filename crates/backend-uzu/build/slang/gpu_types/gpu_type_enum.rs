use std::iter::once;

use itertools::Itertools;

use crate::common::gpu_types::GpuTypeEnum;

pub fn gpu_type_gen_enum(gpu_type_enum: &GpuTypeEnum) -> anyhow::Result<String> {
    Ok(once(format!("enum {} : uint32_t {{", gpu_type_enum.name.as_ref()))
        .chain(gpu_type_enum.variants.iter().map(|variant| format!("  {} = {},", variant.name, variant.discriminant)))
        .chain(once("};".into()))
        .join("\n"))
}
