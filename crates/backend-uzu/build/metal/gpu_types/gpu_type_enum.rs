use std::iter::once;

use igata::gpu_types::{
    GpuTypeEnum,
    tile_geometry::{self, ACCESSORS},
};
use itertools::Itertools;

pub fn gpu_type_gen_enum(gpu_type_enum: &GpuTypeEnum) -> anyhow::Result<String> {
    let declaration = once(format!("enum class {} : uint32_t {{", gpu_type_enum.name.as_ref()))
        .chain(gpu_type_enum.variants.iter().map(|variant| format!("  {} = {},", variant.name, variant.discriminant)))
        .chain(once("};".into()))
        .join("\n");

    Ok(match tile_geometry::geometries(gpu_type_enum) {
        Some(geometries) => format!("{declaration}\n\n{}", tile_accessors(gpu_type_enum, &geometries)),
        None => declaration,
    })
}

/// Metal accessors for a tile enum, as a chain of ternaries so they stay constexpr.
fn tile_accessors(
    gpu_type_enum: &GpuTypeEnum,
    geometries: &[(&str, tile_geometry::TileGeometry)],
) -> String {
    let type_name = gpu_type_enum.name.as_ref();
    let prefix = tile_geometry::metal_prefix(type_name);

    ACCESSORS
        .iter()
        .map(|(_, metal_suffix, value_of)| {
            let arms = geometries
                .iter()
                .map(|(variant, geometry)| format!("      t == {type_name}::{variant} ? {}\n    :", value_of(geometry)))
                .join("\n");

            format!("constexpr uint {prefix}_{metal_suffix}({type_name} t) {{\n  return\n{arms} 0;\n}}")
        })
        .join("\n\n")
}
