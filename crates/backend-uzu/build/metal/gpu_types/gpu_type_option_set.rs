use std::iter::once;

use itertools::Itertools;

use crate::common::gpu_types::GpuTypeOptionSet;

fn r2c(ty: &str) -> anyhow::Result<&'static str> {
    Ok(match ty {
        "u8" => "uint8_t",
        "u16" => "uint16_t",
        "u32" => "uint",
        "u64" => "uint64_t",
        "i8" => "int8_t",
        "i16" => "int16_t",
        "i32" => "int",
        "i64" => "int64_t",
        unknown => anyhow::bail!("Unsupported option-set underlying type: {unknown}"),
    })
}

/// Emit a Metal-side option-set as a struct wrapping the underlying primitive,
/// with named static constants for each flag and a `contains()` helper.
/// Implicit conversions from the underlying type allow a `uint` function
/// constant to be passed where the struct is expected.
pub fn gpu_type_gen_option_set(option_set: &GpuTypeOptionSet) -> anyhow::Result<String> {
    let name = option_set.name.as_ref();
    let underlying_c = r2c(option_set.underlying_type.as_ref())?;

    let constants = option_set
        .variants
        .iter()
        .map(|variant| {
            format!(
                "  static constant constexpr {underlying_c} {} = {};",
                variant.name, variant.value_expression
            )
        });

    Ok(once(format!("struct {name} {{"))
        .chain(once(format!("  {underlying_c} raw_value;")))
        .chain(once(format!("  constexpr {name}() thread : raw_value(0) {{}}")))
        .chain(once(format!(
            "  constexpr {name}({underlying_c} __dsl_v) thread : raw_value(__dsl_v) {{}}"
        )))
        .chain(constants)
        .chain(once(format!(
            "  constexpr bool contains({underlying_c} flag) const thread {{ return (raw_value & flag) != 0; }}"
        )))
        .chain(once("};".into()))
        .join("\n"))
}
