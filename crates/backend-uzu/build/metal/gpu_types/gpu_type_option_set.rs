use std::fmt::Write;

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
    let name = &option_set.name;
    let underlying_c = r2c(&option_set.underlying_type)?;

    let mut out = String::new();
    writeln!(out, "struct {name} {{").unwrap();
    writeln!(out, "  {underlying_c} raw_value;").unwrap();
    writeln!(out, "  constexpr {name}() thread : raw_value(0) {{}}").unwrap();
    writeln!(out, "  constexpr {name}({underlying_c} __dsl_v) thread : raw_value(__dsl_v) {{}}").unwrap();
    for variant in &option_set.variants {
        writeln!(out, "  static constant constexpr {underlying_c} {} = {};", variant.name, variant.value_expression)
            .unwrap();
    }
    writeln!(out, "  constexpr bool contains({underlying_c} flag) const thread {{ return (raw_value & flag) != 0; }}")
        .unwrap();
    out.push_str("};");
    Ok(out)
}
