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

    let variants = option_set
        .variants
        .iter()
        .map(|variant| {
            format!("  static constant constexpr {underlying_c} {} = {};\n", variant.name, variant.value_expression)
        })
        .collect::<String>();

    Ok(format!(
        "struct {name} {{\n\
         \x20 {underlying_c} raw_value;\n\
         \x20 constexpr {name}() thread : raw_value(0) {{}}\n\
         \x20 constexpr {name}({underlying_c} __dsl_v) thread : raw_value(__dsl_v) {{}}\n\
         {variants}\
         \x20 constexpr bool contains({underlying_c} flag) const thread {{ return (raw_value & flag) != 0; }}\n\
         }};"
    ))
}
