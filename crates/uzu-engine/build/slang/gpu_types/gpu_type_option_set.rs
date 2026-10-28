use std::iter::once;

use anyhow::Context;
use itertools::Itertools;

use crate::{common::gpu_types::GpuTypeOptionSet, slang::types::rust2slang};

/// Emit a Slang-side option-set as a struct wrapping the underlying primitive,
/// with named static constants for each flag and a `contains()` helper,
/// mirroring the Metal-side generation.
pub fn gpu_type_gen_option_set(option_set: &GpuTypeOptionSet) -> anyhow::Result<String> {
    let name = &option_set.name;
    let underlying_slang = rust2slang(&option_set.underlying_type)
        .with_context(|| format!("cannot convert {} to slang type", option_set.underlying_type))?;

    once(Ok(format!("struct {name} {{")))
        .chain(once(Ok(format!("  {underlying_slang} raw_value;"))))
        .chain(option_set.variants.iter().map(|variant| {
            Ok(format!("  static const {underlying_slang} {} = {};", variant.name, variant.value_expression))
        }))
        .chain(once(Ok(format!("  bool contains({underlying_slang} flag) {{ return (raw_value & flag) != 0; }}"))))
        .chain(once(Ok(format!("  {underlying_slang} bits() {{ return raw_value; }}"))))
        .chain(once(Ok("};".to_string())))
        .process_results(|mut it| it.join("\n"))
}
