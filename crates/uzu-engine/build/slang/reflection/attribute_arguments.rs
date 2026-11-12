use anyhow::{Context, ensure};
use shader_slang::reflection::UserAttribute;

pub fn string_arguments<const N: usize>(attribute: &UserAttribute) -> anyhow::Result<[&str; N]> {
    let name = attribute.name().context("attribute has no name")?;
    let count = attribute.argument_count();
    ensure!(count as usize == N, "{name} expects {N} arguments, found {count}");

    let mut arguments = [""; N];
    for (index, argument) in arguments.iter_mut().enumerate() {
        *argument = attribute
            .argument_value_string(index as u32)
            .with_context(|| format!("{name} argument {index} has no string value"))?;
    }
    Ok(arguments)
}
