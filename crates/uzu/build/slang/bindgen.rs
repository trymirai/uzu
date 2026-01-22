use anyhow::Context;

use crate::slang::reflection::SlangKernelInfo;

pub fn bindgen(kernel: &SlangKernelInfo) -> anyhow::Result<String> {
    let mut lines = vec![format!("// {}", kernel.name())];

    for argument in kernel.arguments() {
        lines.push(format!(
            "// {}: {} ({:?})",
            argument.name(),
            argument.slang_type(),
            argument.argument_type().context("cannot parse argument type")?
        ));
    }

    Ok(lines.join("\n"))
}
