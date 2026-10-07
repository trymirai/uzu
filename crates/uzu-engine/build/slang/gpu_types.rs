use std::{fs, path::Path};

use anyhow::{Context, bail};

use super::Error;
use crate::common::gpu_types::{GpuType, GpuTypes};

/// Writes `generated/<file>.slang` under `out_dir` for every GPU types file that declares constants, so shaders
/// `import generated.<file>;`. Only scalar constants are emitted; enums, structs and option sets are not.
pub fn generate_constants(
    gpu_types: &GpuTypes,
    out_dir: &Path,
) -> Result<(), Error> {
    let generated = out_dir.join("generated");
    if generated.exists() {
        fs::remove_dir_all(&generated)?;
    }
    fs::create_dir_all(&generated)?;
    for file in &gpu_types.files {
        let mut constants = Vec::new();
        for gpu_type in &file.types {
            let GpuType::Constant(constant) = gpu_type else {
                continue;
            };
            let ty = match constant.ty.as_ref() {
                "u32" => "uint",
                "f32" => "float",
                other => bail!("gpu_types/{}.rs: constant {} has unsupported type '{other}'", file.name, constant.name),
            };
            constants.push(format!("public static const {ty} {} = {};\n", constant.name, constant.value_expression));
        }
        if !constants.is_empty() {
            let path = generated.join(file.name.as_ref()).with_extension("slang");
            let contents = format!(
                "// Generated from gpu_types/{}.rs\nmodule {};\n\n{}",
                file.name,
                file.name,
                constants.concat()
            );
            fs::write(&path, contents).with_context(|| format!("cannot write {}", path.display()))?;
        }
    }
    Ok(())
}
