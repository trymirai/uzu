use std::{fs, path::Path};

use anyhow::{Context, bail};

use super::Error;
use crate::common::gpu_types::{GpuType, GpuTypeStructFieldType, GpuTypes};

/// Writes `generated/<file>.slang` under `out_dir` for every GPU types file that declares constants, enums or structs,
/// so shaders `import generated.<file>;`. Enums are `uint`-backed with the canonical discriminants; option sets are not
/// emitted.
pub fn generate_types(
    gpu_types: &GpuTypes,
    out_dir: &Path,
) -> Result<(), Error> {
    let generated = out_dir.join("generated");
    if generated.exists() {
        fs::remove_dir_all(&generated)?;
    }
    fs::create_dir_all(&generated)?;
    for file in &gpu_types.files {
        let mut declarations = Vec::new();
        for gpu_type in &file.types {
            match gpu_type {
                GpuType::Constant(constant) => {
                    let ty = match constant.ty.as_ref() {
                        "u32" => "uint",
                        "f32" => "float",
                        other => bail!(
                            "gpu_types/{}.rs: constant {} has unsupported type '{other}'",
                            file.name,
                            constant.name
                        ),
                    };
                    declarations
                        .push(format!("public static const {ty} {} = {};\n", constant.name, constant.value_expression));
                },
                GpuType::Enum(gpu_enum) => {
                    let variants = gpu_enum
                        .variants
                        .iter()
                        .map(|variant| format!("  {} = {},\n", variant.name, variant.discriminant))
                        .collect::<String>();
                    declarations.push(format!("public enum {} : uint {{\n{variants}}}\n", gpu_enum.name));
                },
                // Fields of 32-bit scalars or their arrays. Slang stores `bool` in 4 bytes and Rust in 1, so a struct
                // with one has no shared layout; the binding's layout guard rejects it as a host slice.
                GpuType::Struct(gpu_struct) => {
                    let fields = gpu_struct
                        .fields
                        .iter()
                        .map(|field| {
                            let (element, length) = match &field.ty {
                                GpuTypeStructFieldType::Scalar(element) => (element, String::new()),
                                GpuTypeStructFieldType::Array {
                                    element,
                                    length,
                                } => (element, format!("[{length}]")),
                            };
                            let ty = match element.as_ref() {
                                "u32" => "uint",
                                "f32" => "float",
                                "bool" => "bool",
                                other => bail!(
                                    "gpu_types/{}.rs: field {}.{} has unsupported type '{other}'",
                                    file.name,
                                    gpu_struct.name,
                                    field.name
                                ),
                            };
                            Ok(format!("  public {ty} {}{length};\n", field.name))
                        })
                        .collect::<Result<String, Error>>()?;
                    declarations.push(format!("public struct {} {{\n{fields}}}\n", gpu_struct.name));
                },
                GpuType::OptionSet(_) => {},
            }
        }
        if !declarations.is_empty() {
            let path = generated.join(file.name.as_ref()).with_extension("slang");
            let contents = format!(
                "// Generated from gpu_types/{}.rs\nmodule {};\n\n{}",
                file.name,
                file.name,
                declarations.concat()
            );
            fs::write(&path, contents).with_context(|| format!("cannot write {}", path.display()))?;
        }
    }
    Ok(())
}
