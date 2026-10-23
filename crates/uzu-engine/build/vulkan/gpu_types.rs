use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

use anyhow::{Context, bail};

use super::Error;
use crate::common::gpu_types::{GpuType, GpuTypeStructFieldType, GpuTypes};

/// Writes the `generated.<file>` Slang module of each GPU types file's constants, enums and structs.
pub fn generate_types(
    gpu_types: &[&GpuTypes],
    out_dir: &Path,
) -> Result<(), Error> {
    let generated = out_dir.join("generated");
    fs::create_dir_all(&generated)?;
    let mut files: BTreeMap<&str, Vec<&GpuType>> = BTreeMap::new();
    for gpu_types in gpu_types {
        for file in &gpu_types.files {
            files.entry(&file.name).or_default().extend(file.types.iter());
        }
    }
    let mut written = HashSet::new();
    for (file_name, types) in files {
        let mut declarations = Vec::new();
        for gpu_type in types {
            match gpu_type {
                GpuType::Constant(constant) => {
                    let ty = match constant.ty.as_ref() {
                        "i16" => "int16_t",
                        "u32" => "uint",
                        "f32" => "float",
                        other => bail!(
                            "gpu_types/{}.rs: constant {} has unsupported type '{other}'",
                            file_name,
                            constant.name
                        ),
                    };
                    let array_suffix =
                        constant.array_length.as_ref().map(|length| format!("[{length}]")).unwrap_or_default();
                    declarations.push(format!(
                        "public static const {ty} {}{array_suffix} = {};\n",
                        constant.name, constant.value_expression
                    ));
                },
                GpuType::Enum(gpu_enum) => {
                    let variants = gpu_enum
                        .variants
                        .iter()
                        .map(|variant| format!("  {} = {},\n", variant.name, variant.discriminant))
                        .collect::<String>();
                    declarations.push(format!("public enum {} : uint {{\n{variants}}}\n", gpu_enum.name));
                },
                // Slang stores `bool` in 4 bytes and Rust in 1; the binding's layout guard rejects such structs.
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
                                    file_name,
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
            let path = generated.join(file_name).with_extension("slang");
            let contents = format!(
                "// Generated from gpu_types/{}.rs\nmodule {};\n\n{}",
                file_name,
                file_name,
                declarations.concat()
            );
            fs::write(&path, contents)?;
            written.insert(path);
        }
    }
    // A types file removed, or now without declarations, must not leave its module behind.
    for entry in fs::read_dir(&generated)? {
        let path = entry?.path();
        if !written.contains(&path) {
            fs::remove_file(&path).with_context(|| format!("cannot remove {}", path.display()))?;
        }
    }
    Ok(())
}
