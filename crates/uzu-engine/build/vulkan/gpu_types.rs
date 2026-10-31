use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::Path,
};

use anyhow::Context;

use crate::{
    common::gpu_types::{GpuType, GpuTypeStructFieldType, GpuTypes},
    slang::rust2slang,
};

/// Writes the `generated.<file>` Slang module of each GPU types file's constants, enums and structs, and returns the
/// canonical path of each named type.
pub fn generate_types(
    gpu_types: &[&GpuTypes],
    out_dir: &Path,
) -> anyhow::Result<HashMap<String, String>> {
    let generated = out_dir.join("generated");
    fs::create_dir_all(&generated)?;
    let mut files: BTreeMap<&str, Vec<&GpuType>> = BTreeMap::new();
    let mut gpu_type_map = HashMap::new();
    for gpu_types in gpu_types {
        for file in &gpu_types.files {
            files.entry(&file.name).or_default().extend(file.types.iter());
            for gpu_type in &file.types {
                let name = match gpu_type {
                    GpuType::Enum(gpu_enum) => &*gpu_enum.name,
                    GpuType::Struct(gpu_struct) => &*gpu_struct.name,
                    GpuType::OptionSet(option_set) => &*option_set.name,
                    GpuType::Constant(_) => continue,
                };
                gpu_type_map.insert(name.to_string(), format!("{}::{}::{name}", gpu_types.module_path, file.name));
            }
        }
    }
    let mut written = HashSet::new();
    for (file_name, types) in files {
        let mut declarations = Vec::new();
        for gpu_type in types {
            match gpu_type {
                GpuType::Constant(constant) => {
                    let ty = rust2slang(&constant.ty).with_context(|| {
                        format!(
                            "gpu_types/{file_name}.rs: constant {} has unsupported type '{}'",
                            constant.name, constant.ty
                        )
                    })?;
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
                            let ty = rust2slang(element).with_context(|| {
                                format!(
                                    "gpu_types/{file_name}.rs: field {}.{} has unsupported type '{element}'",
                                    gpu_struct.name, field.name
                                )
                            })?;
                            Ok(format!("  public {ty} {}{length};\n", field.name))
                        })
                        .collect::<anyhow::Result<String>>()?;
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
            // Unchanged modules keep their modification time.
            if !fs::read(&path).is_ok_and(|old| old == contents.as_bytes()) {
                fs::write(&path, contents)?;
            }
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
    Ok(gpu_type_map)
}
