use std::{collections::HashMap, env, fs, path::PathBuf};

use anyhow::Context;
use async_trait::async_trait;
use syn::{Expr, ExprLit, Fields, Item, ItemStruct, Lit, Meta, Type, TypeArray, TypePath};
use tokio::task::spawn_blocking;

use crate::{
    common::{compiler::Compiler, kernel::Kernel},
    debug_log,
};

#[derive(Debug)]
pub struct GpuTypesCompiler {
    crate_dir: PathBuf,
    generated_dir: PathBuf,
    build_dir: PathBuf,
}

impl GpuTypesCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);

        let generated_dir = crate_dir.join("src/backends/metal/generated");
        fs::create_dir_all(&generated_dir).with_context(|| format!("cannot create {}", generated_dir.display()))?;

        let build_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("gpu_types");
        fs::create_dir_all(&build_dir).with_context(|| format!("cannot create {}", build_dir.display()))?;

        Ok(Self {
            crate_dir,
            generated_dir,
            build_dir,
        })
    }

    pub fn generated_header_dir(&self) -> &PathBuf {
        &self.generated_dir
    }

    fn generate(&self) -> anyhow::Result<()> {
        let gpu_types_dir = self.crate_dir.join("src/backends/common/gpu_types");

        let source_files: Vec<PathBuf> = fs::read_dir(&gpu_types_dir)
            .with_context(|| format!("cannot read {}", gpu_types_dir.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().map(|ext| ext == "rs").unwrap_or(false) && path.file_stem() != Some("mod".as_ref())
            })
            .collect();

        let mut hasher = blake3::Hasher::new();
        for path in &source_files {
            let content = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
            hasher.update(&content);
        }

        let source_hash: [u8; blake3::OUT_LEN] = hasher.finalize().into();
        let cached_hash_path = self.build_dir.join("gpu_types.hash");

        if let Ok(cached_hash) = fs::read(&cached_hash_path) {
            if cached_hash == source_hash {
                debug_log!("gpu_types cached");
                return Ok(());
            }
        }

        debug_log!("gpu_types generation started");

        for src_path in &source_files {
            self.generate_header_for_file(src_path)?;
        }

        fs::write(&cached_hash_path, &source_hash)
            .with_context(|| format!("cannot write {}", cached_hash_path.display()))?;

        debug_log!("gpu_types generation done");

        Ok(())
    }

    fn generate_header_for_file(
        &self,
        src_path: &PathBuf,
    ) -> anyhow::Result<()> {
        let module_name = src_path.file_stem().and_then(|s| s.to_str()).context("invalid source file name")?;

        let source = fs::read_to_string(src_path).with_context(|| format!("cannot read {}", src_path.display()))?;

        let syntax = syn::parse_file(&source).with_context(|| format!("cannot parse {}", src_path.display()))?;

        let mut repr_c_structs: Vec<ItemStruct> = Vec::new();
        for item in syntax.items {
            if let Item::Struct(item_struct) = item {
                let has_repr_c = item_struct.attrs.iter().any(|attribute| {
                    if attribute.path().is_ident("repr") {
                        if let Ok(ident) = attribute.parse_args::<syn::Ident>() {
                            return ident == "C";
                        }
                    }
                    false
                });

                if has_repr_c {
                    repr_c_structs.push(item_struct);
                }
            }
        }

        let mut c_struct_definitions = String::new();
        for rust_struct in &repr_c_structs {
            for attribute in &rust_struct.attrs {
                if attribute.path().is_ident("doc") {
                    if let Meta::NameValue(meta_name_value) = &attribute.meta {
                        if let Expr::Lit(ExprLit {
                            lit: Lit::Str(literal_string),
                            ..
                        }) = &meta_name_value.value
                        {
                            let doc_comment = literal_string.value();
                            let doc_comment = doc_comment.trim();
                            if !doc_comment.is_empty() {
                                c_struct_definitions.push_str("/**");
                                c_struct_definitions.push_str(doc_comment);
                                c_struct_definitions.push_str(" */\n");
                            }
                        }
                    }
                }
            }

            c_struct_definitions.push_str("typedef struct {\n");

            if let Fields::Named(named_fields) = &rust_struct.fields {
                for field in &named_fields.named {
                    let field_name = field.ident.as_ref().map(|ident| ident.to_string()).unwrap_or_default();
                    let c_type = rust_type_to_c(&field.ty);

                    for attribute in &field.attrs {
                        if attribute.path().is_ident("doc") {
                            if let Meta::NameValue(meta_name_value) = &attribute.meta {
                                if let Expr::Lit(ExprLit {
                                    lit: Lit::Str(literal_string),
                                    ..
                                }) = &meta_name_value.value
                                {
                                    let doc_comment = literal_string.value();
                                    let doc_comment = doc_comment.trim();
                                    if !doc_comment.is_empty() {
                                        c_struct_definitions.push_str("  /**");
                                        c_struct_definitions.push_str(doc_comment);
                                        c_struct_definitions.push_str(" */\n");
                                    }
                                }
                            }
                        }
                    }

                    if let Some((base_type, array_size)) = c_type.split_once('[') {
                        c_struct_definitions.push_str(&format!("  {base_type} {field_name}[{array_size};\n"));
                    } else {
                        c_struct_definitions.push_str(&format!("  {c_type} {field_name};\n"));
                    }
                }
            }

            let struct_name = rust_struct.ident.to_string();
            c_struct_definitions.push_str(&format!("}} {struct_name};\n\n"));
        }

        self.write_header(module_name, c_struct_definitions.trim())?;

        Ok(())
    }

    fn write_header(
        &self,
        module_name: &str,
        structs: &str,
    ) -> anyhow::Result<()> {
        let include_guard = format!("UZU_{}_H", module_name.to_uppercase());

        let header_content = format!(
            r#"// Auto-generated from gpu_types/{module_name}.rs - do not edit manually
#pragma once

#ifndef {include_guard}
#define {include_guard}

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {{
namespace {module_name} {{
#else
#include <stdint.h>
#endif

{structs}

#ifdef __METAL_VERSION__
}} // namespace {module_name}
}} // namespace uzu
#endif

#endif // {include_guard}
"#
        );

        let output_path = self.generated_dir.join(format!("{module_name}.h"));
        fs::write(&output_path, &header_content).with_context(|| format!("cannot write {}", output_path.display()))?;

        debug_log!("  -> {}", output_path.display());

        Ok(())
    }
}

fn rust_type_to_c(rust_type: &Type) -> String {
    match rust_type {
        Type::Path(TypePath {
            path,
            ..
        }) => {
            let segment = path.segments.last().unwrap();
            let type_name = segment.ident.to_string();

            match type_name.as_str() {
                "i8" => "int8_t".to_string(),
                "i16" => "int16_t".to_string(),
                "i32" => "int32_t".to_string(),
                "i64" => "int64_t".to_string(),
                "u8" => "uint8_t".to_string(),
                "u16" => "uint16_t".to_string(),
                "u32" => "uint32_t".to_string(),
                "u64" => "uint64_t".to_string(),
                "f32" => "float".to_string(),
                "f64" => "double".to_string(),
                "bool" => "bool".to_string(),
                "usize" => "size_t".to_string(),
                "isize" => "ptrdiff_t".to_string(),
                _ => type_name,
            }
        },
        Type::Array(TypeArray {
            elem,
            len,
            ..
        }) => {
            let element_type = rust_type_to_c(elem);
            if let Expr::Lit(ExprLit {
                lit: Lit::Int(literal_int),
                ..
            }) = len
            {
                let array_length = literal_int.base10_digits();
                format!("{element_type}[{array_length}]")
            } else {
                format!("{element_type}[]")
            }
        },
        _ => "/* unknown */".to_string(),
    }
}

#[async_trait]
impl Compiler for GpuTypesCompiler {
    async fn build(&self) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>> {
        let crate_dir = self.crate_dir.clone();
        let generated_dir = self.generated_dir.clone();
        let build_dir = self.build_dir.clone();

        spawn_blocking(move || {
            GpuTypesCompiler {
                crate_dir,
                generated_dir,
                build_dir,
            }
            .generate()
        })
        .await??;

        Ok(HashMap::new())
    }
}
