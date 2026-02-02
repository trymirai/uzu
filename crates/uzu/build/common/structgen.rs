use std::fs;

use anyhow::Context;
use proc_macro2::TokenStream;
use quote::quote;

use super::kernel::Struct;

pub fn structgen(
    structs: &[Struct],
    build_dir: &std::path::Path,
) -> anyhow::Result<TokenStream> {
    if structs.is_empty() {
        return Ok(quote! {});
    }

    let mut header_content = String::from("#include <stdint.h>\n\n");
    for struct_info in structs {
        header_content.push_str(&format!("struct {} {{\n", struct_info.name));
        for field in struct_info.fields.iter() {
            header_content
                .push_str(&format!("    {} {};\n", field.ty, field.name));
        }
        header_content.push_str("};\n\n");
    }

    let temp_header = build_dir.join("dsl_structs.h");
    fs::write(&temp_header, header_content)
        .context("cannot write temp header for struct bindgen")?;

    let mut builder = bindgen::Builder::default()
        .header(temp_header.to_string_lossy())
        .clang_arg("-x")
        .clang_arg("c")
        .derive_default(true)
        .derive_copy(true)
        .derive_debug(true)
        .use_core()
        .layout_tests(true);

    for struct_info in structs {
        builder = builder.allowlist_type(struct_info.name.as_ref());
    }

    let bindings = builder
        .generate()
        .context("bindgen failed to generate struct bindings")?;

    let bindings_str = bindings.to_string();
    let syntax_tree: syn::File = syn::parse_str(&bindings_str)
        .context("failed to parse bindgen output")?;

    Ok(quote! { #syntax_tree })
}
