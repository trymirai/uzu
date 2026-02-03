use std::{env, fs, path::PathBuf};

use anyhow::Context;
use quote::quote;

use super::{codegen::write_tokens, kernel::Struct};

pub fn structgen_all(structs: &[Struct]) -> anyhow::Result<()> {
    let out_dir =
        PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);

    if structs.is_empty() {
        write_tokens(quote! {}, out_dir.join("dsl_structs.rs"))
            .context("cannot write empty struct bindings")?;
        return Ok(());
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

    let temp_header = out_dir.join("dsl_structs.h");
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

    write_tokens(quote! { #syntax_tree }, out_dir.join("dsl_structs.rs"))
        .context("cannot write struct bindings")?;

    Ok(())
}
