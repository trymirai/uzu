use anyhow::{Context, Result};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, Type};

use super::super::{
    ast::{MetalArgumentType, MetalKernelInfo},
    enum_path_rewrite::gpu_type_kind_for_c_type,
};
use crate::common::enum_paths::{EnumPaths, GpuTypeKind};

enum SpecializeLowering {
    Direct,
    Enum,
    OptionSet,
}

struct SpecializeArgument {
    name: Ident,
    rust_type: Type,
    function_constant_index: usize,
    lowering: SpecializeLowering,
}

pub struct SpecializeEmission {
    arguments: Vec<SpecializeArgument>,
}

pub fn parse(
    kernel: &MetalKernelInfo,
    base_function_constant_index: Option<usize>,
    kernel_name: &str,
    enum_paths: &EnumPaths,
) -> Result<SpecializeEmission> {
    let arguments = kernel
        .arguments
        .iter()
        .filter_map(|argument| match &argument.argument_type {
            MetalArgumentType::Specialize(rust_type_text) => Some((argument, rust_type_text)),
            _ => None,
        })
        .enumerate()
        .map(|(offset, (argument, rust_type_text))| -> Result<SpecializeArgument> {
            let name = format_ident!("{}", argument.name.as_ref());
            let rust_type: Type = syn::parse_str(&rust_type_text).with_context(|| {
                format!("specialize type `{}` in kernel `{}` cannot be parsed", rust_type_text, kernel_name)
            })?;
            let function_constant_index = base_function_constant_index.unwrap_or(0) + offset;
            let lowering = match gpu_type_kind_for_c_type(enum_paths, &argument.c_type) {
                Some(GpuTypeKind::Enum) => SpecializeLowering::Enum,
                Some(GpuTypeKind::OptionSet) => SpecializeLowering::OptionSet,
                None => SpecializeLowering::Direct,
            };
            Ok(SpecializeArgument {
                name,
                rust_type,
                function_constant_index,
                lowering,
            })
        })
        .collect::<Result<_>>()?;

    Ok(SpecializeEmission {
        arguments,
    })
}

impl SpecializeEmission {
    pub fn constructor_arguments(&self) -> Vec<TokenStream> {
        self.arguments
            .iter()
            .map(|argument| {
                let name = &argument.name;
                let rust_type = &argument.rust_type;
                quote! { #name: #rust_type }
            })
            .collect()
    }

    pub fn function_constants_initialization(&self) -> TokenStream {
        if self.arguments.is_empty() {
            return quote! {};
        }
        let value_assignments: Vec<TokenStream> = self
            .arguments
            .iter()
            .map(|argument| {
                let name = &argument.name;
                let index = argument.function_constant_index;
                match &argument.lowering {
                    SpecializeLowering::Direct => {
                        quote! { function_constants.set_value(&#name, #index); }
                    },
                    SpecializeLowering::Enum => {
                        quote! { function_constants.set_value(&(#name as u32), #index); }
                    },
                    SpecializeLowering::OptionSet => {
                        quote! { function_constants.set_value(&#name.bits(), #index); }
                    },
                }
            })
            .collect();
        quote! {
            let function_constants = MTLFunctionConstantValues::new();
            #(#value_assignments)*
        }
    }

    pub fn function_constants_argument(&self) -> TokenStream {
        if self.arguments.is_empty() {
            quote! { None }
        } else {
            quote! { Some(&function_constants) }
        }
    }

    pub fn cache_key(&self) -> TokenStream {
        if self.arguments.is_empty() {
            return quote! { &entry_name };
        }
        let argument_names: Vec<&Ident> = self.arguments.iter().map(|argument| &argument.name).collect();
        let format_string = format!("{{}}{}", "_{:?}".repeat(argument_names.len()));
        quote! { &format!(#format_string, &entry_name #(, #argument_names)*) }
    }
}
