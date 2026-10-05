//! Rust bindings of AMDGPU kernels: `<Kernel>AmdgpuKernel` structs that resolve a code object symbol
//! in `new` and pack the kernel arguments (addresses, specializations, AXIS sizes) in `encode`.

mod host_expression_rewriter;
#[path = "../../metal/bindgen/specialize.rs"]
#[allow(dead_code)]
mod specialize;
#[path = "../../metal/bindgen/variants.rs"]
mod variants;

use std::iter::repeat_n;

use anyhow::{Context, Result, bail};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Expr, LitInt, Type};

use self::host_expression_rewriter::HostExpressionRewriter;
use super::{
    ast::{MetalArgument, MetalArgumentType, MetalBufferAccess, MetalConstantType, MetalGroupsType, MetalKernelInfo},
    enum_path_rewrite::{gpu_type_kind_for_c_type, rewrite_for_rust},
    wrapper::{KernargParameter, SpecializeKind, kernarg_parameters},
};
use crate::common::{
    enum_paths::{EnumPaths, GpuTypeKind},
    expr_rewrite::rewrite_paths_with,
    kernel::{Kernel, KernelArgumentType, KernelBufferAccess, KernelParameterType},
    mangling::dynamic_mangle,
};

fn backend() -> TokenStream {
    quote! { crate::backends::amdgpu::Amdgpu }
}

fn encoding_type() -> TokenStream {
    let backend = backend();
    quote! {
        <<#backend as crate::backends::common::Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding
    }
}

fn buffer_type(access: KernelBufferAccess) -> TokenStream {
    let backend = backend();
    match access {
        KernelBufferAccess::Read => quote! { impl crate::backends::common::BufferRef<Backend = #backend> },
        KernelBufferAccess::ReadWrite => quote! { impl crate::backends::common::BufferMut<Backend = #backend> },
    }
}

struct ArgumentCondition {
    field_name: syn::Ident,
    rust_expression: TokenStream,
}

fn argument_condition(
    argument: &MetalArgument,
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
) -> Result<Option<ArgumentCondition>> {
    let Some(condition_text) = argument.condition.as_deref() else {
        return Ok(None);
    };
    let field_name = format_ident!("has_{}", argument.name.as_ref());
    let rust_expression = rewrite_for_rust(enum_paths, condition_text)
        .with_context(|| format!("OPTIONAL condition `{condition_text}` cannot be parsed as a rust expression"))?;
    // conditions see constructor arguments; optional specializations are `Option`s there
    let mut expression: Expr = syn::parse2(rust_expression)?;
    rewrite_paths_with(&mut expression, |path| {
        let ident = path.get_ident()?;
        kernel
            .arguments
            .iter()
            .any(|a| {
                ident == a.name.as_ref()
                    && matches!(a.argument_type, MetalArgumentType::Specialize(_))
                    && a.condition.is_some()
            })
            .then(|| syn::parse_quote! { #ident.unwrap() })
    });
    Ok(Some(ArgumentCondition {
        field_name,
        rust_expression: quote! { #expression },
    }))
}

fn is_positive_integer_literal(expression: &TokenStream) -> bool {
    syn::parse2::<LitInt>(expression.clone())
        .ok()
        .and_then(|literal| literal.base10_parse::<u32>().ok())
        .is_some_and(|value| value != 0)
}

pub fn bindgen(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
    code_objects_const: &syn::Ident,
) -> Result<(TokenStream, Option<TokenStream>)> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{}Kernel", kernel_name);
    let struct_name = format_ident!("{}AmdgpuKernel", kernel_name);
    let backend = backend();

    let variant_binds = variants::parse(kernel)?;
    let specialize_emission = specialize::parse(kernel, None, kernel_name, enum_paths)?;
    let mut host =
        HostExpressionRewriter::new(&variant_binds, enum_paths, specialize_emission.argument_names(), kernel_name);
    let kernarg = kernarg_parameters(kernel, enum_paths)?;

    // --- dispatch geometry (host expressions); AXIS grids are rounded up to whole workgroups
    let pad = |mut values: Vec<TokenStream>| -> Vec<TokenStream> {
        let missing = 3 - values.len();
        values.extend(repeat_n(quote! { 1u32 }, missing));
        values
    };
    let mut axis_totals = Vec::new();
    let mut size_expressions = Vec::new();
    let (groups, threads) = if kernel.has_axis() {
        let mut groups = Vec::new();
        let mut threads = Vec::new();
        for argument in kernel.arguments.iter() {
            if let MetalArgumentType::Axis(total_text, per_group_text) = &argument.argument_type {
                let total = host.rewrite(total_text)?;
                let per_group = host.rewrite(per_group_text)?;
                groups.push(quote! { ((#total) as u32).div_ceil((#per_group) as u32) });
                threads.push(quote! { (#per_group) as u32 });
                size_expressions.push(total.clone());
                size_expressions.push(per_group);
                axis_totals.push(total);
            }
        }
        (pad(groups), pad(threads))
    } else {
        let mut groups = Vec::new();
        let mut threads = Vec::new();
        for argument in kernel.arguments.iter() {
            match &argument.argument_type {
                MetalArgumentType::Groups(MetalGroupsType::Direct(text)) => {
                    let expression = host.rewrite(text)?;
                    groups.push(quote! { (#expression) as u32 });
                    size_expressions.push(expression);
                },
                MetalArgumentType::Groups(MetalGroupsType::Indirect) => bail!("indirect dispatch is not supported"),
                MetalArgumentType::Threads(text) => {
                    let expression = host.rewrite(text)?;
                    threads.push(quote! { (#expression) as u32 });
                    size_expressions.push(expression);
                },
                _ => {},
            }
        }
        (pad(groups), pad(threads))
    };

    let empty_dispatch_guard = size_expressions
        .iter()
        .filter(|e| !is_positive_integer_literal(e))
        .map(|e| quote! { (#e) == 0 })
        .reduce(|left, right| quote! { #left || #right })
        .map(|guard| quote! { if #guard { return; } })
        .unwrap_or_default();

    let referenced = host.finish();

    // --- arguments
    let mut encode_arguments = Vec::new();
    let mut deconstructs = Vec::new();
    let mut pushes = Vec::new();
    let mut condition_fields = Vec::new();
    let mut condition_initializers = Vec::new();

    for parameter in kernarg.iter() {
        match parameter {
            KernargParameter::Address(argument) => {
                let name = format_ident!("{}", argument.name.as_ref());
                let condition = argument_condition(argument, kernel, enum_paths)?;
                if let Some(condition) = &condition {
                    let field = &condition.field_name;
                    let expression = &condition.rust_expression;
                    condition_fields.push(quote! { #field: bool });
                    condition_initializers.push(quote! { #field: #expression });
                }
                match &argument.argument_type {
                    MetalArgumentType::Buffer(access) => {
                        let ty = buffer_type(match access {
                            MetalBufferAccess::Read => KernelBufferAccess::Read,
                            MetalBufferAccess::ReadWrite => KernelBufferAccess::ReadWrite,
                        });
                        match &condition {
                            Some(condition) => {
                                let field = &condition.field_name;
                                encode_arguments.push(quote! { #name: Option<#ty> });
                                deconstructs.push(quote! { let #name = #name.map(|#name| #name.parts()); });
                                pushes.push(quote! {
                                    assert!(#name.is_some() == self.#field, concat!("unexpected presence of ", stringify!(#name)));
                                    __dsl_kernarg.push_address(#name.map_or(0, |#name| #name.0.device_address() + #name.1.start as u64));
                                });
                            },
                            None => {
                                encode_arguments.push(quote! { #name: #ty });
                                deconstructs.push(quote! { let #name = #name.parts(); });
                                pushes.push(quote! {
                                    __dsl_kernarg.push_address(#name.0.device_address() + #name.1.start as u64);
                                });
                            },
                        }
                    },
                    MetalArgumentType::Constant((rust_type_text, constant_type)) => {
                        let element_type: Type = syn::parse_str(rust_type_text)
                            .with_context(|| format!("constant rust type `{rust_type_text}` cannot be parsed"))?;
                        let (base_type, byte_size, byte_view) = match constant_type {
                            MetalConstantType::Scalar => (
                                quote! { #element_type },
                                quote! { std::mem::size_of::<#element_type>() },
                                quote! { unsafe { std::slice::from_raw_parts((&raw const #name) as *const u8, std::mem::size_of::<#element_type>()) } },
                            ),
                            MetalConstantType::Array(None) => (
                                quote! { &[#element_type] },
                                quote! { std::mem::size_of_val::<[#element_type]>(#name).max(std::mem::size_of::<#element_type>()) },
                                quote! { unsafe { std::slice::from_raw_parts(#name.as_ptr() as *const u8, std::mem::size_of_val::<[#element_type]>(#name)) } },
                            ),
                            MetalConstantType::Array(Some(size_text)) => {
                                let size: Expr = syn::parse_str(size_text)
                                    .with_context(|| format!("constant array size `{size_text}` cannot be parsed"))?;
                                (
                                    quote! { &[#element_type; #size] },
                                    quote! { std::mem::size_of::<[#element_type; #size]>() },
                                    quote! { unsafe { std::slice::from_raw_parts(#name.as_ptr() as *const u8, std::mem::size_of::<[#element_type; #size]>()) } },
                                )
                            },
                        };
                        let buffer_name = format_ident!("__dsl_argument_buffer_{}", argument.name.as_ref());
                        match &condition {
                            Some(condition) => {
                                let field = &condition.field_name;
                                encode_arguments.push(quote! { #name: Option<#base_type> });
                                // An absent constant still binds a zeroed allocation: the kernel takes a
                                // reference, which the compiler may load from speculatively.
                                deconstructs.push(quote! {
                                    assert!(#name.is_some() == self.#field, concat!("unexpected presence of ", stringify!(#name)));
                                    let mut #buffer_name;
                                    if let Some(#name) = #name {
                                        #buffer_name = command_buffer.allocate_constant(#byte_size).unwrap();
                                        let __dsl_bytes: &[u8] = #byte_view;
                                        if !__dsl_bytes.is_empty() { #buffer_name.copyin(__dsl_bytes); }
                                    } else {
                                        #buffer_name = command_buffer.allocate_constant(std::mem::size_of::<#element_type>().max(16)).unwrap();
                                        #buffer_name.as_slice_mut::<u8>().fill(0);
                                    }
                                });
                            },
                            None => {
                                encode_arguments.push(quote! { #name: #base_type });
                                deconstructs.push(quote! {
                                    let mut #buffer_name = command_buffer.allocate_constant(#byte_size).unwrap();
                                    let __dsl_bytes: &[u8] = #byte_view;
                                    if !__dsl_bytes.is_empty() { #buffer_name.copyin(__dsl_bytes); }
                                });
                            },
                        }
                        pushes.push(quote! {
                            {
                                let __dsl_parts = (&#buffer_name).parts();
                                __dsl_kernarg.push_address(__dsl_parts.0.device_address() + __dsl_parts.1.start as u64);
                            }
                        });
                    },
                    _ => unreachable!(),
                }
            },
            KernargParameter::Specialize(argument, kind) => {
                let field = format_ident!("specialize_{}", argument.name.as_ref());
                let lowered = |value: TokenStream| match kind {
                    SpecializeKind::Bool => quote! { (#value) as u32 },
                    SpecializeKind::U32 => quote! { #value },
                    SpecializeKind::I32 => quote! { (#value) as u32 },
                    SpecializeKind::F32 => quote! { (#value).to_bits() },
                    SpecializeKind::Gpu => match gpu_type_kind_for_c_type(enum_paths, &argument.c_type) {
                        Some(GpuTypeKind::OptionSet) => quote! { (#value).bits() },
                        _ => quote! { (#value) as u32 },
                    },
                };
                if argument.condition.is_some() {
                    let value = lowered(quote! { value });
                    pushes.push(quote! { __dsl_kernarg.push_u32(self.#field.map_or(0, |value| #value)); });
                } else {
                    let value = lowered(quote! { self.#field });
                    pushes.push(quote! { __dsl_kernarg.push_u32(#value); });
                }
            },
            KernargParameter::AxisTotal(dimension) => {
                let total = &axis_totals[*dimension];
                pushes.push(quote! { __dsl_kernarg.push_u32((#total) as u32); });
            },
        }
    }

    // all specializations are kept: they are kernel arguments
    let specialize_fields: Vec<TokenStream> = kernel
        .arguments
        .iter()
        .filter_map(|a| match &a.argument_type {
            MetalArgumentType::Specialize(rust_type_text) => Some((a, rust_type_text)),
            _ => None,
        })
        .map(|(a, rust_type_text)| -> Result<(TokenStream, TokenStream)> {
            let name = format_ident!("{}", a.name.as_ref());
            let field = format_ident!("specialize_{}", a.name.as_ref());
            let ty: Type = syn::parse_str(rust_type_text)?;
            Ok((quote! { #field: #ty }, quote! { #field: #name }))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(|(field, init)| {
            condition_initializers.push(init);
            field
        })
        .collect();

    let variant_fields: Vec<TokenStream> = variant_binds.iter().filter_map(|v| v.struct_field(&referenced)).collect();
    let variant_initializers: Vec<TokenStream> =
        variant_binds.iter().filter_map(|v| v.struct_initializer(&referenced)).collect();
    let variant_constructor_arguments: Vec<TokenStream> =
        variant_binds.iter().map(|v| v.constructor_argument()).collect();
    let entry_name = dynamic_mangle(kernel_name, variant_binds.iter().map(|v| v.kernel_format()));
    let specialize_constructor_arguments = specialize_emission.constructor_arguments();
    let presence_checks = &specialize_emission.presence_checks;

    let (implementation_for, associate_backend, method_visibility, associated_type) = if kernel.public {
        (
            quote! { crate::backends::common::kernel::#trait_name for },
            quote! { type Backend = #backend; },
            quote! {},
            Some(quote! { type #trait_name = #struct_name; }),
        )
    } else {
        (quote! {}, quote! {}, quote! { pub(crate) }, None)
    };

    let encoding = encoding_type();
    // Kernels used only by composites that are not ported yet stay unused.
    let tokens = quote! {
        #[allow(dead_code)]
        pub struct #struct_name {
            function: crate::backends::amdgpu::kernel::AmdgpuFunction,
            #(#condition_fields,)*
            #(#variant_fields,)*
            #(#specialize_fields,)*
        }

        #[allow(clippy::style, clippy::complexity, clippy::perf, unused_variables, unused_mut, dead_code)]
        impl #implementation_for #struct_name {
            #associate_backend

            #method_visibility fn new(
                context: &AmdgpuContext
                #(, #variant_constructor_arguments)*
                #(, #specialize_constructor_arguments)*
            ) -> Result<Self, AmdgpuError> {
                #(#presence_checks)*
                let entry_name = #entry_name;
                let function = context.function(&#code_objects_const, &entry_name)?;
                Ok(Self {
                    function
                    #(, #condition_initializers)*
                    #(, #variant_initializers)*
                })
            }

            #method_visibility fn encode(
                &self,
                #(#encode_arguments,)*
                command_buffer: &mut #encoding
            ) {
                #empty_dispatch_guard
                #(#deconstructs)*
                let mut __dsl_kernarg = crate::backends::amdgpu::kernel::Kernarg::new();
                #(#pushes)*
                command_buffer.dispatch(&self.function, [#(#groups),*], [#(#threads),*], __dsl_kernarg, #kernel_name);
            }
        }
    };

    Ok((tokens, associated_type))
}

/// Bindings for a public kernel that has no AMDGPU implementation yet: the signature comes from
/// another backend, `new` reports the kernel as unavailable.
pub fn bindgen_stub(kernel: &Kernel) -> Result<TokenStream> {
    let trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
    let struct_name = format_ident!("{}AmdgpuKernel", kernel.name.as_ref());
    let kernel_name = kernel.name.as_ref();
    let backend = backend();
    let encoding = encoding_type();

    let parameters = kernel
        .parameters
        .iter()
        .map(|parameter| -> Result<TokenStream> {
            let name = format_ident!("{}", parameter.name.as_ref());
            Ok(match &parameter.ty {
                KernelParameterType::Type => quote! { #name: crate::data_type::DataType },
                KernelParameterType::Value(ty) => {
                    let ty: Type = syn::parse_str(ty)?;
                    quote! { #name: #ty }
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let arguments = kernel
        .arguments
        .iter()
        .map(|argument| -> Result<TokenStream> {
            let name = format_ident!("{}", argument.name.as_ref());
            let ty = match &argument.ty {
                KernelArgumentType::Buffer(access) => buffer_type(access.clone()),
                KernelArgumentType::Constant(ty) => {
                    let ty: Type = syn::parse_str(ty)?;
                    quote! { #ty }
                },
            };
            Ok(if argument.conditional {
                quote! { #name: Option<#ty> }
            } else {
                quote! { #name: #ty }
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(quote! {
        pub struct #struct_name;

        #[allow(clippy::style, clippy::complexity, clippy::perf, unused_variables, non_snake_case)]
        impl crate::backends::common::kernel::#trait_name for #struct_name {
            type Backend = #backend;

            fn new(context: &AmdgpuContext #(, #parameters)*) -> Result<Self, AmdgpuError> {
                Err(AmdgpuError::KernelUnavailable(#kernel_name.into()))
            }

            fn encode(&self, #(#arguments,)* command_buffer: &mut #encoding) {
                unreachable!("{} is not available on AMDGPU", #kernel_name)
            }
        }
    })
}

pub fn bindgen_global(
    files: &[(impl AsRef<std::path::Path>, Vec<syn::Ident>)],
    public_kernels: &[&Kernel],
) -> Result<TokenStream> {
    let includes = files.iter().map(|(path, _)| {
        let path = path.as_ref().to_str().expect("bindings path is not utf-8");
        quote! { include!(#path); }
    });

    let associated_types = public_kernels.iter().map(|kernel| {
        let trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
        let struct_name = format_ident!("{}AmdgpuKernel", kernel.name.as_ref());
        quote! { type #trait_name = #struct_name; }
    });

    Ok(quote! {
        #[allow(unused_imports)]
        use crate::backends::{
            amdgpu::{buffer::AmdgpuBufferExt, context::AmdgpuContext, error::AmdgpuError},
            common::{BufferMut, BufferRef, CommandBufferEncoding},
        };

        #(#includes)*

        macro_rules! autogen_kernels {
            () => {
                #(#associated_types)*
            }
        }
    })
}
