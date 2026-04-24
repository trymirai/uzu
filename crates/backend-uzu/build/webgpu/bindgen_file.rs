use std::{collections::HashMap, fmt::format, path::Path};

use anyhow::Context;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use shader_slang::reflection::Shader;
use syn::{Ident, Lifetime, Type};

use crate::{
    debug_log,
    slang::{SlangArgumentType, SlangBufferAccess, SlangKernel, SlangParameterType, slang2rust},
};

pub fn bindgen_file(
    shader: &Shader,
    kernels: &[SlangKernel],
    gpu_type_map: &HashMap<String, String>,
    object_file: &Path,
) -> anyhow::Result<TokenStream> {
    let object_file = object_file.to_str().unwrap();
    let object_constant = format_ident!("WGSL_{}", blake3::hash(object_file.as_bytes()).to_hex().to_uppercase());

    let generated_kernels = kernels
        .iter()
        .map(|kernel| bindgen_kernel(shader, kernel, gpu_type_map, &object_constant))
        .collect::<anyhow::Result<Vec<TokenStream>>>()?;

    Ok(quote! {
        const #object_constant: &str = include_str!(#object_file);

        #(#generated_kernels)*
    })
}

fn bindgen_kernel(
    shader: &Shader,
    kernel: &SlangKernel,
    gpu_type_map: &HashMap<String, String>,
    object_constant: &Ident,
) -> anyhow::Result<TokenStream> {
    let trait_name = format_ident!("{}Kernel", kernel.name.as_str());
    let struct_name = format_ident!("{}WebGPUKernel", kernel.name.as_str());

    let new_arguments_definitions = kernel
        .parameters
        .iter()
        .map(|parameter| match &parameter.ty {
            SlangParameterType::Type {
                variants: _,
            } => {
                let name = format_ident!("{}", parameter.name.as_str());
                Ok(Some(quote! { #[allow(non_snake_case)] #name: crate::DataType }))
            },
            SlangParameterType::Value {
                value_type,
                variants: _,
            } => {
                let name = format_ident!("{}", parameter.name.as_str());

                let ty: Type = slang2rust(value_type, gpu_type_map)
                    .with_context(|| format!("cannot convert to rust the type of parameter {}", parameter.name))
                    .and_then(|ty| {
                        syn::parse_str(&ty)
                            .with_context(|| format!("cannot parse the type of parameter {}", parameter.name))
                    })?;

                Ok(Some(quote! { #name: #ty }))
            },
            SlangParameterType::GroupShared {
                ..
            } => Ok(None),
        })
        .chain(kernel.arguments.iter().map(|argument| {
            if let SlangArgumentType::Specialize = argument.argument_type {
                let name = format_ident!("{}", argument.name.as_str());
                let ty: Type = argument.rust_type(gpu_type_map).and_then(|ty| {
                    syn::parse_str(&ty).with_context(|| format!("cannot parse the type of argument {}", argument.name))
                })?;
                Ok(Some(quote! { #name: #ty }))
            } else {
                Ok(None)
            }
        }))
        .filter_map(|x| x.transpose())
        .collect::<anyhow::Result<Vec<TokenStream>>>()?;

    let mut indirect_used = false;

    let (encode_generics, encode_arguments_definitions) = kernel
        .arguments
        .iter()
        .map(|argument| {
            match &argument.argument_type {
                SlangArgumentType::Buffer {
                    access_type: _,
                    condition,
                }
                | SlangArgumentType::Constant {
                    condition,
                } => {
                    let argument_name = format_ident!("{}", argument.name);

                    let (generic, mut argument_type) = match &argument.argument_type {
                        SlangArgumentType::Buffer {
                            access_type:
                                access_type @ (SlangBufferAccess::Read {
                                    is_constant: false,
                                }
                                | SlangBufferAccess::ReadWrite),
                            condition: _,
                        } => {
                            let buffer_lifetime = Lifetime::new(&format!("'{}", argument.name.as_str()), Span::call_site());

                            let buffer_arg_type = match access_type {
                                SlangBufferAccess::Read { .. } => quote! { BufferArg },
                                SlangBufferAccess::ReadWrite => quote! { BufferArgMut },
                            };

                            (
                                Some(quote! { #buffer_lifetime }),
                                quote! { impl crate::backends::common::kernel::#buffer_arg_type<#buffer_lifetime, crate::backends::webgpu::buffer::WebGPUBuffer> },
                            )
                        },
                        SlangArgumentType::Constant {
                            condition: _,
                        }
                        | SlangArgumentType::Buffer {
                            access_type:
                                SlangBufferAccess::Read {
                                    is_constant: true,
                                },
                            condition: _,
                        } => {
                            let ty: Type = argument.rust_type(gpu_type_map).and_then(|ty| {
                                syn::parse_str(&ty).with_context(|| format!("cannot parse the type of argument {}", argument.name))
                            })?;

                            (None, quote! { #ty })
                        },
                        _ => unreachable!(),
                    };

                    if let Some(_condition) = condition {
                        argument_type = quote! { Option<#argument_type> };
                    }

                    Ok(Some((generic, quote! { #argument_name: #argument_type })))
                },
                SlangArgumentType::Groups { groups } if !indirect_used && groups == "INDIRECT" => {
                    indirect_used = true;

                    Ok(Some((
                        Some(quote! { '__dsl_indirect_dispatch_buffer }),
                        quote! { __dsl_indirect_dispatch_buffer: impl crate::backends::common::kernel::BufferArg<'__dsl_indirect_dispatch_buffer, crate::backends::webgpu::buffer::WebGPUBuffer> },
                    )))
                },
                _ => Ok(None)
            }
        })
        .filter_map(|x| x.transpose())
        .collect::<anyhow::Result<(Vec<Option<TokenStream>>, Vec<TokenStream>)>>()?;

    let encode_generics: Vec<TokenStream> = encode_generics.into_iter().flatten().collect();

    let (public_trait_impl, public_type_backend, private_pub) = if kernel.public {
        (
            quote! { crate::backends::common::kernel::#trait_name for },
            quote! { type Backend = crate::backends::webgpu::WebGPU; },
            quote! {},
        )
    } else {
        (quote! {}, quote! {}, quote! { pub })
    };

    Ok(quote! {
        pub struct #struct_name {
            pipeline: wgpu::ComputePipeline,
        }

        impl #public_trait_impl #struct_name {
            #public_type_backend

            #private_pub fn new(
                context: &crate::backends::webgpu::context::WebGPUContext,
                #(#new_arguments_definitions ,)*
            ) -> Result<Self, crate::backends::webgpu::error::WebGPUError> {
                let shader_module = context.get_shader_module(#object_constant);
                // let pipeline = context.get_pipeline();
                todo!()
            }
            #private_pub fn encode<#(#encode_generics ,)* 'encoder>(
                &self,
                #(#encode_arguments_definitions ,)*
                encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::webgpu::WebGPU>,
            ) {
                todo!()
            }
        }
    })
}
