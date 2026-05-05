use anyhow::{Context, Result};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Expr, Ident, Lifetime, Type};

use super::super::{
    ast::{MetalArgument, MetalArgumentType, MetalBufferAccess, MetalConstantType, MetalGroupsType, MetalKernelInfo},
    enum_path_rewrite::EnumPathRewriter,
};

pub enum ArgumentEmission {
    Buffer(BufferArgument),
    Constant(ConstantArgument),
    IndirectDispatch(IndirectDispatchArgument),
}

pub struct BufferArgument {
    name: Ident,
    buffer_index: usize,
    access: MetalBufferAccess,
    lifetime: Lifetime,
    condition: Option<ArgumentCondition>,
}

pub struct ConstantArgument {
    name: Ident,
    buffer_index: usize,
    shape: ConstantShape,
    condition: Option<ArgumentCondition>,
}

enum ConstantShape {
    Scalar(Type),
    UnsizedSlice(Type),
    SizedArray {
        element_type: Type,
        size: Expr,
    },
}

struct ArgumentCondition {
    field_name: Ident,
    rust_expression: TokenStream,
}

pub struct IndirectDispatchArgument;

pub fn parse(
    kernel: &MetalKernelInfo,
    enum_path_rewriter: &EnumPathRewriter,
) -> Result<Vec<ArgumentEmission>> {
    let mut emissions = Vec::new();
    let mut next_buffer_index = 0usize;
    let mut indirect_dispatch_emitted = false;

    for argument in kernel.arguments.iter() {
        let argument_type = argument.argument_type().unwrap();
        match argument_type {
            MetalArgumentType::Buffer(access) => {
                let buffer = parse_buffer_argument(argument, access, next_buffer_index, enum_path_rewriter)?;
                emissions.push(ArgumentEmission::Buffer(buffer));
                next_buffer_index += 1;
            },
            MetalArgumentType::Constant((rust_type_text, constant_type)) => {
                let constant = parse_constant_argument(
                    argument,
                    &rust_type_text,
                    &constant_type,
                    next_buffer_index,
                    enum_path_rewriter,
                )?;
                emissions.push(ArgumentEmission::Constant(constant));
                next_buffer_index += 1;
            },
            MetalArgumentType::Groups(MetalGroupsType::Indirect) if !indirect_dispatch_emitted => {
                emissions.push(ArgumentEmission::IndirectDispatch(IndirectDispatchArgument));
                indirect_dispatch_emitted = true;
            },
            _ => {},
        }
    }

    Ok(emissions)
}

fn parse_buffer_argument(
    argument: &MetalArgument,
    access: MetalBufferAccess,
    buffer_index: usize,
    enum_path_rewriter: &EnumPathRewriter,
) -> Result<BufferArgument> {
    let name = format_ident!("{}", argument.name.as_ref());
    let lifetime = Lifetime::new(&format!("'{}", argument.name.as_ref()), Span::call_site());
    let condition = parse_argument_condition(argument, enum_path_rewriter)?;
    Ok(BufferArgument {
        name,
        buffer_index,
        access,
        lifetime,
        condition,
    })
}

fn parse_constant_argument(
    argument: &MetalArgument,
    rust_type_text: &str,
    constant_type: &MetalConstantType,
    buffer_index: usize,
    enum_path_rewriter: &EnumPathRewriter,
) -> Result<ConstantArgument> {
    let name = format_ident!("{}", argument.name.as_ref());
    let element_type: Type = syn::parse_str(rust_type_text)
        .with_context(|| format!("constant rust type `{}` cannot be parsed", rust_type_text))?;
    let shape = match constant_type {
        MetalConstantType::Scalar => ConstantShape::Scalar(element_type),
        MetalConstantType::Array(None) => ConstantShape::UnsizedSlice(element_type),
        MetalConstantType::Array(Some(size_text)) => {
            let size: Expr = syn::parse_str(size_text)
                .with_context(|| format!("constant array size `{}` cannot be parsed", size_text))?;
            ConstantShape::SizedArray {
                element_type,
                size,
            }
        },
    };
    let condition = parse_argument_condition(argument, enum_path_rewriter)?;
    Ok(ConstantArgument {
        name,
        buffer_index,
        shape,
        condition,
    })
}

fn parse_argument_condition(
    argument: &MetalArgument,
    enum_path_rewriter: &EnumPathRewriter,
) -> Result<Option<ArgumentCondition>> {
    match argument.argument_condition().unwrap() {
        Some(condition_text) => {
            let field_name = format_ident!("has_{}", argument.name.as_ref());
            let rust_expression = enum_path_rewriter.rewrite_for_rust(condition_text).with_context(|| {
                format!("OPTIONAL condition `{}` cannot be parsed as a rust expression", condition_text)
            })?;
            Ok(Some(ArgumentCondition {
                field_name,
                rust_expression,
            }))
        },
        None => Ok(None),
    }
}

impl ArgumentEmission {
    pub fn struct_field(&self) -> Option<TokenStream> {
        let condition = self.condition()?;
        let field_name = &condition.field_name;
        Some(quote! { #field_name: bool })
    }

    pub fn struct_initializer(&self) -> Option<TokenStream> {
        let condition = self.condition()?;
        let field_name = &condition.field_name;
        let rust_expression = &condition.rust_expression;
        Some(quote! { #field_name: #rust_expression })
    }

    pub fn encode_argument_definition(&self) -> TokenStream {
        match self {
            ArgumentEmission::Buffer(buffer) => emit_buffer_argument_definition(buffer),
            ArgumentEmission::Constant(constant) => emit_constant_argument_definition(constant),
            ArgumentEmission::IndirectDispatch(_) => quote! {
                __dsl_indirect_dispatch_buffer: impl crate::backends::common::kernel::BufferArg<
                    '__dsl_indirect_dispatch_buffer,
                    Retained<ProtocolObject<dyn MTLBuffer>>,
                >
            },
        }
    }

    pub fn encode_lifetime(&self) -> Option<TokenStream> {
        match self {
            ArgumentEmission::Buffer(buffer) => {
                let lifetime = &buffer.lifetime;
                Some(quote! { #lifetime })
            },
            ArgumentEmission::IndirectDispatch(_) => Some(quote! { '__dsl_indirect_dispatch_buffer }),
            ArgumentEmission::Constant(_) => None,
        }
    }

    pub fn encode_deconstruct(&self) -> Option<TokenStream> {
        match self {
            ArgumentEmission::Buffer(buffer) => {
                let name = &buffer.name;
                Some(if buffer.condition.is_some() {
                    quote! { let #name = #name.map(|#name| #name.into_parts()); }
                } else {
                    quote! { let #name = #name.into_parts(); }
                })
            },
            ArgumentEmission::IndirectDispatch(_) => Some(quote! {
                let __dsl_indirect_dispatch_buffer = __dsl_indirect_dispatch_buffer.into_parts();
            }),
            ArgumentEmission::Constant(_) => None,
        }
    }

    pub fn encode_access(&self) -> Option<TokenStream> {
        match self {
            ArgumentEmission::Buffer(buffer) => Some(emit_buffer_access(buffer)),
            ArgumentEmission::IndirectDispatch(_) => Some(quote! {
                Some(crate::backends::common::Access {
                    range: __dsl_indirect_dispatch_buffer.0.gpu_address_subrange(
                        (__dsl_indirect_dispatch_buffer.1)..(__dsl_indirect_dispatch_buffer.1 + 12),
                    ),
                    flags: crate::backends::common::AccessFlags::compute_read(),
                })
            }),
            ArgumentEmission::Constant(_) => None,
        }
    }

    pub fn encode_set(&self) -> TokenStream {
        match self {
            ArgumentEmission::Buffer(buffer) => emit_buffer_set(buffer),
            ArgumentEmission::Constant(constant) => emit_constant_set(constant),
            ArgumentEmission::IndirectDispatch(_) => quote! {},
        }
    }

    fn condition(&self) -> Option<&ArgumentCondition> {
        match self {
            ArgumentEmission::Buffer(buffer) => buffer.condition.as_ref(),
            ArgumentEmission::Constant(constant) => constant.condition.as_ref(),
            ArgumentEmission::IndirectDispatch(_) => None,
        }
    }
}

fn emit_buffer_argument_definition(buffer: &BufferArgument) -> TokenStream {
    let name = &buffer.name;
    let lifetime = &buffer.lifetime;
    let trait_path = match buffer.access {
        MetalBufferAccess::Read => quote! { crate::backends::common::kernel::BufferArg },
        MetalBufferAccess::ReadWrite => quote! { crate::backends::common::kernel::BufferArgMut },
    };
    let buffer_argument_type = quote! { impl #trait_path<#lifetime, Retained<ProtocolObject<dyn MTLBuffer>>> };
    if buffer.condition.is_some() {
        quote! { #name: Option<#buffer_argument_type> }
    } else {
        quote! { #name: #buffer_argument_type }
    }
}

fn emit_constant_argument_definition(constant: &ConstantArgument) -> TokenStream {
    let name = &constant.name;
    let base_type = match &constant.shape {
        ConstantShape::Scalar(element_type) => quote! { #element_type },
        ConstantShape::UnsizedSlice(element_type) => quote! { &[#element_type] },
        ConstantShape::SizedArray {
            element_type,
            size,
        } => quote! { &[#element_type; #size] },
    };
    if constant.condition.is_some() {
        quote! { #name: Option<#base_type> }
    } else {
        quote! { #name: #base_type }
    }
}

fn emit_buffer_access(buffer: &BufferArgument) -> TokenStream {
    let name = &buffer.name;
    let compute_write = matches!(buffer.access, MetalBufferAccess::ReadWrite);
    let access_expression = quote! {
        crate::backends::common::Access {
            range: #name.0.gpu_address_subrange((#name.1)..(#name.1 + #name.2)),
            flags: crate::backends::common::AccessFlags {
                compute_read: true,
                compute_write: #compute_write,
                copy_read: false,
                copy_write: false,
            },
        }
    };
    if buffer.condition.is_some() {
        quote! { #name.as_ref().map(|#name| #access_expression) }
    } else {
        quote! { Some(#access_expression) }
    }
}

fn emit_buffer_set(buffer: &BufferArgument) -> TokenStream {
    let name = &buffer.name;
    let buffer_index = buffer.buffer_index;
    let unconditional_set = quote! {
        compute_encoder.set_buffer(Some(#name.0), #name.1, #buffer_index);
    };
    match &buffer.condition {
        Some(condition) => {
            let field_name = &condition.field_name;
            quote! {
                assert!(#name.is_some() == self.#field_name);
                if let Some(#name) = #name {
                    #unconditional_set
                }
            }
        },
        None => unconditional_set,
    }
}

fn emit_constant_set(constant: &ConstantArgument) -> TokenStream {
    let name = &constant.name;
    let buffer_index = constant.buffer_index;
    let unconditional_set = match &constant.shape {
        ConstantShape::Scalar(_) => quote! { compute_encoder.set_value(&#name, #buffer_index); },
        ConstantShape::UnsizedSlice(_)
        | ConstantShape::SizedArray {
            ..
        } => {
            quote! { compute_encoder.set_slice(#name, #buffer_index); }
        },
    };
    match &constant.condition {
        Some(condition) => {
            let field_name = &condition.field_name;
            quote! {
                assert!(#name.is_some() == self.#field_name);
                if let Some(#name) = #name {
                    #unconditional_set
                }
            }
        },
        None => unconditional_set,
    }
}

pub fn encode_accesses_call(arguments: &[ArgumentEmission]) -> TokenStream {
    let access_expressions: Vec<TokenStream> =
        arguments.iter().filter_map(|argument| argument.encode_access()).collect();
    if access_expressions.is_empty() {
        quote! {}
    } else {
        quote! {
            encoder.access(&[#(#access_expressions),*].into_iter().flatten().collect::<Vec<_>>());
        }
    }
}
