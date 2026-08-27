use anyhow::{Context, Result};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Expr, Ident, Lifetime, Type};

use super::{
    super::{
        ast::{
            MetalArgument, MetalArgumentType, MetalBufferAccess, MetalConstantType, MetalGroupsType, MetalKernelInfo,
            shared_element_byte_size,
        },
        enum_path_rewrite::rewrite_for_rust,
    },
    host_expression_rewriter::HostExpressionRewriter,
};
use crate::common::enum_paths::EnumPaths;

pub enum ArgumentEmission {
    Buffer(BufferArgument),
    Constant(ConstantArgument),
    Shared(SharedArgument),
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

pub struct SharedArgument {
    condition: ArgumentCondition,
    threadgroup_index: usize,
    length_expression: TokenStream,
}

pub struct IndirectDispatchArgument;

pub fn parse(
    kernel: &MetalKernelInfo,
    enum_paths: &EnumPaths,
    host_expression_rewriter: &mut HostExpressionRewriter,
) -> Result<Vec<ArgumentEmission>> {
    let mut emissions = Vec::new();
    let mut next_buffer_index = 0usize;
    let mut next_threadgroup_index = 0usize;
    let mut indirect_dispatch_emitted = false;

    for argument in kernel.arguments.iter() {
        match &argument.argument_type {
            MetalArgumentType::Buffer(access) => {
                let buffer = parse_buffer_argument(argument, *access, next_buffer_index, enum_paths)?;
                emissions.push(ArgumentEmission::Buffer(buffer));
                next_buffer_index += 1;
            },
            MetalArgumentType::Constant((rust_type_text, constant_type)) => {
                let constant =
                    parse_constant_argument(argument, rust_type_text, constant_type, next_buffer_index, enum_paths)?;
                emissions.push(ArgumentEmission::Constant(constant));
                next_buffer_index += 1;
            },
            MetalArgumentType::Shared(dimensions) if argument.condition.is_some() => {
                let shared = parse_shared_argument(
                    argument,
                    dimensions.as_deref(),
                    next_threadgroup_index,
                    enum_paths,
                    host_expression_rewriter,
                )?;
                emissions.push(ArgumentEmission::Shared(shared));
                next_threadgroup_index += 1;
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

fn parse_shared_argument(
    argument: &MetalArgument,
    dimensions: Option<&str>,
    threadgroup_index: usize,
    enum_paths: &EnumPaths,
    host_expression_rewriter: &mut HostExpressionRewriter,
) -> Result<SharedArgument> {
    let condition = parse_argument_condition(argument, enum_paths)?
        .context("optional threadgroup argument must carry a condition")?;
    let dimensions = dimensions.context("optional threadgroup argument must have explicit array dimensions")?;
    let element_size = shared_element_byte_size(&argument.c_type)?;
    let dimension_factors = dimensions
        .split("][")
        .map(|dimension| host_expression_rewriter.rewrite(dimension))
        .collect::<Result<Vec<TokenStream>>>()?;
    let element_count = dimension_factors
        .into_iter()
        .reduce(|left, right| quote! { (#left) * (#right) })
        .expect("split always yields at least one dimension");
    let length_expression = quote! { (((#element_count) as usize) * #element_size).div_ceil(16) * 16 };
    Ok(SharedArgument {
        condition,
        threadgroup_index,
        length_expression,
    })
}

fn parse_buffer_argument(
    argument: &MetalArgument,
    access: MetalBufferAccess,
    buffer_index: usize,
    enum_paths: &EnumPaths,
) -> Result<BufferArgument> {
    let name = format_ident!("{}", argument.name.as_ref());
    let lifetime = Lifetime::new(&format!("'{}", argument.name.as_ref()), Span::call_site());
    let condition = parse_argument_condition(argument, enum_paths)?;
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
    enum_paths: &EnumPaths,
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
    let condition = parse_argument_condition(argument, enum_paths)?;
    Ok(ConstantArgument {
        name,
        buffer_index,
        shape,
        condition,
    })
}

fn parse_argument_condition(
    argument: &MetalArgument,
    enum_paths: &EnumPaths,
) -> Result<Option<ArgumentCondition>> {
    match argument.condition.as_deref() {
        Some(condition_text) => {
            let field_name = format_ident!("has_{}", argument.name.as_ref());
            let rust_expression = rewrite_for_rust(enum_paths, condition_text).with_context(|| {
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

    pub fn encode_argument_definition(&self) -> Option<TokenStream> {
        match self {
            ArgumentEmission::Buffer(buffer) => Some(emit_buffer_argument_definition(buffer)),
            ArgumentEmission::Constant(constant) => Some(emit_constant_argument_definition(constant)),
            ArgumentEmission::Shared(_) => None,
            ArgumentEmission::IndirectDispatch(_) => Some(quote! {
                __dsl_indirect_dispatch_buffer: impl crate::backends::common::BufferArg<
                    '__dsl_indirect_dispatch_buffer, crate::backends::metal::Metal
                >
            }),
        }
    }

    pub fn encode_lifetime(&self) -> Option<TokenStream> {
        match self {
            ArgumentEmission::Buffer(buffer) => {
                let lifetime = &buffer.lifetime;
                Some(quote! { #lifetime })
            },
            ArgumentEmission::IndirectDispatch(_) => Some(quote! { '__dsl_indirect_dispatch_buffer }),
            ArgumentEmission::Constant(_) | ArgumentEmission::Shared(_) => None,
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
            ArgumentEmission::Constant(constant) => {
                let constant_name = &constant.name;
                let buffer_name = format_ident!("__dsl_argument_buffer_{}", constant_name);
                let parts_name = format_ident!("__dsl_argument_buffer_parts_{}", constant_name);
                let (buffer_size, buffer_copy) = match &constant.shape {
                    ConstantShape::Scalar(ty) => (
                        quote! { std::mem::size_of::<#ty>() },
                        quote! { #buffer_name.copyin(unsafe { std::slice::from_raw_parts((&raw const #constant_name) as *const u8, std::mem::size_of::<#ty>()) }); },
                    ),
                    ConstantShape::UnsizedSlice(ty) => (
                        quote! { std::mem::size_of_val::<[#ty]>(#constant_name).max(std::mem::size_of::<#ty>()) },
                        quote! {
                            if !#constant_name.is_empty() {
                                #buffer_name.copyin(unsafe { std::slice::from_raw_parts(#constant_name.as_ptr() as *const u8, std::mem::size_of_val::<[#ty]>(#constant_name)) });
                            }
                        },
                    ),
                    ConstantShape::SizedArray {
                        element_type,
                        size,
                    } => (
                        quote! { std::mem::size_of::<[#element_type; #size]>() },
                        quote! { #buffer_name.copyin(unsafe { std::slice::from_raw_parts(#constant_name.as_ptr() as *const u8, std::mem::size_of::<[#element_type; #size]>()) }); },
                    ),
                };
                Some(if constant.condition.is_none() {
                    quote! {
                        let mut #buffer_name = encoder.allocate_constant(#buffer_size).unwrap();
                        #buffer_copy
                        let #parts_name = (&#buffer_name).into_parts();
                    }
                } else {
                    quote! {
                        let mut #buffer_name;
                        let #parts_name = if let Some(#constant_name) = #constant_name {
                            #buffer_name = encoder.allocate_constant(#buffer_size).unwrap();
                            #buffer_copy
                            Some((&#buffer_name).into_parts())
                        } else {
                            None
                        };
                    }
                })
            },
            ArgumentEmission::Shared(_) => None,
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
            ArgumentEmission::Constant(constant) => Some(emit_constant_access(constant)),
            ArgumentEmission::Shared(_) => None,
        }
    }

    pub fn encode_set(&self) -> Option<(TokenStream, bool)> {
        match self {
            ArgumentEmission::Buffer(buffer) => Some((emit_buffer_set(buffer), true)),
            ArgumentEmission::Constant(constant) => Some((emit_constant_set(constant), true)),
            ArgumentEmission::Shared(shared) => Some((emit_shared_set(shared), false)),
            ArgumentEmission::IndirectDispatch(_) => None,
        }
    }

    fn condition(&self) -> Option<&ArgumentCondition> {
        match self {
            ArgumentEmission::Buffer(buffer) => buffer.condition.as_ref(),
            ArgumentEmission::Constant(constant) => constant.condition.as_ref(),
            ArgumentEmission::Shared(shared) => Some(&shared.condition),
            ArgumentEmission::IndirectDispatch(_) => None,
        }
    }
}

fn emit_buffer_argument_definition(buffer: &BufferArgument) -> TokenStream {
    let name = &buffer.name;
    let lifetime = &buffer.lifetime;
    let trait_path = match buffer.access {
        MetalBufferAccess::Read => quote! { crate::backends::common::BufferArg },
        MetalBufferAccess::ReadWrite => quote! { crate::backends::common::BufferArgMut },
    };
    let buffer_argument_type = quote! { impl #trait_path<#lifetime, crate::backends::metal::Metal> };
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

fn emit_constant_access(constant: &ConstantArgument) -> TokenStream {
    let parts_name = format_ident!("__dsl_argument_buffer_parts_{}", constant.name);
    let access_expression = quote! {
        crate::backends::common::Access {
            range: #parts_name.0.gpu_address_subrange((#parts_name.1)..(#parts_name.1 + #parts_name.2)),
            flags: crate::backends::common::AccessFlags::compute_read(),
        }
    };
    if constant.condition.is_some() {
        quote! { #parts_name.as_ref().map(|#parts_name| #access_expression) }
    } else {
        quote! { Some(#access_expression) }
    }
}

fn emit_buffer_set(buffer: &BufferArgument) -> TokenStream {
    let name = &buffer.name;
    let buffer_index = buffer.buffer_index;
    let unconditional_set = quote! {
        self.argument_table.set_address_at_index(#name.0.downcast().gpu_address() + #name.1 as u64, #buffer_index);
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
    let parts_name = format_ident!("__dsl_argument_buffer_parts_{}", constant.name);
    let buffer_index = constant.buffer_index;
    let unconditional_set = quote! {
        self.argument_table.set_address_at_index(#parts_name.0.downcast().gpu_address() + #parts_name.1 as u64, #buffer_index);
    };
    match &constant.condition {
        Some(condition) => {
            let field_name = &condition.field_name;
            quote! {
                assert!(#parts_name.is_some() == self.#field_name);
                if let Some(#parts_name) = #parts_name {
                    #unconditional_set
                }
            }
        },
        None => unconditional_set,
    }
}

fn emit_shared_set(shared: &SharedArgument) -> TokenStream {
    let field_name = &shared.condition.field_name;
    let threadgroup_index = shared.threadgroup_index;
    let length_expression = &shared.length_expression;
    quote! {
        if self.#field_name {
            compute_encoder.set_threadgroup_memory_length_at_index(#length_expression, #threadgroup_index);
        }
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
