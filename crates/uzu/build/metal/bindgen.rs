use std::iter::repeat_n;

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Lifetime, LitInt, Type};

use super::{
    ast::{MetalArgumentType, MetalConstantType, MetalKernelInfo},
    wrapper::SpecializeBaseIndices,
};
use crate::{
    common::mangling::dynamic_mangle,
    metal::ast::{MetalGroupsType, MetalTemplateParameterType},
};

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
) -> anyhow::Result<(TokenStream, TokenStream)> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{kernel_name}Kernel");
    let struct_name = format_ident!("{kernel_name}MetalKernel");

    let parse_expr = |expr: &str| -> anyhow::Result<TokenStream> {
        syn::parse_str(expr.as_ref())
            .with_context(|| format!("cannot parse rust expression `{}` in kernel `{}`", expr, kernel_name))
    };

    let (variants_extra_arguments, variants_kernel_format) = if let Some(variants) = &kernel.variants {
        variants
            .iter()
            .map(|variant| {
                let name = Ident::new(variant.name.as_ref(), Span::call_site());

                match &variant.ty {
                    MetalTemplateParameterType::Type => {
                        Ok((quote! { #[allow(non_snake_case)] #name: crate::DataType }, quote! { #name.metal_type() }))
                    },
                    MetalTemplateParameterType::Value(ty) => {
                        let ty: Type = syn::parse_str(ty.as_ref())?;
                        Ok((quote! { #[allow(non_snake_case)] #name: #ty }, quote! { #name.to_string() }))
                    },
                }
            })
            .collect::<anyhow::Result<_>>()?
    } else {
        (Vec::new(), Vec::new())
    };

    let entry_name = dynamic_mangle(kernel.name.as_ref(), variants_kernel_format);

    let base_index = specialize_indices.get(&kernel.name).copied();
    let (specialize_args, specialize_setup): (Vec<TokenStream>, Vec<TokenStream>) = kernel
        .arguments
        .iter()
        .filter(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Specialize(_))))
        .enumerate()
        .map(|(i, a)| {
            let arg_name = format_ident!("{}", a.name.as_ref());
            let rust_type = match a.argument_type().unwrap() {
                MetalArgumentType::Specialize(t) => format_ident!("{t}"),
                _ => unreachable!(),
            };
            let idx = base_index.unwrap_or(0) + i;
            let arg_def = quote! { #arg_name: #rust_type };
            let setup = quote! {
                function_constants.set_value(&#arg_name, #idx);
            };
            (arg_def, setup)
        })
        .unzip();

    let has_specialize = !specialize_args.is_empty();
    let function_constants_init = if has_specialize {
        quote! {
            let function_constants = MTLFunctionConstantValues::new();
            #(#specialize_setup)*
        }
    } else {
        quote! {}
    };
    let function_constants_arg = if has_specialize {
        quote! { Some(&function_constants) }
    } else {
        quote! { None }
    };

    let mut arg_count: usize = 0;
    let mut indirect_flag = false;

    let (conditional_buffer_fields,conditional_buffer_sets, encode_generics, encode_args_defs, encode_args_sets, encode_args_names): (
        Vec<Option<TokenStream>>,
        Vec<Option<TokenStream>>,
        Vec<Option<TokenStream>>,
        Vec<TokenStream>,
        Vec<TokenStream>,
        Vec<TokenStream>,
    ) = kernel
        .arguments
        .iter()
        .filter_map(|ka| {
            let arg_name = format_ident!("{}", ka.name.as_ref());

            match ka.argument_type().unwrap() {
                arg_type @ (MetalArgumentType::Buffer | MetalArgumentType::Constant(_)) => {
                    let (mut ty, mut set, generic) = match arg_type {
                        MetalArgumentType::Buffer => {
                            let buffer_lifetime = Lifetime::new(&format!("'{}", ka.name.as_ref()), Span::call_site());
                            (
                                quote! { impl crate::backends::common::kernel::BufferArg<#buffer_lifetime, Retained<ProtocolObject<dyn MTLBuffer>>> },
                                quote! {
                                    let (__dsl_buffer, __dsl_offset) = #arg_name.into_parts();
                                    compute_encoder.set_buffer(Some(__dsl_buffer), __dsl_offset, #arg_count);
                                },
                                Some(quote! { #buffer_lifetime }),
                            )
                        },
                        MetalArgumentType::Constant((r_type, constant_type)) => {
                            let arg_dtype: Type = syn::parse_str(&r_type).unwrap();
                            match constant_type {
                                MetalConstantType::Scalar => (
                                    quote! { #arg_dtype },
                                    quote! { compute_encoder.set_value(&#arg_name, #arg_count); },
                                    None,
                                ),
                                MetalConstantType::Array => (
                                    quote! { &[#arg_dtype] },
                                    quote! { compute_encoder.set_slice(#arg_name, #arg_count); },
                                    None,
                                ),
                            }
                        },
                        _ => unreachable!(),
                    };

                    let (conditional_buffer_field, conditional_buffer_set) =
                        if let Some(condition) = ka.argument_condition().unwrap() {
                            let conditional_field_name = format_ident!("has_{}", ka.name.as_ref());
                            let condition = parse_expr(condition.as_ref()).unwrap();

                            ty = quote! { Option<#ty> };
                            set = quote! {
                                assert!(#arg_name.is_some() == (self.#conditional_field_name));
                                if let Some(#arg_name) = #arg_name {
                                    #set
                                }
                            };

                            (
                                Some(quote! { #conditional_field_name: bool }),
                                Some(quote! { #conditional_field_name: #condition }),
                            )
                        } else {
                            (None, None)
                        };

                    arg_count += 1;

                    Some((
                        conditional_buffer_field,
                        conditional_buffer_set,
                        generic,
                        quote! { #arg_name: #ty },
                        set,
                        quote! { #arg_name },
                    ))
                }
                MetalArgumentType::Groups(MetalGroupsType::Indirect) if !indirect_flag => {
                    indirect_flag = true;

                    Some((None, None, Some(quote! { '__dsl_indirect_dispatch_buffer }), quote! { __dsl_indirect_dispatch_buffer: impl crate::backends::common::kernel::BufferArg<'__dsl_indirect_dispatch_buffer, Retained<ProtocolObject<dyn MTLBuffer>>> }, quote! {}, quote! { __dsl_indirect_dispatch_buffer }))
                }
                _ => None,
            }
        })
        .multiunzip();

    let conditional_buffer_fields = conditional_buffer_fields.into_iter().flatten().collect::<Vec<_>>();
    let conditional_buffer_sets = conditional_buffer_sets.into_iter().flatten().collect::<Vec<_>>();
    let encode_generics = encode_generics.into_iter().flatten().collect::<Vec<_>>();

    let (dispatch, elements) = if kernel.has_axis() {
        if kernel.has_groups() || kernel.has_threads() {
            anyhow::bail!("mixing groups/threads and axis is not supported");
        }

        let mut axis = kernel
            .arguments
            .iter()
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Axis(threads_rexprs, threads_per_group_rexprs)) => {
                    Some((threads_rexprs, threads_per_group_rexprs))
                },
                _ => None,
            })
            .map(|(threads_rexprs, threads_per_group_rexprs)| {
                let threads = parse_expr(&threads_rexprs)?;
                let threads_per_group = parse_expr(&threads_per_group_rexprs)?;
                Ok((threads, threads_per_group))
            })
            .collect::<anyhow::Result<Vec<(TokenStream, TokenStream)>>>()?;
        axis.extend(repeat_n((quote! {1}, quote! {1}), 3 - axis.len()));

        let (threads, threads_per_group): (Vec<TokenStream>, Vec<TokenStream>) = axis.into_iter().unzip();

        (
            quote! {
                compute_encoder.dispatch_threads(
                    MTLSize::new(#((#threads) as usize, )*),
                    MTLSize::new(#((#threads_per_group) as usize, )*),
                );
            },
            threads.into_iter().chain(threads_per_group.into_iter()),
        )
    } else {
        let mut threads = kernel
            .arguments
            .iter()
            .filter_map(|a| {
                if let Ok(MetalArgumentType::Threads(rexprs)) = a.argument_type() {
                    Some(parse_expr(&rexprs))
                } else {
                    None
                }
            })
            .collect::<anyhow::Result<Vec<TokenStream>>>()?;
        threads.extend(repeat_n(quote! {1}, 3 - threads.len()));

        if kernel.has_groups_indirect() {
            if kernel.has_groups_direct() {
                anyhow::bail!("cannot mix indirect and direct groups");
            }

            (
                quote! {
                    let (__dsl_buffer, __dsl_offset) = __dsl_indirect_dispatch_buffer.into_parts();
                    compute_encoder.dispatch_threadgroups_indirect(
                        __dsl_buffer,
                        __dsl_offset,
                        MTLSize::new(#((#threads) as usize, )*),
                    );
                },
                vec![].into_iter().chain(threads.into_iter()),
            )
        } else {
            let mut groups = kernel
                .arguments
                .iter()
                .filter_map(|a| {
                    if let Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(rexprs))) = a.argument_type() {
                        Some(parse_expr(&rexprs))
                    } else {
                        None
                    }
                })
                .collect::<anyhow::Result<Vec<TokenStream>>>()?;
            groups.extend(repeat_n(quote! {1}, 3 - groups.len()));

            (
                quote! {
                    compute_encoder.dispatch_threadgroups(
                        MTLSize::new(#((#groups) as usize, )*),
                        MTLSize::new(#((#threads) as usize, )*),
                    );
                },
                groups.into_iter().chain(threads.into_iter()),
            )
        }
    };

    let guards = elements
        .flat_map(|e| {
            if let Ok(lit_int) = syn::parse2::<LitInt>(e.clone())
                && let Ok(int) = lit_int.base10_parse::<u32>()
                && int != 0
            {
                None
            } else {
                Some(quote! { (#e) == 0 })
            }
        })
        .fold(quote! {}, |a, b| {
            if !a.is_empty() && !b.is_empty() {
                quote! {#a || #b}
            } else {
                quote! {#a #b}
            }
        });

    let empty_dispatch_guards = if !guards.is_empty() {
        quote! { if #guards { return; }; }
    } else {
        quote! {}
    };

    let kernel = quote! {
        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
            #(#conditional_buffer_fields,)*
        }

        impl crate::backends::common::kernel::#trait_name for #struct_name {
            type Backend = crate::backends::metal::Metal;

            fn new(context: &MetalContext #(, #variants_extra_arguments)* #(, #specialize_args)*) -> Result<Self, MetalError> {
                #function_constants_init
                let pipeline = context.compute_pipeline_state(&#entry_name, #function_constants_arg)?;
                Ok(Self { pipeline #(, #conditional_buffer_sets)* })
            }

            fn encode<#(#encode_generics, )* 'encoder>(&self, #(#encode_args_defs, )* compute_encoder: &'encoder ProtocolObject<dyn MTLComputeCommandEncoder>) {
                #empty_dispatch_guards
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_args_sets)*
                #dispatch
            }
            fn encode_if<#(#encode_generics, )* 'encoder, 'predicate>(&self, #(#encode_args_defs, )* compute_encoder: &'encoder ProtocolObject<dyn MTLComputeCommandEncoder>, predicate: Option<impl crate::backends::common::kernel::BufferArg<'predicate, Retained<ProtocolObject<dyn MTLBuffer>>>>) {
                #empty_dispatch_guards

                if let Some(predicate) = predicate {
                    let (__dsl_buffer, __dsl_offset) = predicate.into_parts();
                    compute_encoder.condition(
                        __dsl_buffer,
                        __dsl_offset,
                        || {
                            self.encode(#(#encode_args_names, )* compute_encoder);
                        },
                        None::<fn()>,
                    );
                } else {
                    self.encode(#(#encode_args_names, )* compute_encoder);
                }
            }
        }
    };

    let associated_type = quote! { type #trait_name = #struct_name; };

    Ok((kernel, associated_type))
}
