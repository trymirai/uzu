use std::iter::repeat_n;

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, LitInt, Type};

use super::{
    ast::{MetalArgumentType, MetalConstantType, MetalKernelInfo},
    wrapper::SpecializeBaseIndices,
};
use crate::{
    common::mangling::dynamic_mangle,
    metal::ast::{MetalBufferAccess, MetalGroupsType, MetalTemplateParameterType},
};

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
) -> anyhow::Result<(TokenStream, Option<TokenStream>)> {
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
    let (specialize_args, (specialize_setup, specialize_arg_names)): (
        Vec<TokenStream>,
        (Vec<TokenStream>, Vec<Ident>),
    ) = kernel
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
            (arg_def, (setup, arg_name))
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
    let cache_key = if has_specialize {
        let format_str = format!("{{}}{}", repeat_n("_{}", specialize_arg_names.len()).join(""));
        quote! { &format!(#format_str, &entry_name #(, #specialize_arg_names)*) }
    } else {
        quote! { &entry_name }
    };

    let mut arg_count: usize = 0;
    let mut indirect_flag = false;

    let (conditional_buffer_fields,conditional_buffer_sets, encode_args_defs, encode_deconstructs, encode_accesses, encode_args_sets): (
        Vec<Option<TokenStream>>,
        Vec<Option<TokenStream>>,
        Vec<TokenStream>,
        Vec<Option<TokenStream>>,
        Vec<Option<TokenStream>>,
        Vec<TokenStream>,
    ) = kernel
        .arguments
        .iter()
        .filter_map(|ka| {
            let arg_name = format_ident!("{}", ka.name.as_ref());

            match ka.argument_type().unwrap() {
                arg_type @ (MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_)) => {
                    let (mut ty, deconstruct, access_type, mut set) = match arg_type {
                        MetalArgumentType::Buffer(access) => {
                            (
                                match access {
                                    MetalBufferAccess::Read => {
                                        quote! { &crate::backends::common::Allocation<crate::backends::metal::Metal> }
                                    },
                                    MetalBufferAccess::ReadWrite => {
                                        quote! { &mut crate::backends::common::Allocation<crate::backends::metal::Metal> }
                                    },
                                },
                                Some(if ka.argument_condition().unwrap().is_some() {
                                    quote! {
                                        let #arg_name = #arg_name.map(|#arg_name| #arg_name.as_buffer_range());
                                    }
                                } else {
                                    quote! {
                                        let #arg_name = #arg_name.as_buffer_range();
                                    }
                                }),
                                Some(access),
                                quote! {
                                    compute_encoder.set_buffer(Some(#arg_name.0), #arg_name.1.start, #arg_count);
                                },
                            )
                        },
                        MetalArgumentType::Constant((r_type, constant_type)) => {
                            let arg_dtype: Type = syn::parse_str(&r_type).unwrap();
                            match constant_type {
                                MetalConstantType::Scalar => (
                                    quote! { #arg_dtype },
                                    None,
                                    None,
                                    quote! { compute_encoder.set_value(&#arg_name, #arg_count); },
                                ),
                                MetalConstantType::Array => (
                                    quote! { &[#arg_dtype] },
                                    None,
                                    None,
                                    quote! { compute_encoder.set_slice(#arg_name, #arg_count); },
                                ),
                            }
                        },
                        _ => unreachable!(),
                    };

                    let mut access = access_type.map(|access| {
                        let compute_write = matches!(access, MetalBufferAccess::ReadWrite);

                        quote! {
                            crate::backends::common::Access {
                                range: #arg_name.0.gpu_address_subrange((#arg_name.1.start)..(#arg_name.0.length())),
                                flags: crate::backends::common::AccessFlags {
                                    compute_read: true,
                                    compute_write: #compute_write,
                                    copy_read: false,
                                    copy_write: false,
                                },
                            }
                        }
                    });

                    let (conditional_buffer_field, conditional_buffer_set) =
                        if let Some(condition) = ka.argument_condition().unwrap() {
                            let conditional_field_name = format_ident!("has_{}", ka.name.as_ref());
                            let condition = parse_expr(condition.as_ref()).unwrap();

                            ty = quote! { Option<#ty> };
                            access = access.map(|access| quote! { #arg_name.as_ref().map(|#arg_name| #access)});
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
                            access = access.map(|access| quote! { Some(#access)});

                            (None, None)
                        };

                    arg_count += 1;

                    Some((
                        conditional_buffer_field,
                        conditional_buffer_set,
                        quote! { #arg_name: #ty },
                        deconstruct,
                        access,
                        set,
                    ))
                }
                MetalArgumentType::Groups(MetalGroupsType::Indirect) if !indirect_flag => {
                    indirect_flag = true;

                    Some((
                        None,
                        None,
                        quote! { __dsl_indirect_dispatch_buffer: &crate::backends::common::Allocation<crate::backends::metal::Metal> },
                        Some(quote! { let __dsl_indirect_dispatch_buffer = __dsl_indirect_dispatch_buffer.as_buffer_range(); }),
                        Some(quote! {
                            Some(crate::backends::common::Access {
                                range: __dsl_indirect_dispatch_buffer.0.gpu_address_subrange(
                                    (__dsl_indirect_dispatch_buffer.1.start)..(__dsl_indirect_dispatch_buffer.1.start + 12)
                                ),
                                flags: crate::backends::common::AccessFlags::compute_read(),
                            })
                        }),
                        quote! {},
                    ))
                }
                _ => None,
            }
        })
        .multiunzip();

    let conditional_buffer_fields = conditional_buffer_fields.into_iter().flatten().collect::<Vec<_>>();
    let conditional_buffer_sets = conditional_buffer_sets.into_iter().flatten().collect::<Vec<_>>();

    let encode_deconstructs = encode_deconstructs.into_iter().flatten().collect::<Vec<_>>();

    let encode_accesses = encode_accesses.into_iter().flatten().collect::<Vec<_>>();
    let encode_accesses = if !encode_accesses.is_empty() {
        let encode_accesses = encode_accesses.iter().fold(quote! {}, |a, b| {
            if !a.is_empty() && !b.is_empty() {
                quote! {#a , #b}
            } else {
                quote! {#a #b}
            }
        });
        quote! { encoder.access(&[#encode_accesses].into_iter().flatten().collect::<Vec<_>>()); }
    } else {
        quote! {}
    };

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
                    compute_encoder.dispatch_threadgroups_indirect(
                        __dsl_indirect_dispatch_buffer.0,
                        __dsl_indirect_dispatch_buffer.1.start,
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

    let (maybe_trait_impl, maybe_associate_backend, associated_type) = if kernel.public {
        (
            quote! {crate::backends::common::kernel::#trait_name for},
            quote! { type Backend = crate::backends::metal::Metal; },
            Some(quote! { type #trait_name = #struct_name; }),
        )
    } else {
        (quote! {}, quote! {}, None)
    };

    let method_visibility = if kernel.public {
        quote! {}
    } else {
        quote! { pub(crate) }
    };

    let kernel = quote! {
        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
            #(#conditional_buffer_fields,)*
        }

        impl #maybe_trait_impl #struct_name {
            #maybe_associate_backend

            #method_visibility fn new(context: &MetalContext #(, #variants_extra_arguments)* #(, #specialize_args)*) -> Result<Self, MetalError> {
                let entry_name = #entry_name;
                #function_constants_init
                let pipeline = context.compute_pipeline_state(#cache_key, &entry_name, #function_constants_arg)?;
                Ok(Self { pipeline #(, #conditional_buffer_sets)* })
            }

            #method_visibility fn encode(&self, #(#encode_args_defs, )* encoder: &mut crate::backends::common::Encoder<crate::backends::metal::Metal>) {
                #empty_dispatch_guards
                #(#encode_deconstructs)*
                #encode_accesses
                let compute_encoder = encoder.as_command_buffer_mut().ensure_compute();
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_args_sets)*
                #dispatch
            }
        }
    };

    Ok((kernel, associated_type))
}
