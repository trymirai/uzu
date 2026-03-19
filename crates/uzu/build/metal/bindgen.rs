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

    let (conditional_buffer_fields,conditional_buffer_sets, encode_generics, encode_args_defs, encode_deconstructs, encode_accesses, encode_args_sets): (
        Vec<Option<TokenStream>>,
        Vec<Option<TokenStream>>,
        Vec<Option<TokenStream>>,
        Vec<TokenStream>,
        Vec<TokenStream>,
        Vec<TokenStream>,
        Vec<TokenStream>,
    ) = kernel
        .arguments
        .iter()
        .filter_map(|ka| {
            let arg_name = format_ident!("{}", ka.name.as_ref());
            let arg_deconstructed_name = format_ident!("__dsl_arg_deconstructed_{}", ka.name.as_ref());

            match ka.argument_type().unwrap() {
                arg_type @ (MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_)) => {
                    let (mut ty, mut deconstruct, mut access, mut set, generic) = match arg_type {
                        MetalArgumentType::Buffer(access) => {
                            let buffer_lifetime = Lifetime::new(&format!("'{}", ka.name.as_ref()), Span::call_site());
                            let compute_write = matches!(access, MetalBufferAccess::ReadWrite);

                            (
                                match access {
                                    MetalBufferAccess::Read => quote! { impl crate::backends::common::kernel::BufferArg<#buffer_lifetime, Retained<ProtocolObject<dyn MTLBuffer>>> },
                                    MetalBufferAccess::ReadWrite => quote! { impl crate::backends::common::kernel::BufferArgMut<#buffer_lifetime, Retained<ProtocolObject<dyn MTLBuffer>>> },
                                },
                                quote! {
                                    let #arg_deconstructed_name = #arg_name.into_parts();
                                    let #arg_deconstructed_name = #arg_deconstructed_name.0.gpu_address_subrange((#arg_deconstructed_name.1)..(#arg_deconstructed_name.0.length()));
                                },
                                quote! {
                                    crate::backends::common::Access {
                                        range: #arg_deconstructed_name.clone(),
                                        flags: crate::backends::common::AccessFlags {
                                            compute_read: true,
                                            compute_write: #compute_write,
                                            copy_read: false,
                                            copy_write: false,
                                        },
                                    }
                                },
                                quote! {
                                    __dsl_argument_table.set_address_at_index(#arg_deconstructed_name.start as u64, #arg_count);
                                },
                                Some(quote! { #buffer_lifetime }),
                            )
                        },
                        MetalArgumentType::Constant((r_type, constant_type)) => {
                            let arg_dtype: Type = syn::parse_str(&r_type).unwrap();
                            match constant_type {
                                MetalConstantType::Scalar => (
                                    quote! { #arg_dtype },
                                    quote! {
                                        let #arg_deconstructed_name = encoder.allocate_constant(std::mem::size_of::<#arg_dtype>()).unwrap();
                                        let #arg_deconstructed_name = #arg_deconstructed_name.as_buffer_range();
                                        unsafe {
                                            *(#arg_deconstructed_name.0.cpu_ptr().as_ptr().byte_add(#arg_deconstructed_name.1.start) as *mut #arg_dtype) = #arg_name;
                                        };
                                        let #arg_deconstructed_name = #arg_deconstructed_name.0.gpu_address_subrange(#arg_deconstructed_name.1);
                                    },
                                    quote! {
                                        crate::backends::common::Access {
                                            range: #arg_deconstructed_name.clone(),
                                            flags: crate::backends::common::AccessFlags::compute_read(),
                                        }
                                    },
                                    quote! {
                                        __dsl_argument_table.set_address_at_index(#arg_deconstructed_name.start as u64, #arg_count);
                                    },
                                    None,
                                ),
                                MetalConstantType::Array => (
                                    quote! { &[#arg_dtype] },
                                    quote! {
                                        let #arg_deconstructed_name = encoder.allocate_constant(#arg_name.len() * std::mem::size_of::<#arg_dtype>()).unwrap();
                                        let #arg_deconstructed_name = #arg_deconstructed_name.as_buffer_range();
                                        unsafe {
                                            std::slice::from_raw_parts_mut::<#arg_dtype>(#arg_deconstructed_name.0.cpu_ptr().as_ptr().byte_add(#arg_deconstructed_name.1.start) as *mut #arg_dtype, #arg_name.len()).copy_from_slice(#arg_name);
                                        };
                                        let #arg_deconstructed_name = #arg_deconstructed_name.0.gpu_address_subrange(#arg_deconstructed_name.1);
                                    },
                                    quote! {
                                        crate::backends::common::Access {
                                            range: #arg_deconstructed_name.clone(),
                                            flags: crate::backends::common::AccessFlags::compute_read(),
                                        }
                                    },
                                    quote! {
                                        __dsl_argument_table.set_address_at_index(#arg_deconstructed_name.start as u64, #arg_count);
                                    },
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
                            deconstruct = quote! {
                                let #arg_deconstructed_name = #arg_name.map(|#arg_name| {
                                    #deconstruct
                                    #arg_deconstructed_name
                                });
                            };
                            access = quote! { #arg_deconstructed_name.as_ref().map(|#arg_deconstructed_name| #access)};
                            set = quote! {
                                assert!(#arg_deconstructed_name.is_some() == (self.#conditional_field_name));
                                if let Some(#arg_deconstructed_name) = #arg_deconstructed_name {
                                    #set
                                }
                            };

                            (
                                Some(quote! { #conditional_field_name: bool }),
                                Some(quote! { #conditional_field_name: #condition }),
                            )
                        } else {
                            access = quote! { Some(#access) };

                            (None, None)
                        };

                    arg_count += 1;

                    Some((
                        conditional_buffer_field,
                        conditional_buffer_set,
                        generic,
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
                        Some(quote! { '__dsl_indirect_dispatch_buffer }),
                        quote! { __dsl_indirect_dispatch_buffer: impl crate::backends::common::kernel::BufferArg<'__dsl_indirect_dispatch_buffer, Retained<ProtocolObject<dyn MTLBuffer>>> },
                        quote! { let __dsl_indirect_dispatch_buffer = __dsl_indirect_dispatch_buffer.into_parts(); },
                        quote! {
                            Some(crate::backends::common::Access {
                                range: __dsl_indirect_dispatch_buffer.0.gpu_address_subrange((__dsl_indirect_dispatch_buffer.1)..(__dsl_indirect_dispatch_buffer.1+12)),
                                flags: crate::backends::common::AccessFlags::compute_read(),
                            })
                        },
                        quote! {},
                    ))
                }
                _ => None,
            }
        })
        .multiunzip();

    let conditional_buffer_fields = conditional_buffer_fields.into_iter().flatten().collect::<Vec<_>>();
    let conditional_buffer_sets = conditional_buffer_sets.into_iter().flatten().collect::<Vec<_>>();
    let encode_generics = encode_generics.into_iter().flatten().collect::<Vec<_>>();

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
                command_encoder.dispatch_threads_threads_per_threadgroup(
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
                    command_encoder.dispatch_threadgroups_with_indirect_buffer_threads_per_threadgroup(
                        __dsl_indirect_dispatch_buffer.0.gpu_address() + __dsl_indirect_dispatch_buffer.1 as u64,
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
                    command_encoder.dispatch_threadgroups_threads_per_threadgroup(
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

    let max_buffers = encode_args_sets.len();

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

            #method_visibility fn encode<#(#encode_generics, )* 'encoder>(&self, #(#encode_args_defs, )* encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::metal::Metal>) {
                #empty_dispatch_guards
                #(#encode_deconstructs)*
                #encode_accesses
                let command_encoder = encoder.as_command_buffer_mut().command_encoder();
                command_encoder.set_compute_pipeline_state(&self.pipeline);
                let __dsl_argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
                __dsl_argument_table_descriptor.set_max_buffer_bind_count(#max_buffers);
                let __dsl_argument_table = self.pipeline.device().new_argument_table_with_descriptor(&__dsl_argument_table_descriptor).unwrap();
                #(#encode_args_sets)*
                command_encoder.set_argument_table(Some(&__dsl_argument_table));
                #dispatch
            }
        }
    };

    Ok((kernel, associated_type))
}
