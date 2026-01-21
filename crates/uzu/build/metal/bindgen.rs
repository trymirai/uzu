use std::iter::repeat_n;

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::ast::{MetalArgumentType, MetalKernelInfo};

pub fn bindgen(kernel: &MetalKernelInfo) -> anyhow::Result<TokenStream> {
    let kernel_name = kernel.name.as_ref();
    let struct_name = format_ident!("{kernel_name}Kernel");

    let parse_expr = |expr: &Box<str>| -> anyhow::Result<TokenStream> {
        syn::parse_str(expr.as_ref()).with_context(|| {
            format!(
                "cannot parse rust expression `{}` in kernel `{}`",
                expr, kernel_name
            )
        })
    };

    let (specialize_extra_argument, specialize_kernel_format) = if kernel
        .specializations
        .is_some()
    {
        (
            quote! { , data_type: crate::backends::metal::KernelDataType },
            quote! { &format!("{}_{}", #kernel_name, data_type.function_name_suffix()) },
        )
    } else {
        (quote! {}, quote! { #kernel_name })
    };

    let (encode_args_defs, encode_args_sets, encode_args_names): (
        Vec<TokenStream>,
        Vec<TokenStream>,
        Vec<TokenStream>,
    ) = kernel
        .arguments
        .iter()
        .filter(|k| {
            matches!(
                k.argument_type(),
                Ok(MetalArgumentType::Buffer | MetalArgumentType::Constant(_))
            )
        })
        .enumerate()
        .map(|(i, ka)| {
            let arg_index = i as u64;
            let arg_name = format_ident!("{}", ka.name.as_ref());

            match ka.argument_type().unwrap() {
                MetalArgumentType::Buffer => {
                    let def = quote! { #arg_name: crate::backends::metal::BufferRef<'_> };
                    let set = quote! {
                        compute_encoder.set_buffer(#arg_index, Some(#arg_name), 0);
                    };

                    (def, set, quote! { #arg_name })
                },
                MetalArgumentType::Constant(r_type) => {
                    let arg_dtype = format_ident!("{r_type}");
                    let def = quote! { #arg_name: #arg_dtype };
                    let set = quote! {
                        compute_encoder.set_bytes(#arg_index, std::mem::size_of::<#arg_dtype>() as u64, std::ptr::addr_of!(#arg_name).cast::<std::ffi::c_void>());
                    };

                    (def, set, quote! { #arg_name })
                }
                _ => unreachable!(),
            }
        })
        .multiunzip();

    let dispatch = if kernel.has_axis() {
        if kernel.has_groups() || kernel.has_threads() {
            anyhow::bail!("mixing groups/threads and axis is not supported");
        }

        let mut axis = kernel
            .arguments
            .iter()
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Axis(
                    threads_rexprs,
                    threads_per_group_rexprs,
                )) => Some((threads_rexprs, threads_per_group_rexprs)),
                _ => None,
            })
            .map(|(threads_rexprs, threads_per_group_rexprs)| {
                let threads = parse_expr(&threads_rexprs)?;
                let threads_per_group = parse_expr(&threads_per_group_rexprs)?;
                Ok((threads, threads_per_group))
            })
            .collect::<anyhow::Result<Vec<(TokenStream, TokenStream)>>>()?;
        axis.extend(repeat_n((quote! {1}, quote! {1}), 3 - axis.len()));

        let (threads, threads_per_group): (Vec<TokenStream>, Vec<TokenStream>) =
            axis.into_iter().unzip();

        quote! {
            compute_encoder.dispatch_threads(
                crate::backends::metal::mtl_size(#((#threads) as u64, )*),
                crate::backends::metal::mtl_size(#((#threads_per_group) as u64, )*),
            );
        }
    } else {
        let mut groups = kernel
            .arguments
            .iter()
            .filter_map(|a| {
                if let Ok(MetalArgumentType::Groups(rexprs)) = a.argument_type()
                {
                    Some(parse_expr(&rexprs))
                } else {
                    None
                }
            })
            .collect::<anyhow::Result<Vec<TokenStream>>>()?;
        groups.extend(repeat_n(quote! {1}, 3 - groups.len()));

        let mut threads = kernel
            .arguments
            .iter()
            .filter_map(|a| {
                if let Ok(MetalArgumentType::Threads(rexprs)) =
                    a.argument_type()
                {
                    Some(parse_expr(&rexprs))
                } else {
                    None
                }
            })
            .collect::<anyhow::Result<Vec<TokenStream>>>()?;
        threads.extend(repeat_n(quote! {1}, 3 - threads.len()));

        quote! {
            compute_encoder.dispatch_thread_groups(
                crate::backends::metal::mtl_size(#((#groups) as u64, )*),
                crate::backends::metal::mtl_size(#((#threads) as u64, )*),
            );
        }
    };

    Ok(quote! {
        pub struct #struct_name {
            pipeline: crate::backends::metal::ComputePipelineState,
        }

        impl #struct_name {
            pub fn new(context: &crate::backends::metal::MTLContext #specialize_extra_argument) -> Result<Self, crate::backends::metal::MTLError> {
                use crate::backends::metal::metal_extensions::LibraryPipelineExtensions;
                let pipeline = context.library.compute_pipeline_state(#specialize_kernel_format, None)?;
                Ok(Self { pipeline })
            }

            pub fn encode(&self, #(#encode_args_defs, )*compute_encoder: crate::backends::metal::ComputeCommandEncoderRef<'_>) {
                use crate::backends::metal::ComputeEncoderLegacy;
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_args_sets)*
                #dispatch
            }

            pub fn encode_if(&self, #(#encode_args_defs, )*compute_encoder: crate::backends::metal::ComputeCommandEncoderRef<'_>, predicate: Option<crate::backends::metal::BufferRef<'_>>) {
                use crate::backends::metal::metal_extensions::ComputeEncoderConditional;
                compute_encoder.condition(
                    predicate,
                    || {
                        self.encode(#(#encode_args_names, )*compute_encoder);
                    },
                    None::<fn()>,
                );
            }
        }
    })
}
