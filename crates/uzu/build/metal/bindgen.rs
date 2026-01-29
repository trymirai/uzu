use std::iter::repeat_n;

use anyhow::Context;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::LitInt;

use super::{
    ast::{MetalArgumentType, MetalConstantType, MetalKernelInfo},
    wrapper::SpecializeBaseIndices,
};

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
) -> anyhow::Result<TokenStream> {
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

    let (variants_extra_arguments, variants_kernel_format) = if let Some(
        variants,
    ) =
        &kernel.variants
    {
        let variant_names = variants
            .iter()
            .map(|type_parameter| {
                format_ident!("{}", type_parameter.name.as_ref())
            })
            .collect::<Vec<_>>();

        let kernel_format = repeat_n("{}", variant_names.len() + 1).join("_");

        (
            variant_names
                .iter()
                .map(|name| quote! { #[allow(non_snake_case)] #name: KernelDataType })
                .collect(),
            quote! { &format!(#kernel_format, #kernel_name #(, #variant_names.function_name_suffix())*) },
        )
    } else {
        (Vec::new(), quote! { #kernel_name })
    };

    let base_index = specialize_indices.get(&kernel.name).copied();
    let (specialize_args, specialize_setup): (
        Vec<TokenStream>,
        Vec<TokenStream>,
    ) = kernel
        .arguments
        .iter()
        .filter(|a| {
            matches!(a.argument_type(), Ok(MetalArgumentType::Specialize(_)))
        })
        .enumerate()
        .map(|(i, a)| {
            let arg_name = format_ident!("{}", a.name.as_ref());
            let rust_type = match a.argument_type().unwrap() {
                MetalArgumentType::Specialize(t) => format_ident!("{t}"),
                _ => unreachable!(),
            };
            let mtl_type = match a.argument_type().unwrap() {
                MetalArgumentType::Specialize(ref t) => match t.as_ref() {
                    "bool" => quote! { Bool },
                    "u32" => quote! { UInt },
                    "i32" => quote! { Int },
                    "f32" => quote! { Float },
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let idx = base_index.unwrap_or(0) + i;
            let arg_def = quote! { #arg_name: #rust_type };
            let setup = quote! {
                function_constants.set_constant_value_type_at_index(
                    std::ptr::NonNull::from(&#arg_name).cast(),
                    MTLDataType::#mtl_type,
                    #idx,
                );
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
            let arg_index = i;
            let arg_name = format_ident!("{}", ka.name.as_ref());

            match ka.argument_type().unwrap() {
                MetalArgumentType::Buffer => {
                    let def = quote! { #arg_name: &ProtocolObject<dyn MTLBuffer> };
                    let set = quote! {
                        compute_encoder.set_buffer(Some(#arg_name), 0, #arg_index);
                    };

                    (def, set, quote! { #arg_name })
                }
                MetalArgumentType::Constant((r_type, constant_type)) => {
                    let arg_dtype = format_ident!("{r_type}");

                    let (def, set) = match constant_type {
                        MetalConstantType::Scalar => (
                            quote! { #arg_name: #arg_dtype },
                            quote! { compute_encoder.set_value(&#arg_name, #arg_index); },
                        ),
                        MetalConstantType::Array => (
                            quote! { #arg_name: &[#arg_dtype] },
                            quote! { compute_encoder.set_slice(#arg_name, #arg_index); },
                        ),
                    };

                    (def, set, quote! { #arg_name })
                }
                _ => unreachable!(),
            }
        })
        .multiunzip();

    let (dispatch, elements) = if kernel.has_axis() {
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

        (
            quote! {
                compute_encoder.dispatch_threadgroups(
                    MTLSize::new(#((#groups) as usize, )*),
                    MTLSize::new(#((#threads) as usize, )*),
                );
            },
            groups.into_iter().chain(threads.into_iter()),
        )
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

    Ok(quote! {
        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        }

        impl #struct_name {
            pub fn new(context: &MTLContext #(, #variants_extra_arguments)* #(, #specialize_args)*) -> Result<Self, MTLError> {
                #function_constants_init
                let pipeline = context.library.compute_pipeline_state(#variants_kernel_format, #function_constants_arg)?;
                Ok(Self { pipeline })
            }

            pub fn encode(&self, #(#encode_args_defs, )* compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
                #empty_dispatch_guards
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_args_sets)*
                #dispatch
            }

            pub fn encode_if(&self, #(#encode_args_defs, )* compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, predicate: Option<&ProtocolObject<dyn MTLBuffer>>) {
                #empty_dispatch_guards
                compute_encoder.condition(
                    predicate,
                    || {
                        self.encode(#(#encode_args_names, )* compute_encoder);
                    },
                    None::<fn()>,
                );
            }
        }
    })
}
